# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import datetime
import json
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import override

import eccodes
import netCDF4
import numpy as np
import zarr
from numpy.typing import NDArray

from weathergen.common.config import parse_timedelta
from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon
from weathergen.datasets.data_reader_base import (
    NPDT64,
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)

# Native cadence: one 3 h tp accumulation per message, 8 per daily file.
ICON_DREAM_PERIOD = np.timedelta64(3, "h")

# The only channel served. tp (GRIB shortName tp, surface, stepRange 0-3, kg m-2) is a 3 h
# accumulation valid at the END of its window; offset-index key is (dataTime, "tp", 0, "3").
ICON_DREAM_CHANNELS: list[str] = ["tp"]
_TP_KEY = ("tp", 0, "3")

_TP_UNITS_SCALE = {"mm": 1.0, "m": 1e-3}  # native kg m-2 == mm

# EXACT tp statistics in mm, keyed by (number of 3 h messages accumulated, grid served):
# every 3 h field of 2016-01-01..2022-12-31 (no negatives, no NaN). Scaled to `tp_units`.
#   native: all 2,949,120 R03B07 cells (Jupiter job 2215775)
#   n320:   remapped onto the 542,080-point N320 grid of the anemoi stores (job 2216380);
#           6 h max 464 mm, 58.4% below 0.1 mm
_TP_STATS_MM: dict[tuple[int, str], tuple[float, float]] = {
    (1, "native"): (0.37424, 1.24527),
    (2, "native"): (0.74847, 2.27550),
    (1, "n320"): (0.37441, 1.20978),
    (2, "n320"): (0.74882, 2.22578),
}


def _unit_xyz(lat_deg: NDArray, lon_deg: NDArray) -> NDArray[np.float64]:
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def nearest_on_reduced_gaussian(
    lat_q: NDArray, lon_q: NDArray, lat_g: NDArray, lon_g: NDArray
) -> NDArray[np.int64]:
    """
    Exact nearest grid point (great circle) for query points on a reduced Gaussian grid.

    The grid must be stored row by row (constant latitude per row), each row holding points
    equally spaced in longitude from the row's first longitude -- the layout of the anemoi
    N320 stores. Within one row the nearest point is the one nearest in longitude, so only
    that point of each of the 4 rows bracketing the query latitude is a candidate.
    """
    lat_g = np.asarray(lat_g, dtype=np.float64)
    lon_g = np.asarray(lon_g, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, lat_g[1:] != lat_g[:-1]])
    counts = np.diff(np.r_[starts, len(lat_g)])
    row_lat = lat_g[starts]
    row_lon0 = lon_g[starts]
    if len(np.unique(row_lat)) != len(row_lat):
        raise ValueError("Grid is not stored row by row (a latitude appears in two runs).")
    for s, n in zip(
        starts[[0, len(starts) // 2, -1]], counts[[0, len(starts) // 2, -1]], strict=True
    ):
        if not np.allclose(np.diff(lon_g[s : s + n]) % 360.0, 360.0 / n, atol=1e-6):
            raise ValueError("Grid rows are not equally spaced in longitude.")

    # rows sorted by latitude for searchsorted, mapped back to storage order
    order = np.argsort(row_lat)
    lat_sorted = row_lat[order]
    j = np.searchsorted(lat_sorted, lat_q)
    q_xyz = _unit_xyz(lat_q, lon_q)
    g_xyz = _unit_xyz(lat_g, lon_g)
    best = np.zeros(len(lat_q), dtype=np.int64)
    best_d = np.full(len(lat_q), np.inf)
    for dj in (-2, -1, 0, 1):
        r = order[np.clip(j + dj, 0, len(order) - 1)]
        dlon = 360.0 / counts[r]
        k = np.rint(((lon_q - row_lon0[r]) % 360.0) / dlon).astype(np.int64) % counts[r]
        idx = starts[r] + k
        d = np.sum((q_xyz - g_xyz[idx]) ** 2, axis=1)
        better = d < best_d
        best[better] = idx[better]
        best_d[better] = d[better]
    return best


class DataReaderIconDream(DataReaderTimestep):
    """
    Total precipitation (tp) from the ICON-DREAM global reanalysis (GRIB2, ICON R03B07).

    Only tp is served -- the dataset's other fields are deliberately not exposed.

    Layout (verified on Jupiter /e/data1/slmet/ml_training/ICON-DREAM, 2010-2024, no gaps):

    * One GRIB2 file per day: ``{base}/{YYYY}/fc_R03B07_rea_ml.YYYYMMDD`` (~3.2 GB, 568 messages).
    * tp: 3 h accumulation (stepRange 0-3) in kg m-2, valid at ``dataTime + 3h``. The message
      valid at 00:00 of day Y lives in **file Y** with ``dataDate=Y-1, dataTime=2100``; files are
      looked up by valid date, so the boundary is transparent.
    * Unstructured grid of 2,949,120 cells; coordinates and cell areas come from the ICON grid
      file in ``const/`` (same grid uuid as the GRIB messages).

    Options (``stream_info``):

    * ``accumulation_period`` (e.g. ``06:00:00``): accumulate on the fly. The reader's timestep
      becomes this period on a grid aligned to 00 UTC, and tp valid at T is the sum of the 3 h
      accumulations valid at T-3h, ..., T: the total over (T - period, T]. That is the
      convention of the anemoi 6-hourly stores (ERA5, OPERAN, IMERG), so a 6 h window starting
      at T yields ONE field valid at T covering the same interval as those streams. Without it
      a 6 h window returns two 3 h fields valid at T and T+3h, i.e. (T-3h, T+3h].
    * ``tp_units``: ``mm`` (native kg m-2, default) or ``m``, the unit of the anemoi stores.
    * ``regrid_to``: an anemoi zarr store (absolute, or relative to the ICON base directory's
      parent). tp is remapped onto that store's grid by nearest-centre, area-weighted
      aggregation ("drop-in-bucket"): each ICON cell goes WHOLE to its nearest store point, and
      a point's value is the area-weighted mean of its cells. This keeps the integral over the
      assigned ICON areas exactly, but it is NOT an overlap-based conservative remap: cells are
      never split, so a point's footprint differs from its true grid box by about one boundary
      cell (~6 ICON cells per N320 point). Values are unbiased means; only the area each one
      represents is approximate. Served at that store's exact coordinates, so it is directly
      comparable with the other streams on that grid, in training, inference and evaluation.
    * ``index_dir``: where per-file GRIB offset indexes are persisted (JSON). Pre-built for
      every file on Jupiter in /e/project1/e-ext-2025e01-128/almikaeel1/icon_dream_index.
    * ``parallel_reads`` (default 2): threads decoding messages within one ``_get``. A 6 h window
      needs 2 tp messages, so 2 threads decode both at once (~0.1 s instead of ~0.2 s per window).
    * ``offset_cache_size`` (default 64): day files whose offset index is kept in memory per
      process. Training samples days at random, so a larger cache rarely hits; a miss reloads that
      day's index JSON (~40 ms), and each cached day costs ~80 KB per dataloader worker.
    * ``normalization``: ``{"tp": [mean, stdev]}`` in ``tp_units``. Defaults to the measured
      statistics for the configured accumulation and grid; any other combination must set it.

    A requested message that is not on disk or cannot be decoded raises; it is never silently
    filled with NaN (the loss masks NaN targets, so the window would quietly go unsupervised).
    """

    _DEFAULT_MAX_CACHED_FILES: int = 64
    _DEFAULT_PARALLEL_READS: int = 2
    # Thread dispatch pays off from 2 messages on (~100 ms of CCSDS decoding each).
    _PARALLEL_MIN_REQUESTS: int = 2

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        **kwargs,
    ) -> None:
        self.base_path = Path(filename)
        if not self.base_path.exists():
            raise FileNotFoundError(f"ICON-DREAM base path not found: {self.base_path}")
        name = stream_info["name"]
        for unsupported in ("geoinfo_channels", "spatial_bbox", "spatial_stride"):
            if stream_info.get(unsupported):
                raise ValueError(f"{name}: `{unsupported}` is not supported by the tp-only reader.")

        const_dir = stream_info.get("const_path")
        const_dir = Path(const_dir) if const_dir else (self.base_path / "const")
        self._grid_path = const_dir / stream_info.get("grid_file", "icon_grid_0026_R03B07_G.nc")

        self.file_index: list[dict] = self._build_file_index()
        if not self.file_index:
            _logger.warning(
                f"No ICON-DREAM files found in {self.base_path}. Stream {name} is empty."
            )
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return
        self._date_to_path: dict[np.datetime64, Path] = {
            entry["date"]: entry["path"] for entry in self.file_index
        }

        data_start_time = self.file_index[0]["start"]
        data_end_time = self.file_index[-1]["end"]
        period = ICON_DREAM_PERIOD.astype("timedelta64[ns]")

        # On-the-fly accumulation: `_accum_steps` native 3 h messages per value.
        self._accum_steps = 1
        accum = stream_info.get("accumulation_period")
        if accum is not None:
            period = parse_timedelta(accum).astype("timedelta64[ns]")
            n_steps, rem = divmod(period, ICON_DREAM_PERIOD.astype("timedelta64[ns]"))
            day = np.timedelta64(24, "h").astype("timedelta64[ns]")
            zero = np.timedelta64(0, "ns")
            if n_steps < 1 or rem != zero or day % period != zero:
                raise ValueError(
                    f"accumulation_period {accum} must be a multiple of 3 h dividing 24 h."
                )
            self._accum_steps = int(n_steps)
            # First valid time on the period grid whose whole window is on disk (the earliest
            # tp message is valid at the first file's 00:00), and last valid time on the grid.
            lead = (self._accum_steps - 1) * ICON_DREAM_PERIOD.astype("timedelta64[ns]")
            data_start_time = data_start_time + -(-lead // period) * period
            last_day = data_end_time.astype("datetime64[D]").astype("datetime64[ns]")
            data_end_time = last_day + ((data_end_time - last_day) // period) * period

        units = stream_info.get("tp_units", "mm")
        if units not in _TP_UNITS_SCALE:
            raise ValueError(
                f"{name}: tp_units must be one of {list(_TP_UNITS_SCALE)}, got {units}."
            )
        self._scale = _TP_UNITS_SCALE[units]

        # Output grid: native ICON cells, or the points of an anemoi store (regrid_to).
        self._load_native_grid()
        self._regrid = None
        regrid_to = stream_info.get("regrid_to")
        self._index_dir = self._resolve_index_dir(stream_info.get("index_dir"))
        if regrid_to:
            ref = Path(regrid_to)
            ref = ref if ref.is_absolute() else self.base_path.parent / ref
            self._setup_regrid(ref)
            grid_tag = "n320" if self.n_grid_points == 542080 else ref.name
        else:
            self.coords_template = np.stack([self._lat_native, self._lon_native], 1).astype(
                np.float32
            )
            self.n_grid_points = len(self.coords_template)
            grid_tag = "native"
        # Only the cell count is needed from here on; don't ship the arrays to every worker.
        self._n_native = len(self._lat_native)
        del self._lat_native, self._lon_native, self._area_native

        super().__init__(tw_handler, stream_info, data_start_time, data_end_time, period)

        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            self.init_empty()
            return

        self.len = int((data_end_time - data_start_time) / period) + 1

        self.source_channels = self.select_channels(stream_info.get("source"))
        self.target_channels = self.select_channels(stream_info.get("target"))
        self.source_idx = np.arange(len(self.source_channels), dtype=np.int64)
        self.target_idx = np.arange(len(self.target_channels), dtype=np.int64)

        user_norm = (stream_info.get("normalization") or {}).get("tp")
        if user_norm is not None:
            mean, stdev = float(user_norm[0]), float(user_norm[1])
        elif (self._accum_steps, grid_tag) in _TP_STATS_MM:
            mean, stdev = (v * self._scale for v in _TP_STATS_MM[(self._accum_steps, grid_tag)])
        else:
            raise ValueError(
                f"{name}: no measured tp statistics for accumulation of {self._accum_steps} x 3 h "
                f"on grid '{grid_tag}'; set normalization: {{tp: [mean, stdev]}} in {units}."
            )
        self.mean = np.array([mean], dtype=np.float32)
        self.stdev = np.array([stdev], dtype=np.float32)

        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.mean_geoinfo = np.zeros(0)
        self.stdev_geoinfo = np.ones(0)

        self.target_channel_weights = self.parse_target_channel_weights()
        self.properties = {"stream_id": stream_info.get("stream_id", 0)}

        self._max_cached_files = int(
            stream_info.get("offset_cache_size", self._DEFAULT_MAX_CACHED_FILES)
        )
        self._parallel_reads = int(stream_info.get("parallel_reads", self._DEFAULT_PARALLEL_READS))
        self._executor: ThreadPoolExecutor | None = None
        self._offset_cache: OrderedDict[Path, dict[tuple[int, str, int, str], int]] = OrderedDict()

        _logger.info(
            f"{name}: tp in {units}, accumulation {self._accum_steps} x 3 h, grid {grid_tag} "
            f"({self.n_grid_points:,} points), norm mean {mean:.6g} stdev {stdev:.6g}; "
            f"source {self.source_channels}, target {self.target_channels}; "
            f"{data_start_time} to {data_end_time}, period {period}; {len(self.file_index)} files"
        )

    # ------------------------------------------------------------------
    # File index, grids
    # ------------------------------------------------------------------

    def _build_file_index(self) -> list[dict]:
        """Scan year directories for daily GRIB files; uses filename dates only."""
        index: list[dict] = []
        year_dirs = sorted(d for d in self.base_path.iterdir() if d.is_dir() and d.name.isdigit())
        for year_dir in year_dirs:
            for fpath in sorted(year_dir.glob("fc_R03B07_rea_ml.*")):
                date_str = fpath.name.rsplit(".", 1)[-1]
                if len(date_str) != 8 or not date_str.isdigit():
                    continue
                try:
                    file_date = datetime.datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    continue
                start_dt = np.datetime64(file_date).astype("datetime64[ns]")
                index.append(
                    {
                        "path": fpath,
                        "date": np.datetime64(file_date.date()),
                        "start": start_dt,
                        "end": start_dt + np.timedelta64(21, "h"),
                    }
                )
        return sorted(index, key=lambda x: x["start"])

    def _load_native_grid(self) -> None:
        """Cell-centre lat/lon (degrees) and cell areas of the ICON grid."""
        if not self._grid_path.exists():
            raise FileNotFoundError(f"ICON grid file not found: {self._grid_path}")
        with netCDF4.Dataset(self._grid_path, "r") as ds:
            self._lat_native = np.degrees(np.asarray(ds["clat"][:], dtype=np.float64))
            self._lon_native = np.degrees(np.asarray(ds["clon"][:], dtype=np.float64))
            self._area_native = np.asarray(ds["cell_area"][:], dtype=np.float64)

    def _setup_regrid(self, ref: Path) -> None:
        """
        Map every ICON cell to its nearest point of the reference grid. A reference point's
        value is the area-weighted mean over the cells mapped to it. Cached in index_dir.
        """
        if not ref.exists():
            raise FileNotFoundError(f"regrid_to store not found: {ref}")
        store = zarr.open(str(ref), mode="r")
        lat_ref = np.asarray(store["latitudes"][:], dtype=np.float64)
        lon_ref = np.asarray(store["longitudes"][:], dtype=np.float64)
        n_ref = len(lat_ref)

        cache = (
            self._index_dir / f"regrid_{self._grid_path.stem}_to_{ref.name}.npz"
            if self._index_dir
            else None
        )
        target = None
        if cache is not None and cache.exists():
            with np.load(cache) as z:
                if len(z["target"]) == len(self._lat_native) and int(z["n_ref"]) == n_ref:
                    target = z["target"]
        if target is None:
            target = nearest_on_reduced_gaussian(
                self._lat_native, self._lon_native, lat_ref, lon_ref
            ).astype(np.int32)
            if cache is not None:
                tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
                try:
                    np.savez(tmp, target=target, n_ref=n_ref)
                    tmp.replace(cache)
                except OSError as e:
                    _logger.warning(f"Could not cache regrid mapping at {cache}: {e}")
                    with contextlib.suppress(OSError):
                        tmp.unlink(missing_ok=True)

        wsum = np.bincount(target, weights=self._area_native, minlength=n_ref)
        # Target points no ICON cell maps to: N320's polar rows are spaced more finely (~8 km)
        # than ICON cells (~13 km), so a few polar points own no cell. They take the value of
        # their nearest ICON cell (brute force -- there are only a handful).
        empty = np.flatnonzero(wsum == 0)
        if len(empty) > 1000:
            raise ValueError(
                f"{len(empty)} points of {ref.name} receive no ICON cell; the ICON grid is too "
                "coarse for this target grid."
            )
        cells_xyz = _unit_xyz(self._lat_native, self._lon_native)
        empty_xyz = _unit_xyz(lat_ref[empty], lon_ref[empty])
        fill = np.array(
            [np.argmin(np.sum((cells_xyz - q) ** 2, axis=1)) for q in empty_xyz], dtype=np.int64
        )
        if len(empty):
            _logger.info(
                f"regrid to {ref.name}: {len(empty)} points own no ICON cell (lat "
                f"{np.round(lat_ref[empty], 2).tolist()}); filled from their nearest cell."
            )
        wsum[empty] = 1.0  # unused for empty points, avoids 0-division below
        self._regrid = {
            "target": target,
            "weight": (self._area_native / wsum[target]).astype(np.float32),
            "n_ref": n_ref,
            "empty": empty,
            "fill": fill,
        }
        # Exactly the coordinates DataReaderAnemoi serves for the same store.
        self.coords_template = np.stack([_clip_lat(lat_ref), _clip_lon(lon_ref)], 1).astype(
            np.float32
        )
        self.n_grid_points = n_ref

    def _to_output_grid(self, field: NDArray) -> NDArray[np.float32]:
        if self._regrid is None:
            return field.astype(np.float32)
        r = self._regrid
        out = np.bincount(r["target"], weights=field * r["weight"], minlength=r["n_ref"])
        out[r["empty"]] = field[r["fill"]]
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    # GRIB offset index
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_index_dir(idx_dir: str | None) -> Path | None:
        if idx_dir is None:
            xdg = os.environ.get("XDG_CACHE_HOME")
            base = Path(xdg) if xdg else Path.home() / ".cache"
            idx_dir = base / "weathergen" / "icon_dream_index"
        path = Path(idx_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _logger.warning(f"Could not create index dir {path}: {e}; nothing will be persisted.")
            return None
        return path

    def _build_offset_cache(self, file_path: Path) -> dict[tuple[int, str, int, str], int]:
        """Byte offset of every message in a daily GRIB file.

        headers_only skips each message's data section, so a scan reads a few KB per message
        instead of the whole ~3.2 GB file; offsets are identical to a full read.
        """
        cache: dict[tuple[int, str, int, str], int] = {}
        with open(file_path, "rb") as f:
            while True:
                offset = f.tell()
                msg = eccodes.codes_grib_new_from_file(f, headers_only=True)
                if msg is None:
                    break
                try:
                    sn = eccodes.codes_get(msg, "shortName")
                    lev = int(eccodes.codes_get(msg, "level"))
                    dt = int(eccodes.codes_get(msg, "dataTime"))
                    es = str(eccodes.codes_get(msg, "endStep"))
                    cache[(dt, sn, lev, es)] = offset
                finally:
                    eccodes.codes_release(msg)
        return cache

    def _index_file_path(self, file_path: Path) -> Path | None:
        if self._index_dir is None:
            return None
        return self._index_dir / f"{file_path.name}.idx.json"

    def _load_offset_cache_from_disk(
        self, file_path: Path
    ) -> dict[tuple[int, str, int, str], int] | None:
        """Load a persisted offset cache, or None if missing/stale/corrupt."""
        idx_file = self._index_file_path(file_path)
        if idx_file is None or not idx_file.exists():
            return None
        try:
            if idx_file.stat().st_mtime < file_path.stat().st_mtime:
                return None
            with idx_file.open("r") as f:
                raw = json.load(f)
            return {(int(dt), sn, int(lev), str(es)): int(off) for dt, sn, lev, es, off in raw}
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            _logger.debug(f"Discarding unreadable offset cache {idx_file}: {e}")
            return None

    def _save_offset_cache_to_disk(
        self, file_path: Path, cache: dict[tuple[int, str, int, str], int]
    ) -> None:
        idx_file = self._index_file_path(file_path)
        if idx_file is None:
            return
        # Per-process tmp name: dataloader workers and ranks scan the same day concurrently.
        tmp = idx_file.with_suffix(f"{idx_file.suffix}.{os.getpid()}.tmp")
        try:
            with tmp.open("w") as f:
                json.dump([[dt, sn, lev, es, off] for (dt, sn, lev, es), off in cache.items()], f)
            tmp.replace(idx_file)
        except OSError as e:
            _logger.warning(f"Failed to persist offset cache for {file_path}: {e}")
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def _get_offset_cache(self, file_path: Path) -> dict[tuple[int, str, int, str], int]:
        """In-memory LRU, then disk, then (last resort) a headers-only scan of the file."""
        cache = self._offset_cache.get(file_path)
        if cache is not None:
            self._offset_cache.move_to_end(file_path)
            return cache
        cache = self._load_offset_cache_from_disk(file_path)
        if cache is None:
            cache = self._build_offset_cache(file_path)
            self._save_offset_cache_to_disk(file_path, cache)
        self._offset_cache[file_path] = cache
        while len(self._offset_cache) > self._max_cached_files:
            self._offset_cache.popitem(last=False)
        return cache

    def _locate_message(self, valid_time: NPDT64) -> tuple[Path, int] | None:
        """(file, byte offset) of the native 3 h tp message ending at valid_time, or None."""
        file_path = self._date_to_path.get(valid_time.astype("datetime64[D]"))
        if file_path is None:
            return None
        hour = valid_time.astype("datetime64[s]").astype(datetime.datetime).hour
        data_time = ((hour - 3) % 24) * 100
        offset = self._get_offset_cache(file_path).get((data_time, *_TP_KEY))
        return None if offset is None else (file_path, offset)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _ensure_executor(self) -> ThreadPoolExecutor | None:
        if self._parallel_reads <= 1:
            return None
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._parallel_reads,
                thread_name_prefix=f"icon_dream_decode_{id(self)}",
            )
        return self._executor

    def _read_file_requests(
        self,
        file_path: Path,
        requests: list[tuple[int, int, int]],
        buf: NDArray[np.float32],
    ) -> None:
        """
        Decode requests ``(ti, k, byte_offset)`` from one file into ``buf[ti, k]`` (native
        cells). Requests write disjoint slices, so concurrent calls are safe. Failures raise.
        """
        with open(file_path, "rb") as f:
            for ti, k, offset in requests:
                f.seek(offset)
                msg = eccodes.codes_grib_new_from_file(f)
                if msg is None:
                    raise OSError(f"No GRIB message at offset {offset} in {file_path}")
                try:
                    values = eccodes.codes_get_array(msg, "values")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to decode GRIB message at offset {offset} in {file_path}"
                    ) from e
                finally:
                    eccodes.codes_release(msg)
                if len(values) != buf.shape[2]:
                    raise ValueError(
                        f"GRIB message at offset {offset} in {file_path} has {len(values)} "
                        f"values, the grid has {buf.shape[2]} cells."
                    )
                buf[ti, k] = values

    # ------------------------------------------------------------------
    # Base-class overrides
    # ------------------------------------------------------------------

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0
        self.coords_template = np.zeros((0, 2), dtype=np.float32)
        self.n_grid_points = 0
        self.file_index = []
        self._date_to_path = {}
        self._offset_cache = OrderedDict()
        self._index_dir = None
        self._executor = None
        self._max_cached_files = self._DEFAULT_MAX_CACHED_FILES
        self._parallel_reads = 0
        self._accum_steps = 1
        self._regrid = None
        self.properties = {"stream_id": 0}

    @override
    def length(self) -> int:
        return self.len

    def select_channels(self, cols_select: list[str] | None) -> list[str]:
        """tp is the only channel: selected when listed (None means all, i.e. tp)."""
        if cols_select is None:
            return list(ICON_DREAM_CHANNELS)
        unknown = [c for c in cols_select if c not in ICON_DREAM_CHANNELS]
        if unknown:
            raise ValueError(f"ICON-DREAM reader serves only {ICON_DREAM_CHANNELS}; got {unknown}.")
        return [c for c in ICON_DREAM_CHANNELS if c in cols_select]

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.len == 0 or len(t_idxs) == 0 or len(channels_idx) == 0 or self.n_grid_points == 0:
            return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

        valid_times = self.data_start_time.astype("datetime64[ns]") + t_idxs * self.period.astype(
            "timedelta64[ns]"
        )
        n_times = len(valid_times)
        n_comp = self._accum_steps
        step3 = ICON_DREAM_PERIOD.astype("timedelta64[ns]")

        # Native 3 h components: those ending at vt-(n-1)*3h, ..., vt cover (vt - period, vt].
        buf = np.empty((n_times, n_comp, self._n_native), dtype=np.float32)
        requests_by_file: dict[Path, list[tuple[int, int, int]]] = {}
        for ti, vt in enumerate(valid_times):
            for k in range(n_comp):
                t_msg = vt - (n_comp - 1 - k) * step3
                loc = self._locate_message(t_msg)
                if loc is None:
                    raise FileNotFoundError(
                        f"{self.stream_info['name']}: no tp message valid at {t_msg} (window "
                        f"{dtr.start} - {dtr.end}): day file missing under {self.base_path} "
                        "or message absent from it."
                    )
                requests_by_file.setdefault(loc[0], []).append((ti, k, loc[1]))
        for reqs in requests_by_file.values():
            reqs.sort(key=lambda r: r[2])

        total = sum(len(r) for r in requests_by_file.values())
        executor = self._ensure_executor()
        if executor is None or total < self._PARALLEL_MIN_REQUESTS:
            for fpath, reqs in requests_by_file.items():
                self._read_file_requests(fpath, reqs, buf)
        else:
            futures = []
            for fpath, reqs in requests_by_file.items():
                n_workers = min(self._parallel_reads, len(reqs))
                chunk = (len(reqs) + n_workers - 1) // n_workers
                for i in range(0, len(reqs), chunk):
                    futures.append(
                        executor.submit(self._read_file_requests, fpath, reqs[i : i + chunk], buf)
                    )
            for fut in futures:
                fut.result()

        # Accumulate (float64), remap, convert units.
        out = np.stack(
            [self._to_output_grid(buf[ti].sum(axis=0, dtype=np.float64)) for ti in range(n_times)]
        )
        out *= np.float32(self._scale)
        n_out = self.n_grid_points
        # tp is the only channel; every requested channel index refers to it.
        data = np.repeat(out.reshape(n_times * n_out, 1), len(channels_idx), axis=1)
        coords = np.vstack([self.coords_template] * n_times)
        datetimes = np.repeat(valid_times, n_out)
        geoinfos = np.zeros((len(data), 0), dtype=np.float32)

        rd = ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes)
        check_reader_data(rd, dtr)
        return rd

    def __getstate__(self) -> dict:
        # A thread pool cannot be pickled, and spawned dataloader workers receive the reader
        # by pickle. Drop it; each process rebuilds its own lazily in _ensure_executor.
        state = self.__dict__.copy()
        state["_executor"] = None
        return state

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            with contextlib.suppress(Exception):
                executor.shutdown(wait=False)
