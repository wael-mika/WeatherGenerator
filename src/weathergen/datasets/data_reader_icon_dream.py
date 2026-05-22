# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import override

import eccodes
import netCDF4 as nc
import numpy as np
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    NPDT64,
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


# ICON model-level indices available in the dataset (out of 120 total)
# Higher index = closer to the surface.
ICON_DREAM_MODEL_LEVELS: list[int] = [49, 57, 64, 70, 75, 79, 86, 91, 96, 101, 108, 112, 119]

# 3-hourly cadence: 8 timesteps per day.
ICON_DREAM_PERIOD = np.timedelta64(3, "h")
ICON_DREAM_DATA_TIMES: list[int] = [0, 300, 600, 900, 1200, 1500, 1800, 2100]

# 3D variables on model levels (instantaneous).
_ML_VARS = ("q", "t", "u", "v", "pres")

# Surface variables: (shortName, level, endStep_string).
# Level values match the GRIB ``level`` key for the heightAboveGround / surface levels.
_SURFACE_INSTANT = (
    ("10u", 10, "0s"),
    ("10v", 10, "0s"),
    ("2t", 2, "0s"),
    ("2d", 2, "0s"),
    ("sp", 0, "0s"),
)
# Accumulated total precipitation: 3h window, valid at end of accumulation.
_TP = ("tp", 0, "3")


def _build_channels() -> tuple[list[str], dict[str, tuple[str, int, str]]]:
    """Return (ordered channel names, channel_name -> (shortName, level, endStep) spec)."""
    names: list[str] = []
    spec: dict[str, tuple[str, int, str]] = {}
    for sn, lev, es in _SURFACE_INSTANT:
        names.append(sn)
        spec[sn] = (sn, lev, es)
    names.append(_TP[0])
    spec[_TP[0]] = (_TP[0], _TP[1], _TP[2])
    for var in _ML_VARS:
        for lev in ICON_DREAM_MODEL_LEVELS:
            ch = f"{var}_L{lev:03d}"
            names.append(ch)
            spec[ch] = (var, lev, "0s")
    return names, spec


ICON_DREAM_CHANNELS, ICON_DREAM_CHANNEL_SPEC = _build_channels()


# Reasonable normalisation defaults per variable family (mean, stdev).
# Can be overridden per channel via stream_info["normalization"].
_DEFAULT_NORM: dict[str, tuple[float, float]] = {
    "10u": (0.0, 5.0),
    "10v": (0.0, 5.0),
    "2t": (273.15, 25.0),
    "2d": (270.0, 20.0),
    "sp": (90000.0, 10000.0),
    "tp": (0.5, 2.0),
    "q": (0.005, 0.005),
    "t": (250.0, 30.0),
    "u": (5.0, 15.0),
    "v": (0.0, 12.0),
    "pres": (50000.0, 30000.0),
}


def _channel_default_norm(ch: str) -> tuple[float, float]:
    if ch in _DEFAULT_NORM:
        return _DEFAULT_NORM[ch]
    family = ch.split("_L")[0]
    return _DEFAULT_NORM.get(family, (0.0, 1.0))


class DataReaderIconDream(DataReaderTimestep):
    """
    Data reader for the ICON-DREAM global reanalysis dataset (GRIB2 on ICON R03B07).

    Layout (verified on /p/scratch/hclimrep/wahl2/ICON-DREAM/):

    * One GRIB2 file per day: ``{base}/{YYYY}/fc_R03B07_rea_ml.YYYYMMDD`` (~3.1 GB / day).
    * 3-hourly cadence: 8 timesteps per file (00, 03, ..., 21 UTC).
    * 71 channels per timestep: 5 instantaneous surface fields (``10u``, ``10v``, ``2t``,
      ``2d``, ``sp``), one accumulated 3h precip (``tp``), and 5 model-level fields
      (``q``, ``t``, ``u``, ``v``, ``pres``) at 13 ICON levels each.
    * Grid: unstructured ICON R03B07 with 2,949,120 cells. Coordinates come from a
      separate NetCDF extpar file in ``const/``.

    Notes
    -----
    cfgrib cannot decode unstructured grids, so the GRIB messages are read with the raw
    ``eccodes`` API using a per-file byte-offset cache.

    Performance options (all optional, set via ``stream_info``):

    * ``parallel_reads`` (default 4): number of threads used to decode GRIB messages
      within one ``_get`` call. GRIB decompression is the dominant per-sample cost
      (~30 ms per 2.9M-cell message); threads release the GIL inside eccodes' C code,
      so 4-8 workers typically give a 3-5× speed-up.
    * ``index_dir`` (default ``$XDG_CACHE_HOME/weathergen/icon_dream_index`` or
      ``~/.cache/weathergen/icon_dream_index``): directory where per-file offset
      caches are persisted as JSON. First access scans the file (~400 ms / 568 msgs);
      every subsequent reader (or new run) loads the cache in <1 ms.
    * ``offset_cache_size`` (default 512): in-memory LRU bound for offset caches.

    The ``tp`` field uses ``stepRange=0-3``: its valid time is ``dataTime + 3h``.
    The ``tp`` message valid at 00:00 of day Y lives in **file Y** with
    ``dataDate=Y-1, dataTime=2100, endStep=3``. The file lookup is by valid-time date,
    not GRIB dataDate, so this boundary is transparent to callers.
    """

    # Default in-memory LRU bound for offset caches (overridable per stream).
    _DEFAULT_MAX_CACHED_FILES: int = 512
    # Default number of decoder threads (overridable per stream).
    _DEFAULT_PARALLEL_READS: int = 4
    # Threshold below which parallel dispatch overhead isn't worth it.
    _PARALLEL_MIN_REQUESTS: int = 4

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

        # Locate the const/extpar file (used for lat/lon and optional geoinfo).
        const_dir = stream_info.get("const_path")
        const_dir = Path(const_dir) if const_dir else (self.base_path / "const")
        extpar_filename = stream_info.get(
            "extpar_file", "icon_extpar_0026_R03B07_G_20220601_tiles.nc"
        )
        self._extpar_path = const_dir / extpar_filename

        # Build the daily file index from filenames only (fast, no GRIB opening).
        self.file_index: list[dict] = self._build_file_index()

        if not self.file_index:
            name = stream_info["name"]
            _logger.warning(
                f"No ICON-DREAM files found in {self.base_path}. Stream {name} is empty."
            )
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Map date -> file path (used for fast lookup in _get).
        self._date_to_path: dict[np.datetime64, Path] = {
            entry["date"]: entry["path"] for entry in self.file_index
        }

        data_start_time = self.file_index[0]["start"]
        data_end_time = self.file_index[-1]["end"]
        period = ICON_DREAM_PERIOD

        # Load lat/lon and apply spatial subsampling.
        self._load_coordinates(self._extpar_path)
        self._apply_spatial_filters(stream_info)

        # Initialise the timestep base class with discovered temporal extent.
        super().__init__(tw_handler, stream_info, data_start_time, data_end_time, period)

        # Bail out if requested training window has no overlap.
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            name = stream_info["name"]
            _logger.warning(
                f"{name} is not supported over data loader window. Stream is skipped."
            )
            self.init_empty()
            return

        # Total 3-hourly steps from data_start to data_end inclusive.
        self.len = int((data_end_time - data_start_time) / period) + 1

        # Channel selection (uses substring-match semantics, like the other readers).
        available = ICON_DREAM_CHANNELS
        s_chs = stream_info.get("source")
        s_chs_exclude = stream_info.get("source_exclude", [])
        t_chs = stream_info.get("target")
        t_chs_exclude = stream_info.get("target_exclude", [])

        self.source_channels = self.select_channels(available, s_chs, s_chs_exclude)
        self.source_idx = np.array(
            [i for i, ch in enumerate(available) if ch in self.source_channels],
            dtype=np.int64,
        )
        self.target_channels = self.select_channels(available, t_chs, t_chs_exclude)
        self.target_idx = np.array(
            [i for i, ch in enumerate(available) if ch in self.target_channels],
            dtype=np.int64,
        )

        # Normalisation arrays cover ALL channels so source_idx/target_idx index in.
        norm_means = np.array(
            [_channel_default_norm(ch)[0] for ch in available], dtype=np.float32
        )
        norm_stdevs = np.array(
            [_channel_default_norm(ch)[1] for ch in available], dtype=np.float32
        )
        for ch, ms in (stream_info.get("normalization") or {}).items():
            if ch in available:
                i = available.index(ch)
                norm_means[i] = float(ms[0])
                norm_stdevs[i] = float(ms[1])
        self.mean = norm_means
        self.stdev = norm_stdevs

        # Static geoinfo from extpar (e.g. topography, land fraction).
        geoinfo_names = list(stream_info.get("geoinfo_channels", []) or [])
        self._load_geoinfo(geoinfo_names)

        self.target_channel_weights = self.parse_target_channel_weights()
        self.properties = {"stream_id": stream_info.get("stream_id", 0)}

        # ---- Performance knobs (see class docstring) ----
        self._max_cached_files: int = int(
            stream_info.get("offset_cache_size", self._DEFAULT_MAX_CACHED_FILES)
        )
        self._parallel_reads: int = int(
            stream_info.get("parallel_reads", self._DEFAULT_PARALLEL_READS)
        )
        self._executor: ThreadPoolExecutor | None = None  # built lazily

        # In-memory LRU offset cache:
        # file_path -> {(dataTime, shortName, level, endStep_str): byte_offset}
        self._offset_cache: OrderedDict[Path, dict[tuple[int, str, int, str], int]] = OrderedDict()

        # Disk-persisted offset cache directory.
        idx_dir = stream_info.get("index_dir")
        if idx_dir is None:
            xdg = os.environ.get("XDG_CACHE_HOME")
            idx_dir = (
                Path(xdg) / "weathergen" / "icon_dream_index"
                if xdg
                else Path.home() / ".cache" / "weathergen" / "icon_dream_index"
            )
        self._index_dir: Path | None = Path(idx_dir)
        try:
            self._index_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _logger.warning(
                f"Could not create offset index dir {self._index_dir}: {e}; "
                "offset caches won't be persisted across runs."
            )
            self._index_dir = None

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")
        _logger.info(
            f"{ds_name}: data period: {data_start_time} to {data_end_time}, period={period}"
        )
        _logger.info(
            f"{ds_name}: {len(self.file_index)} files, {self.n_grid_points:,} grid cells"
        )

    # ------------------------------------------------------------------
    # File-index, coordinates, spatial filtering
    # ------------------------------------------------------------------

    def _build_file_index(self) -> list[dict]:
        """Scan year directories for daily GRIB files; uses filename dates only."""
        index: list[dict] = []
        year_dirs = sorted(
            d for d in self.base_path.iterdir() if d.is_dir() and d.name.isdigit()
        )
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
                end_dt = start_dt + np.timedelta64(21, "h")
                index.append(
                    {
                        "path": fpath,
                        "date": np.datetime64(file_date.date()),
                        "start": start_dt,
                        "end": end_dt,
                    }
                )
        return sorted(index, key=lambda x: x["start"])

    def _load_coordinates(self, extpar_path: Path) -> None:
        """Read cell-centre lat/lon from the ICON extpar NetCDF file."""
        if not extpar_path.exists():
            raise FileNotFoundError(f"ICON extpar file not found: {extpar_path}")
        _logger.info(f"Loading coordinates from {extpar_path}")
        with nc.Dataset(extpar_path, "r") as ds:
            self.latitudes_full = ds.variables["lat"][:].astype(np.float32)
            self.longitudes_full = ds.variables["lon"][:].astype(np.float32)
        _logger.info(f"Loaded {len(self.latitudes_full):,} grid cells from extpar")

    def _apply_spatial_filters(self, stream_info: dict) -> None:
        """Select a subset of grid cells via bounding box and/or stride."""
        n_orig = len(self.latitudes_full)
        bbox = stream_info.get("spatial_bbox")
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            mask = (
                (self.latitudes_full >= lat_min)
                & (self.latitudes_full <= lat_max)
                & (self.longitudes_full >= lon_min)
                & (self.longitudes_full <= lon_max)
            )
            self.cell_indices = np.where(mask)[0]
        else:
            self.cell_indices = np.arange(n_orig, dtype=np.int64)

        stride = int(stream_info.get("spatial_stride", 1))
        if stride > 1:
            self.cell_indices = self.cell_indices[::stride]

        self.latitudes = self.latitudes_full[self.cell_indices]
        self.longitudes = self.longitudes_full[self.cell_indices]
        self.coords_template = np.stack(
            [self.latitudes, self.longitudes], axis=1
        ).astype(np.float32)
        self.n_grid_points = len(self.coords_template)

        if bbox or stride > 1:
            reduction = (1 - self.n_grid_points / n_orig) * 100
            _logger.info(
                f"Spatial filtering: {n_orig:,} -> {self.n_grid_points:,} cells "
                f"({reduction:.1f}% reduction)"
            )

    def _load_geoinfo(self, names: list[str]) -> None:
        """Load static per-cell fields from the extpar NetCDF (e.g. topography)."""
        if not names:
            self.geoinfo_channels = []
            self.geoinfo_idx = []
            self.mean_geoinfo = np.zeros(0)
            self.stdev_geoinfo = np.ones(0)
            self._geoinfo_static = np.zeros((self.n_grid_points, 0), dtype=np.float32)
            return

        arrays: list[NDArray[np.float32]] = []
        kept: list[str] = []
        with nc.Dataset(self._extpar_path, "r") as ds:
            for v in names:
                if v not in ds.variables:
                    _logger.warning(f"Geoinfo variable '{v}' not in extpar file; skipping")
                    continue
                arr = ds.variables[v][:]
                if arr.ndim != 1 or arr.shape[0] != len(self.latitudes_full):
                    _logger.warning(
                        f"Geoinfo variable '{v}' has incompatible shape {arr.shape}; skipping"
                    )
                    continue
                arrays.append(arr.astype(np.float32))
                kept.append(v)

        if not arrays:
            self.geoinfo_channels = []
            self.geoinfo_idx = []
            self.mean_geoinfo = np.zeros(0)
            self.stdev_geoinfo = np.ones(0)
            self._geoinfo_static = np.zeros((self.n_grid_points, 0), dtype=np.float32)
            return

        stacked = np.stack(arrays, axis=1)[self.cell_indices]
        self._geoinfo_static = stacked.astype(np.float32)
        self.geoinfo_channels = kept
        self.geoinfo_idx = list(range(len(kept)))
        # Per-channel stats from the global field (computed before subsampling not needed —
        # subsampling is a deterministic spatial selection, not a sample of a distribution).
        self.mean_geoinfo = stacked.mean(axis=0).astype(np.float32)
        std = stacked.std(axis=0).astype(np.float32)
        std[std == 0] = 1.0
        self.stdev_geoinfo = std

    # ------------------------------------------------------------------
    # GRIB message lookup
    # ------------------------------------------------------------------

    def _build_offset_cache(self, file_path: Path) -> dict[tuple[int, str, int, str], int]:
        """Scan a daily GRIB file once and record byte offset of each message."""
        cache: dict[tuple[int, str, int, str], int] = {}
        with open(file_path, "rb") as f:
            while True:
                offset = f.tell()
                msg = eccodes.codes_grib_new_from_file(f)
                if msg is None:
                    break
                try:
                    sn = eccodes.codes_get(msg, "shortName")
                    lev = int(eccodes.codes_get(msg, "level"))
                    dt = int(eccodes.codes_get(msg, "dataTime"))
                    es = str(eccodes.codes_get(msg, "endStep"))
                    cache[(dt, sn, lev, es)] = offset
                except Exception as e:
                    _logger.warning(f"Skipping malformed message in {file_path}: {e}")
                finally:
                    eccodes.codes_release(msg)
        return cache

    def _index_file_path(self, file_path: Path) -> Path | None:
        """Per-GRIB-file location of the persisted offset cache."""
        if self._index_dir is None:
            return None
        return self._index_dir / f"{file_path.name}.idx.json"

    def _load_offset_cache_from_disk(
        self, file_path: Path
    ) -> dict[tuple[int, str, int, str], int] | None:
        """Load a previously-persisted offset cache, or None if missing/stale/corrupt."""
        idx_file = self._index_file_path(file_path)
        if idx_file is None or not idx_file.exists():
            return None
        try:
            if idx_file.stat().st_mtime < file_path.stat().st_mtime:
                # GRIB file modified since the index was written — rebuild.
                return None
            with idx_file.open("r") as f:
                raw = json.load(f)
            return {
                (int(dt), sn, int(lev), str(es)): int(off)
                for dt, sn, lev, es, off in raw
            }
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            _logger.debug(f"Discarding unreadable offset cache {idx_file}: {e}")
            return None

    def _save_offset_cache_to_disk(
        self, file_path: Path, cache: dict[tuple[int, str, int, str], int]
    ) -> None:
        """Persist an offset cache. Failures are warnings, not fatal."""
        idx_file = self._index_file_path(file_path)
        if idx_file is None:
            return
        tmp = idx_file.with_suffix(idx_file.suffix + ".tmp")
        try:
            with tmp.open("w") as f:
                json.dump(
                    [[dt, sn, lev, es, off] for (dt, sn, lev, es), off in cache.items()],
                    f,
                )
            tmp.replace(idx_file)
        except OSError as e:
            _logger.warning(f"Failed to persist offset cache for {file_path}: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _get_offset_cache(self, file_path: Path) -> dict[tuple[int, str, int, str], int]:
        """
        Return cached offsets for a file.

        Hits the in-memory LRU first, then the disk cache, and only as a last resort
        scans the GRIB file (and writes the result to disk).
        """
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

    def _ensure_executor(self) -> ThreadPoolExecutor | None:
        """Lazily create the decode thread pool (returns None when disabled)."""
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
        data_buf: NDArray[np.float32],
    ) -> None:
        """
        Decode all requested messages from one file and write into ``data_buf``.

        Each request is ``(ti, ci, byte_offset)``. Different requests write to
        disjoint ``data_buf[ti, :, ci]`` slices, so concurrent calls into this
        method from multiple threads on the same buffer are safe.
        """
        with open(file_path, "rb") as f:
            for ti, ci, offset in requests:
                f.seek(offset)
                msg = eccodes.codes_grib_new_from_file(f)
                if msg is None:
                    _logger.warning(f"Null message at offset {offset} in {file_path}")
                    continue
                try:
                    values = eccodes.codes_get_array(msg, "values")
                    # Direct assignment auto-casts float64 -> float32 (data_buf is f32).
                    data_buf[ti, :, ci] = values[self.cell_indices]
                except Exception as e:
                    _logger.warning(
                        f"Failed to read at offset {offset} in {file_path}: {e}"
                    )
                finally:
                    eccodes.codes_release(msg)

    def _locate_message(
        self, valid_time: NPDT64, ch_name: str
    ) -> tuple[Path, int] | None:
        """Resolve (valid_time, channel) -> (file_path, byte_offset), or None if missing."""
        short_name, level, end_step = ICON_DREAM_CHANNEL_SPEC[ch_name]
        # The file holding valid_time T is named after T's calendar date, even for tp
        # at 00:00 (whose GRIB dataDate is the previous day — but it still lives in T's file).
        file_date = valid_time.astype("datetime64[D]")
        file_path = self._date_to_path.get(file_date)
        if file_path is None:
            return None
        vt_dt = valid_time.astype("datetime64[s]").astype(datetime.datetime)
        if end_step == "3":
            # tp: GRIB dataTime is (valid_time - 3h)'s hour, mod 24.
            data_time = ((vt_dt.hour - 3) % 24) * 100
        else:
            data_time = vt_dt.hour * 100
        offset = self._get_offset_cache(file_path).get((data_time, short_name, level, end_step))
        if offset is None:
            return None
        return file_path, offset

    # ------------------------------------------------------------------
    # Base-class overrides
    # ------------------------------------------------------------------

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0
        self.coords_template = np.zeros((0, 2), dtype=np.float32)
        self.n_grid_points = 0
        self.cell_indices = np.array([], dtype=np.int64)
        self.file_index = []
        self._date_to_path = {}
        self._geoinfo_static = np.zeros((0, 0), dtype=np.float32)
        self._offset_cache = OrderedDict()
        self._index_dir = None
        self._executor = None
        self._max_cached_files = self._DEFAULT_MAX_CACHED_FILES
        self._parallel_reads = 0
        self.properties = {"stream_id": 0}

    @override
    def length(self) -> int:
        return self.len

    def select_channels(
        self,
        colnames: list[str],
        cols_select: list[str] | None,
        cols_exclude: list[str] | None,
    ) -> list[str]:
        """Filter channel names by include/exclude substring patterns."""
        if cols_select is not None and len(cols_select) == 0:
            return []
        return [
            c
            for c in colnames
            if (
                (any(sel in c for sel in cols_select) if cols_select else True)
                and not (any(excl in c for excl in cols_exclude) if cols_exclude else False)
            )
        ]

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if (
            self.len == 0
            or len(t_idxs) == 0
            or len(channels_idx) == 0
            or self.n_grid_points == 0
        ):
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # Valid times = absolute datetimes for the dataset indices in this window.
        valid_times = (
            self.data_start_time.astype("datetime64[ns]")
            + t_idxs * self.period.astype("timedelta64[ns]")
        )

        ch_names = [ICON_DREAM_CHANNELS[i] for i in channels_idx]
        n_times = len(valid_times)
        n_ch = len(ch_names)
        n_grid = self.n_grid_points

        # Pre-allocate; missing messages stay as NaN.
        data_buf = np.full((n_times, n_grid, n_ch), np.nan, dtype=np.float32)

        # Group reads by file to minimise open()/seek().
        # request: list of (ti, ci, offset) for one file.
        requests_by_file: dict[Path, list[tuple[int, int, int]]] = {}
        for ti, vt in enumerate(valid_times):
            for ci, ch in enumerate(ch_names):
                loc = self._locate_message(vt, ch)
                if loc is None:
                    continue
                fpath, offset = loc
                requests_by_file.setdefault(fpath, []).append((ti, ci, offset))

        # Sort each file's requests by offset for sequential I/O within a worker.
        for reqs in requests_by_file.values():
            reqs.sort(key=lambda r: r[2])

        total_requests = sum(len(r) for r in requests_by_file.values())
        executor = self._ensure_executor()

        if executor is None or total_requests < self._PARALLEL_MIN_REQUESTS:
            # Serial path — small workloads don't recoup thread dispatch overhead.
            for fpath, requests in requests_by_file.items():
                self._read_file_requests(fpath, requests, data_buf)
        else:
            # Parallel path: split each file's requests across workers so that
            # GRIB decompression (the dominant cost) runs concurrently. Different
            # (ti, ci) slots of data_buf are disjoint, so writes are race-free.
            futures = []
            for fpath, requests in requests_by_file.items():
                n_workers = min(self._parallel_reads, len(requests))
                chunk = (len(requests) + n_workers - 1) // n_workers
                for i in range(0, len(requests), chunk):
                    futures.append(
                        executor.submit(
                            self._read_file_requests,
                            fpath,
                            requests[i : i + chunk],
                            data_buf,
                        )
                    )
            for fut in futures:
                fut.result()

        # Flatten (n_times, n_grid, n_ch) -> (n_times * n_grid, n_ch).
        data = data_buf.reshape(n_times * n_grid, n_ch)
        coords = np.vstack([self.coords_template] * n_times)
        datetimes = np.repeat(valid_times.astype("datetime64[ns]"), n_grid)

        if self._geoinfo_static.shape[1] > 0:
            geoinfos = np.vstack([self._geoinfo_static] * n_times)
        else:
            geoinfos = np.zeros((len(data), 0), dtype=np.float32)

        rd = ReaderData(
            coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes
        )
        check_reader_data(rd, dtr)
        return rd

    def __del__(self) -> None:
        # Best-effort shutdown of the decode pool — never raise from a destructor.
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass
