# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import contextlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class RadklimDirectReader(DataReaderTimestep):
    """
    WeatherGenerator data reader for RADKLIM RW monthly NetCDF files (direct read).

    Directory layout expected (one subdirectory per year):
        base_path/
          2018/RW_2017.002_201801.nc
          2018/RW_2017.002_201802.nc
          ...
          2019/RW_2017.002_201901.nc
          ...

    Dataset essentials (per DWD RW inspection):
      - Variables: RR(time, y, x) [hourly kg m-2], lat(y,x), lon(y,x)
      - Dimensions: time (regular 3600 s periods), y=1100, x=900 (curvilinear grid)
      - CRS: RADOLAN stereographic (use provided lat/lon for geo)

    This reader:
      * Lazily opens each monthly NetCDF with h5netcdf (no dask graph), keeps a small LRU cache.
      * Outputs point-cloud windows (coords, data, datetimes) compatible with WG samplers.
      * Optionally applies per-channel standard score normalization.
    """

    # Construction / metadata
    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,  # base_path
        stream_info: dict,
    ) -> None:
        self._empty = False
        self.stream_info = dict(stream_info)

        # Base path
        self.base_path = Path(filename)
        if not self.base_path.exists():
            raise FileNotFoundError(f"RADKLIM base path not found: {self.base_path}")

        # Config
        self.file_pattern: str = self.stream_info.get(
            "file_pattern", "RW_2017.002_{year}{month:02d}.nc"
        )
        self.source_channels = list(self.stream_info.get("source_channels", ["RR"]))
        self.target_channels = list(self.stream_info.get("target_channels", ["RR"]))
        self.geoinfo_channels = []  # none by default

        self.source_idx = list(range(len(self.source_channels)))
        self.target_idx = list(range(len(self.target_channels)))
        self.geoinfo_idx = list(range(len(self.geoinfo_channels)))

        self.apply_norm: bool = bool(self.stream_info.get("apply_norm", False))
        stats_path = self.stream_info.get("stats_path")
        if self.apply_norm and not stats_path:
            raise ValueError("apply_norm=True but 'stats_path' not provided in stream_info")

        # Init stats (optional)
        self.mean = np.zeros((len(self.source_channels),), dtype=np.float32)
        self.stdev = np.ones((len(self.source_channels),), dtype=np.float32)
        if stats_path:
            norm_path = Path(stats_path)
            if not norm_path.exists():
                raise FileNotFoundError(f"Normalization JSON not found: {norm_path}")
            stats = json.loads(norm_path.read_text())
            m = np.asarray(stats.get("mean", []), dtype=np.float32)
            s = np.asarray(stats.get("std", []), dtype=np.float32)
            if m.size and m.size != len(self.source_channels):
                raise ValueError(
                    f"Stats mean len ({m.size}) ≠ num source channels ({len(self.source_channels)})"
                )
            if s.size and s.size != len(self.source_channels):
                raise ValueError(
                    f"Stats std len ({s.size}) ≠ num source channels ({len(self.source_channels)})"
                )
            if m.size:
                self.mean = m
            if s.size:
                self.stdev = s

        # Caches
        self._ds_cache: OrderedDict[tuple[int,int], xr.Dataset] = OrderedDict()
        self._cache_size_limit: int = int(self.stream_info.get("cache_size", 6))

        # Grid / coords
        self.lat2d: NDArray[np.float_] | None = None
        self.lon2d: NDArray[np.float_] | None = None
        self.ny: int | None = None
        self.nx: int | None = None
        self.points_per_slice: int | None = None

        # Discover time index & build mapping
        _logger.info("Scanning RADKLIM base path: %s", self.base_path)
        self._build_time_index()

        # If nothing found
        if getattr(self, "time_array", None) is None or self.time_array.size == 0:
            super().__init__(tw_handler, stream_info, None, None, None)
            self.init_empty()
            return

        # Period check (expect 3600 s)
        deltas = np.unique(np.diff(self.time_array.astype("datetime64[s]")))
        if deltas.size != 1:
            raise ValueError(f"Irregular time steps detected: {deltas}")
        period = deltas[0]
        _logger.info("Detected period: %s seconds", int(period.astype(int)))

        # Init base-class temporal bounds
        super().__init__(tw_handler, stream_info, self.time_array[0], self.time_array[-1], period)

        # If global TW outside data range,
        if tw_handler.t_start >= self.time_array[-1] or tw_handler.t_end <= self.time_array[0]:
            self.init_empty()
            return

        # Compute absolute index range in full timeline
        self.start_idx = int(np.searchsorted(self.time_array, tw_handler.t_start, side="left"))
        self.end_idx = int(np.searchsorted(self.time_array, tw_handler.t_end, side="right"))

        # Steps per window: floor (avoid rounding up past end)
        steps_ratio = tw_handler.t_window_len / period
        self.num_steps_per_window = int(np.floor(float(steps_ratio)))
        if not np.isclose(steps_ratio, self.num_steps_per_window):
            _logger.warning(
                "Window len not integer multiple of period: %s / %s = %.6f; using %d steps.",
                tw_handler.t_window_len,
                period,
                steps_ratio,
                self.num_steps_per_window,
            )
        if self.num_steps_per_window <= 0:
            raise ValueError(f"Non-positive steps per window: {self.num_steps_per_window}")

        _logger.info(
            "RADKLIM reader ready: %d timesteps, window_steps=%d, grid=%sx%s",
            len(self.time_array),
            self.num_steps_per_window,
            self.ny,
            self.nx,
        )

    # Discovery helpers
    def _build_time_index(self) -> None:
        """
        Manifest-only: build global time index and timestamp→file mapping from JSON.
        The reader will not scan the filesystem.
        """
        manifest_path = self.stream_info.get("manifest_path", None)
        if not manifest_path:
            raise ValueError(
                        "RADKLIM reader requires 'manifest_path' in stream_info "
                        "(manifest-only mode)"
                    )
        mp = Path(manifest_path)
        if not mp.exists():
            raise FileNotFoundError(
                f"RADKLIM manifest not found: {mp}. Generate it with generate_radklim_manifest.py"
            )

        try:
            man = json.loads(mp.read_text())
            months = man["months"]
            period_seconds = int(man.get("period_seconds", 3600))
        except Exception as e:
            raise RuntimeError(f"Failed to parse RADKLIM manifest {mp}: {e}") from e

        # Keep exact file paths from the manifest for fast resolution later
        self._month_path = {(int(m["year"]), int(m["month"])): m["path"] for m in months}

        self.time_to_file = {}
        all_times = []
        for m in months:
            year = int(m["year"]) 
            month = int(m["month"])
            n = int(m["n"])
            start = np.datetime64(m["start"], "ns")
            ts = start + np.arange(n, dtype="int64") * np.timedelta64(period_seconds, "s")
            for local_idx, t in enumerate(ts):
                self.time_to_file[t] = (year, month, local_idx)
                all_times.append(t)

        if not all_times:
            self.time_array = np.array([], dtype="datetime64[ns]")
            _logger.warning("Empty manifest produced no timestamps")
            return

        self.time_array = np.array(sorted(all_times), dtype="datetime64[ns]")

        # Load 2D lat/lon once from the first file path in manifest
        if self.lat2d is None or self.lon2d is None:
            fp = self.base_path / months[0]["path"]
            # 222
            with xr.open_dataset(
                fp,
                engine="h5netcdf",
                decode_times=True,
                mask_and_scale=True,
                chunks=None,
            ) as ds0:
                if "lat" not in ds0 or "lon" not in ds0:
                    raise KeyError(f"lat/lon missing in {fp}")
                self.lat2d = ds0["lat"].values.astype(np.float32)
                self.lon2d = ds0["lon"].values.astype(np.float32)
                self.ny, self.nx = self.lat2d.shape
                for vn in self.source_channels:
                    if vn not in ds0:
                        raise KeyError(f"Variable '{vn}' missing in {fp}")
                    if tuple(ds0[vn].dims[-2:]) != ("y", "x"):
                        raise ValueError(f"{vn} dims expected (..., y, x) in {fp}")
        #_logger.info("Loaded timeline from manifest with %d months", len(months))

    # Dataset cache
    def _get_dataset(self, year: int, month: int) -> xr.Dataset:
        key = (year, month)
        if key in self._ds_cache:
            ds = self._ds_cache.pop(key)  # move to MRU
            self._ds_cache[key] = ds
            return ds

        # Prefer exact relative path from manifest; fall back to pattern if missing
        rel = getattr(self, "_month_path", {}).get(key)
        if rel is not None:
            file_path = self.base_path / rel
        else:
            file_path = (
                self.base_path / str(year) /
                self.file_pattern.format(year=year, month=month)
            )


        if not file_path.exists():
            raise FileNotFoundError(f"Expected file not found: {file_path}")

        _logger.debug("Opening NetCDF: %s", file_path)
        ds = xr.open_dataset(
            file_path,
            engine="h5netcdf",
            decode_times=True,
            mask_and_scale=True,
            chunks=None,
        )

        # Keep only the requested variables (drop coords to save memory)
        missing = [v for v in self.source_channels if v not in ds]
        if missing:
            ds.close()
            raise KeyError(f"Variables not found in {file_path}: {missing}")
        ds = ds[self.source_channels]

        self._ds_cache[key] = ds
        if len(self._ds_cache) > self._cache_size_limit:
            oldest_key, oldest_ds = self._ds_cache.popitem(last=False)
            with contextlib.suppress(Exception):
                oldest_ds.close()
            # _logger.debug("Evicted from cache: %s", oldest_key)
        return ds

    # ----------------------------
    # Public API required by WG
    # ----------------------------
    @override
    def init_empty(self) -> None:
        self._empty = True
        super().init_empty()

    @override
    def length(self) -> int:
        if self._empty:
            return 0
        nt = self.end_idx - self.start_idx
        return max(0, nt - self.num_steps_per_window + 1)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        if self._empty:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # Absolute time indices for this logical window
        t_idxs_abs, dtr = self._get_dataset_idxs(idx)
        if t_idxs_abs.size == 0:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        requested_times = self.time_array[t_idxs_abs]

        data_chunks = []
        coords_chunks = []
        times_chunks = []

        # precompute mean/std slice for selected channels
        if self.apply_norm:
            mean_sel = self.mean[channels_idx]
            std_sel = np.maximum(self.stdev[channels_idx], 1e-6)

        # Extract step-by-step (keeps peak memory under control)
        for t in requested_times:
            mapping = self.time_to_file.get(t)
            if not mapping:
                _logger.warning("Timestamp %s not mapped to any file", t)
                continue
            year, month, local_idx = mapping

            try:
                ds = self._get_dataset(year, month)
                # select time slice -> (y,x) per var
                ds_slice = ds.isel(time=local_idx)

                var_names = [self.source_channels[i] for i in channels_idx]
                var_arrays = []
                for vn in var_names:
                    a = ds_slice[vn].values  # (y,x)
                    var_arrays.append(a)

                # stack -> (y,x,C) then flatten -> (N,C)
                if len(var_arrays) == 1:
                    stacked = var_arrays[0][..., np.newaxis]
                else:
                    stacked = np.stack(var_arrays, axis=-1)
                flat_data = stacked.reshape(-1, len(var_names)).astype(np.float32, copy=False)

                # mask invalid rows across all selected channels
                valid_rows = np.isfinite(flat_data).all(axis=1)
                if not valid_rows.any():
                    continue
                flat_data = flat_data[valid_rows]

                # optional normalization (per channel)
                if self.apply_norm:
                    flat_data = (flat_data - mean_sel) / std_sel

                # coords (lat, lon) masked the same way
                flat_lat = self.lat2d.ravel()[valid_rows]
                flat_lon = self.lon2d.ravel()[valid_rows]
                flat_coords = np.stack([flat_lat, flat_lon], axis=1)

                # datetimes
                flat_times = np.full(flat_data.shape[0], t, dtype="datetime64[ns]")

                data_chunks.append(flat_data)
                coords_chunks.append(flat_coords)
                times_chunks.append(flat_times)

            except Exception as e:
                _logger.error(
                    "Failed reading %d-%02d idx=%d (t=%s): %s", year, month, local_idx, t, e
                )
                continue

        if not data_chunks:
            _logger.debug("No valid data for window idx=%s", idx)
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        data = np.concatenate(data_chunks, axis=0)
        coords = np.concatenate(coords_chunks, axis=0)
        dts = np.concatenate(times_chunks, axis=0)

        r = ReaderData(
            coords=coords,
            geoinfos=np.zeros((data.shape[0], len(self.geoinfo_idx)), dtype=np.float32),
            data=data,
            datetimes=dts,
        )
        check_reader_data(r, dtr)
        return r   

    def __del__(self):
        try:
            for ds in list(self._ds_cache.values()):
                with contextlib.suppress(Exception):
                    ds.close()
        except Exception:
            pass