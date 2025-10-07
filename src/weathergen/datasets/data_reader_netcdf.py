# (C) 2025 WeatherGenerator contributors — Apache 2.0
from __future__ import annotations

import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_log = logging.getLogger(__name__)

# - Files are ERA5 pressure-level monthly NetCDF on the O96 grid
# - Dims per var: (time, pressure_level, values); coords latitude(values), longitude(values)
# - Variable names inside files are short codes: q, t, u, v, w, z (we’ll auto-map from long names)
# - Time is decode_cf’d to numpy datetime64
# - One directory with files like: era5_pl_{var}_{yyyymm}_L50-1000.nc

ALIASES = {
    "specific_humidity": "q",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "geopotential": "z",
}
SHORTS = {"q", "t", "u", "v", "w", "z"}
SHORT2LONG = {v: k for k, v in ALIASES.items()}
DEFAULT_VARS = ["q", "t", "u", "v", "w", "z"]
DEFAULT_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
DEFAULT_TEMPLATE = "era5_pl_{var}_{yyyymm}_L50-1000.nc"


def _clip_lat(x: np.ndarray) -> np.ndarray:
    return (2 * np.clip(x, -90.0, 90.0) - x).astype(np.float32)


def _clip_lon(x: np.ndarray) -> np.ndarray:
    return ((x + 180.0) % 360.0 - 180.0).astype(np.float32)


def _yyyymm_range(t0: np.datetime64, t1: np.datetime64) -> list[str]:
    y0, m0 = int(str(t0)[:4]), int(str(t0)[5:7])
    y1, m1 = int(str(t1)[:4]), int(str(t1)[5:7])
    out, y, m = [], y0, m0
    while (y < y1) or (y == y1 and m <= m1):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


class DataReaderERA5NetCDF(DataReaderTimestep):
    """
    Minimal, vectorized ERA5 PL NetCDF reader for WeatherGenerator.

    Stream config (all optional; we default smartly):
      name            : str (for logs)
      variables       : list of 'q','t','u','v','w','z' or long names (mapped to shorts)
      levels          : list of ints (pressure levels in hPa)
      file_template   : e.g. 'era5_pl_{var}_{yyyymm}_L50-1000.nc'
      engine          : 'h5netcdf' | 'netcdf4' (default: h5netcdf)
      chunks          : dict like {'time': 64}  (optional)
      source / target : lists of substrings (e.g., ['t_500','u_700']) and *_exclude lists
      stats_path      : optional json with mean/stdev (else identity)
    """

    def __init__(self, tw_handler: TimeWindowHandler, filename: Path, stream_info: dict) -> None:
        self.stream_info = stream_info
        self.data_dir = Path(filename)

        # Resolve config (hard defaults kept small)
        vars_cfg = stream_info.get("variables", DEFAULT_VARS)
        # Map any long names to shorts, keep order
        self.vars_short = [ALIASES.get(v, v) for v in vars_cfg]
        # Keep only known vars
        self.vars_short = [v for v in self.vars_short if v in SHORTS]
        if not self.vars_short:
            self.vars_short = DEFAULT_VARS

        self.levels = stream_info.get("levels", DEFAULT_LEVELS)
        self.file_template = stream_info.get("file_template", DEFAULT_TEMPLATE)
        self.engine = stream_info.get("engine", "h5netcdf")
        self.chunks = stream_info.get("chunks", {})
        self.name = stream_info.get("name", "era5_netcdf")

        # Build file lists for requested months/variables
        t0 = np.datetime64(tw_handler.t_start, "ns")
        t1 = np.datetime64(tw_handler.t_end, "ns")
        months = _yyyymm_range(t0, t1)

        def paths_for(var_short: str) -> list[str]:
            # try both short and long tokens in the filename template
            tokens = [var_short]
            long_tok = SHORT2LONG.get(var_short)
            if long_tok and long_tok not in tokens:
                tokens.append(long_tok)

            out, seen = [], set()
            for yyyymm in months:
                for tok in tokens:
                    p = self.data_dir / self.file_template.format(var=tok, yyyymm=yyyymm)
                    if p.exists():
                        sp = str(p)
                        if sp not in seen:
                            out.append(sp)
                            seen.add(sp)
            return out


        # Open per-variable datasets (lazy) and merge
        datasets = []
        for v in self.vars_short:
            pp = paths_for(v)
            if not pp:
                _log.warning("%s: no files for var=%s in requested window.", self.name, v)
                continue
            ds_v = xr.open_mfdataset(
                pp,
                combine="by_coords",
                engine=self.engine,
                chunks=self.chunks or {},
                decode_cf=True,
                parallel=False,
                data_vars="minimal",
                coords="minimal",
                compat="override",
            )
            # normalize dims/coords cheaply (we assume O96)
            if "valid_time" in ds_v:  # rare
                ds_v = ds_v.rename({"valid_time": "time"})
            if "level" in ds_v.dims:
                ds_v = ds_v.rename({"level": "pressure_level"})
            # if "values" not in ds_v.dims:
            #     # reduced name variants (unlikely in your set)
            #     for cand in ("number_of_points",):
            #         if cand in ds_v.dims:
            #             ds_v = ds_v.rename({cand: "values"})
            #             break
            if "values" not in ds_v.dims:
                # Some files may expose (latitude, longitude) as separate dims
                if {"latitude", "longitude"}.issubset(ds_v.dims):
                    ds_v = ds_v.stack(values=("latitude", "longitude"))
                elif {"lat", "lon"}.issubset(ds_v.dims):
                    ds_v = ds_v.stack(values=("lat", "lon"))
                else:
                    # last resort: look for any 2D spatial dims and stack them
                    cand_spatial = [d for d in ds_v.dims if d not in {"time", "pressure_level"}]
                    if len(cand_spatial) >= 2:
                        ds_v = ds_v.stack(values=tuple(cand_spatial[:2]))
                    else:
                        raise ValueError(f"Cannot find spatial dims to stack in {list(ds_v.dims)}")
            # keep only this variable's data array (short name inside files)
            keep = v if v in ds_v.data_vars else list(ds_v.data_vars)[0]
            datasets.append(ds_v[[keep]])

        if not datasets:
            _log.warning("%s: no datasets opened in [%s,%s].", self.name, t0, t1)
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        ds = xr.merge(datasets, compat="override", join="outer")

        # Time slice + level subset
        ds = ds.sel(time=slice(np.datetime_as_string(t0, "s"), np.datetime_as_string(t1, "s")))
        if ds.sizes.get("time", 0) == 0:
            _log.warning("%s: no timesteps after slice; skipping.", self.name)
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Ensure level order & presence
        present_levels = [int(x) for x in np.asarray(ds["pressure_level"].values)]
        keep_levels = [L for L in self.levels if L in present_levels]
        ds = ds.sel(pressure_level=keep_levels).sortby("pressure_level")

        # Cache shape/coords
        keep_vars = [v for v in self.vars_short if v in ds.data_vars]
        self.ds = ds[keep_vars]

        self.times = self.ds["time"].values
        self.n_values = int(self.ds.sizes["values"])
        latname = "latitude" if "latitude" in self.ds else "lat"
        lonname = "longitude" if "longitude" in self.ds else "lon"
        self.latitudes = _clip_lat(np.asarray(self.ds[latname].values))
        self.longitudes = _clip_lon(np.asarray(self.ds[lonname].values))

        # Period (6-hourly expected)
        period = (self.times[1] - self.times[0]) if self.times.size >= 2 else np.timedelta64(6, "h")
        super().__init__(tw_handler, stream_info, self.times[0], self.times[-1], np.timedelta64(period))

        # Build flattened channel list & index mapping (var-major then level)
        self.levels_present = [int(x) for x in np.asarray(self.ds["pressure_level"].values)]
        self.all_channels: list[str] = []
        self._channel_pos: dict[tuple[str, int], int] = {}
        k = 0
        for v in self.vars_short:
            for L in self.levels_present:
                self.all_channels.append(f"{v}_{L}")
                self._channel_pos[(v, int(L))] = k
                k += 1

        # Selections from YAML (substring semantics, like Anemoi)
        self.source_idx = self._select("source")
        self.target_idx = self._select("target")
        self.source_channels = [self.all_channels[i] for i in self.source_idx]
        self.target_channels = [self.all_channels[i] for i in self.target_idx]
        self.geoinfo_channels: list[str] = []
        self.geoinfo_idx: list[int] = []

        # Identity normalization (fast default; can be overridden by stats_path)
        nC = len(self.all_channels)
        self.mean = np.zeros(nC, dtype=np.float32)
        self.stdev = np.ones(nC, dtype=np.float32)
        sp = stream_info.get("stats_path")
        if sp:
            try:
                import json

                st = json.load(open(sp, "r"))
                def _mk(arr, key):
                    if isinstance(arr, dict):
                        return np.array([arr.get(ch, 0.0 if key=="mean" else 1.0) for ch in self.all_channels], np.float32)
                    arr = np.array(arr, np.float32)
                    if arr.size != nC:
                        raise ValueError(f"stats[{key}] size {arr.size} != {nC}")
                    return arr
                self.mean = _mk(st.get("mean", self.mean), "mean")
                self.stdev = _mk(st.get("stdev", self.stdev), "stdev")
            except Exception as e:
                _log.warning("%s: failed to load stats %s (%s); using identity.", self.name, sp, e)

        self.len = int(self.times.size)

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0
        self.times = np.array([], dtype="datetime64[ns]")
        self.latitudes = np.array([], dtype=np.float32)
        self.longitudes = np.array([], dtype=np.float32)
        self.n_values = 0
        self.all_channels = []
        self.source_idx = np.array([], dtype=int)
        self.target_idx = np.array([], dtype=int)
        self.geoinfo_idx = []

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Vectorized: slice time → to_array('var') → stack (var, level) → isel(channel)
        Returns flattened (T*V, C_sel) with repeated coords & datetimes.
        """
        t_idxs, dtr = self._get_dataset_idxs(idx)
        if self.ds is None or self.len == 0 or len(t_idxs) == 0 or len(channels_idx) == 0:
            return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

        i0, i1 = int(t_idxs[0]), int(t_idxs[-1]) + 1
        T, V = (i1 - i0), self.n_values

        ds_slice = self.ds.isel(time=slice(i0, i1))  # keep var order

        # -> DataArray dims: ('var','time','pressure_level','values')
        arr = ds_slice.to_array("var").transpose("time", "values", "var", "pressure_level")
        # Stack ('var','pressure_level') to a single 'channel' dim in canonical order
        arr = arr.stack(channel=("var", "pressure_level"))

        # Map requested channel indices (global) onto 'channel' positions
        # Our 'channel' order is exactly vars_short × levels_present
        # so channel i is consistent with self.all_channels
        csel = np.asarray(channels_idx, dtype=int)
        arr = arr.isel(channel=csel)  # (T, V, C_sel)

        data = arr.values.astype(np.float32).reshape(T * V, csel.size)

        latlon = np.stack([self.latitudes, self.longitudes], axis=1).astype(np.float32)  # (V,2)
        coords = np.vstack([latlon] * T)  # (T*V, 2)

        geoinfos = np.zeros((T * V, 0), dtype=np.float32)
        times = self.times[i0:i1]
        datetimes = np.repeat(times, V)

        rd = ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes)
        check_reader_data(rd, dtr)
        return rd

    def _select(self, kind: str) -> np.ndarray:
        filt = self.stream_info.get(kind)
        excl = self.stream_info.get(f"{kind}_exclude", []) or []
        if filt is None:
            mask = np.ones(len(self.all_channels), bool)
        else:
            mask = np.array([any(s in ch for s in filt) for ch in self.all_channels], bool)
            if len(filt) == 0:
                _log.warning("%s: no channel for %s.", self.name, kind)
        if excl:
            exm = np.array([any(s in ch for s in excl) for ch in self.all_channels], bool)
            mask &= ~exm
        return np.nonzero(mask)[0].astype(int)

