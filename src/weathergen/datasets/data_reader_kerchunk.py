# # (C) 2025 WeatherGenerator contributors — Apache 2.0
# from __future__ import annotations

# import json
# import logging
# from pathlib import Path
# from typing import Any, override
# import zarr

# import numpy as np
# import xarray as xr
# import fsspec

# from weathergen.datasets.data_reader_base import (
#     DataReaderTimestep,
#     ReaderData,
#     TimeWindowHandler,
#     TIndex,
# )

# _log = logging.getLogger(__name__)

# # ---- ERA5 naming --------------------------------------------------------

# ALIASES = {
#     "specific_humidity": "q",
#     "temperature": "t",
#     "u_component_of_wind": "u",
#     "v_component_of_wind": "v",
#     "vertical_velocity": "w",
#     "geopotential": "z",
# }
# SHORTS = {"q", "t", "u", "v", "w", "z"}
# DEFAULT_VARS = ["q", "t", "u", "v", "w", "z"]
# DEFAULT_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# def _clip_lat(x: np.ndarray) -> np.ndarray:
#     return (2 * np.clip(x, -90.0, 90.0) - x).astype(np.float32)

# def _clip_lon(x: np.ndarray) -> np.ndarray:
#     return ((x + 180.0) % 360.0 - 180.0).astype(np.float32)


# # ========================================================================
# # Manifest-only, fast kerchunk reader for ERA5 PL
# # ========================================================================

# class DataReaderERA5Kerchunk(DataReaderTimestep):
#     """
#     Fast ERA5 PL data reader backed by a *kerchunk manifest* JSON.

#     Assumptions (hard-coded for simplicity & speed):
#       - Reference file is a *manifest* with fields: {"kind":"kerchunk-manifest","refs":[...]}.
#       - Concat dimension is 'valid_time'.
#       - Dims: (valid_time, pressure_level, values)
#       - Coords on 'values': latitude(values), longitude(values)
#       - Variables named by short codes: q, t, u, v, w, z
#       - Some datasets may include 'expver'; we select expver=5 if present, else first.

#     Stream config keys used:
#       - name            : str (optional)
#       - variables       : list[str] subset of {q,t,u,v,w,z} or long names (mapped to shorts)
#       - levels          : list[int] pressure levels (defaults below)
#       - reference_path  : str path to the manifest JSON (required)
#       - prefer_expver   : int, default 5
#       - chunks          : dict for logical rechunk after open (optional; default keep source)
#     """
    
    
    
#     def _open_ref_fast(self, ref: dict, vars_to_open: list[str]) -> xr.Dataset:
#         """
#         Open one kerchunk ref quickly:
#         - consolidate metadata in-memory (adds .zmetadata to the ref mapping)
#         - open with consolidated=True
#         - standardize dim names and subset vars
#         """
#         fs = fsspec.filesystem("reference", fo=ref)
#         mapper = fs.get_mapper("")
#         try:
#             zarr.consolidate_metadata(mapper)  # mutates ref mapping (in-memory)
#             ds = xr.open_dataset(mapper, engine="zarr", consolidated=True, chunks={})
#         except Exception:
#             ds = xr.open_dataset(mapper, engine="zarr", consolidated=False, chunks={})

#         # Standardize names
#         if "time" in ds.dims and "valid_time" not in ds.dims:
#             ds = ds.rename({"time": "valid_time"})
#         if "level" in ds.dims and "pressure_level" not in ds.dims:
#             ds = ds.rename({"level": "pressure_level"})
#         if "values" not in ds.dims and "number_of_points" in ds.dims:
#             ds = ds.rename({"number_of_points": "values"})

#         # Subset vars (keep only those present)
#         keep = [v for v in vars_to_open if v in ds.data_vars]
#         if keep:
#             ds = ds[keep]
#         else:
#             # coords-only dataset to keep concat happy
#             ds = xr.Dataset(coords={k: v for k, v in ds.coords.items()})

#         # Optional: drop noisy attrs that can slow merges
#         for k in ("history",):
#             ds.attrs.pop(k, None)
#         return ds


#     def __init__(self, tw_handler: TimeWindowHandler, filename: Path, stream_info: dict) -> None:
#         self.stream_info = stream_info
#         self.name = stream_info.get("name", "era5_kerchunk")

#         # Resolve variables (map long → short, keep order & filter)
#         vars_cfg = stream_info.get("variables", DEFAULT_VARS)
#         vars_short = [ALIASES.get(v, v) for v in vars_cfg]
#         self.vars_short = [v for v in vars_short if v in SHORTS] or DEFAULT_VARS

#         self.levels = stream_info.get("levels", DEFAULT_LEVELS)
#         self.prefer_expver = int(stream_info.get("prefer_expver", 5))
#         self.user_chunks: dict[str, int] = stream_info.get("chunks", {}) or {}

#         # Time window (as strings & ns)
#         t0 = np.datetime64(tw_handler.t_start, "ns")
#         t1 = np.datetime64(tw_handler.t_end, "ns")
#         self._t0_ns = t0
#         self._t1_ns = t1
#         t0s = np.datetime_as_string(t0, "s")
#         t1s = np.datetime_as_string(t1, "s")

#         # Load manifest (required)
#         ref_path = Path(stream_info.get("reference_path", filename))
#         m = json.load(open(ref_path, "r"))
#         if not (isinstance(m, dict) and m.get("kind") == "kerchunk-manifest" and isinstance(m.get("refs"), list)):
#             raise ValueError(f"{ref_path} is not a kerchunk-manifest JSON with a 'refs' array.")

#         refs = m["refs"]
#         vars_to_open = self.vars_short[:]  # only requested vars
#         dsets = [self._open_ref_fast(ref, vars_to_open) for ref in refs]


#         # ---- concat lazily by valid_time (no global alignment heuristics)
#         ds = xr.concat(
#             dsets,
#             dim="valid_time",
#             data_vars="minimal",
#             coords="minimal",
#             compat="no_conflicts",
#             join="exact",         # assume timeline is consistent across months
#             combine_attrs="drop",
#         )

#         # expver preference (if available)
#         if "expver" in ds.dims:
#             expvers = np.asarray(ds["expver"].values)
#             if self.prefer_expver in expvers:
#                 ds = ds.sel(expver=self.prefer_expver)
#             else:
#                 ds = ds.isel(expver=0)

#         # Sort and slice window (boolean mask avoids pandas index rules)
#         ds = ds.sortby("valid_time")
#         mask = (ds["valid_time"] >= self._t0_ns) & (ds["valid_time"] <= self._t1_ns)
#         ds = ds.isel(valid_time=mask)

#         # Normalize to 'time'
#         ds = ds.rename({"valid_time": "time"})

#         # Subset levels & order
#         if "pressure_level" in ds.dims:
#             present_levels = [int(x) for x in np.asarray(ds["pressure_level"].values)]
#             keep_levels = [L for L in self.levels if L in present_levels]
#             ds = ds.sel(pressure_level=keep_levels).sortby("pressure_level")

#         # Optional logical rechunk for downstream ops
#         if self.user_chunks:
#             ds = ds.chunk(self.user_chunks)

#         # ---- cache metadata for fast _get
#         self.ds = ds
#         self.times = ds["time"].values
#         self.len = int(self.times.size)

#         self.n_values = int(ds.sizes["values"])
#         latname = "latitude" if "latitude" in ds.coords else "lat"
#         lonname = "longitude" if "longitude" in ds.coords else "lon"
#         self.latitudes = _clip_lat(np.asarray(ds[latname].values))
#         self.longitudes = _clip_lon(np.asarray(ds[lonname].values))

#         # Period estimate from first two steps (fallback to 6h)
#         period = (self.times[1] - self.times[0]) if self.len >= 2 else np.timedelta64(6, "h")
#         super().__init__(tw_handler, stream_info, self.times[0], self.times[-1], np.timedelta64(period))

#         # Build channels (var-major then level)
#         self.levels_present = [int(x) for x in (ds["pressure_level"].values if "pressure_level" in ds.dims else [])]
#         self.all_channels: list[str] = []
#         self._channel_pos: dict[tuple[str, int], int] = {}
#         k = 0
#         for v in self.vars_short:
#             if v not in ds.data_vars:
#                 continue
#             for L in self.levels_present:
#                 self.all_channels.append(f"{v}_{int(L)}")
#                 self._channel_pos[(v, int(L))] = k
#                 k += 1

#         # Channel selections (substring match)
#         self.source_idx = self._select("source")
#         self.target_idx = self._select("target")
#         self.source_channels = [self.all_channels[i] for i in self.source_idx]
#         self.target_channels = [self.all_channels[i] for i in self.target_idx]
#         self.geoinfo_channels: list[str] = []
#         self.geoinfo_idx: list[int] = []

#         # Identity normalization (can be overridden via stats_path externally)
#         nC = len(self.all_channels)
#         self.mean = np.zeros(nC, dtype=np.float32)
#         self.stdev = np.ones(nC, dtype=np.float32)

#     # ---------------- DataReaderTimestep protocol ----------------

#     @override
#     def init_empty(self) -> None:
#         super().init_empty()
#         self.ds = None
#         self.len = 0
#         self.times = np.array([], dtype="datetime64[ns]")
#         self.latitudes = np.array([], dtype=np.float32)
#         self.longitudes = np.array([], dtype=np.float32)
#         self.n_values = 0
#         self.all_channels = []
#         self.source_idx = np.array([], dtype=int)
#         self.target_idx = np.array([], dtype=int)
#         self.geoinfo_idx = []

#     @override
#     def length(self) -> int:
#         return self.len

#     # Simple, explicit index normalization (slice | (start,end) | list | int)
#     def _norm_idx(self, idx: TIndex, n: int) -> np.ndarray:
#         if isinstance(idx, slice):
#             start = 0 if idx.start is None else int(idx.start)
#             stop  = n if idx.stop  is None else int(idx.stop)
#             step  = 1 if idx.step  is None else int(idx.step)
#             return np.arange(max(0, start), min(stop, n), step, dtype=int)
#         if isinstance(idx, tuple) and len(idx) == 2:
#             a, b = int(idx[0]), int(idx[1])
#             if b < a:
#                 return np.arange(0, dtype=int)
#             return np.arange(max(0, a), min(b + 1, n), dtype=int)
#         arr = np.atleast_1d(np.asarray(idx, dtype=int))
#         return arr[(arr >= 0) & (arr < n)]

#     @override
#     def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
#         if self.ds is None or self.len == 0 or len(channels_idx) == 0:
#             return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

#         t_idxs = self._norm_idx(idx, self.len)
#         if t_idxs.size == 0:
#             return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

#         i0, i1 = int(t_idxs[0]), int(t_idxs[-1]) + 1
#         T, V = (i1 - i0), self.n_values

#         # Keep var-major order and flatten like your NetCDF reader
#         ds_slice = self.ds.isel(time=slice(i0, i1))               # (time, level, values)
#         arr = ds_slice.to_array("var")                             # ('var','time','pressure_level','values')
#         arr = arr.transpose("time", "values", "var", "pressure_level")
#         arr = arr.stack(channel=("var", "pressure_level"))
#         csel = np.asarray(channels_idx, dtype=int)
#         arr = arr.isel(channel=csel)                               # (T, V, C_sel)

#         data = arr.values.astype(np.float32).reshape(T * V, csel.size)

#         latlon = np.stack([self.latitudes, self.longitudes], axis=1).astype(np.float32)  # (V,2)
#         coords = np.broadcast_to(latlon, (T, V, 2)).reshape(T * V, 2)

#         geoinfos = np.zeros((T * V, 0), dtype=np.float32)
#         times = self.times[i0:i1]
#         datetimes = np.repeat(times, V)

#         return ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes)

#     # --------------- channel selection helpers ---------------

#     def _select(self, kind: str) -> np.ndarray:
#         filt = self.stream_info.get(kind)
#         excl = self.stream_info.get(f"{kind}_exclude", []) or []
#         if filt is None:
#             mask = np.ones(len(self.all_channels), bool)
#         else:
#             mask = np.array([any(s in ch for s in filt) for ch in self.all_channels], bool)
#         if excl:
#             exm = np.array([any(s in ch for s in excl) for ch in self.all_channels], bool)
#             mask &= ~exm
#         return np.nonzero(mask)[0].astype(int)


# (C) 2025 WeatherGenerator contributors — Apache 2.0
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, override

import numpy as np
import xarray as xr
import fsspec
import zarr
import re
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
)

_log = logging.getLogger(__name__)

# ---- ERA5 naming
ALIASES = {
    "specific_humidity": "q",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "geopotential": "z",
}
SHORTS = {"q", "t", "u", "v", "w", "z"}
DEFAULT_VARS = ["q", "t", "u", "v", "w", "z"]
DEFAULT_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

def _clip_lat(x: np.ndarray) -> np.ndarray:
    return (2 * np.clip(x, -90.0, 90.0) - x).astype(np.float32)

def _clip_lon(x: np.ndarray) -> np.ndarray:
    return ((x + 180.0) % 360.0 - 180.0).astype(np.float32)


class DataReaderERA5Kerchunk(DataReaderTimestep):
    """
    Fast ERA5-PL reader over a *kerchunk manifest*.

    Assumptions (by design, for speed):
      - Reference is a manifest: {"kind":"kerchunk-manifest","refs":[...]}.
      - Concat dim is 'valid_time'; dims are (valid_time, pressure_level, values).
      - Coords: latitude(values), longitude(values).
      - Vars are {q,t,u,v,w,z} (long names mapped to shorts).
    """

    # ---------- helpers ----------
    def _open_ref_fast(self, ref: dict, vars_to_open: list[str]) -> xr.Dataset:
        fs = fsspec.filesystem("reference", fo=ref)
        m = fs.get_mapper("")
        # consolidate in-memory to slash key lookups, then open consolidated
        has_zmeta = isinstance(ref.get("refs"), dict) and (".zmetadata" in ref["refs"])
        if has_zmeta:
            ds = xr.open_dataset(m, engine="zarr", consolidated=True, chunks={})
        else:
            try:
                zarr.consolidate_metadata(m)  # add .zmetadata in-memory
                ds = xr.open_dataset(m, engine="zarr", consolidated=True, chunks={})
            except Exception:
                ds = xr.open_dataset(m, engine="zarr", consolidated=False, chunks={})

        # standardize names
        if "time" in ds.dims and "valid_time" not in ds.dims:
            ds = ds.rename({"time": "valid_time"})
        if "level" in ds.dims and "pressure_level" not in ds.dims:
            ds = ds.rename({"level": "pressure_level"})
        if "values" not in ds.dims and "number_of_points" in ds.dims:
            ds = ds.rename({"number_of_points": "values"})

        # subset vars present here
        keep = [v for v in vars_to_open if v in ds.data_vars]
        if keep:
            ds = ds[keep]
        else:
            ds = xr.Dataset(coords={k: v for k, v in ds.coords.items()})

        ds.attrs.pop("history", None)
        return ds

    def _norm_idx(self, idx: TIndex, n: int) -> np.ndarray:
        if isinstance(idx, slice):
            start = 0 if idx.start is None else int(idx.start)
            stop  = n if idx.stop  is None else int(idx.stop)
            step  = 1 if idx.step  is None else int(idx.step)
            return np.arange(max(0, start), min(stop, n), step, dtype=int)
        if isinstance(idx, tuple) and len(idx) == 2:
            a, b = int(idx[0]), int(idx[1])
            if b < a:
                return np.arange(0, dtype=int)
            return np.arange(max(0, a), min(b + 1, n), dtype=int)
        arr = np.atleast_1d(np.asarray(idx, dtype=int))
        return arr[(arr >= 0) & (arr < n)]

    def _select(self, kind: str) -> np.ndarray:
        filt = self.stream_info.get(kind)
        excl = self.stream_info.get(f"{kind}_exclude", []) or []
        if filt is None:
            mask = np.ones(len(self.all_channels), bool)
        else:
            mask = np.array([any(s in ch for s in filt) for ch in self.all_channels], bool)
        if excl:
            exm = np.array([any(s in ch for s in excl) for ch in self.all_channels], bool)
            mask &= ~exm
        return np.nonzero(mask)[0].astype(int)


    def _months_in_range(self, t0: np.datetime64, t1: np.datetime64) -> set[str]:
        y0, m0 = int(str(t0)[:4]), int(str(t0)[5:7])
        y1, m1 = int(str(t1)[:4]), int(str(t1)[5:7])
        out, y, m = set(), y0, m0
        while (y < y1) or (y == y1 and m <= m1):
            out.add(f"{y:04d}{m:02d}")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        return out

    def _ref_yyyymm(self, ref: dict) -> str | None:
        refs_map = ref.get("refs", {})
        key = next((k for k in refs_map if k.startswith("valid_time/") and not k.endswith("/.zarray") and not k.endswith("/.zattrs")), None)
        if not key:
            return None
        v = refs_map.get(key)
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            fname = v[0].rsplit("/", 1)[-1]
            m = re.search(r"(\d{6})", fname)
            return m.group(1) if m else None
        return None

    def _ref_has_any_var(self, ref: dict, vars_short: list[str]) -> bool:
        refs_map = ref.get("refs", {})
        return any((f"{v}/.zarray" in refs_map) or (f"{v}/.zattrs" in refs_map) for v in vars_short)

    # ---------- ctor ----------
    @override
    def __init__(self, tw_handler: TimeWindowHandler, filename: Path, stream_info: dict) -> None:
        self.stream_info = stream_info
        self.name = stream_info.get("name", "era5_kerchunk")

        # variables
        vars_cfg = stream_info.get("variables", DEFAULT_VARS)
        vars_short = [ALIASES.get(v, v) for v in vars_cfg]
        self.vars_short = [v for v in vars_short if v in SHORTS] or DEFAULT_VARS

        self.levels = stream_info.get("levels", DEFAULT_LEVELS)
        self.prefer_expver = int(stream_info.get("prefer_expver", 5))
        self.user_chunks: dict[str, int] = stream_info.get("chunks", {}) or {}

        # window
        t0 = np.datetime64(tw_handler.t_start, "ns")
        t1 = np.datetime64(tw_handler.t_end, "ns")

        # load manifest
        ref_path = Path(stream_info.get("reference_path", filename))
        m = json.loads(Path(ref_path).read_text())
        # refs = m.get("refs", [])
        # if not (isinstance(m, dict) and m.get("kind") == "kerchunk-manifest" and isinstance(refs, list) and refs):
        #     raise ValueError(f"{ref_path} is not a non-empty kerchunk-manifest.")
        if isinstance(m, dict) and m.get("kind") == "kerchunk-manifest" and isinstance(m.get("refs"), list) and m["refs"]:
            refs = m["refs"]
            single_ref_mode = False
        elif isinstance(m, dict) and isinstance(m.get("refs"), dict):
            # single combined kerchunk reference (MZZ output)
            refs = [m]
            single_ref_mode = True
        else:
            raise ValueError(f"{ref_path} is not a kerchunk-manifest or a single combined reference.")

        wanted_months = self._months_in_range(np.datetime64(tw_handler.t_start, "ns"),
                                            np.datetime64(tw_handler.t_end, "ns"))
        # vars_to_open = self.vars_short[:]

        # filtered_refs = []
        # for r in refs:
        #     if not self._ref_has_any_var(r, vars_to_open):
        #         continue
        #     mm = self._ref_yyyymm(r)
        #     if (mm is not None) and (mm not in wanted_months):
        #         continue
        #     filtered_refs.append(r)
        vars_to_open = self.vars_short[:]
        filtered_refs = []
        if single_ref_mode:
            filtered_refs = refs[:]               # cannot filter by month; time slicing happens later
        else:
            wanted_months = self._months_in_range(np.datetime64(tw_handler.t_start, "ns"),
                                                np.datetime64(tw_handler.t_end, "ns"))
            for r in refs:
                if not self._ref_has_any_var(r, vars_to_open):
                    continue
                mm = self._ref_yyyymm(r)
                if (mm is not None) and (mm not in wanted_months):
                    continue
                filtered_refs.append(r)

        if not filtered_refs:
            _log.warning("%s: manifest filtering removed all refs; window/vars mismatch?", self.name)
            self.init_empty()
            return

        # Open only the subset we actually need:
        dsets = [self._open_ref_fast(r, vars_to_open) for r in filtered_refs]

        # concat by valid_time (minimal alignment)
        ds = xr.concat(
            dsets, dim="valid_time",
            data_vars="minimal", coords="minimal",
            compat="no_conflicts", join="exact",
            combine_attrs="drop",
        )

        # choose expver if present
        # Prefer expver=5 but fill gaps from expver=1, else fall back to the only member
        if "expver" in ds.dims:
            exp_vals = np.asarray(ds["expver"].values).tolist()
            if (5 in exp_vals) and (1 in exp_vals):
                # take 5 where present, otherwise 1
                ds5 = ds.sel(expver=5)
                ds1 = ds.sel(expver=1)
                ds = ds5.combine_first(ds1)
            else:
                # single-member expver; pick it
                ds = ds.isel(expver=0)


        # sort & window via boolean mask (no pandas slicing)
        ds = ds.sortby("valid_time")
        mask = (ds["valid_time"] >= t0) & (ds["valid_time"] <= t1)
        ds = ds.isel(valid_time=mask).rename({"valid_time": "time"})

        if ds.sizes.get("time", 0) == 0:
            _log.warning("%s: empty time selection", self.name)
            self.init_empty()
            return

        # subset & order levels
        if "pressure_level" in ds.dims:
            present_levels = [int(x) for x in np.asarray(ds["pressure_level"].values)]
            keep_levels = [L for L in self.levels if L in present_levels]
            ds = ds.sel(pressure_level=keep_levels).sortby("pressure_level")

        # choose sensible chunks: align time to window size if possible
        times = ds["time"].values
        if times.size >= 2:
            deltas = times[1:] - times[:-1]
            vals, counts = np.unique(deltas, return_counts=True)
            period = vals[np.argmax(counts)]
            if period <= np.timedelta64(0, "ns"):
                period = np.timedelta64(6, "h")
        else:
            period = np.timedelta64(6, "h")

        try:
            win_steps = int(round(np.float64((np.array(tw_handler.t_window_len) / np.array(period)))))
            if win_steps <= 0:
                win_steps = 1
        except Exception:
            win_steps = 1


        if self.user_chunks:
            ds = ds.chunk(self.user_chunks)
        else:
            # typical ERA5 chunks: (time≈40, level≈4, values=13440); for small windows prefer time=win_steps
            chunks = {"time": win_steps}
            if "pressure_level" in ds.dims:
                chunks["pressure_level"] = -1  # single chunk across levels
            ds = ds.chunk(chunks)

        # cache coords and times
        self.ds = ds
        self.times = ds["time"].values
        self.len = int(self.times.size)
        self.n_values = int(ds.sizes["values"])
        latname = "latitude" if "latitude" in ds.coords else "lat"
        lonname = "longitude" if "longitude" in ds.coords else "lon"
        # self.latitudes = _clip_lat(np.asarray(ds[latname].values))
        # self.longitudes = _clip_lon(np.asarray(ds[lonname].values))
        lat = np.asarray(ds[latname].values)
        lon = np.asarray(ds[lonname].values)
        # handle possible shapes: (V,), (2,V), (V,1), (1,V), even extra leading dims
        lat = np.squeeze(lat)
        lon = np.squeeze(lon)
        if lat.ndim == 2 and lat.shape[0] == 2 and lat.shape[1] != 2:
            # extremely rare: some pipelines store a stacked [lat,lon] array
            # pick the first row as latitude in that case
            lat = lat[0]
        if lon.ndim == 2 and lon.shape[0] == 2 and lon.shape[1] != 2:
            lon = lon[1]

        lat = lat.ravel().astype(np.float32)
        lon = lon.ravel().astype(np.float32)

        if lat.size != lon.size:
            raise ValueError(f"lat/lon size mismatch: {lat.shape} vs {lon.shape}")
        self.latitudes  = _clip_lat(lat)
        self.longitudes = _clip_lon(lon)
        self.n_values   = int(lat.size)  # keep V in sync with coords
        # build channel map (var-major × levels)
        keep_vars = [v for v in self.vars_short if v in ds.data_vars]
        self.levels_present = [int(x) for x in (ds["pressure_level"].values if "pressure_level" in ds.dims else [])]
        self.all_channels: list[str] = []
        self._channel_pos: dict[tuple[str, int], int] = {}
        k = 0
        for v in keep_vars:
            for L in self.levels_present:
                self.all_channels.append(f"{v}_{int(L)}")
                self._channel_pos[(v, int(L))] = k
                k += 1

        # pre-stack once: (time, values, channel)
        arr = ds[keep_vars].to_array("var").transpose("time", "values", "var", "pressure_level")
        arr = arr.stack(channel=("var", "pressure_level"))
        self._arr = arr  # lazy dask-backed DataArray

        # selections from YAML (substring matching)
        self.source_idx = self._select("source")
        self.target_idx = self._select("target")
        self.source_channels = [self.all_channels[i] for i in self.source_idx]
        self.target_channels = [self.all_channels[i] for i in self.target_idx]
        self.geoinfo_channels: list[str] = []
        self.geoinfo_idx: list[int] = []

        # identity normalization (can be filled externally)
        nC = len(self.all_channels)
        self.mean = np.zeros(nC, dtype=np.float32)
        self.stdev = np.ones(nC, dtype=np.float32)

        # init base
        super().__init__(tw_handler, stream_info, self.times[0], self.times[-1], period)

    # ---------- protocol ----------
    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self._arr = None
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
        return getattr(self, "len", 0)

    # @override
    # def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
    #     if self._arr is None or self.len == 0 or len(channels_idx) == 0:
    #         return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

    #     t_idxs = self._norm_idx(idx, self.len)
    #     if t_idxs.size == 0:
    #         return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

    #     i0, i1 = int(t_idxs[0]), int(t_idxs[-1]) + 1
    #     T, V = (i1 - i0), self.n_values

    #     csel = np.asarray(channels_idx, dtype=int)
    #     # slice pre-stacked array → (T, V, C_sel)
    #     block = self._arr.isel(time=slice(i0, i1), channel=csel).data
    #     data = np.asarray(block.compute(), dtype=np.float32).reshape(T * V, csel.size)

    #     # coords and times
    #     # latlon = np.stack([self.latitudes, self.longitudes], axis=1).astype(np.float32)  # (V,2)
    #     # coords = np.broadcast_to(latlon, (T, V, 2)).reshape(T * V, 2)
    #     lat = self.latitudes.reshape(-1, 1)                    # (V,1)
    #     lon = self.longitudes.reshape(-1, 1)                   # (V,1)
    #     latlon = np.concatenate([lat, lon], axis=1)            # (V,2)
    #     coords = np.repeat(latlon[None, :, :], T, axis=0)      # (T,V,2)
    #     coords = coords.reshape(T * V, 2).astype(np.float32)
    #     times = self.times[i0:i1]
    #     datetimes = np.repeat(times, V)

    #     return ReaderData(
    #         coords=coords,
    #         geoinfos=np.zeros((T * V, 0), dtype=np.float32),
    #         data=data,
    #         datetimes=datetimes,
    #     )
    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        if self._arr is None or self.len == 0 or len(channels_idx) == 0:
            return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

        # Use base helper: may return a slice OR an array of ints + a debug tuple
        t_idxs, dtr = self._get_dataset_idxs(idx)
        if t_idxs is None:
            return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

        # Channel select
        csel = np.asarray(channels_idx, dtype=int)

        # Time select: pass slice through unchanged; else pass the int array
        time_sel = t_idxs if isinstance(t_idxs, slice) else np.asarray(t_idxs, dtype=int)

        # (T, V, C_sel) dask array
        block = self._arr.isel(time=time_sel, channel=csel).data

        # Get shapes from the actual selection (robust for slice/array)
        T = int(block.shape[0])
        V = int(block.shape[1])
        C = int(block.shape[2])

        # Materialize data
        data = np.asarray(block.compute(), dtype=np.float32).reshape(T * V, C)

        # --- FIX: build coords using V from block and tile, no 3D broadcast/reshape ---
        lat = self.latitudes.ravel().astype(np.float32)
        lon = self.longitudes.ravel().astype(np.float32)
        if lat.size != lon.size:
            raise ValueError(f"lat/lon size mismatch: {lat.shape} vs {lon.shape}")

        latlon = np.column_stack([lat, lon])          # (V_full, 2)
        if latlon.shape[0] < V:
            raise ValueError(f"coords shorter than V: {latlon.shape[0]} < {V}")

        # Tile exactly to T*V rows
        coords = np.tile(latlon[:V], (T, 1)).astype(np.float32)   # (T*V, 2)

        # Datetimes (works for slice or ndarray)
        times = self.times[time_sel]
        datetimes = np.repeat(np.asarray(times), V)

        # Optional consistency checks
        # assert data.shape[0] == coords.shape[0] == datetimes.shape[0]

        return ReaderData(
            coords=coords,
            geoinfos=np.zeros((T * V, 0), dtype=np.float32),
            data=data,
            datetimes=datetimes,
        )
