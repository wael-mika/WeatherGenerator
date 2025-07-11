# import json
# import logging
# from pathlib import Path
# from typing import override

# import fsspec
# import numpy as np
# import xarray as xr

# from weathergen.datasets.data_reader_base import (
#     DataReaderTimestep,
#     ReaderData,
#     TimeWindowHandler,
#     TIndex,
#     check_reader_data,
# )

# _logger = logging.getLogger(__name__)

# class RadklimKerchunkReader(DataReaderTimestep):
#     """
#     Reader for RADKLIM data accessed via a Kerchunk reference.
#     Handles empty windows and out-of-range requests robustly.
#     """

#     def __init__(
#         self,
#         tw_handler: TimeWindowHandler,
#         filename: Path,
#         stream_info: dict,
#     ) -> None:
#         self._empty = False
#         self.ds: xr.Dataset | None = None

#         # Paths
#         self.ref_path = Path(stream_info.get("reference", filename))
#         self.norm_path = Path(stream_info.get("stats_path"))
#         self.sample_coord_file = Path(stream_info["sample_coord_file"])
#         self.stream_info = stream_info

#         # Channels
#         self.source_channels = ["RR"]
#         self.target_channels = ["RR"]
#         self.geoinfo_channels: list[str] = []

#         self.source_idx = list(range(len(self.source_channels)))
#         self.target_idx = list(range(len(self.target_channels)))
#         self.geoinfo_idx = list(range(len(self.geoinfo_channels)))

#         # File existence checks
#         if not self.ref_path.exists():
#             raise FileNotFoundError(f"Kerchunk reference not found: {self.ref_path}")
#         if not self.norm_path.exists():
#             raise FileNotFoundError(f"Normalization JSON not found: {self.norm_path}")
#         if not self.sample_coord_file.exists():
#             raise FileNotFoundError(f"Sample coord file not found: {self.sample_coord_file}")

#         # Load available time axis
#         _logger.info("Reading time metadata from: %s", self.ref_path)
#         with fsspec.open(self.ref_path, "rt") as f:
#             kerchunk_ref = json.load(f)
#         fs_meta = fsspec.filesystem("reference", fo=kerchunk_ref)
#         mapper_meta = fs_meta.get_mapper("")
#         with xr.open_dataset(mapper_meta, engine="zarr", consolidated=False, chunks={}) as ds_meta:
#             times_full = ds_meta["time"].values

#         # Handle empty dataset (no time dimension)
#         if times_full.size == 0:
#             super().__init__(tw_handler, stream_info, None, None, None)
#             self.init_empty()
#             return

#         # Ensure regular timesteps
#         deltas = np.unique(np.diff(times_full.astype("datetime64[s]")))
#         if deltas.size != 1:
#             raise ValueError("Irregular time steps in Kerchunk reference")
#         period = deltas[0]
#         super().__init__(
#             tw_handler,
#             stream_info,
#             times_full[0],
#             times_full[-1],
#             period,
#         )

#         # If requested window does not overlap, go empty
#         if tw_handler.t_start >= times_full[-1] or tw_handler.t_end <= times_full[0]:
#             super().__init__(tw_handler, stream_info, None, None, None)
#             self.init_empty()
#             return

#         self.start_idx = int(np.searchsorted(times_full, tw_handler.t_start, "left"))
#         self.end_idx = int(np.searchsorted(times_full, tw_handler.t_end, "right"))
#         self.num_steps_per_window = int(tw_handler.t_window_len / period)

#         # Load normalization stats
#         stats = json.loads(self.norm_path.read_text())
#         self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
#         self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)
#         self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
#         self.stdev_geoinfo = np.asarray(stats.get("std_geoinfo", []), dtype=np.float32)

#         if len(self.mean) != len(self.source_channels):
#             raise ValueError("Stats length ≠ number of source channels")
        
#     def is_empty(self) -> bool:
#         return self._empty
    
#     @override
#     def init_empty(self) -> None:
#         self._empty = True
#         self.ds = None
#         self.start_idx = None
#         self.end_idx = None
#         super().init_empty()

#     @override
#     def length(self) -> int:
#         if self._empty:
#             return 0
#         nt = self.end_idx - self.start_idx
#         return max(0, nt - self.num_steps_per_window + 1)

#     def _lazy_open(self):
#         if self._empty or self.ds is not None:
#             return

#         # Open dataset
#         with open(self.ref_path) as f:
#             kerchunk_ref = json.load(f)
#         fs = fsspec.filesystem("reference", fo=kerchunk_ref)
#         mapper = fs.get_mapper("")
#         ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=False)

#         # Drop existing lat/lon and subset to time window
#         ds_full = ds_full.drop_vars(["lat", "lon"], errors="ignore")
#         subset = ds_full[self.source_channels].isel(time=slice(self.start_idx, self.end_idx))
#         if "chunks" in self.stream_info:
#             subset = subset.chunk(self.stream_info["chunks"])

#         ny = subset.sizes["y"]
#         nx = subset.sizes["x"]
#         y_slice = slice(0, ny)
#         x_slice = slice(0, nx)
#         self.ds = subset.isel(y=y_slice, x=x_slice)

#         # Inject static coordinates
#         with xr.open_dataset(self.sample_coord_file) as ds_sample:
#             lat2d = ds_sample["lat"].values.astype(np.float32)[y_slice, x_slice]
#             lon2d = ds_sample["lon"].values.astype(np.float32)[y_slice, x_slice]
#             self.ds = self.ds.assign_coords(
#                 lat=(("y", "x"), lat2d),
#                 lon=(("y", "x"), lon2d),
#             )
#         self.ny, self.nx = lat2d.shape
#         self.points_per_slice = self.ny * self.nx

#     @override
#     def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
#         """
#         Fetch one training window.

#         Returns a ReaderData with:
#         - coords: [N, 2] (lat, lon) for each kept point
#         - geoinfos: [N, G] (here empty -> zeros)
#         - data: [N, C] (C == len(channels_idx))
#         - datetimes: [N] np.datetime64[ns]
#         Handles sparse RADKLIM by keeping all positive RR and a small random
#         fraction of zeros; guarantees a non-empty return.
#         """
#         # Fast path for empty reader
#         if self._empty:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         # Make sure dataset is opened
#         try:
#             self._lazy_open()
#         except Exception as e:
#             _logger.error("[RADKLIM] Lazy open failed: %s", e)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         if self.ds is None or "time" not in self.ds.dims:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))


#         # Resolve absolute time indices for this sample
#         try:
#             t_idxs_abs, dtr = self._get_dataset_idxs(idx)
#         except Exception as e:
#             _logger.error("[RADKLIM] _get_dataset_idxs failed: %s", e)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         if t_idxs_abs is None or len(t_idxs_abs) == 0:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         # Convert absolute -> relative indices in the already-opened subset
#         # _logger.debug(
#         #     "[RADKLIM][%s] t_idxs_abs=%s (len=%d), start_idx=%s, end_idx=%s, ds_time_len=%d",
#         #     str(idx), t_idxs_abs.tolist(), len(t_idxs_abs), self.start_idx, self.end_idx, self.ds.sizes.get("time", -1)
#         # )
#         t_rel = t_idxs_abs - self.start_idx
#         if (
#             t_rel.size == 0
#             or np.any(t_rel < 0)
#             or np.any(t_rel >= self.ds.sizes["time"])
#         ):
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         # Slice the time window
#         t0 = int(t_rel[0])
#         t1 = int(t_rel[-1]) + 1  # inclusive end
#         ds_win = self.ds.isel(time=slice(t0, t1))

#         # To array -> [time, y, x, var]
#         da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
#         raw = da.values
#         if np.isnan(raw).any():
#             raw = np.nan_to_num(raw, nan=0.0)

#         # Flatten spatial/time -> [T*Y*X, var]
#         flat_all = raw.reshape(-1, raw.shape[-1]).astype(np.float32)
#         # Select requested variables (channels_idx refers to var dimension in ds_win)
#         flat_data = flat_all[..., channels_idx]

#         # Build coords/times aligned with flattened data
#         lat2d = self.ds["lat"].values.astype(np.float32)
#         lon2d = self.ds["lon"].values.astype(np.float32)
#         flat_coords_one = np.stack([lat2d.ravel(), lon2d.ravel()], axis=1)  # [Y*X, 2]
#         full_coords = np.tile(flat_coords_one, (ds_win.sizes["time"], 1)).astype(np.float32)
#         full_times = np.repeat(
#             ds_win["time"].values.astype("datetime64[ns]"),
#             flat_coords_one.shape[0],
#         )

#         # -------- Sparse filtering for RADKLIM RR --------
#         # Threshold and keep fraction can be tuned via stream_info
#         eps = float(0.0)        # e.g., 0.05
#         keep_zero_frac = float( 0.01)  # 1%
#         max_points = int(2_000_000)

#         # We only have one source channel "RR"; after selection it's column 0.
#         rr = flat_data[:, 0] if flat_data.shape[1] > 0 else np.zeros((flat_data.shape[0],), dtype=np.float32)

#         pos_mask = rr > eps
#         neg_mask = ~pos_mask

#         if keep_zero_frac > 0.0 and neg_mask.any():
#             rng = np.random.default_rng()
#             neg_idxs = np.flatnonzero(neg_mask)
#             sampled = rng.random(neg_idxs.size) < keep_zero_frac
#             keep_mask = np.zeros_like(pos_mask, dtype=bool)
#             keep_mask[pos_mask] = True
#             if sampled.any():
#                 keep_mask[neg_idxs[sampled]] = True
#         else:
#             keep_mask = pos_mask

#         # Ensure we never return an empty sample
#         if not np.any(keep_mask):
#             fallback = min(4096, flat_data.shape[0])
#             keep_mask[:fallback] = True

#         # Optional cap to protect memory
#         kept_count = int(keep_mask.sum())
#         if kept_count > max_points:
#             rng = np.random.default_rng()
#             idxs = np.flatnonzero(keep_mask)
#             choose = rng.choice(idxs, size=max_points, replace=False)
#             keep_mask = np.zeros_like(keep_mask, dtype=bool)
#             keep_mask[choose] = True
#             kept_count = max_points

#         # Apply mask
#         flat_data = flat_data[keep_mask]
#     #     _logger.debug(
#     #     "[RADKLIM][%s] raw window shape: %s, flat_data.shape=%s, RR stats: min=%.3f, max=%.3f, nz_frac=%.4f",
#     #     str(idx), raw.shape, flat_data.shape,
#     #     rr.min() if rr.size else -1,
#     #     rr.max() if rr.size else -1,
#     #     float((rr > eps).sum()) / rr.size if rr.size else 0.0
#     # )

#         full_coords = full_coords[keep_mask]
#         full_times = full_times[keep_mask]

#         # Debug/trace
#         nz_kept = int((rr[keep_mask] > eps).sum()) if rr.size else 0
#         _logger.info(
#             "[RADKLIM][%s] kept=%d (nz=%d, zeros=%d) out of %d (eps=%.3f, keep_zero=%.3f)",
#             str(idx), kept_count, nz_kept, kept_count - nz_kept, rr.size, eps, keep_zero_frac
#         )

#         # Final safety checks
#         assert flat_data.shape[0] == full_coords.shape[0] == full_times.shape[0], \
#             f"row mismatch data={flat_data.shape} coords={full_coords.shape} times={full_times.shape}"
#         assert flat_data.ndim == 2 and full_coords.shape[1] == 2, \
#             f"bad shapes data={flat_data.shape} coords={full_coords.shape}"

#         rdata = ReaderData(
#             coords=full_coords.astype(np.float32),
#             geoinfos=np.zeros((flat_data.shape[0], len(self.geoinfo_idx)), dtype=np.float32),
#             data=flat_data.astype(np.float32),
#             datetimes=full_times,
#         )
#         check_reader_data(rdata, dtr)
#         return rdata

# import json
# import logging
# from pathlib import Path
# from typing import Optional, override

# import fsspec
# import numpy as np
# import xarray as xr

# from weathergen.datasets.data_reader_base import (
#     DataReaderTimestep,
#     ReaderData,
#     TimeWindowHandler,
#     TIndex,
#     check_reader_data,
# )

# _logger = logging.getLogger(__name__)


# class RadklimKerchunkReader(DataReaderTimestep):
#     """
#     Minimal RADKLIM kerchunk reader.

#     Constraints:
#       - Uses dataset-provided geodetic coordinates only (lat/lon in degrees).
#       - If lat/lon are missing or unusable, the reader returns empty for that window.
#       - No subsampling: returns all pixels in the sliced window.
#       - Reads normalization stats from JSON if provided.
#     """

#     def __init__(
#         self,
#         tw_handler: TimeWindowHandler,
#         filename: Path,
#         stream_info: dict,
#     ) -> None:
#         self._empty: bool = False
#         self.ds: Optional[xr.Dataset] = None
#         self.stream_info = dict(stream_info or {})

#         # ---- Paths ----
#         self.ref_path = Path(self.stream_info.get("reference", filename))
#         stats_path = self.stream_info.get("stats_path")
#         if not self.ref_path.exists():
#             raise FileNotFoundError(f"Kerchunk reference not found: {self.ref_path}")

#         # ---- Channels ----
#         self.source_channels: list[str] = list(self.stream_info.get("source_channels", ["RR"]))
#         self.target_channels: list[str] = list(self.stream_info.get("target_channels", ["RR"]))
#         self.geoinfo_channels: list[str] = list(self.stream_info.get("geoinfo_channels", []))
#         self.source_idx = list(range(len(self.source_channels)))
#         self.target_idx = list(range(len(self.target_channels)))
#         self.geoinfo_idx = list(range(len(self.geoinfo_channels)))
#         self._coords_ready: bool = False
#         self._base_coords: Optional[np.ndarray] = None   # [ny*nx, 2] float32
#         self._valid_px: Optional[np.ndarray] = None      # [ny*nx] bool
#         self._ny: Optional[int] = None
#         self._nx: Optional[int] = None


#         # ---- Time axis & base init ----
#         _logger.info("Reading time metadata from kerchunk reference: %s", self.ref_path)
#         with fsspec.open(self.ref_path, "rt") as f:
#             kerchunk_ref = json.load(f)
#         fs_meta = fsspec.filesystem("reference", fo=kerchunk_ref)
#         mapper_meta = fs_meta.get_mapper("")
#         with xr.open_dataset(mapper_meta, engine="zarr", consolidated=False, chunks={}) as ds_meta:
#             if "time" not in ds_meta:
#                 super().__init__(tw_handler, stream_info, None, None, None)
#                 self.init_empty()
#                 return
#             times_full = ds_meta["time"].values

#         if times_full.size == 0:
#             super().__init__(tw_handler, stream_info, None, None, None)
#             self.init_empty()
#             return

#         deltas = np.unique(np.diff(times_full.astype("datetime64[s]")))
#         if deltas.size != 1:
#             raise ValueError("Irregular time steps in Kerchunk reference (non-constant Δtime).")
#         period = deltas[0]

#         if tw_handler.t_start >= times_full[-1] or tw_handler.t_end <= times_full[0]:
#             super().__init__(tw_handler, stream_info, None, None, None)
#             self.init_empty()
#             return

#         super().__init__(tw_handler, stream_info, times_full[0], times_full[-1], period)
#         self.start_idx = int(np.searchsorted(times_full, tw_handler.t_start, "left"))
#         self.end_idx = int(np.searchsorted(times_full, tw_handler.t_end, "right"))
#         self.num_steps_per_window = max(1, int(tw_handler.t_window_len / period))

#         # ---- Optional stats ----
#         self.mean = np.asarray([], dtype=np.float32)
#         self.stdev = np.asarray([], dtype=np.float32)
#         self.mean_geoinfo = np.asarray([], dtype=np.float32)
#         self.stdev_geoinfo = np.asarray([], dtype=np.float32)
#         if stats_path:
#             try:
#                 stats = json.loads(Path(stats_path).read_text())
#                 self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
#                 self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)
#                 self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
#                 self.stdev_geoinfo = np.asarray(stats.get("std_geoinfo", []), dtype=np.float32)
#                 if self.mean.size and (self.mean.size != len(self.source_channels)):
#                     _logger.warning(
#                         "Stats length (%d) != #source channels (%d). Ignoring stats.",
#                         self.mean.size, len(self.source_channels)
#                     )
#                     self.mean = self.stdev = np.asarray([], dtype=np.float32)
#             except Exception as e:
#                 _logger.warning("Failed to load stats: %s. Continuing without stats.", e)

#     def is_empty(self) -> bool:
#         return self._empty

#     @override
#     def init_empty(self) -> None:
#         self._empty = True
#         self.ds = None
#         self.start_idx = None
#         self.end_idx = None
#         super().init_empty()

#     @override
#     def length(self) -> int:
#         if self._empty or self.start_idx is None or self.end_idx is None:
#             return 0
#         nt = self.end_idx - self.start_idx
#         return max(0, nt - self.num_steps_per_window + 1)

#     # ---- internals ----

#     def _lazy_open(self) -> None:
#         if self._empty or self.ds is not None:
#             return
#         with fsspec.open(self.ref_path, "rt") as f:
#             kerchunk_ref = json.load(f)
#         fs = fsspec.filesystem("reference", fo=kerchunk_ref)
#         mapper = fs.get_mapper("")
#         ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=False)

#         # Keep only channels + lat/lon if present
#         vars_to_keep = list(self.source_channels)
#         for v in ("lat", "lon"):  # require geodetic coords only
#             if v in ds_full.variables and v not in vars_to_keep:
#                 vars_to_keep.append(v)
#             if v in ds_full.coords and v not in vars_to_keep:
#                 vars_to_keep.append(v)

#         missing = [v for v in self.source_channels if v not in ds_full.variables]
#         if missing:
#             _logger.error("Missing expected variables in dataset: %s", missing)
#             kept = [v for v in vars_to_keep if (v in ds_full.variables or v in ds_full.coords)]
#             if not kept:
#                 self.init_empty()
#                 return
#             ds_full = ds_full[kept]
#         else:
#             ds_full = ds_full[vars_to_keep]

#         if "time" not in ds_full.dims:
#             self.init_empty()
#             return

#         ds_win = ds_full.isel(time=slice(int(self.start_idx), int(self.end_idx)))

#         self.ds = ds_win
            
#     @override
#     def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
#         try:
#             self._lazy_open()
#         except Exception as e:
#             _logger.error("[RADKLIM] Lazy open failed: %s", e)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         if self._empty or self.ds is None or "time" not in self.ds.dims:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         try:
#             t_idxs_abs, dtr = self._get_dataset_idxs(idx)
#         except Exception as e:
#             _logger.error("[RADKLIM] _get_dataset_idxs failed: %s", e)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#         if t_idxs_abs is None or t_idxs_abs.size == 0 or self.start_idx is None:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         t_rel = t_idxs_abs - self.start_idx
#         T_avail = int(self.ds.sizes.get("time", 0))
#         if t_rel.size == 0 or np.any(t_rel < 0) or np.any(t_rel >= T_avail):
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         t0 = int(t_rel[0]); t1 = int(t_rel[-1]) + 1
#         ds_win = self.ds.isel(time=slice(t0, t1))
#         T = int(ds_win.sizes.get("time", 0))
#         if T == 0:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         if not self._ensure_coords_cache():
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#         ny, nx = self._ny, self._nx
#         pts = int(ny * nx)

#         # Data → [T,y,x,V]
#         try:
#             da = ds_win[self.source_channels].to_array(dim="var").transpose("time","y","x","var")
#         except Exception as e:
#             _logger.error("[RADKLIM] to_array/transpose failed: %s", e)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         raw = da.values
#         if raw.size == 0:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#         if np.isnan(raw).any():
#             raw = np.nan_to_num(raw, nan=0.0)

#         _, y_, x_, V = raw.shape
#         if y_ != ny or x_ != nx:
#             _logger.error("Data grid (%d,%d) != cached coords grid (%d,%d).", y_, x_, ny, nx)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         flat_all = raw.reshape(-1, V).astype(np.float32, copy=False)
#         try:
#             flat_data = flat_all[:, channels_idx]
#         except Exception:
#             keep = min(V, len(channels_idx))
#             flat_data = flat_all[:, :keep]

#         # Coords/times (dense, no filtering)
#         base_coords = self._base_coords  # [pts,2]
#         full_coords = np.tile(base_coords, (T, 1))                       # [T*pts, 2]
#         times = ds_win["time"].values.astype("datetime64[ns]")
#         full_times = np.repeat(times, pts)                                # [T*pts]

#         # Hard shape agreement checks (prevent downstream index errors)
#         N = flat_data.shape[0]
#         if not (N == full_coords.shape[0] == full_times.shape[0] == T * pts):
#             _logger.error("Row mismatch: data=%s coords=%s times=%s (expected %d)",
#                         flat_data.shape, full_coords.shape, full_times.shape, T * pts)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         rdata = ReaderData(
#             coords=full_coords,
#             geoinfos=np.zeros((N, len(self.geoinfo_idx)), dtype=np.float32),
#             data=flat_data,
#             datetimes=full_times,
#         )
#         check_reader_data(rdata, dtr)
#         return rdata


#     def _ensure_coords_cache(self) -> bool:
#         if self._coords_ready:
#             return True
#         if self.ds is None:
#             return False

#         ny = self.ds.sizes.get("y"); nx = self.ds.sizes.get("x")
#         if ny is None or nx is None:
#             _logger.error("Dataset missing y/x dims; cannot cache coordinates.")
#             return False

#         ds0 = self.ds

#         def _get_da(*names: str):
#             for n in names:
#                 if n in ds0: return ds0[n]
#                 if n in ds0.coords: return ds0.coords[n]
#                 if n in ds0.data_vars: return ds0.data_vars[n]
#             return None

#         lat_da = _get_da("lat", "latitude")
#         lon_da = _get_da("lon", "longitude")
#         if lat_da is None or lon_da is None:
#             _logger.error("Required geodetic coordinates missing (lat/lon).")
#             return False

#         def _reduce_to_yx(da: xr.DataArray) -> xr.DataArray | None:
#             for d in list(da.dims):
#                 if d not in ("y", "x"):
#                     da = da.isel({d: 0})
#             if set(da.dims) == {"y","x"}:
#                 return da.transpose("y","x") if da.dims != ("y","x") else da
#             if da.dims in (("y",), ("x",)):
#                 return da
#             return None

#         lat2 = _reduce_to_yx(lat_da)
#         lon2 = _reduce_to_yx(lon_da)
#         if lat2 is None or lon2 is None:
#             _logger.error("Could not reduce lat/lon to (y,x); lat dims=%s lon dims=%s", lat_da.dims, lon_da.dims)
#             return False

#         if lat2.ndim == 1 and lon2.ndim == 1:
#             if lat2.sizes.get("y") != ny or lon2.sizes.get("x") != nx:
#                 _logger.error("1D lat(y)/lon(x) sizes don’t match grid (%d,%d).", ny, nx)
#                 return False
#             yy, xx = np.meshgrid(lat2.values, lon2.values, indexing="ij")
#             lat2d = yy.astype(np.float32, copy=False)
#             lon2d = xx.astype(np.float32, copy=False)
#         else:
#             if tuple(lat2.shape) != (ny, nx) or tuple(lon2.shape) != (ny, nx):
#                 _logger.error("2D lat/lon shapes %s %s don’t match grid (%d,%d).",
#                             lat2.shape, lon2.shape, ny, nx)
#                 return False
#             lat2d = lat2.values.astype(np.float32, copy=False)
#             lon2d = lon2.values.astype(np.float32, copy=False)

#         # (Optional) sanity once — but DO NOT mask later
#         if not (np.isfinite(lat2d).all() and np.isfinite(lon2d).all()):
#             _logger.error("Non-finite lat/lon in grid; refusing to proceed.")
#             return False
#         if not (-90 <= lat2d.min() <= 90 and -90 <= lat2d.max() <= 90
#                 and -180 <= lon2d.min() <= 180 and -180 <= lon2d.max() <= 180):
#             _logger.error("lat/lon outside geodetic bounds.")
#             return False

#         self._base_coords = np.stack([lat2d.ravel(), lon2d.ravel()], axis=1).astype(np.float32, copy=False)
#         self._ny, self._nx = int(ny), int(nx)
#         self._coords_ready = True
#         _logger.info("[RADKLIM] Cached coords (%d,%d) -> base_coords %s", ny, nx, self._base_coords.shape)
#         return True

# import json
# import logging
# from pathlib import Path
# from typing import Optional, override

# import fsspec
# import numpy as np
# import xarray as xr

# from weathergen.datasets.data_reader_base import (
#     DataReaderTimestep,
#     ReaderData,
#     TimeWindowHandler,
#     TIndex,
#     check_reader_data,
# )

# _logger = logging.getLogger(__name__)


# class RadklimKerchunkReader(DataReaderTimestep):
#     """
#     Reader for RADKLIM kerchunk references.

#     Key properties:
#     - Channels taken from stream_info (defaults to ["RR"]).
#     - Requires geodetic coordinates (lat/lon in degrees). If missing in the dataset,
#       injects them from stream_info["sample_coord_file"]. If still unavailable, returns empty.
#     - Robust on empty/out-of-range windows.
#     - Optional sparse sampling to control sample size.
#     """

#     def __init__(
#         self,
#         tw_handler: TimeWindowHandler,
#         filename: Path,
#         stream_info: dict,
#     ) -> None:
#         self._empty: bool = False
#         self.ds: Optional[xr.Dataset] = None
#         self.stream_info = dict(stream_info or {})

#         # -------- Paths --------
#         self.ref_path = Path(self.stream_info.get("reference", filename))
#         self.sample_coord_file = self.stream_info.get("sample_coord_file")
#         stats_path = self.stream_info.get("stats_path")  # optional

#         if not self.ref_path.exists():
#             raise FileNotFoundError(f"Kerchunk reference not found: {self.ref_path}")

#         # -------- Channels (from config) --------
#         self.source_channels: list[str] = list(self.stream_info.get("source_channels", ["RR"]))
#         self.target_channels: list[str] = list(self.stream_info.get("target_channels", ["RR"]))
#         self.geoinfo_channels: list[str] = list(self.stream_info.get("geoinfo_channels", []))

#         # Save indices for the base class compatibility
#         self.source_idx = list(range(len(self.source_channels)))
#         self.target_idx = list(range(len(self.target_channels)))
#         self.geoinfo_idx = list(range(len(self.geoinfo_channels)))

#         # -------- Sparse sampling config --------
#         ssc = self.stream_info.get("sparse_sampling", {}) or {}
#         self.sparse_cfg = {
#             "keep_zero_frac": float(ssc.get("keep_zero_frac", 0.01)),     # ~1% of zeros
#             "positive_threshold": float(ssc.get("positive_threshold", 0.0)),
#             "max_points": int(ssc.get("max_points", 2_000_000)),          # 0 → uncapped
#             "target_points": int(ssc.get("target_points", 0)),            # 0 → off
#             "zero_per_pos": float(ssc.get("zero_per_pos", 0.0)),          # 0 → off
#             "seed": int(ssc.get("seed", 0)),                              # 0 → nondeterministic
#         }

#         # -------- Read time axis & initialize base --------
#         _logger.info("Reading time metadata from kerchunk reference: %s", self.ref_path)
#         with fsspec.open(self.ref_path, "rt") as f:
#             kerchunk_ref = json.load(f)

#         fs_meta = fsspec.filesystem("reference", fo=kerchunk_ref)
#         mapper_meta = fs_meta.get_mapper("")

#         with xr.open_dataset(mapper_meta, engine="zarr", consolidated=False, chunks={}) as ds_meta:
#             if "time" not in ds_meta:
#                 super().__init__(tw_handler, stream_info, None, None, None)
#                 self.init_empty()
#                 return
#             times_full = ds_meta["time"].values

#         if times_full.size == 0:
#             super().__init__(tw_handler, stream_info, None, None, None)
#             self.init_empty()
#             return

#         # Ensure regular sampling
#         deltas = np.unique(np.diff(times_full.astype("datetime64[s]")))
#         if deltas.size != 1:
#             raise ValueError("Irregular time steps in Kerchunk reference (non-constant Δtime).")
#         period = deltas[0]

#         # If requested window doesn't overlap at all, initialize empty
#         if tw_handler.t_start >= times_full[-1] or tw_handler.t_end <= times_full[0]:
#             super().__init__(tw_handler, stream_info, None, None, None)
#             self.init_empty()
#             return

#         # Normal, non-empty init
#         super().__init__(
#             tw_handler,
#             stream_info,
#             times_full[0],
#             times_full[-1],
#             period,
#         )

#         # Precompute time slicing indices
#         self.start_idx = int(np.searchsorted(times_full, tw_handler.t_start, "left"))
#         self.end_idx = int(np.searchsorted(times_full, tw_handler.t_end, "right"))
#         self.num_steps_per_window = max(1, int(tw_handler.t_window_len / period))

#         # -------- Optional normalization stats --------
#         self.mean = np.asarray([], dtype=np.float32)
#         self.stdev = np.asarray([], dtype=np.float32)
#         self.mean_geoinfo = np.asarray([], dtype=np.float32)
#         self.stdev_geoinfo = np.asarray([], dtype=np.float32)
#         if stats_path:
#             try:
#                 stats = json.loads(Path(stats_path).read_text())
#                 self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
#                 self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)
#                 self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
#                 self.stdev_geoinfo = np.asarray(stats.get("std_geoinfo", []), dtype=np.float32)
#                 if len(self.mean) and (len(self.mean) != len(self.source_channels)):
#                     _logger.warning(
#                         "Stats length (%d) != number of source channels (%d). Ignoring stats.",
#                         len(self.mean), len(self.source_channels)
#                     )
#                     self.mean = np.asarray([], dtype=np.float32)
#                     self.stdev = np.asarray([], dtype=np.float32)
#             except Exception as e:
#                 _logger.warning("Failed to load stats: %s. Continuing without stats.", e)

#     def is_empty(self) -> bool:
#         return self._empty

#     @override
#     def init_empty(self) -> None:
#         self._empty = True
#         self.ds = None
#         self.start_idx = None
#         self.end_idx = None
#         super().init_empty()

#     @override
#     def length(self) -> int:
#         if self._empty or self.start_idx is None or self.end_idx is None:
#             return 0
#         nt = self.end_idx - self.start_idx
#         return max(0, nt - self.num_steps_per_window + 1)

#     # ---------- Internals ----------

#     def _rng_for(self, idx: TIndex) -> np.random.Generator:
#         """Deterministic RNG per-sample if seed != 0, else nondeterministic."""
#         seed = int(self.sparse_cfg["seed"])
#         if seed == 0:
#             return np.random.default_rng()
#         # mix idx into seed to vary across samples
#         idx_hash = int(np.int64(abs(hash(idx))) & 0x7FFFFFFF)
#         mixed = (seed * 1_000_003) ^ idx_hash
#         return np.random.default_rng(mixed)

#     def _lazy_open(self) -> None:
#         """Open the dataset once and subset the requested time range."""
#         if self._empty or self.ds is not None:
#             return

#         with fsspec.open(self.ref_path, "rt") as f:
#             kerchunk_ref = json.load(f)
#         fs = fsspec.filesystem("reference", fo=kerchunk_ref)
#         mapper = fs.get_mapper("")

#         ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=False)

#         # Keep only channels + any coord helpers if present
#         vars_to_keep = list(self.source_channels)
#         for v in ("lat", "lon", "y", "x"):
#             if v in ds_full.variables and v not in vars_to_keep:
#                 vars_to_keep.append(v)

#         missing = [v for v in self.source_channels if v not in ds_full.variables]
#         if missing:
#             _logger.error("Missing expected variables in dataset: %s", missing)
#             kept = [v for v in vars_to_keep if v in ds_full.variables]
#             if kept:
#                 ds_full = ds_full[kept]
#             else:
#                 self.init_empty()
#                 return
#         else:
#             ds_full = ds_full[vars_to_keep]

#         # Subset time range only
#         if "time" not in ds_full.dims:
#             self.init_empty()
#             return
#         ds_win = ds_full.isel(time=slice(int(self.start_idx), int(self.end_idx)))

#         # Optional chunking
#         chunks_cfg = self.stream_info.get("chunks")
#         if chunks_cfg:
#             try:
#                 ds_win = ds_win.chunk(chunks_cfg)
#             except Exception as e:
#                 _logger.warning("Failed to apply chunks=%s: %s. Proceeding unchunked.", chunks_cfg, e)

#         self.ds = ds_win

#     # ---------- Public API ----------

#     @override
#     def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
#         """Fetch one training/eval window.

#         Returns ReaderData with aligned [coords, geoinfos, data, datetimes].
#         Requires geodetic (lat, lon) either in dataset or injected from sample_coord_file.
#         """
#         if self._empty:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         try:
#             self._lazy_open()
#         except Exception as e:
#             _logger.error("[RADKLIM] Lazy open failed: %s", e)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         if self.ds is None or "time" not in self.ds.dims:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         # Resolve absolute -> relative time indices for this sample
#         try:
#             t_idxs_abs, dtr = self._get_dataset_idxs(idx)
#         except Exception as e:
#             _logger.error("[RADKLIM] _get_dataset_idxs failed: %s", e)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         if t_idxs_abs is None or len(t_idxs_abs) == 0 or self.start_idx is None:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         t_rel = t_idxs_abs - self.start_idx
#         if (
#             t_rel.size == 0
#             or np.any(t_rel < 0)
#             or np.any(t_rel >= self.ds.sizes.get("time", 0))
#         ):
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         # Slice dataset for this window
#         t0 = int(t_rel[0])
#         t1 = int(t_rel[-1]) + 1  # inclusive end
#         ds_win = self.ds.isel(time=slice(t0, t1))

#         if ds_win.sizes.get("time", 0) == 0:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         # To [time, y, x, var]
#         try:
#             da = ds_win[self.source_channels].to_array(dim="var").transpose("time", "y", "x", "var")
#         except Exception as e:
#             _logger.error("[RADKLIM] to_array/transpose failed: %s", e)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         raw = da.values
#         if raw.size == 0:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#         if np.isnan(raw).any():
#             raw = np.nan_to_num(raw, nan=0.0)

#         # Flatten to [N, V] and select requested vars by channels_idx
#         flat_all = raw.reshape(-1, raw.shape[-1]).astype(np.float32)
#         try:
#             flat_data = flat_all[..., channels_idx]
#         except Exception:
#             _logger.warning("channels_idx invalid; falling back to sequential selection.")
#             keep = min(flat_all.shape[-1], len(channels_idx))
#             flat_data = flat_all[..., :keep]

#         # --- Build coords aligned with flattened data (REQUIRES geodetic lat/lon) ---
#         ny = ds_win.sizes.get("y", None)
#         nx = ds_win.sizes.get("x", None)
#         if ny is None or nx is None:
#             _logger.error("Dataset missing y/x dims; refusing to proceed without spatial grid.")
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#         points_per_time = ny * nx

#         def _latlon_from_dataset() -> Optional[tuple[np.ndarray, np.ndarray]]:
#             lat_da = None
#             lon_da = None
#             if "lat" in ds_win.variables:
#                 lat_da = ds_win.variables["lat"]
#             elif "lat" in ds_win.coords:
#                 lat_da = ds_win.coords["lat"]
#             if "lon" in ds_win.variables:
#                 lon_da = ds_win.variables["lon"]
#             elif "lon" in ds_win.coords:
#                 lon_da = ds_win.coords["lon"]
#             if lat_da is None or lon_da is None:
#                 return None

#             lat = np.asarray(lat_da.values)
#             lon = np.asarray(lon_da.values)
#             # Accept 2D lat[y,x], lon[y,x]; or 1D lat[y], lon[x]; or broadcastable
#             if lat.ndim == 2 and lon.ndim == 2:
#                 lat2d, lon2d = lat, lon
#             elif lat.ndim == 1 and lon.ndim == 1 and lat.shape[0] == ny and lon.shape[0] == nx:
#                 yy, xx = np.meshgrid(lat, lon, indexing="ij")
#                 lat2d, lon2d = yy, xx
#             else:
#                 try:
#                     lat2d = np.broadcast_to(lat, (ny, nx))
#                     lon2d = np.broadcast_to(lon, (ny, nx))
#                 except Exception:
#                     return None
#             return lat2d.astype(np.float32), lon2d.astype(np.float32)

#         latlon = _latlon_from_dataset()
#         if latlon is None:
#             # Fallback: inject from a static coord file
#             if not self.sample_coord_file:
#                 _logger.error("No (lat,lon) in dataset and no sample_coord_file provided.")
#                 return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#             try:
#                 with xr.open_dataset(self.sample_coord_file) as ds_coord:
#                     lat2d = ds_coord["lat"].values.astype(np.float32)
#                     lon2d = ds_coord["lon"].values.astype(np.float32)
#             except Exception as e:
#                 _logger.error("Failed to load sample_coord_file: %s", e)
#                 return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#             if lat2d.shape != (ny, nx) or lon2d.shape != (ny, nx):
#                 _logger.error(
#                     "Coord shape mismatch: coord (%s,%s) vs data (%d,%d).",
#                     lat2d.shape, lon2d.shape, ny, nx
#                 )
#                 return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#         else:
#             lat2d, lon2d = latlon

#         # Geodetic sanity check
#         if not (np.isfinite(lat2d).all() and np.isfinite(lon2d).all()):
#             _logger.error("Non-finite values in lat/lon.")
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
#         min_lat, max_lat = float(lat2d.min()), float(lat2d.max())
#         min_lon, max_lon = float(lon2d.min()), float(lon2d.max())
#         if not (-90.1 <= min_lat <= 90.1 and -90.1 <= max_lat <= 90.1 and
#                 -180.1 <= min_lon <= 180.1 and -180.1 <= max_lon <= 180.1):
#             _logger.error("Lat/lon out of geodetic bounds: lat[%f,%f], lon[%f,%f].",
#                           min_lat, max_lat, min_lon, max_lon)
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         base_coords = np.stack([lat2d.ravel(), lon2d.ravel()], axis=1).astype(np.float32)
#         full_coords = np.tile(base_coords, (ds_win.sizes["time"], 1)).astype(np.float32)

#         # Times aligned with flattened rows
#         times = ds_win["time"].values.astype("datetime64[ns]")
#         full_times = np.repeat(times, points_per_time)

#         # -------- Sparse subsampling --------
#         keep_zero_frac = float(self.sparse_cfg["keep_zero_frac"])
#         positive_threshold = float(self.sparse_cfg["positive_threshold"])
#         max_points = int(self.sparse_cfg["max_points"])
#         target_points = int(self.sparse_cfg["target_points"])
#         zero_per_pos = float(self.sparse_cfg["zero_per_pos"])

#         rr = flat_data[:, 0] if flat_data.shape[1] > 0 else np.zeros(flat_data.shape[0], np.float32)
#         pos_mask = rr > positive_threshold
#         neg_mask = ~pos_mask
#         pos_idx = np.flatnonzero(pos_mask)
#         neg_idx = np.flatnonzero(neg_mask)

#         keep = None
#         rng = self._rng_for(idx)

#         if target_points > 0:
#             # Start with positives, then fill/cap with zeros up to target_points
#             keep = np.zeros_like(pos_mask, dtype=bool)
#             if pos_idx.size > target_points:
#                 choose = rng.choice(pos_idx, size=target_points, replace=False)
#                 keep[choose] = True
#             else:
#                 keep[pos_idx] = True
#                 remaining = target_points - pos_idx.size
#                 if zero_per_pos > 0 and pos_idx.size > 0:
#                     k_zero = min(int(zero_per_pos * pos_idx.size), neg_idx.size, remaining)
#                 elif remaining > 0:
#                     k_zero = min(remaining, neg_idx.size)
#                 else:
#                     k_zero = 0
#                 if k_zero > 0:
#                     choose = rng.choice(neg_idx, size=k_zero, replace=False)
#                     keep[choose] = True
#         else:
#             # Default policy: keep all positives + random fraction of zeros
#             if keep_zero_frac > 0.0:
#                 keep = np.zeros_like(pos_mask, dtype=bool)
#                 keep[pos_mask] = True
#                 if neg_idx.size > 0:
#                     sampled = rng.random(neg_idx.size) < keep_zero_frac
#                     if sampled.any():
#                         keep[neg_idx[sampled]] = True

#         if keep is None:
#             keep = pos_mask.copy()

#         # Non-empty safety
#         if not keep.any():
#             take = min(4096, flat_data.shape[0])
#             keep[:take] = True

#         # Global cap
#         if max_points > 0 and int(keep.sum()) > max_points:
#             all_idx = np.flatnonzero(keep)
#             choose = rng.choice(all_idx, size=max_points, replace=False)
#             keep[:] = False
#             keep[choose] = True

#         # Apply mask
#         flat_data = flat_data[keep]
#         full_coords = full_coords[keep]
#         full_times = full_times[keep]

#         # If still empty, bail out gracefully
#         if flat_data.shape[0] == 0:
#             return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

#         # Final safety & pack
#         N = flat_data.shape[0]
#         assert N == full_coords.shape[0] == full_times.shape[0], \
#             f"row mismatch data={flat_data.shape} coords={full_coords.shape} times={full_times.shape}"
#         assert flat_data.ndim == 2 and full_coords.shape[1] == 2, \
#             f"bad shapes data={flat_data.shape} coords={full_coords.shape}"

#         rdata = ReaderData(
#             coords=full_coords.astype(np.float32),
#             geoinfos=np.zeros((N, len(self.geoinfo_idx)), dtype=np.float32),
#             data=flat_data.astype(np.float32),
#             datetimes=full_times,
#         )
#         check_reader_data(rdata, dtr)
#         return rdata


import json
import logging
from pathlib import Path
from typing import override

import fsspec
import numpy as np
import xarray as xr

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class RadklimKerchunkReader(DataReaderTimestep):
    """
    Reader for RADKLIM data accessed via a Kerchunk reference.

    This reader:
    1. Drops lat/lon from the Zarr reference,
    2. Lazily injects correct static lat/lon from a hardcoded sample file,
    3. Verifies that coords load successfully.
    4. (ADDED) Optional subsampling of points per window (positives + sampled zeros).
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        self._empty = False
        self.ds: xr.Dataset | None = None

        # Paths from config
        self.ref_path = Path(stream_info.get("reference", filename))
        self.norm_path = Path(stream_info.get("stats_path"))
        self.sample_coord_file = Path(stream_info["sample_coord_file"])
        self.stream_info = stream_info

        # Channels
        self.source_channels = ["RR"]
        self.target_channels = ["RR"]
        self.geoinfo_channels: list[str] = []

        self.source_idx = list(range(len(self.source_channels)))
        self.target_idx = list(range(len(self.target_channels)))
        self.geoinfo_idx = list(range(len(self.geoinfo_channels)))

        # (ADDED) Subsampling config (all optional; defaults keep behavior close to "off")
        ssc = (self.stream_info.get("sparse_sampling") or {})
        self.sparse_cfg = {
            "keep_zero_frac": float(ssc.get("keep_zero_frac", 0.7)),   # 0.0 → keep no extra zeros by default
            "positive_threshold": float(ssc.get("positive_threshold", 0.0)),
            "max_points": int(ssc.get("max_points", 20)),               # 0 → no global cap
            "target_points": int(ssc.get("target_points", 0)),         # 0 → disabled
            "zero_per_pos": float(ssc.get("zero_per_pos", 2.0)),       # 0 → disabled
            "seed": int(ssc.get("seed", 1234)),                           # 0 → nondeterministic per worker
        }

        # Ensure files exist
        if not self.ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference not found: {self.ref_path}")
        if not self.norm_path.exists():
            raise FileNotFoundError(f"Normalization JSON not found: {self.norm_path}")
        if not self.sample_coord_file.exists():
            raise FileNotFoundError(f"Sample coord file not found: {self.sample_coord_file}")

        # Read full time axis for window setup
        _logger.info("Reading time metadata from: %s", self.ref_path)
        with fsspec.open(self.ref_path, "rt") as f:
            kerchunk_ref = json.load(f)
        fs_meta = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper_meta = fs_meta.get_mapper("")
        try:
            with xr.open_dataset(
                mapper_meta, engine="zarr", consolidated=False, chunks={}
            ) as ds_meta:
                times_full = ds_meta["time"].values
        except Exception:
            _logger.error("Failed to open reference for time axis; corrupt?")
            raise

        # Empty dataset check
        if times_full.size == 0:
            super().__init__(tw_handler, stream_info, None, None, None)
            self.init_empty()
            return

        # Check regular time steps
        deltas = np.unique(np.diff(times_full.astype("datetime64[s]")))
        if deltas.size != 1:
            raise ValueError("Irregular time steps in Kerchunk reference")
        period = deltas[0]
        super().__init__(
            tw_handler,
            stream_info,
            times_full[0],
            times_full[-1],
            period,
        )

        # Window indices
        if tw_handler.t_start >= times_full[-1] or tw_handler.t_end <= times_full[0]:
            self.init_empty()
            return
        self.start_idx = int(np.searchsorted(times_full, tw_handler.t_start, "left"))
        self.end_idx = int(np.searchsorted(times_full, tw_handler.t_end, "right"))
        self.num_steps_per_window = int(tw_handler.t_window_len / period)

        # Load normalization stats
        stats = json.loads(self.norm_path.read_text())
        self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
        self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)
        self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
        self.stdev_geoinfo = np.asarray(stats.get("std_geoinfo", []), dtype=np.float32)

        if len(self.mean) != len(self.source_channels):
            raise ValueError("Stats length ≠ number of source channels")

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

    def _lazy_open(self):
        if self.ds is not None:
            return

        _logger.info("Lazy loading Kerchunk dataset...")

        # Load Kerchunk reference
        with open(self.ref_path) as f:
            kerchunk_ref = json.load(f)
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")
        ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=False)

        # 1) DROP existing lat/lon
        ds_full = ds_full.drop_vars(["lat", "lon"], errors="ignore")

        # 2) Subset variables & time
        subset = ds_full[self.source_channels].isel(time=slice(self.start_idx, self.end_idx))
        if "chunks" in self.stream_info:
            subset = subset.chunk(self.stream_info["chunks"])

        # ---- existing spatial slice (unchanged) ----
        ny = subset.sizes["y"]
        nx = subset.sizes["x"]
        y_slice = slice(0, ny )
        x_slice = slice(0, nx )
        self.ds = subset.isel(y=y_slice, x=x_slice)

        with xr.open_dataset(self.sample_coord_file) as ds_sample:
            lat2d = ds_sample["lat"].values.astype(np.float32)[y_slice, x_slice]
            lon2d = ds_sample["lon"].values.astype(np.float32)[y_slice, x_slice]

            _logger.debug(
                "lat2d shape=%s  min=%s  max=%s",
                lat2d.shape,
                np.min(lat2d),
                np.max(lat2d),
            )
            _logger.debug(
                "lon2d shape=%s  min=%s  max=%s",
                lon2d.shape,
                np.min(lon2d),
                np.max(lon2d),
            )

            self.ds = self.ds.assign_coords(
                lat=(("y", "x"), lat2d),
                lon=(("y", "x"), lon2d),
            )
            _logger.info(
                "Injected static lat/lon from %s",
                self.sample_coord_file,
            )

        # Save dims
        self.ny, self.nx = lat2d.shape
        self.points_per_slice = self.ny * self.nx

    # (ADDED) tiny helper for deterministic RNG per-sample (if seed provided)
    def _rng_for(self, idx: TIndex) -> np.random.Generator:
        seed = int(self.sparse_cfg.get("seed", 0))
        if seed == 0:
            return np.random.default_rng()
        # mix idx into seed so different samples get different but reproducible streams
        idx_hash = int(np.int64(abs(hash(idx))) & 0x7FFFFFFF)
        mixed = (seed * 1_000_003) ^ idx_hash
        return np.random.default_rng(mixed)

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Fetch data for a specific time window and channels.
        """
        # Crucial step: ensure dataset is open in the current process
        self._lazy_open()
        if self._empty or self.ds is None:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        t_idxs_abs, dtr = self._get_dataset_idxs(idx)
        t_rel = t_idxs_abs - self.start_idx
        if t_rel.size == 0 or np.any(t_rel < 0) or np.any(t_rel >= self.ds.sizes["time"]):
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        ds_win = self.ds.isel(time=slice(int(t_rel[0]), int(t_rel[-1]) + 1))

        # Flatten data
        da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
        raw = da.values
        flat_all = raw.reshape(-1, raw.shape[-1]).astype(np.float32)
        try:
            flat_data = flat_all[..., channels_idx]
        except Exception:
            # conservative fallback: sequential selection
            keep = min(flat_all.shape[-1], len(channels_idx))
            flat_data = flat_all[..., :keep]

        # Coords and times (dense, before subsampling)
        lat2d = self.ds["lat"].values
        lon2d = self.ds["lon"].values
        flat_coords = np.stack([lat2d.ravel(), lon2d.ravel()], axis=1).astype(np.float32)
        full_coords = np.tile(flat_coords, (ds_win.sizes["time"], 1))

        full_times = np.repeat(
            ds_win["time"].values.astype("datetime64[ns]"),
            self.points_per_slice,
        )

        # -------------------- SUBSAMPLING (added) --------------------
        # Strategy: keep all positives (RR > positive_threshold) and
        # add a fraction of zeros, then optional caps.
        s_cfg = self.sparse_cfg
        keep_zero_frac = float(s_cfg.get("keep_zero_frac", 0.0))
        positive_threshold = float(s_cfg.get("positive_threshold", 0.0))
        max_points = int(s_cfg.get("max_points", 0))
        target_points = int(s_cfg.get("target_points", 0))
        zero_per_pos = float(s_cfg.get("zero_per_pos", 0.0))

        # First channel is RR (by construction in this class)
        rr = flat_data[:, 0] if flat_data.shape[1] > 0 else np.zeros(flat_data.shape[0], np.float32)
        pos_mask = rr > positive_threshold
        neg_mask = ~pos_mask
        pos_idx = np.flatnonzero(pos_mask)
        neg_idx = np.flatnonzero(neg_mask)

        keep_mask = None
        rng = self._rng_for(idx)

        if target_points > 0:
            keep_mask = np.zeros_like(pos_mask, dtype=bool)
            if pos_idx.size >= target_points:
                choose = rng.choice(pos_idx, size=target_points, replace=False)
                keep_mask[choose] = True
            else:
                keep_mask[pos_idx] = True
                remaining = target_points - pos_idx.size
                if remaining > 0:
                    if zero_per_pos > 0 and pos_idx.size > 0:
                        k_zero = min(int(zero_per_pos * pos_idx.size), neg_idx.size, remaining)
                    else:
                        k_zero = min(remaining, neg_idx.size)
                    if k_zero > 0:
                        choose = rng.choice(neg_idx, size=k_zero, replace=False)
                        keep_mask[choose] = True
        else:
            # default: all positives + random fraction of negatives
            keep_mask = np.zeros_like(pos_mask, dtype=bool)
            if pos_idx.size > 0:
                keep_mask[pos_idx] = True
            if keep_zero_frac > 0.0 and neg_idx.size > 0:
                sampled = rng.random(neg_idx.size) < keep_zero_frac
                if sampled.any():
                    keep_mask[neg_idx[sampled]] = True

        # safety: ensure non-empty
        if not keep_mask.any():
            take = min(4096, flat_data.shape[0])
            keep_mask[:take] = True

        # global cap
        if max_points > 0:
            k = int(keep_mask.sum())
            if k > max_points:
                all_idx = np.flatnonzero(keep_mask)
                choose = rng.choice(all_idx, size=max_points, replace=False)
                keep_mask[:] = False
                keep_mask[choose] = True

        # Apply subsampling consistently to all arrays
        flat_data = flat_data[keep_mask]
        full_coords = full_coords[keep_mask]
        full_times = full_times[keep_mask]

        # ------------------ END SUBSAMPLING (added) ------------------

        # Package ReaderData
        length = flat_data.shape[0]
        if length == 0:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        rdata = ReaderData(
            coords=full_coords,
            geoinfos=np.zeros((length, len(self.geoinfo_idx)), dtype=np.float32),
            data=flat_data,
            datetimes=full_times,
        )
        check_reader_data(rdata, dtr)
        return rdata
