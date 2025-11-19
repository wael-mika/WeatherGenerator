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
