# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
    RADKLIM reader via Kerchunk.

    * Validates presence of the Kerchunk reference, stats JSON, and a trusted
      `sample_coord_file` that supplies correct 2D `lat`/`lon`.
    * Loads `lat`/`lon` **once** (in `__init__`) and stores them for reuse.
    * Opens the Kerchunk-referenced dataset lazily on first call to `_get(...)`.

    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct a Kerchunk-backed RADKLIM reader.

        Parameters
        ----------
        tw_handler : TimeWindowHandler
            Global time-window configuration (start/end, step, window length).
            `DataReaderTimestep` uses this to compute per-window indices.
        filename : Path
            Fallback path used if `stream_info["reference"]` is not provided.
        stream_info : dict
            Stream configuration dictionary. Expected keys:
              - "reference": Path-like string to Kerchunk JSON (required or
                `filename` fallback is used).
              - "stats_path": Path to normalization stats JSON (required).
              - "sample_coord_file": Path to a NetCDF containing `lat` and `lon`
                2D arrays that match the dataset spatial grid (required).
              - "source_channels": list[str], variables to read (default ["RR"]).
              - "target_channels": list[str], variables to mark as target (default ["RR"]).

        Raises
        ------
        FileNotFoundError
            If any of the required files do not exist.
        KeyError
            If `lat`/`lon` are missing in the `sample_coord_file`.
        ValueError
            If the dataset time steps are irregular, or if the spatial shape in
            Kerchunk does not match the shape of `lat`/`lon`, or if stats length
            mismatches the number of source channels.
        """
        self._empty = False
        self.ds: xr.Dataset | None = None
        self.stream_info = stream_info

        # Paths / config
        self.ref_path = Path(stream_info.get("reference", filename))
        self.norm_path = Path(stream_info.get("stats_path"))
        self.sample_coord_file = Path(stream_info["sample_coord_file"])

        # Existence checks (fail fast on misconfiguration)
        if not self.ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference not found: {self.ref_path}")
        if not self.norm_path.exists():
            raise FileNotFoundError(f"Normalization JSON not found: {self.norm_path}")
        if not self.sample_coord_file.exists():
            raise FileNotFoundError(f"Sample coord file not found: {self.sample_coord_file}")

        # Channel configuration
        self.source_channels = list(stream_info.get("source_channels", ["RR"]))
        self.target_channels = list(stream_info.get("target_channels", ["RR"]))
        self.geoinfo_channels: list[str] = []

        # Pre-computed indices per channel kind (mirrors WG conventions)
        self.source_idx = list(range(len(self.source_channels)))
        self.target_idx = list(range(len(self.target_channels)))
        self.geoinfo_idx = list(range(len(self.geoinfo_channels)))

        # Load lat/lon ONCE from the trusted file to avoid inconsistent coords in Kerchunk
        with xr.open_dataset(self.sample_coord_file) as ds_sample:
            if "lat" not in ds_sample or "lon" not in ds_sample:
                raise KeyError(f"'lat'/'lon' not found in {self.sample_coord_file}")
            self.lat2d = ds_sample["lat"].values.astype(np.float32)
            self.lon2d = ds_sample["lon"].values.astype(np.float32)

        # Cache spatial shape and per-slice point count
        self.ny, self.nx = self.lat2d.shape
        self.points_per_slice = self.ny * self.nx
        _logger.info(
            "Loaded static lat/lon once from %s (ny=%d, nx=%d).",
            self.sample_coord_file,
            self.ny,
            self.nx,
        )

        # ---- Read and validate the time axis from the Kerchunk reference
        _logger.info("Reading time metadata from: %s", self.ref_path)
        with fsspec.open(self.ref_path, "rt") as f:
            kerchunk_ref = json.load(f)
        fs_meta = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper_meta = fs_meta.get_mapper("")

        try:
            # Use small in-memory open to extract time axis and dims
            with xr.open_dataset(
                mapper_meta, engine="zarr", consolidated=False, chunks={}
            ) as ds_meta:
                times_full = np.asarray(ds_meta["time"].values)

                # Optional sanity check: spatial dims present and matching lat/lon
                if "y" in ds_meta.dims and "x" in ds_meta.dims:
                    if ds_meta.dims["y"] != self.ny or ds_meta.dims["x"] != self.nx:
                        raise ValueError(
                            f"Lat/lon shape {self.lat2d.shape} does not match dataset "
                            f"({ds_meta.dims['y']}, {ds_meta.dims['x']})"
                        )
        except Exception as e:
            _logger.error("Failed to open kerchunk reference for time axis: %s", e)
            raise

        # If the dataset contains no time entries, initialize as empty
        if times_full.size == 0:
            super().__init__(tw_handler, stream_info, None, None, None)
            self.init_empty()
            return

        # Require regular (constant-step) time axis; derive the period
        deltas = np.unique(np.diff(times_full.astype("datetime64[s]")))
        if deltas.size != 1:
            raise ValueError("Irregular time steps in Kerchunk reference")
        period = deltas[0]  # numpy.timedelta64[s]

        # Initialize the base class (positional args – API requirement)
        super().__init__(
            tw_handler,
            stream_info,
            times_full[0],
            times_full[-1],
            period,
        )

        # Early-exit: requested global window does not intersect the dataset
        if tw_handler.t_start >= times_full[-1] or tw_handler.t_end <= times_full[0]:
            self.init_empty()
            return

        # Compute absolute time-index range [start_idx, end_idx) for our global span
        self.start_idx = int(np.searchsorted(times_full, tw_handler.t_start, side="left"))
        self.end_idx = int(np.searchsorted(times_full, tw_handler.t_end, side="right"))

        # Compute number of steps per sliding window (rounded, with a warning if fractional)
        steps_float = float(tw_handler.t_window_len / period)
        self.num_steps_per_window = int(round(steps_float))
        if not np.isclose(self.num_steps_per_window, steps_float):
            _logger.warning(
                "Window len not integer multiple of period: %s / %s = %.6f; using %d steps.",
                tw_handler.t_window_len,
                period,
                steps_float,
                self.num_steps_per_window,
            )
        if self.num_steps_per_window <= 0:
            raise ValueError(f"Computed non-positive steps per window: {self.num_steps_per_window}")

        # Load normalization stats (shape check only; no in-reader normalization)
        stats = json.loads(self.norm_path.read_text())
        self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
        self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)

        if len(self.mean) != len(self.source_channels):
            raise ValueError(
                f"Stats len ({len(self.mean)}) ≠ num of src channels ({len(self.source_channels)})"
            )


    @override
    def init_empty(self) -> None:
        """
        Switch the reader into an "empty" mode.

        Called when there is no overlap between the requested global time range
        and the dataset, or if the dataset has no time entries. Ensures that
        `length()` returns 0 and `_get(...)` yields empty `ReaderData`.
        """
        self._empty = True
        super().init_empty()

    @override
    def length(self) -> int:
        """
        Number of available sliding windows within the configured global span.

        Returns
        -------
        int
            `max(0, (end_idx - start_idx) - num_steps_per_window + 1)`. If the
            reader is empty, returns 0.
        """
        if self._empty:
            return 0
        nt = self.end_idx - self.start_idx
        return max(0, nt - self.num_steps_per_window + 1)

    # ---------------- internal helpers ----------------

    def _lazy_open(self) -> None:
        """
        Open and cache the dataset view covering the global time span.

        This performs the following (only on first call):
        1) Opens the Kerchunk-backed Zarr dataset.
        2) Drops any `lat` / `lon` variables present in the reference (they may
           be wrong or inconsistent).
        3) Subsets **only in time** to `[start_idx, end_idx)`.
        4) Injects the trusted 2D `lat` / `lon` coordinates loaded in `__init__`.
        5) Stores the resulting `xarray.Dataset` in `self.ds`.

        """
        if self.ds is not None:
            return

        _logger.info("Lazy loading Kerchunk dataset...")

        with fsspec.open(self.ref_path, "rt") as f:
            kerchunk_ref = json.load(f)
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")

        ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=False)

        # Remove coordinates that might be incorrect in the reference
        ds_full = ds_full.drop_vars(["lat", "lon"], errors="ignore")

        # Keep only the requested variables and the global time span
        subset = ds_full[self.source_channels].isel(time=slice(self.start_idx, self.end_idx))

        # Inject trusted lat/lon as 2D coords (shape-checked in __init__)
        subset = subset.assign_coords(
            lat=(("y", "x"), self.lat2d),
            lon=(("y", "x"), self.lon2d),
        )

        self.ds = subset
        del ds_full

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Materialize a single sliding time window as `ReaderData`.

        Parameters
        ----------
        idx : TIndex
            Logical window index (0-based) within the available span.
        channels_idx : list[int]
            Indices into `self.source_channels` specifying which variables to
            include (order-preserving). The returned `data` array has shape
            `(N, len(channels_idx))`.

        Returns
        -------
        ReaderData
            If valid and non-empty: a bundle with `data`, `coords`, `datetimes`,
            and empty `geoinfos`. Each row corresponds to one `(y, x)` grid point
            at one time in the window. If the request is out of range or all
            values in the window are invalid after filtering, returns an **empty**
            `ReaderData` with zero rows.

        Processing Steps
        ----------------
        1) Lazily open/cache the dataset (`_lazy_open()`).
        2) Convert the logical window index to absolute time indices and sanity-check.
        3) Extract `ds_win = self.ds.isel(time=slice(t0, t1))`.
        4) Stack variables → `(time, y, x, var)`; flatten → `(N_all, C_all)`.
        5) Select requested channels → `(N_all, C)`.
        6) **Drop any rows** containing NaN/Inf across the selected channels.
        7) Build coordinates by tiling the cached 2D `lat`/`lon` over time and
           applying the same row mask.
        8) Build datetimes by repeating the window time stamps per grid point and
           applying the same row mask.
        9) Run `check_reader_data(...)` to validate shapes and time range.

        """
        self._lazy_open()
        if self._empty or self.ds is None:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # Translate logical index to absolute dataset time indices
        t_idxs_abs, dtr = self._get_dataset_idxs(idx)
        if t_idxs_abs.size == 0:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # Convert to indices relative to currently opened [start_idx, end_idx)
        t_rel = t_idxs_abs - self.start_idx
        t0 = int(t_rel[0])
        t1 = int(t_rel[-1]) + 1

        if t0 < 0 or t1 > int(self.ds.sizes["time"]) or t0 >= t1:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # Extract window
        ds_win = self.ds.isel(time=slice(t0, t1))
        if ds_win.sizes.get("time", 0) == 0:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # (time, y, x, var) -> numpy
        da = ds_win.to_array(dim="var").transpose("time", "y", "x", "var")
        raw = da.values
        flat_data = raw.reshape(-1, raw.shape[-1])[..., channels_idx].astype(np.float32, copy=False)

        # Filter out any rows that contain NaN/Inf across selected channels
        valid_rows = np.isfinite(flat_data).all(axis=1)
        if not valid_rows.any():
            _logger.debug("All values NaN in window idx=%s", idx)
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))
        flat_data = flat_data[valid_rows]

        # Coordinates: reuse preloaded arrays tiled per time, with the same mask
        flat_coords = np.stack([self.lat2d.ravel(), self.lon2d.ravel()], axis=1).astype(
            np.float32, copy=False
        )
        full_coords = np.tile(flat_coords, (ds_win.sizes["time"], 1))[valid_rows]

        # Datetimes: repeat time stamps per grid point, then mask
        full_times = np.repeat(
            ds_win["time"].values.astype("datetime64[ns]"),
            self.points_per_slice,
        )[valid_rows]

        # Package result
        length = flat_data.shape[0]
        rdata = ReaderData(
            coords=full_coords,
            geoinfos=np.zeros((length, len(self.geoinfo_idx)), dtype=np.float32),
            data=flat_data,
            datetimes=full_times,
        )
        check_reader_data(rdata, dtr)
        return rdata
