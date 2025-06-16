# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
from pathlib import Path
from typing import Any, Union, override

import fsspec
import numpy as np
import xarray as xr
import zarr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DType,
    DataReaderTimestep,
    ReaderData,
    TIndex,
    TimeWindowHandler,
    check_reader_data,
)

class RadklimKerchunkReader(DataReaderTimestep):
    '''
    Construct data reader for RADKLIM radar-rain product

    The class implements a time-window-based `get` API for training and evaluation.

    Notes
    -----
    * Assumes regular hourly time steps. Irregularities will raise a `ValueError`.
    * Only the 'reflectivity rain-rate' (RR) channel is available.
    * Geo-information fields are not provided and are returned as empty arrays.
    '''
    # abstract‑interface metadata ------------------------------------------------
    source_channels: list[str] = ["RR"]
    target_channels: list[str] = ["RR"]
    geoinfo_channels: list[str] = []

    source_idx: list[int] = [0]
    target_idx: list[int] = [0]
    geoinfo_idx: list[int] = []

    # ---------------------------------------------------------------------
    # constructor ----------------------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        reference_json: Union[str, Path],
        normalization_json: Union[str, Path],
        *,
        chunks: dict[str, Any] | None = None,
    ) -> None:
        # flag must be defined *before* any early‑exit so that it always exists
        self._empty: bool = False

        # ---------------------------------------------------------------
        # 1. load normalisation statistics -----------------------------
        # ---------------------------------------------------------------
        norm_path = Path(normalization_json)
        if not norm_path.exists():
            raise FileNotFoundError(f"normalisation JSON not found: {norm_path}")

        stats = json.loads(norm_path.read_text())
        self.mean = np.asarray(stats.get("mean", []), dtype=np.float32)
        self.stdev = np.asarray(stats.get("std", []), dtype=np.float32)
        self.mean_geoinfo = np.asarray(stats.get("mean_geoinfo", []), dtype=np.float32)
        self.stdev_geoinfo = np.asarray(
            stats.get("std_geoinfo", []), dtype=np.float32
        )

        if len(self.mean) != len(self.source_channels):
            raise ValueError(
                "normalisation stats length does not match number of variables"
            )

        # ---------------------------------------------------------------
        # 2. open the Kerchunk reference & inspect global time axis -----
        # ---------------------------------------------------------------
        ref_path = Path(reference_json)
        if not ref_path.exists():
            raise FileNotFoundError(f"Kerchunk reference JSON not found: {ref_path}")

        kerchunk_ref = json.loads(ref_path.read_text())
        fs = fsspec.filesystem("reference", fo=kerchunk_ref)
        mapper = fs.get_mapper("")
        # consolidate metadata – if the reference already has a .zmetadata this
        # is a no‑op; otherwise it creates an in‑memory view.
        try:
            zarr.consolidate_metadata(mapper)  
        except Exception: 
            pass

        ds_full = xr.open_dataset(mapper, engine="zarr", consolidated=True)

        # pull out *numpy* datetime64 array for speed
        times_full: NDArray[np.datetime64] = ds_full["time"].values
        if times_full.size == 0:
            # initialise as empty and bail out early
            super().__init__(tw_handler, None, None, None)
            self.init_empty()
            return

        # verify regular sampling ------------------------------------------------
        # Using seconds ensures a uniform unit that fits into int64. Converting to
        # seconds before diff avoids the ns→s cast overflow when the dataset spans
        # centuries.
        deltas_sec = np.diff(times_full.astype("datetime64[s]"))
        unique_deltas = np.unique(deltas_sec)
        if unique_deltas.size != 1:
            raise ValueError("RADKLIM Kerchunk reference has irregular time steps")
        period = unique_deltas[0]

        data_start = times_full[0]
        data_end = times_full[-1]

        # -------------------------------
        # 3. call parent initialiser ----
        # -------------------------------
        super().__init__(tw_handler, data_start, data_end, period)

        # early‑exit if the requested window sits outside the dataset -------------
        if tw_handler.t_start >= data_end or tw_handler.t_end <= data_start:
            self.init_empty()
            return

        # --------------------------------------------------------------------------------
        # 4. keep only the time slice that overlaps with the training time span ---------
        # --------------------------------------------------------------------------------
        self.start_idx = int(np.searchsorted(times_full, tw_handler.t_start, side="left"))
        self.end_idx = int(np.searchsorted(times_full, tw_handler.t_end, side="right"))

        subset = ds_full[self.source_channels].isel(time=slice(self.start_idx, self.end_idx))
        if chunks is not None:
            subset = subset.chunk(chunks)
        self.ds = subset

        # --------------------------------------------------------------------------------
        # 5. geometry – prepare once, reuse often ---------------------------------------
        # --------------------------------------------------------------------------------
        y1d = self.ds["y"].values.astype(np.float32)
        x1d = self.ds["x"].values.astype(np.float32)
        self.ny: int = len(y1d)
        self.nx: int = len(x1d)
        self.points_per_slice: int = self.ny * self.nx

        lat_var = self.ds["lat"]
        lon_var = self.ds["lon"]
        # Some RADKLIM flavours have *time* in the lat/lon variables, some don't.
        raw_lat = lat_var.isel(time=0).values if "time" in lat_var.dims else lat_var.values
        raw_lon = lon_var.isel(time=0).values if "time" in lon_var.dims else lon_var.values

        self.latitudes = _clip_lat(raw_lat)
        self.longitudes = _clip_lon(raw_lon)

        # flattened coordinate array (ny*nx, 2) – reused for every window ----------
        self._base_coords = np.column_stack(
            (self.latitudes.reshape(-1), self.longitudes.reshape(-1))
        ).astype(DType)

        # number of dataset timesteps per logical *weathergen* window --------------
        self.num_steps_per_window = int(tw_handler.t_window_len / period)

    # ------------------------------------------------------------------
    # public API --------------------------------------------------------
    # ------------------------------------------------------------------

    @override
    def init_empty(self) -> None: 
        """Transform this reader into an *always‑empty* stub."""
        self._empty = True
        super().init_empty()

    # ------------------------------------------------------------------

    @override
    def length(self) -> int:
        """Number of *weathergen* windows this reader can deliver."""
        if self._empty:
            return 0
        nt: int = int(self.ds.sizes["time"])
        return max(0, nt - self.num_steps_per_window + 1)

    # ------------------------------------------------------------------

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData: 
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """
        
        # 1. translate window index → absolute dataset indices --------------
        t_idxs_abs, dtr = self._get_dataset_idxs(idx)

        if self._empty or t_idxs_abs.size == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # shift from *absolute* to *subset‑relative* indices ----------------
        t_idxs_rel = t_idxs_abs - self.start_idx
        if np.any(t_idxs_rel < 0) or np.any(t_idxs_rel >= self.ds.sizes["time"]):
            # This can pop up if the parent index calc spilled across the subset
            # boundaries due to rounding errors. Safer to bail out as empty.
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        # 2. slice dataset --------------------------------------------------
        start, stop = int(t_idxs_rel[0]), int(t_idxs_rel[-1]) + 1
        ds_win = self.ds.isel(time=slice(start, stop))

        # 3. bring to ndarray shape (t, y, x, var) --------------------------
        arr4 = (
            ds_win.to_array(dim="var") 
            .transpose("time", "y", "x", "var")
            .values
            .astype(np.float32, copy=False)
        )
        nt, ny, nx, nvars = arr4.shape

        # sanity for channel indices ---------------------------------------
        if not channels_idx:
            raise ValueError("channels_idx cannot be empty")
        if min(channels_idx) < 0 or max(channels_idx) >= nvars:
            raise IndexError("channels_idx out of bounds for this dataset slice")
        if len(set(channels_idx)) != len(channels_idx):
            raise ValueError("channels_idx must be unique")

        flat_vars = arr4.reshape(-1, nvars)

        # 4. coordinates & timestamps --------------------------------------
        coords = np.tile(self._base_coords, (nt, 1))  # shape: (nt*ny*nx, 2)

        time_vals = ds_win["time"].values.astype("datetime64[ns]")  # per timestep
        times = np.repeat(time_vals, self.points_per_slice)

        # 5. NaN mask -------------------------------------------------------
        valid = ~np.any(np.isnan(flat_vars[:, channels_idx]), axis=1)
        if not np.any(valid):
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=len(self.geoinfo_idx),
            )

        coords_sel = coords[valid]
        data_sel = flat_vars[valid][:, channels_idx].astype(DType, copy=False)
        times_sel = times[valid]

        rdata = ReaderData(
            coords=coords_sel,
            geoinfos=np.zeros((coords_sel.shape[0], 0), dtype=DType),  # none for RADKLIM
            data=data_sel,
            datetimes=times_sel,
        )
        check_reader_data(rdata, dtr)
        return rdata

# -----------------------------------------------------------------------------
# helper functions -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _clip_lat(lats: NDArray[np.floating]) -> NDArray[np.float32]:
    """Mirror latitudes into the range ``[-90, 90]`` so that out‑of‑bounds values
    wrap around the poles (periodicity). Returned array is *copied* and cast to
    ``float32``.
    """
    return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(np.float32)


def _clip_lon(lons: NDArray[np.floating]) -> NDArray[np.float32]:
    """Wrap longitudes to the range ``[-180, 180]``. Returned array is *copied*
    and cast to ``float32``.
    """
    return ((lons + 180.0) % 360.0 - 180.0).astype(np.float32)