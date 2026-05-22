# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import re
from pathlib import Path
from typing import override

import numpy as np
import zarr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    NPDT64,
    NPTDel64,
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)

_logger = logging.getLogger(__name__)


class DataReaderImerg(DataReaderTimestep):
    """
    Data reader for IMERG precipitation dataset in zarr format.

    IMERG (Integrated Multi-satellitE Retrievals for GPM) provides global precipitation
    estimates at 0.1° spatial resolution and 30-minute temporal resolution.

    This implementation uses LAZY TIME LOADING - timestamps are computed mathematically
    on-demand rather than loaded from the zarr file, dramatically reducing initialization
    time for large datasets (480K+ timesteps).
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        **kwargs,
    ) -> None:
        """
        Construct data reader for IMERG dataset

        Parameters
        ----------
        tw_handler :
            time window handler
        filename :
            filename (and path) of zarr dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        # Open zarr dataset
        self.filename = filename
        _logger.info(f"Opening IMERG zarr dataset: {filename}")
        self.z = zarr.open(filename, mode="r")

        # Extract coordinates
        self.latitudes = self.z["latitude"][:].astype(np.float32)
        self.longitudes = self.z["longitude"][:].astype(np.float32)
        _logger.info(f"Loaded coordinates: {len(self.latitudes)} lats × {len(self.longitudes)} lons")

        # Parse time coordinate - LAZY LOADING: only read metadata, not actual times
        time_attrs = dict(self.z["time"].attrs)
        self.total_timesteps = self.z["time"].shape[0]

        # Parse time units (e.g., "minutes since 1998-01-01 00:00:00")
        units_str = time_attrs.get("units", "")
        self.base_datetime, self.time_unit = self._parse_time_units(units_str)

        # Store base datetime as numpy datetime64 for efficient computation
        self.base_datetime_np = np.datetime64(self.base_datetime)

        # Sample only first and last points to determine time range (2 values, not 480K+)
        sample_times = self.z["time"].oindex[[0, self.total_timesteps - 1]]
        sample_datetimes = self._convert_time_to_datetime64(
            sample_times, self.base_datetime, self.time_unit
        )

        data_start_time = sample_datetimes[0]
        data_end_time = sample_datetimes[-1]

        # Calculate period from total range (more accurate than 2-point difference)
        period = (data_end_time - data_start_time) / (self.total_timesteps - 1)
        self.period_ns = period.astype("timedelta64[ns]").astype(np.int64)

        _logger.info(
            f"Dataset has {self.total_timesteps:,} timesteps covering "
            f"{data_start_time} to {data_end_time} (period: {period})"
        )

        # Check if training window overlaps with dataset
        if tw_handler.t_end <= data_start_time or tw_handler.t_start >= data_end_time:
            _logger.warning(
                f"Training window [{tw_handler.t_start}, {tw_handler.t_end}) does not overlap "
                f"with dataset [{data_start_time}, {data_end_time}]. Initializing empty dataset."
            )
            self.len = 0
        else:
            # Calculate index range for training window (no loading, just math)
            start_idx = max(
                0, int((tw_handler.t_start - data_start_time) / period)
            )
            end_idx = min(
                self.total_timesteps,
                int((tw_handler.t_end - data_start_time) / period) + 1
            )
            self.len = end_idx - start_idx

            _logger.info(
                f"Training window maps to indices [{start_idx}, {end_idx}) "
                f"= {self.len:,} timesteps ({self.len/self.total_timesteps*100:.1f}% of total)"
            )

        # Store reference to precipitation array (lazy loading)
        self.precip = self.z["precipitation"]

        # Initialize parent class
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )

        # If there is no overlap with the time range, the dataset will be empty
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            self.init_empty()
            return

        # Load statistics for normalization
        self.mean = np.array([self.z["mean"][()]], dtype=np.float32)
        self.stdev = np.array([self.z["stdev"][()]], dtype=np.float32)
        _logger.info(f"Normalization stats: mean={self.mean[0]:.4f}, stdev={self.stdev[0]:.4f}")

        # Apply spatial filtering (bounding box + subsampling)
        self._apply_spatial_filters(stream_info)

        # Create meshgrid for coordinates (computed once, used repeatedly)
        # Uses filtered lat/lon arrays from _apply_spatial_filters
        lon_grid, lat_grid = np.meshgrid(self.longitudes, self.latitudes)
        self.coords_template = np.stack(
            [lat_grid.flatten(), lon_grid.flatten()], axis=1
        ).astype(np.float32)
        self.n_grid_points = len(self.coords_template)

        # Select channels
        available_channels = ["precipitation"]

        s_chs = stream_info.get("source")
        s_chs_exclude = stream_info.get("source_exclude", [])
        t_chs = stream_info.get("target")
        t_chs_exclude = stream_info.get("target_exclude", [])

        self.source_channels = self.select_channels(available_channels, s_chs, s_chs_exclude)
        self.source_idx = np.array(
            [i for i, ch in enumerate(available_channels) if ch in self.source_channels],
            dtype=np.int64,
        )

        self.target_channels = self.select_channels(available_channels, t_chs, t_chs_exclude)
        self.target_idx = np.array(
            [i for i, ch in enumerate(available_channels) if ch in self.target_channels],
            dtype=np.int64,
        )

        # No geoinfo for now (can be extended later with climatology)
        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.mean_geoinfo = np.zeros(0)
        self.stdev_geoinfo = np.ones(0)

        # Get target channel weights
        self.target_channel_weights = self.parse_target_channel_weights()

        ds_name = stream_info["name"]
        _logger.info(
            f"{ds_name} initialized: {len(self.latitudes)}×{len(self.longitudes)} grid, "
            f"{data_start_time} to {data_end_time}, "
            f"source={self.source_channels}, target={self.target_channels}"
        )

    def _apply_spatial_filters(self, stream_info: dict) -> None:
        """
        Apply spatial filtering to reduce memory usage

        Supports:
        1. Bounding box filtering (lat/lon bounds)
        2. Spatial subsampling (stride)

        Parameters
        ----------
        stream_info :
            stream configuration dict
        """
        original_shape = (len(self.latitudes), len(self.longitudes))
        original_points = original_shape[0] * original_shape[1]

        # Get spatial bounding box (if specified)
        bbox = stream_info.get("spatial_bbox", None)
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox

            # Find indices within bounding box
            lat_mask = (self.latitudes >= lat_min) & (self.latitudes <= lat_max)
            lon_mask = (self.longitudes >= lon_min) & (self.longitudes <= lon_max)

            # Filter coordinates
            self.lat_indices = np.where(lat_mask)[0]
            self.lon_indices = np.where(lon_mask)[0]

            self.latitudes = self.latitudes[self.lat_indices]
            self.longitudes = self.longitudes[self.lon_indices]
        else:
            # No bounding box - use full grid
            self.lat_indices = np.arange(len(self.latitudes))
            self.lon_indices = np.arange(len(self.longitudes))

        # Get spatial stride (subsampling)
        spatial_stride = stream_info.get("spatial_stride", 1)
        if spatial_stride > 1:
            # Subsample indices
            self.lat_indices = self.lat_indices[::spatial_stride]
            self.lon_indices = self.lon_indices[::spatial_stride]

            # Subsample coordinates
            self.latitudes = self.latitudes[::spatial_stride]
            self.longitudes = self.longitudes[::spatial_stride]

        # Log spatial filtering results
        final_shape = (len(self.latitudes), len(self.longitudes))
        final_points = final_shape[0] * final_shape[1]
        total_reduction = (1 - final_points / original_points) * 100

        if bbox or spatial_stride > 1:
            _logger.info(
                f"Spatial filtering: {original_shape} → {final_shape} "
                f"({final_points:,} points, {total_reduction:.1f}% reduction)"
            )

    def _parse_time_units(self, units_str: str) -> tuple[datetime.datetime, str]:
        """
        Parse time units string like "minutes since 1998-01-01 00:00:00"

        Parameters
        ----------
        units_str :
            time units string from zarr attributes

        Returns
        -------
        base_datetime :
            base datetime for time coordinate
        time_unit :
            time unit (e.g., 'minutes', 'hours', 'seconds')
        """
        # Pattern: "unit since YYYY-MM-DD HH:MM:SS"
        pattern = r"(\w+)\s+since\s+(.+)"
        match = re.match(pattern, units_str)

        if not match:
            raise ValueError(f"Could not parse time units: {units_str}")

        time_unit = match.group(1).lower()
        datetime_str = match.group(2).strip()

        # Parse datetime
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d",
        ]:
            try:
                base_datetime = datetime.datetime.strptime(datetime_str, fmt)
                return base_datetime, time_unit
            except ValueError:
                continue

        raise ValueError(f"Could not parse datetime from units: {units_str}")

    def _convert_time_to_datetime64(
        self, time_raw: NDArray, base_datetime: datetime.datetime, time_unit: str
    ) -> NDArray[NPDT64]:
        """
        Convert raw time values to numpy datetime64

        Parameters
        ----------
        time_raw :
            raw time values from zarr
        base_datetime :
            base datetime for time coordinate
        time_unit :
            time unit (e.g., 'minutes', 'hours')

        Returns
        -------
        datetimes :
            numpy datetime64 array
        """
        # Map common time units to numpy timedelta units
        unit_map = {
            "seconds": "s",
            "second": "s",
            "minutes": "m",
            "minute": "m",
            "hours": "h",
            "hour": "h",
            "days": "D",
            "day": "D",
        }

        np_unit = unit_map.get(time_unit)
        if np_unit is None:
            raise ValueError(f"Unsupported time unit: {time_unit}")

        # Convert base datetime to numpy datetime64
        base_dt64 = np.datetime64(base_datetime)

        # Create timedelta array and add to base
        time_deltas = time_raw.astype("timedelta64[" + np_unit + "]")
        datetimes = base_dt64 + time_deltas

        return datetimes

    def _compute_datetimes_for_range(self, start_idx: int, end_idx: int) -> NDArray[NPDT64]:
        """
        Compute datetimes for a range of zarr indices (lazy - no disk read).

        Parameters
        ----------
        start_idx :
            Start index (inclusive)
        end_idx :
            End index (exclusive)

        Returns
        -------
        datetimes :
            Array of datetime64 values
        """
        indices = np.arange(start_idx, end_idx)
        deltas = (indices * self.period_ns).astype("timedelta64[ns]")
        return self.base_datetime_np + deltas

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0
        self.coords_template = np.zeros((0, 2), dtype=np.float32)
        self.n_grid_points = 0

    @override
    def length(self) -> int:
        return self.len

    def select_channels(
        self, colnames: list[str], cols_select: list[str] | None, cols_exclude: list[str] | None
    ) -> list[str]:
        """
        Select channels based on include/exclude lists

        Parameters
        ----------
        colnames :
            available channel names
        cols_select :
            list of patterns to include (None = all, [] = none)
        cols_exclude :
            list of patterns to exclude

        Returns
        -------
        selected_colnames :
            filtered list of channel names
        """
        # Handle empty list explicitly: [] means include NONE
        # None means no filter (include all)
        if cols_select is not None and len(cols_select) == 0:
            return []

        selected_colnames = [
            c
            for c in colnames
            if (
                (np.array([c_sel in c for c_sel in cols_select]).any() if cols_select else True)
                and not (
                    np.array([c_excl in c for c_excl in cols_exclude]).any()
                    if cols_exclude
                    else False
                )
            )
        ]

        return selected_colnames

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels (0 for precipitation)

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        # Get dataset time indices for the window
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        # Return empty if no valid data
        if self.len == 0 or len(t_idxs) == 0 or len(channels_idx) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        didx_end = t_idxs[-1] + 1  # End is exclusive

        # Extract precipitation data for time range (with spatial filtering)
        # Shape: (n_times, n_lats, n_lons) -> only load filtered region
        try:
            # didx_start and didx_end from _get_dataset_idxs are already the correct
            # zarr indices (computed relative to data_start_time, which is the first
            # timestamp in the zarr array). No offset adjustment is needed here.
            zarr_start = didx_start
            zarr_end = didx_end

            # Use advanced indexing to load only the filtered spatial region
            # This dramatically reduces memory usage for large datasets like IMERG
            precip_slice = self.precip[zarr_start:zarr_end, self.lat_indices, :]
            precip_slice = precip_slice[:, :, self.lon_indices]
        except Exception as e:
            _logger.error(f"Error reading precipitation data: {e}")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        n_times = precip_slice.shape[0]

        # Flatten precipitation data
        # From (n_times, n_lats, n_lons) to (n_times * n_lats * n_lons, 1)
        data_flat = precip_slice.reshape(-1, 1).astype(np.float32)

        # Select requested channels (for IMERG, only index 0 = precipitation)
        data = data_flat[:, channels_idx]

        # Repeat coordinates for each time step
        # Shape: (n_times * n_grid_points, 2)
        coords = np.vstack([self.coords_template] * n_times)

        # Compute datetimes on-demand (lazy loading - no stored datetime array)
        # Repeat each datetime n_grid_points times
        window_datetimes = self._compute_datetimes_for_range(zarr_start, zarr_end)
        datetimes = np.repeat(window_datetimes, self.n_grid_points)

        # Empty geoinfos for now
        geoinfos = np.zeros((len(data), 0), dtype=np.float32)

        # Apply time mask to ensure [t_start, t_end) convention
        t_mask = np.logical_and(datetimes >= dtr.start, datetimes < dtr.end)

        # Debug: warn if time mask filters everything
        if t_mask.sum() == 0 and len(datetimes) > 0:
            _logger.warning(
                f"IMERG _get: Time mask filtered ALL data! "
                f"zarr_idx=[{zarr_start},{zarr_end}), n_times={n_times}, "
                f"computed_dt=[{window_datetimes[0]}, {window_datetimes[-1]}], "
                f"expected_dt=[{dtr.start}, {dtr.end})"
            )

        rd = ReaderData(
            coords=coords[t_mask],
            geoinfos=geoinfos[t_mask],
            data=data[t_mask],
            datetimes=datetimes[t_mask],
        )

        check_reader_data(rd, dtr)

        return rd
