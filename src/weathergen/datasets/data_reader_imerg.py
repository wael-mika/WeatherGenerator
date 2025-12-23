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
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
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
        self.z = zarr.open(filename, mode="r")

        # Extract coordinates
        self.latitudes = self.z["latitude"][:].astype(np.float32)
        self.longitudes = self.z["longitude"][:].astype(np.float32)

        # Parse time coordinate
        time_raw = self.z["time"][:]
        time_attrs = dict(self.z["time"].attrs)

        # Parse time units (e.g., "minutes since 1998-01-01 00:00:00")
        units_str = time_attrs.get("units", "")
        base_datetime, time_unit = self._parse_time_units(units_str)

        # Convert time to numpy datetime64
        self.datetimes = self._convert_time_to_datetime64(time_raw, base_datetime, time_unit)

        # Store reference to precipitation array (lazy loading)
        self.precip = self.z["precipitation"]

        # Determine data period
        data_start_time = self.datetimes[0]
        data_end_time = self.datetimes[-1]

        # Parse frequency from metadata or infer from time coordinate
        frequency_str = self.z.attrs.get("frequency", "30min")
        period = self._parse_frequency(frequency_str, self.datetimes)

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

        self.len = len(self.datetimes)

        # Load statistics for normalization
        # Using global statistics (Option A from plan)
        self.mean = np.array([self.z["mean"][()]], dtype=np.float32)
        self.stdev = np.array([self.z["stdev"][()]], dtype=np.float32)

        # Create meshgrid for coordinates (computed once, used repeatedly)
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
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: data period: {data_start_time} to {data_end_time}")
        _logger.info(f"{ds_name}: frequency: {period}")
        _logger.info(f"{ds_name}: grid size: {len(self.latitudes)} x {len(self.longitudes)}")

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

    def _parse_frequency(self, frequency_str: str, datetimes: NDArray[NPDT64]) -> NPTDel64:
        """
        Parse frequency string or infer from datetime array

        Parameters
        ----------
        frequency_str :
            frequency string from metadata (e.g., "30min")
        datetimes :
            datetime array

        Returns
        -------
        period :
            numpy timedelta64 representing the frequency
        """
        # Try to parse frequency string
        if frequency_str:
            # Handle common formats: "30min", "1h", "1hour", etc.
            freq_map = {
                "min": "m",
                "minute": "m",
                "h": "h",
                "hour": "h",
                "d": "D",
                "day": "D",
            }

            pattern = r"(\d+)\s*(\w+)"
            match = re.match(pattern, frequency_str)
            if match:
                value = int(match.group(1))
                unit_str = match.group(2).lower()

                for key, np_unit in freq_map.items():
                    if key in unit_str:
                        return np.timedelta64(value, np_unit)

        # Infer from datetime array
        if len(datetimes) > 1:
            period = datetimes[1] - datetimes[0]
            _logger.info(f"Inferred period from datetimes: {period}")
            return period

        # Default fallback
        _logger.warning("Could not determine frequency, using default 30 minutes")
        return np.timedelta64(30, "m")

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
            list of patterns to include (None = all)
        cols_exclude :
            list of patterns to exclude

        Returns
        -------
        selected_colnames :
            filtered list of channel names
        """
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

        # Extract precipitation data for time range
        # Shape: (n_times, n_lats, n_lons)
        try:
            precip_slice = self.precip[didx_start:didx_end, :, :]
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

        # Create datetime array matching data points
        # Repeat each datetime n_grid_points times
        datetimes = np.repeat(self.datetimes[didx_start:didx_end], self.n_grid_points)

        # Empty geoinfos for now
        geoinfos = np.zeros((len(data), 0), dtype=np.float32)

        # Apply time mask to ensure [t_start, t_end) convention
        t_mask = np.logical_and(datetimes >= dtr.start, datetimes < dtr.end)

        rd = ReaderData(
            coords=coords[t_mask],
            geoinfos=geoinfos[t_mask],
            data=data[t_mask],
            datetimes=datetimes[t_mask],
        )

        check_reader_data(rd, dtr)

        return rd
