# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import netCDF4 as nc
import numpy as np
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


class DataReaderRadklim(DataReaderTimestep):
    """
    Data reader for RADKLIM (DWD radar-based precipitation climatology) dataset in netCDF format.

    RADKLIM provides hourly precipitation estimates over Germany based on weather radar data.
    Data is organized in monthly netCDF files across multiple years.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct data reader for RADKLIM dataset

        Parameters
        ----------
        tw_handler :
            time window handler
        filename :
            base directory containing year subdirectories with netCDF files
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        self.base_path = Path(filename)
        if not self.base_path.exists():
            raise FileNotFoundError(f"RADKLIM base path not found: {self.base_path}")

        # Build file index mapping time ranges to file paths
        _logger.info(f"Building file index for RADKLIM data in {self.base_path}")
        self.file_index = self._build_file_index()

        if not self.file_index:
            name = stream_info["name"]
            _logger.warning(f"No RADKLIM files found in {self.base_path}. Stream {name} is empty.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Determine data period from file index
        data_start_time = self.file_index[0]["start"]
        data_end_time = self.file_index[-1]["end"]

        # Frequency is hourly
        period = np.timedelta64(1, "h")

        # Load spatial grid from first file
        first_file = self.file_index[0]["path"]
        with nc.Dataset(first_file, "r") as ds:
            # Read lat/lon arrays (2D)
            lat_2d = ds.variables["lat"][:]
            lon_2d = ds.variables["lon"][:]

            # Flatten and create coordinate template
            lat_flat = lat_2d.flatten()
            lon_flat = lon_2d.flatten()
            self.coords_template = np.stack([lat_flat, lon_flat], axis=1).astype(np.float32)
            self.n_grid_points = len(self.coords_template)

            # Store grid dimensions
            self.grid_shape = lat_2d.shape  # (ny, nx)

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

        # Calculate total length in hours
        total_hours = int((data_end_time - data_start_time) / np.timedelta64(1, "h")) + 1
        self.len = total_hours

        # Simple statistics (mean/std from typical precipitation values)
        # Could be computed from data, but using reasonable defaults for now
        self.mean = np.array([0.1], dtype=np.float32)  # ~0.1 mm/hr average
        self.stdev = np.array([1.0], dtype=np.float32)  # ~1.0 mm/hr std dev

        # Select channels (use netCDF variable name)
        available_channels = ["RR"]

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

        # No geoinfo for now
        self.geoinfo_channels = []
        self.geoinfo_idx = []
        self.mean_geoinfo = np.zeros(0)
        self.stdev_geoinfo = np.ones(0)

        # Get target channel weights
        self.target_channel_weights = self.parse_target_channel_weights()

        # File caching
        self.current_file = None
        self.current_filepath = None

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: data period: {data_start_time} to {data_end_time}")
        _logger.info(f"{ds_name}: frequency: {period}")
        _logger.info(f"{ds_name}: grid size: {self.grid_shape[0]} x {self.grid_shape[1]}")
        _logger.info(f"{ds_name}: number of files: {len(self.file_index)}")

    def _build_file_index(self) -> list[dict]:
        """
        Build index mapping time ranges to file paths

        Returns
        -------
        file_index :
            List of dicts with 'path', 'start', 'end', 'year', 'month'
        """
        file_index = []

        # Scan year directories
        year_dirs = sorted([d for d in self.base_path.iterdir() if d.is_dir() and d.name.isdigit()])

        for year_dir in year_dirs:
            # Scan monthly netCDF files
            nc_files = sorted(year_dir.glob("*.nc"))

            for nc_file in nc_files:
                try:
                    # Open file to get time range
                    with nc.Dataset(nc_file, "r") as ds:
                        time_var = ds.variables["time"]
                        times = nc.num2date(time_var[:], time_var.units, time_var.calendar)

                        start_time = times[0]
                        end_time = times[-1]

                        # Extract year and month from filename
                        # Pattern: RW_2017.002_YYYYMM.nc
                        filename = nc_file.name
                        year_month_part = filename.split("_")[-1].replace(".nc", "")
                        year = int(year_month_part[:4])
                        month = int(year_month_part[4:6])

                        file_index.append(
                            {
                                "path": nc_file,
                                "start": np.datetime64(start_time),
                                "end": np.datetime64(end_time),
                                "year": year,
                                "month": month,
                            }
                        )
                except Exception as e:
                    _logger.warning(f"Could not read file {nc_file}: {e}")
                    continue

        return sorted(file_index, key=lambda x: x["start"])

    def _get_files_for_time_range(self, start: NPDT64, end: NPDT64) -> list[dict]:
        """
        Find all files that overlap with given time range

        Parameters
        ----------
        start :
            start of time range
        end :
            end of time range

        Returns
        -------
        files :
            List of file index entries that overlap with time range
        """
        files = []
        for entry in self.file_index:
            # Check if file's time range overlaps with requested range
            if entry["end"] >= start and entry["start"] < end:
                files.append(entry)
        return files

    def _open_file(self, filepath: Path, cache: bool = True) -> nc.Dataset:
        """
        Open netCDF file, using cache if possible

        Parameters
        ----------
        filepath :
            path to netCDF file
        cache :
            whether to cache the opened file

        Returns
        -------
        dataset :
            opened netCDF dataset
        """
        # Use cached file if same path
        if cache and self.current_filepath == filepath and self.current_file is not None:
            return self.current_file

        # Close previous file
        if self.current_file is not None:
            try:
                self.current_file.close()
            except Exception:
                pass  # Ignore errors when closing

        # Open new file
        self.current_file = nc.Dataset(filepath, "r")
        self.current_filepath = filepath

        return self.current_file

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0
        self.coords_template = np.zeros((0, 2), dtype=np.float32)
        self.n_grid_points = 0
        self.grid_shape = (0, 0)
        self.file_index = []

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
        Get data for window, handling multi-file scenarios

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels (0 for rainfall)

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

        # Find files that contain data for this time range
        files_needed = self._get_files_for_time_range(dtr.start, dtr.end)

        if not files_needed:
            _logger.debug(f"No files found for time range {dtr.start} to {dtr.end}")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # Load data from each file
        data_chunks = []
        coords_chunks = []
        datetime_chunks = []

        for file_entry in files_needed:
            try:
                ds = self._open_file(file_entry["path"])

                # Get time variable and convert to datetime
                time_var = ds.variables["time"]
                file_times = nc.num2date(time_var[:], time_var.units, time_var.calendar)
                file_times_dt64 = np.array([np.datetime64(t) for t in file_times])

                # Find timesteps within requested window
                mask = (file_times_dt64 >= dtr.start) & (file_times_dt64 < dtr.end)
                time_indices = np.where(mask)[0]

                if len(time_indices) > 0:
                    # Load rainfall data for these times
                    # Shape: (n_times, ny, nx)
                    rr_data = ds.variables["RR"][time_indices, :, :]

                    # Convert masked array to regular array, replacing fill values with NaN
                    if np.ma.is_masked(rr_data):
                        rr_data = np.ma.filled(rr_data, np.nan)
                    else:
                        # Replace fill value (999.0) with NaN
                        rr_data = np.where(rr_data >= 900.0, np.nan, rr_data)

                    # Flatten spatial dimensions
                    # From (n_times, ny, nx) to (n_times * ny * nx, 1)
                    data_flat = rr_data.reshape(len(time_indices) * self.n_grid_points, 1)
                    data_chunks.append(data_flat.astype(np.float32))

                    # Repeat coords for each timestep
                    coords_chunk = np.vstack([self.coords_template] * len(time_indices))
                    coords_chunks.append(coords_chunk)

                    # Repeat datetimes for each grid point
                    datetimes_chunk = np.repeat(file_times_dt64[time_indices], self.n_grid_points)
                    datetime_chunks.append(datetimes_chunk)

            except Exception as e:
                _logger.error(f"Error reading file {file_entry['path']}: {e}")
                continue

        # Check if we got any data
        if not data_chunks:
            _logger.debug(f"No valid data found for time range {dtr.start} to {dtr.end}")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # Concatenate all chunks
        data = np.concatenate(data_chunks, axis=0)
        coords = np.concatenate(coords_chunks, axis=0)
        datetimes = np.concatenate(datetime_chunks, axis=0)

        # Select requested channels
        data_selected = data[:, channels_idx]

        # Empty geoinfos for now
        geoinfos = np.zeros((len(data), 0), dtype=np.float32)

        # Create ReaderData
        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data_selected,
            datetimes=datetimes,
        )

        check_reader_data(rd, dtr)

        return rd

    def __del__(self):
        """Close any open files"""
        if hasattr(self, "current_file") and self.current_file is not None:
            try:
                self.current_file.close()
            except Exception:
                pass  # Ignore errors during cleanup
