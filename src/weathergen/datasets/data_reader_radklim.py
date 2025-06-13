import json
import os
from pathlib import Path
from typing import Union, override

import numpy as np
import pandas as pd
import xarray as xr
from icechunk import local_filesystem_storage, Repository

# Import the base classes from your new architecture
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    DType,
    check_reader_data,
)


class RadklimDataReader(DataReaderTimestep):
    """
    RADKLIM data reader adapted to the new base class architecture.
    Lazily loads Radklim data via an Icechunk repository.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        icechunk_repo_path: Union[str, Path],
        normalization_path: Union[str, Path],
        branch_name: str = 'main',
    ):
        """
        Parameters
        ----------
        tw_handler : TimeWindowHandler
            Time window handler from the base architecture
        icechunk_repo_path : Union[str, Path]
            Path to the Icechunk repository
        normalization_path : Union[str, Path]
            Path to normalization statistics JSON file
        branch_name : str
            Branch name in the Icechunk repository
        """
        
        # Open dataset to check compatibility with requested parameters
        self.ds = self._open_icechunk_dataset(icechunk_repo_path, branch_name)
        
        # Get dataset timing information
        times_all = self.ds["time"].values
        data_start_time = times_all[0].astype('datetime64[ns]')
        data_end_time = times_all[-1].astype('datetime64[ns]')
        
        # Determine period from time differences
        dt_arr = np.unique(np.diff(times_all.astype("datetime64[s]")))
        if dt_arr.size != 1:
            raise ValueError("Inconsistent time steps in the dataset.")
        dt_seconds = int(dt_arr[0].item().total_seconds())
        period = np.timedelta64(dt_seconds, 's')
        
        # Check if there's overlap with the time window
        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            super().__init__(tw_handler)
            self.init_empty()
            return
            
        # Call parent constructor with dataset timing info
        super().__init__(
            tw_handler=tw_handler,
            data_start_time=data_start_time,
            data_end_time=data_end_time,
            period=period,
        )
        
        # Subset dataset to the time window of interest
        self._subset_dataset_to_timewindow(tw_handler)
        
        # Load normalization stats
        self._load_normalization_stats(normalization_path)
        
        # Set up channel configuration (single RR channel)
        self._setup_channels()
        
        # Compute and cache spatial coordinates
        self._compute_spatial_coords()

    @override
    def init_empty(self) -> None:
        """Initialize empty dataset."""
        super().init_empty()
        self.len = 0

    def _open_icechunk_dataset(self, icechunk_repo_path: Union[str, Path], branch_name: str) -> xr.Dataset:
        """Open dataset from Icechunk repository."""
        repo_path = Path(icechunk_repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Icechunk repository not found: {repo_path}")
        
        abs_repo_path = os.path.abspath(repo_path)
        storage_config = local_filesystem_storage(abs_repo_path)
        self.repo = Repository.open(storage_config)
        
        self.session = self.repo.writable_session(branch_name)
        zarr_store = self.session.store
        
        ds = xr.open_zarr(
            zarr_store, 
            consolidated=False,
            chunks={"time": "auto", "y": -1, "x": -1}
        )
        return ds

    def _subset_dataset_to_timewindow(self, tw_handler: TimeWindowHandler):
        """Subset dataset to the time window of interest."""
        times_all = self.ds["time"].values
        start_time_np = np.datetime64(tw_handler.t_start)
        end_time_np = np.datetime64(tw_handler.t_end)
        
        start_idx = int(np.searchsorted(times_all, start_time_np, side="left"))
        end_idx = int(np.searchsorted(times_all, end_time_np, side="right"))
        
        if start_idx >= end_idx:
            self.len = 0
            return
            
        # Subset the dataset to our time range and only keep RR variable
        self.ds = self.ds[["RR"]].isel(time=slice(start_idx, end_idx))
        self.len = len(self.ds["time"])

    def _load_normalization_stats(self, normalization_path: Union[str, Path]) -> None:
        """Load normalization statistics from JSON file."""
        path = Path(normalization_path)
        if not path.exists():
            raise FileNotFoundError(f"Normalization JSON not found: {path}")
        
        with open(path, "r") as f:
            stats = json.load(f)
        
        # Set up normalization arrays - single channel (RR)
        self.mean = np.array(stats.get("mean", [0.0]), dtype=DType)
        self.stdev = np.array(stats.get("std", [1.0]), dtype=DType)
        
        # No geoinfo normalization needed for RADKLIM
        self.mean_geoinfo = np.zeros(0, dtype=DType)
        self.stdev_geoinfo = np.ones(0, dtype=DType)
        
        if len(self.mean) != 1 or len(self.stdev) != 1:
            raise ValueError("RADKLIM should have exactly one channel (RR) for normalization")

    def _setup_channels(self) -> None:
        """Set up channel configuration required by base class."""
        # Single RR channel for both source and target
        self.source_channels = ["RR"]
        self.target_channels = ["RR"]
        self.geoinfo_channels = []
        
        # Channel indices - single channel at index 0
        self.source_idx = [0]
        self.target_idx = [0]
        self.geoinfo_idx = []

    def _compute_spatial_coords(self) -> None:
        """Compute and cache spatial coordinates."""
        # Get spatial dimensions
        y1d = self.ds["y"].values.astype(DType)
        x1d = self.ds["x"].values.astype(DType)
        self.ny, self.nx = len(y1d), len(x1d)
        
        # Load lat/lon arrays (handle time-varying coordinates if present)
        lat_var = self.ds["lat"]
        if "time" in lat_var.dims:
            lat2d = lat_var.isel(time=0).values.astype(DType)
        else:
            lat2d = lat_var.values.astype(DType)
            
        lon_var = self.ds["lon"]
        if "time" in lon_var.dims:
            lon2d = lon_var.isel(time=0).values.astype(DType)
        else:
            lon2d = lon_var.values.astype(DType)
        
        # Apply coordinate transformations (same as Anemoi example)
        self.latitudes = self._clip_lat(lat2d.flatten())
        self.longitudes = self._clip_lon(lon2d.flatten())

    def _clip_lat(self, lats: np.ndarray) -> np.ndarray:
        """Clip latitudes to the range [-90, 90] and ensure periodicity."""
        return (2 * np.clip(lats, -90.0, 90.0) - lats).astype(DType)

    def _clip_lon(self, lons: np.ndarray) -> np.ndarray:
        """Clip longitudes to the range [-180, 180] and ensure periodicity."""
        return ((lons + 180.0) % 360.0 - 180.0).astype(DType)

    @override
    def length(self) -> int:
        """Return the length of the dataset."""
        return getattr(self, 'len', 0)

    