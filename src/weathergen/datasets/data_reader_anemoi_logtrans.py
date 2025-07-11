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
from typing import TypeAlias, override

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    TimeWindowHandler,
)

_logger = logging.getLogger(__name__)

DType: TypeAlias = np.float32  # The type for the data in the datasets.


class DataReaderAnemoiLogTrans(DataReaderAnemoi):
    "Wrapper for Anemoi datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        super().__init__(tw_handler, filename, stream_info)

    @override
    def normalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """
        Log-transforms source channels. This is so far used for IMERG data only and also involves scaling of the data to convert from metres to milimetres.

        Parameters
        ----------
        data :
            data to be log-transformed

        Returns
        -------
        Log-transformed data
        """
        assert source.shape[-1] == len(self.source_idx), "incorrect number of channels"
        for i, ch in enumerate(self.source_idx):
            source[..., i] = np.log10(source[..., i] * 1000.0 + 1.0)

        return source

    @override
    def normalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
        """
        Log-transforms target channels. This is so far used for IMERG data only and also involves scaling of the data to convert from metres to milimetres.

        Parameters
        ----------
        data :
            data to be log-transformed

        Returns
        -------
        Log-transformed data
        """
        assert target.shape[-1] == len(self.target_idx), "incorrect number of target channels"
        for i, ch in enumerate(self.target_idx):
            target[..., i] = np.log10(target[..., i] * 1000.0 + 1.0)

        return target

    @override
    def denormalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """
        Denormalize source channels

        Parameters
        ----------
        data :
            data to be denormalized

        Returns
        -------
        Denormalized data
        """
        assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
        for i, ch in enumerate(self.source_idx):
            source[..., i] = (torch.pow(10, source[..., i]) - 1.0) / 1000.0

        return source

    @override
    def denormalize_target_channels(self, data: NDArray[DType]) -> NDArray[DType]:
        """
        Denormalize target channels

        Parameters
        ----------
        data :
            data to be denormalized (target or pred)

        Returns
        -------
        Denormalized data
        """
        assert data.shape[-1] == len(self.target_idx), "incorrect number of target channels"
        for i, ch in enumerate(self.target_idx):
            data[..., i] = (torch.pow(10, data[..., i]) - 1.0) / 1000.0

        return data
