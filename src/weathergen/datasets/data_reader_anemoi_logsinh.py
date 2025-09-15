# # (C) Copyright 2025 WeatherGenerator contributors.
# #
# # This software is licensed under the terms of the Apache Licence Version 2.0
# # which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# #
# # In applying this licence, ECMWF does not waive the privileges and immunities
# # granted to it by virtue of its status as an intergovernmental organisation
# # nor does it submit to any jurisdiction.

# import logging
# from pathlib import Path
# from typing import TypeAlias, override

# import numpy as np
# import torch
# from numpy.typing import NDArray

# from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
# from weathergen.datasets.data_reader_base import (
#     TimeWindowHandler,
# )

# _logger = logging.getLogger(__name__)

# DType: TypeAlias = np.float32  # The type for the data in the datasets.

# def _to_float64(x):
#     if isinstance(x, np.ndarray):
#         return x.astype(np.float64)
#     elif torch.is_tensor(x):
#         return x.to(dtype=torch.float64)
#     else:
#         raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

# def _to_dtype(x, dtype):
#     if isinstance(x, np.ndarray):
#         return x.astype(dtype)
#     elif torch.is_tensor(x):
#         torch_dtype = getattr(torch, np.dtype(dtype).name, None)
#         if torch_dtype is None:
#             torch_dtype = dtype
#         return x.to(dtype=torch_dtype)
#     else:
#         raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

# def _arcsinh(x):
#     if isinstance(x, np.ndarray):
#         return np.arcsinh(x)
#     elif torch.is_tensor(x):
#         return torch.arcsinh(x)
#     else:
#         raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

# def _sinh(x):
#     if isinstance(x, np.ndarray):
#         return np.sinh(x)
#     elif torch.is_tensor(x):
#         return torch.sinh(x)
#     else:
#         raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

# class DataReaderAnemoiLogSinh(DataReaderAnemoi):
#     "Wrapper for Anemoi datasets using Log-Sinh (arcsinh/asinh) + z-score transformation"

#     MU = 0.25682980      # Global mean after arcsinh(x_mm)
#     SIGMA = 0.68649328   # Global std after arcsinh(x_mm)

#     def __init__(
#         self,
#         tw_handler: TimeWindowHandler,
#         filename: Path,
#         stream_info: dict,
#     ) -> None:
#         super().__init__(tw_handler, filename, stream_info)

#     @override
#     def normalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
#         """
#         Applies the arcsinh (inverse hyperbolic sine) transformation to source channels,
#         followed by z-score normalization using precomputed global mean and std.
#         Converts data from metres to millimetres before transformation.

#         Parameters
#         ----------
#         data :
#             data to be transformed

#         Returns
#         -------
#         Transformed and z-scored data
#         """
#         assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
#         for i, ch in enumerate(self.source_idx):
#             # --- CONVERT m to mm ---
#             x = _to_float64(source[..., i]) * 1000.0
#             x_trans = _arcsinh(x)
#             source[..., i] = _to_dtype((x_trans - self.MU) / self.SIGMA, DType)
#         return source

#     @override
#     def normalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
#         """
#         Applies the arcsinh (inverse hyperbolic sine) transformation to target channels,
#         followed by z-score normalization using precomputed global mean and std.
#         Converts data from metres to millimetres before transformation.

#         Parameters
#         ----------
#         data :
#             data to be transformed

#         Returns
#         -------
#         Transformed and z-scored data
#         """
#         assert target.shape[-1] == len(self.target_idx), "incorrect number of target channels"
#         for i, ch in enumerate(self.target_idx):
#             # --- CONVERT m to mm ---
#             x = _to_float64(target[..., i]) * 1000.0
#             x_trans = _arcsinh(x)
#             target[..., i] = _to_dtype((x_trans - self.MU) / self.SIGMA, DType)
#         return target

#     @override
#     def denormalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
#         """
#         Inverts z-score and arcsinh transformation for source channels.
#         Converts data from millimetres back to metres after inversion.

#         Parameters
#         ----------
#         data :
#             data to be denormalized

#         Returns
#         -------
#         Denormalized data
#         """
#         assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
#         for i, ch in enumerate(self.source_idx):
#             y = _to_float64(source[..., i]) * self.SIGMA + self.MU
#             x_mm = _sinh(y)
#             x_m = x_mm / 1000.0
#             source[..., i] = _to_dtype(x_m, DType)
#         return source

#     @override
#     def denormalize_target_channels(self, data: NDArray[DType]) -> NDArray[DType]:
#         """
#         Inverts z-score and arcsinh transformation for target channels.
#         Converts data from millimetres back to metres after inversion.

#         Parameters
#         ----------
#         data :
#             data to be denormalized (target or pred)

#         Returns
#         -------
#         Denormalized data
#         """
#         assert data.shape[-1] == len(self.target_idx), "incorrect number of target channels"
#         for i, ch in enumerate(self.target_idx):
#             y = _to_float64(data[..., i]) * self.SIGMA + self.MU
#             x_mm = _sinh(y)
#             x_m = x_mm / 1000.0
#             data[..., i] = _to_dtype(x_m, DType)
#         return data

# (C) Copyright 2025 WeatherGenerator contributors.
# Licensed under the Apache License, Version 2.0

import logging
from pathlib import Path
from typing import TypeAlias, override

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import TimeWindowHandler

_logger = logging.getLogger(__name__)

DType: TypeAlias = np.float32  # dataset storage dtype

# ---------- helpers ----------
def _to_float64(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    elif torch.is_tensor(x):
        return x.to(dtype=torch.float64)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

def _to_dtype(x, dtype):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif torch.is_tensor(x):
        torch_dtype = getattr(torch, np.dtype(dtype).name, None) or dtype
        return x.to(dtype=torch_dtype)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

def _arcsinh(x):
    if isinstance(x, np.ndarray):
        return np.arcsinh(x)
    elif torch.is_tensor(x):
        return torch.arcsinh(x)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

def _sinh(x):
    if isinstance(x, np.ndarray):
        return np.sinh(x)
    elif torch.is_tensor(x):
        return torch.sinh(x)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


class DataReaderAnemoiLogSinh(DataReaderAnemoi):
    """
    Anemoi wrapper using *scaled* arcsinh transform:

        y = asinh( x_mm / ALPHA )

    with optional z-scoring:
        y_z = (y - MU) / SIGMA

    Inverse:
        x_mm = ALPHA * sinh( y_z * SIGMA + MU )

    Notes
    -----
    - ALPHA is a length scale in mm; here default is 0.31 mm.
    - If MU/SIGMA are not provided (or SIGMA <= 0), z-scoring is skipped.
    - Zero maps to zero (asinh(0)=0), so dry points stay at 0 in transform space.
    """

    # Default scale (mm). You can override via stream_info or subclassing.
    ALPHA_DEFAULT = 0.000381472

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        super().__init__(tw_handler, filename, stream_info)

        # Allow overrides from stream_info (use keys if present)
        self.alpha: float = float(stream_info.get("logsinh_alpha", self.ALPHA_DEFAULT))

        # Optional z-score stats for the *scaled* transform asinh(x_mm/alpha)
        mu = stream_info.get("logsinh_mu", 2.3638699054718018)
        sigma = stream_info.get("logsinh_sigma", 3.6606361865997314)

        # If provided, use them; else, disable z-scoring
        self.mu = float(mu) if mu is not None else None
        self.sigma = float(sigma) if sigma is not None and float(sigma) > 0 else None

        # Convenience flag
        self.apply_zscore = (self.mu is not None) and (self.sigma is not None)

        if self.alpha <= 0:
            raise ValueError(f"logsinh_alpha must be > 0, got {self.alpha}")

        if self.apply_zscore:
            _logger.info(
                f"LogSinhScaled using alpha={self.alpha:.6g}, z-score with mu={self.mu:.6g}, sigma={self.sigma:.6g}"
            )
        else:
            _logger.info(
                f"LogSinhScaled using alpha={self.alpha:.6g}, no z-scoring (mu/sigma not provided)"
            )

    # ---------- forward (normalize) ----------
    @override
    def normalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
        for i, _ch in enumerate(self.source_idx):
            # meters -> mm
            x_mm = _to_float64(source[..., i]) * 1000.0
            # scaled asinh
            y = _arcsinh(x_mm / self.alpha)
            # optional z-score
            if self.apply_zscore:
                y = (y - self.mu) / self.sigma
            source[..., i] = _to_dtype(y, DType)
        return source

    @override
    def normalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
        assert target.shape[-1] == len(self.target_idx), "incorrect number of target channels"
        for i, _ch in enumerate(self.target_idx):
            x_mm = _to_float64(target[..., i]) * 1000.0
            y = _arcsinh(x_mm / self.alpha)
            if self.apply_zscore:
                y = (y - self.mu) / self.sigma
            target[..., i] = _to_dtype(y, DType)
        return target

    # ---------- inverse (denormalize) ----------
    @override
    def denormalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        assert source.shape[-1] == len(self.source_idx), "incorrect number of source channels"
        for i, _ch in enumerate(self.source_idx):
            y = _to_float64(source[..., i])
            # un-zscore if needed
            if self.apply_zscore:
                y = y * self.sigma + self.mu
            # inverse scaled asinh: x_mm = alpha * sinh(y)
            x_mm = self.alpha * _sinh(y)
            # back to meters
            source[..., i] = _to_dtype(x_mm / 1000.0, DType)
        return source

    @override
    def denormalize_target_channels(self, data: NDArray[DType]) -> NDArray[DType]:
        assert data.shape[-1] == len(self.target_idx), "incorrect number of target channels"
        for i, _ch in enumerate(self.target_idx):
            y = _to_float64(data[..., i])
            if self.apply_zscore:
                y = y * self.sigma + self.mu
            x_mm = self.alpha * _sinh(y)
            data[..., i] = _to_dtype(x_mm / 1000.0, DType)
        return data
