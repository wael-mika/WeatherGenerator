# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Unified data reader for Anemoi datasets with configurable transformations.

Supports multiple transformation types for handling highly skewed distributions
(e.g., precipitation data):

- "log10": log10(x * scale + offset) with optional z-score
- "log_eps": ln((x + eps) / eps) with z-score normalization
- "arcsinh": arcsinh(x * scale / alpha) with optional z-score (recommended)
- "none": no transformation, just optional z-score normalization

Configuration via stream_info:
    transform_type: str - one of "log10", "log_eps", "arcsinh", "none"
    transform_scale: float - unit conversion factor (default: 1000.0 for m->mm)
    transform_alpha: float - scale parameter for arcsinh (default: 0.15)
    transform_eps: float - epsilon for log_eps (default: 2.39e-7)
    transform_offset: float - offset for log10 (default: 1.0)
    transform_mu: float - mean for z-score (optional)
    transform_sigma: float - std for z-score (optional, disables z-score if <= 0)
"""

import logging
from pathlib import Path
from typing import TypeAlias, override

import numpy as np
import torch
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import TimeWindowHandler

_logger = logging.getLogger(__name__)

DType: TypeAlias = np.float32


# ---------- Type-agnostic helper functions ----------


def _to_float64(x):
    """Convert to float64 for numerical precision."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    elif torch.is_tensor(x):
        return x.to(dtype=torch.float64)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _to_output_dtype(x, dtype=DType):
    """Convert back to output dtype."""
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif torch.is_tensor(x):
        torch_dtype = getattr(torch, np.dtype(dtype).name, None)
        if torch_dtype is None:
            torch_dtype = torch.float32
        return x.to(dtype=torch_dtype)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _log(x):
    """Natural logarithm (type-agnostic)."""
    if isinstance(x, np.ndarray):
        return np.log(x)
    elif torch.is_tensor(x):
        return torch.log(x)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _log10(x):
    """Base-10 logarithm (type-agnostic)."""
    if isinstance(x, np.ndarray):
        return np.log10(x)
    elif torch.is_tensor(x):
        return torch.log10(x)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _exp(x):
    """Exponential (type-agnostic)."""
    if isinstance(x, np.ndarray):
        return np.exp(x)
    elif torch.is_tensor(x):
        return torch.exp(x)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _pow10(x):
    """10^x (type-agnostic)."""
    if isinstance(x, np.ndarray):
        return np.power(10.0, x)
    elif torch.is_tensor(x):
        return torch.pow(10.0, x)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _arcsinh(x):
    """Inverse hyperbolic sine (type-agnostic)."""
    if isinstance(x, np.ndarray):
        return np.arcsinh(x)
    elif torch.is_tensor(x):
        return torch.arcsinh(x)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


def _sinh(x):
    """Hyperbolic sine (type-agnostic)."""
    if isinstance(x, np.ndarray):
        return np.sinh(x)
    elif torch.is_tensor(x):
        return torch.sinh(x)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


class DataReaderAnemoiTransform(DataReaderAnemoi):
    """
    Unified Anemoi data reader with configurable transformations.

    Supports multiple transformation types for handling highly skewed distributions
    like precipitation data. All parameters are configurable via stream_info.

    Transform types:
    - "arcsinh": y = arcsinh(x_scaled / alpha) - best for precipitation
    - "log10": y = log10(x_scaled + offset)
    - "log_eps": y = ln((x + eps) / eps)
    - "none": no transformation

    All transforms support optional z-score normalization: y_final = (y - mu) / sigma
    """

    # Default values (can be overridden via stream_info)
    DEFAULTS = {
        "transform_type": "arcsinh",
        "transform_scale": 1000.0,  # m -> mm conversion
        "transform_alpha": 0.15,  # arcsinh scale parameter
        "transform_eps": 2.39e-7,  # log_eps epsilon
        "transform_offset": 1.0,  # log10 offset
        "transform_mu": None,  # z-score mean (None = no z-score)
        "transform_sigma": None,  # z-score std (None = no z-score)
    }

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        super().__init__(tw_handler, filename, stream_info)

        # Parse configuration from stream_info with defaults
        self.transform_type = str(
            stream_info.get("transform_type", self.DEFAULTS["transform_type"])
        )
        self.scale = float(stream_info.get("transform_scale", self.DEFAULTS["transform_scale"]))
        self.alpha = float(stream_info.get("transform_alpha", self.DEFAULTS["transform_alpha"]))
        self.eps = float(stream_info.get("transform_eps", self.DEFAULTS["transform_eps"]))
        self.offset = float(stream_info.get("transform_offset", self.DEFAULTS["transform_offset"]))

        # Z-score parameters (optional)
        mu = stream_info.get("transform_mu", self.DEFAULTS["transform_mu"])
        sigma = stream_info.get("transform_sigma", self.DEFAULTS["transform_sigma"])

        self.mu = float(mu) if mu is not None else None
        self.sigma = float(sigma) if sigma is not None and float(sigma) > 0 else None
        self.apply_zscore = (self.mu is not None) and (self.sigma is not None)

        # Validation
        valid_types = ("arcsinh", "log10", "log_eps", "none")
        if self.transform_type not in valid_types:
            raise ValueError(
                f"transform_type must be one of {valid_types}, got '{self.transform_type}'"
            )

        if self.transform_type == "arcsinh" and self.alpha <= 0:
            raise ValueError(f"transform_alpha must be > 0 for arcsinh, got {self.alpha}")

        if self.transform_type == "log_eps" and self.eps <= 0:
            raise ValueError(f"transform_eps must be > 0 for log_eps, got {self.eps}")

        # Log configuration
        zscore_info = (
            f"z-score(mu={self.mu:.6g}, sigma={self.sigma:.6g})"
            if self.apply_zscore
            else "no z-score"
        )
        _logger.info(
            f"DataReaderAnemoiTransform: type={self.transform_type}, scale={self.scale}, "
            f"alpha={self.alpha}, eps={self.eps}, offset={self.offset}, {zscore_info}"
        )

    def _forward_transform(self, x):
        """
        Apply forward transformation: raw data -> normalized space.

        Parameters
        ----------
        x : array-like
            Input data (in original units, e.g., meters)

        Returns
        -------
        Transformed data
        """
        # Convert to float64 for precision
        x = _to_float64(x)

        # Apply unit scaling (e.g., m -> mm)
        x_scaled = x * self.scale

        # Apply transformation based on type
        if self.transform_type == "arcsinh":
            # arcsinh(x_scaled / alpha) - preserves zeros, handles wide range
            y = _arcsinh(x_scaled / self.alpha)

        elif self.transform_type == "log10":
            # log10(x_scaled + offset) - classic log transform
            y = _log10(x_scaled + self.offset)

        elif self.transform_type == "log_eps":
            # ln((x + eps) / eps) - shifted log that maps 0 -> 0
            y = _log((x + self.eps) / self.eps)

        elif self.transform_type == "none":
            y = x_scaled

        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

        # Apply optional z-score normalization
        if self.apply_zscore:
            y = (y - self.mu) / self.sigma

        return _to_output_dtype(y)

    def _inverse_transform(self, y):
        """
        Apply inverse transformation: normalized space -> raw data.

        Parameters
        ----------
        y : array-like
            Transformed data

        Returns
        -------
        Data in original units (e.g., meters)
        """
        # Convert to float64 for precision
        y = _to_float64(y)

        # Undo z-score normalization
        if self.apply_zscore:
            y = y * self.sigma + self.mu

        # Apply inverse transformation based on type
        if self.transform_type == "arcsinh":
            # x_scaled = alpha * sinh(y)
            x_scaled = self.alpha * _sinh(y)

        elif self.transform_type == "log10":
            # x_scaled = 10^y - offset
            x_scaled = _pow10(y) - self.offset

        elif self.transform_type == "log_eps":
            # x = eps * (exp(y) - 1)
            x_scaled = self.eps * (_exp(y) - 1.0)

        elif self.transform_type == "none":
            x_scaled = y

        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

        # Undo unit scaling (e.g., mm -> m)
        x = x_scaled / self.scale

        return _to_output_dtype(x)

    # ---------- Override normalization methods ----------

    @override
    def normalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """Apply forward transformation to source channels."""
        assert source.shape[-1] == len(self.source_idx), (
            f"incorrect number of source channels: {source.shape[-1]} vs {len(self.source_idx)}"
        )

        for i in range(len(self.source_idx)):
            source[..., i] = self._forward_transform(source[..., i])

        return source

    @override
    def normalize_target_channels(self, target: NDArray[DType]) -> NDArray[DType]:
        """Apply forward transformation to target channels."""
        assert target.shape[-1] == len(self.target_idx), (
            f"incorrect number of target channels: {target.shape[-1]} vs {len(self.target_idx)}"
        )

        for i in range(len(self.target_idx)):
            target[..., i] = self._forward_transform(target[..., i])

        return target

    @override
    def denormalize_source_channels(self, source: NDArray[DType]) -> NDArray[DType]:
        """Apply inverse transformation to source channels."""
        assert source.shape[-1] == len(self.source_idx), (
            f"incorrect number of source channels: {source.shape[-1]} vs {len(self.source_idx)}"
        )

        for i in range(len(self.source_idx)):
            source[..., i] = self._inverse_transform(source[..., i])

        return source

    @override
    def denormalize_target_channels(self, data: NDArray[DType]) -> NDArray[DType]:
        """Apply inverse transformation to target channels."""
        assert data.shape[-1] == len(self.target_idx), (
            f"incorrect number of target channels: {data.shape[-1]} vs {len(self.target_idx)}"
        )

        for i in range(len(self.target_idx)):
            data[..., i] = self._inverse_transform(data[..., i])

        return data
