# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Spectral utility functions for the Wavelet-Fourier Composite Loss (WFCL).

Provides grid reconstruction from flat HEALPix/O96 point arrays, Fourier-domain
losses (FAL, FCL), DTCWT wavelet-domain losses (WAL, WCL), and the P_t training
schedule that transitions from amplitude to correlation objectives.
"""

import logging

import torch
from pytorch_wavelets import DTCWTForward

_logger = logging.getLogger(__name__)


def build_grid_mapping(
    coords_latlon: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a mapping from flat point indices to regular (H, W) grid cells.

    Args:
        coords_latlon: (N, 2) tensor with (latitude, longitude) in degrees.
            Latitude in [-90, 90], longitude in [0, 360) or [-180, 180).
        grid_h: Number of rows in the target regular grid.
        grid_w: Number of columns in the target regular grid.

    Returns:
        row_idx: (N,) long tensor -- row index for each point.
        col_idx: (N,) long tensor -- col index for each point.
        cell_counts: (H, W) tensor -- number of source points mapped to each cell
            (used for averaging when multiple points land in the same cell).
    """
    lat = coords_latlon[:, 0].float()
    lon = coords_latlon[:, 1].float()

    # Normalize longitude to [0, 360)
    lon = lon % 360.0

    row = torch.round((90.0 - lat) * (grid_h - 1) / 180.0).long().clamp(0, grid_h - 1)
    col = torch.round(lon * grid_w / 360.0).long() % grid_w

    cell_counts = torch.zeros(grid_h, grid_w, dtype=torch.float32, device=coords_latlon.device)
    cell_counts.index_put_((row, col), torch.ones_like(row, dtype=torch.float32), accumulate=True)

    return row, col, cell_counts


def flat_to_grid(
    values: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    cell_counts: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Scatter flat (N, C) values onto a regular (C, H, W) grid.

    Multiple points mapping to the same cell are averaged.  Empty cells are left
    as zero (no interpolation -- acceptable for spectral loss where the grid is
    densely covered by the O96/HEALPix source points).

    Args:
        values: (N, C) tensor of field values.
        row_idx, col_idx: index tensors from ``build_grid_mapping``.
        cell_counts: (H, W) counts tensor from ``build_grid_mapping``.
        grid_h, grid_w: target grid dimensions.

    Returns:
        grid: (C, H, W) tensor.
    """
    num_channels = values.shape[-1]
    device = values.device

    grid = torch.zeros(num_channels, grid_h, grid_w, dtype=torch.float32, device=device)

    # Scatter-add values into grid cells for each channel
    for c in range(num_channels):
        grid[c].index_put_((row_idx, col_idx), values[:, c].float(), accumulate=True)

    # Average cells with multiple contributing points
    safe_counts = cell_counts.clamp(min=1.0)
    grid = grid / safe_counts.unsqueeze(0)

    return grid


# ---------------------------------------------------------------------------
# Fourier-domain losses
# ---------------------------------------------------------------------------


def fourier_amplitude_loss(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> torch.Tensor:
    """FAL: Mean squared difference of 2-D Fourier magnitudes.

    Args:
        pred_grid: (C, H, W) predicted field on a regular grid (float32).
        target_grid: (C, H, W) target field on a regular grid (float32).

    Returns:
        Scalar FAL loss.
    """
    f_pred = torch.fft.fft2(pred_grid, dim=(-2, -1))
    f_target = torch.fft.fft2(target_grid, dim=(-2, -1))

    amp_pred = torch.abs(f_pred)
    amp_target = torch.abs(f_target)

    return torch.mean((amp_pred - amp_target) ** 2)


def fourier_correlation_loss(
    pred_grid: torch.Tensor,
    target_grid: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """FCL: 1 - normalised complex cross-correlation in Fourier domain.

    Args:
        pred_grid: (C, H, W) predicted field on a regular grid (float32).
        target_grid: (C, H, W) target field on a regular grid (float32).
        eps: Small constant for numerical stability.

    Returns:
        Scalar FCL loss.
    """
    f_pred = torch.fft.fft2(pred_grid, dim=(-2, -1))
    f_target = torch.fft.fft2(target_grid, dim=(-2, -1))

    numerator = torch.sum(torch.real(f_target * torch.conj(f_pred)))
    denom = (
        torch.sqrt(torch.sum(torch.abs(f_target) ** 2))
        * torch.sqrt(torch.sum(torch.abs(f_pred) ** 2))
        + eps
    )
    return 1.0 - numerator / denom


# ---------------------------------------------------------------------------
# Wavelet-domain losses
# ---------------------------------------------------------------------------


def dtcwt_decompose(
    grid: torch.Tensor,
    dtcwt_fwd: DTCWTForward,
) -> list[torch.Tensor]:
    """Run DTCWT forward transform and return high-frequency sub-bands.

    Args:
        grid: (C, H, W) tensor.
        dtcwt_fwd: Pre-initialised ``DTCWTForward`` module.

    Returns:
        List of length J (number of levels).  Each element is a complex tensor
        of shape (C, 6, H_l, W_l) where 6 is the number of directional bands.
    """
    # DTCWTForward expects (B, C, H, W) -- use C as the batch dimension
    x = grid.unsqueeze(0)  # (1, C, H, W)
    _, yh = dtcwt_fwd(x)  # yh: list[Tensor(1, C, 6, H_l, W_l, 2)]

    highpasses = []
    for h in yh:
        # h shape: (1, C, 6, H_l, W_l, 2) -- last dim is (real, imag)
        h = h.squeeze(0)  # (C, 6, H_l, W_l, 2)
        h_complex = torch.complex(h[..., 0], h[..., 1])  # (C, 6, H_l, W_l)
        highpasses.append(h_complex)

    return highpasses


def wavelet_amplitude_loss(h_pred: torch.Tensor, h_target: torch.Tensor) -> torch.Tensor:
    """WAL: Mean squared difference of wavelet coefficient magnitudes.

    Args:
        h_pred: Complex tensor of shape (C, 6, H_l, W_l) for one level.
        h_target: Complex tensor of the same shape.

    Returns:
        Tensor of shape (6,) -- WAL per directional band.
    """
    mag_pred = torch.abs(h_pred)
    mag_target = torch.abs(h_target)
    # Mean over (C, H_l, W_l), keep the 6-band dimension
    return torch.mean((mag_pred - mag_target) ** 2, dim=(0, 2, 3))


def wavelet_correlation_loss(
    h_pred: torch.Tensor,
    h_target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """WCL: 1 - normalised complex cross-correlation in wavelet domain.

    Args:
        h_pred: Complex tensor of shape (C, 6, H_l, W_l) for one level.
        h_target: Complex tensor of the same shape.
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (6,) -- WCL per directional band.
    """
    # Sum over (C, H_l, W_l), keep band dimension
    numerator = torch.sum(torch.real(h_target * torch.conj(h_pred)), dim=(0, 2, 3))
    denom = (
        torch.sqrt(torch.sum(torch.abs(h_target) ** 2, dim=(0, 2, 3)))
        * torch.sqrt(torch.sum(torch.abs(h_pred) ** 2, dim=(0, 2, 3)))
        + eps
    )
    return 1.0 - numerator / denom


# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------


def compute_pt(current_step: int, total_steps: int, alpha: float = 0.1) -> float:
    """Compute the P_t scheduling coefficient.

    P_t starts at 1 (correlation-dominated) and linearly decays to 0
    (amplitude-dominated) over the first (1 - alpha) fraction of training.

    Args:
        current_step: Current global training step.
        total_steps: Total number of training steps.
        alpha: Fraction of training reserved for pure amplitude loss at the end.

    Returns:
        P_t in [0, 1].
    """
    if total_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - current_step / ((1.0 - alpha) * total_steps))
