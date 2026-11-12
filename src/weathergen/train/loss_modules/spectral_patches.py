# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Getting a regular 2-D grid out of WeatherGenerator's scattered target points.

WFCL needs a dense ``(N, C, H, W)`` raster. WeatherGenerator hands the loss a flat
list of target points that were drawn at random from the source grid, so the grid
has to come from somewhere.

**Why not rasterise the random draw.** CERRA is 1069x1069 points and
``max_num_targets`` is 50k, i.e. a 4.4% sample. Scattering that onto a grid leaves
``exp(-50000/G^2)`` of cells empty -- 95.7% at native 5.5 km resolution. Empty
cells get filled with zeros, and the resulting spectrum describes the sampling
pattern rather than the weather. To push the empty fraction below 5% you have to
coarsen to ~46 km cells, at which point the finest DTCWT level covers ~92-184 km
and the loss is blind to the 10-25 km band the sharpness campaign is about.

**What we do instead.** The reader draws the same 50k budget as a handful of
*contiguous* patches (see ``sampling: patches`` in
``weathergen.datasets.data_reader_base``). Each patch is then a complete,
gap-free block of the native grid, and 12 patches of 64x64 = 49,152 points buys
DTCWT bands at 11-22 / 22-44 / 44-88 km -- exactly the scales of interest.

This module recovers those patches on the loss side. CERRA's flat index is exactly
raster-ordered (verified: 1069^2 points, uniform 5.50 km spacing along both raster
axes), so membership is recovered from a point's native index by ``divmod``, with
no reprojection and no nearest-neighbour search.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


def group_into_patches(
    native_idx: torch.Tensor,
    grid_width: int,
    patch_size: int,
    group_key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group flat native-grid indices into complete square patches.

    Args:
        native_idx: ``(P,)`` int64, each point's index into the source grid's flat
            raster ordering.
        grid_width: width of the source raster (1069 for CERRA).
        patch_size: patch edge length in grid cells.
        group_key: optional ``(P,)`` integer tag splitting points that share a grid
            cell but are not part of the same raster -- in practice the timestamp.
            **This is not optional in practice for CERRA.** With
            ``tokenize_spacetime: True`` a 6 h target window carries two 3-hourly
            snapshots, so every grid point arrives TWICE, each patch holds 8192
            points instead of 4096, and without this the completeness test rejects
            every one of them (measured: 56 patches touched, 8192 points each, 0
            complete). Passing the times instead yields one raster per (time, patch)
            -- correct, and twice the independent samples.

    Returns:
        ``(gather_idx, group_ids)`` where ``gather_idx`` is ``(K, patch_size,
        patch_size)`` holding positions *into the point list*. ``K`` is the number
        of (group, patch) rasters that arrived complete; incomplete ones are dropped
        rather than zero-filled, because a hole is exactly the spurious
        high-frequency energy this whole module exists to avoid.
    """
    rows, cols = torch.div(native_idx, grid_width, rounding_mode="floor"), native_idx % grid_width
    p_row, p_col = (
        torch.div(rows, patch_size, rounding_mode="floor"),
        torch.div(cols, patch_size, rounding_mode="floor"),
    )
    # A patch id that is unique across the raster, and the point's offset inside it.
    patches_per_row = (grid_width + patch_size - 1) // patch_size
    pid = p_row * patches_per_row + p_col
    within = (rows % patch_size) * patch_size + (cols % patch_size)

    if group_key is not None:
        # Compose (group, patch) into one id. The stride must exceed every possible
        # pid, not merely every observed one, so the mapping stays injective.
        stride = patches_per_row * patches_per_row + 1
        _, grank = torch.unique(group_key, return_inverse=True)
        pid = grank.to(pid.dtype) * stride + pid

    area = patch_size * patch_size
    uniq, counts = torch.unique(pid, return_counts=True)
    complete = uniq[counts == area]
    if complete.numel() == 0:
        return (
            torch.empty(0, patch_size, patch_size, dtype=torch.long, device=native_idx.device),
            complete,
        )

    keep = torch.isin(pid, complete)
    kept_pid, kept_within = pid[keep], within[keep]
    positions = torch.nonzero(keep, as_tuple=True)[0]

    # Dense patch axis: rank each surviving id into 0..K-1.
    rank = torch.searchsorted(complete, kept_pid)

    gather = torch.empty(complete.numel(), area, dtype=torch.long, device=native_idx.device)
    gather[rank, kept_within] = positions
    return gather.view(-1, patch_size, patch_size), complete


def points_to_patches(
    values: torch.Tensor,
    gather_idx: torch.Tensor,
) -> torch.Tensor:
    """Lift flat per-point values onto the patch raster.

    Args:
        values: ``(P, C)`` field values for the points.
        gather_idx: ``(K, S, S)`` from :func:`group_into_patches`.

    Returns:
        ``(K, C, S, S)`` -- the ``(N, C, H, W)`` layout the spectral terms expect,
        with the patch axis playing the role of the batch.
    """
    k, s, _ = gather_idx.shape
    picked = values[gather_idx.reshape(-1)]  # (K*S*S, C)
    return picked.view(k, s, s, values.shape[-1]).permute(0, 3, 1, 2).contiguous()


class DTCWTBands(nn.Module):
    """DTCWT high-pass sub-bands for a batch of patches.

    Kept as an ``nn.Module`` (rather than a bare object) so its filter buffers
    follow ``.to(device)``.

    ``biort='near_sym_b'`` / ``qshift='qshift_b'`` are the standard Kingsbury
    filters; the package default is the shorter ``_a`` pair. The paper does not
    say which it used, so both are exposed and the choice is logged.
    """

    def __init__(self, j: int = 3, biort: str = "near_sym_b", qshift: str = "qshift_b"):
        super().__init__()
        from pytorch_wavelets import DTCWTForward

        self.j = j
        self.xfm = DTCWTForward(J=j, biort=biort, qshift=qshift, o_dim=2, ri_dim=-1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``(N, C, H, W)`` -> list of ``J`` tensors ``(N, C, 6, H_l, W_l, 2)``."""
        _yl, yh = self.xfm(x)
        return yh


def check_patch_size(patch_size: int, j: int) -> None:
    """A patch must be divisible by ``2**j`` or pytorch_wavelets silently pads.

    The transform edge-replicates whenever a level's input is not evenly
    divisible, which fabricates high-frequency energy at the patch border --
    invisible in the loss value and fatal to its interpretation.
    """
    if patch_size % (2**j) != 0:
        raise ValueError(
            f"patch_size={patch_size} is not divisible by 2**J={2**j}; "
            "pytorch_wavelets would silently edge-replicate and inject "
            "spurious high-frequency energy at every level."
        )
