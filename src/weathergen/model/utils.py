# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re
import warnings

import astropy_healpix as hp
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_context_map(healpix_level: int, level: int) -> torch.Tensor:
    """Map every cell at ``healpix_level`` to the 1-ring of its ancestor at ``level``.

    Used by the multi-scale decode context (see ``engines.MultiScaleContextDecoder``). HEALPix
    nested ordering gives ``ancestor(c) = c // 4**(healpix_level - level)``. Slot 0 of the ring is
    the ancestor itself, slots 1..8 its neighbours in the fixed order ``hp.neighbours`` returns;
    missing neighbours (the polar corners) are filled with the cell itself, matching how
    ``ModelParams`` builds the fine-level table.

    Returns:
        ``[12 * 4**healpix_level, 9]`` int32 of coarse-cell indices.
    """
    assert 0 <= level < healpix_level, (
        f"context level {level} must be in [0, healpix_level={healpix_level}): coarser than the"
        " latent, and not the latent itself."
    )
    num_cells_coarse = 12 * 4**level
    with warnings.catch_warnings(action="ignore"):
        nbrs = hp.neighbours(np.arange(num_cells_coarse), 2**level, order="nested").transpose()
    for i, row in enumerate(nbrs):
        nbrs[i][row == -1] = i
    ring = np.concatenate([np.arange(num_cells_coarse, dtype=nbrs.dtype)[:, None], nbrs], axis=1)

    ancestors = np.arange(12 * 4**healpix_level) // (4 ** (healpix_level - level))
    return torch.from_numpy(ring[ancestors].astype(np.int32))


def pool_latent_levels(
    tokens_bcd: torch.Tensor, healpix_level: int, levels: list[int]
) -> list[torch.Tensor]:
    """Mean-pool a latent onto each coarse HEALPix level.

    Nested ordering makes a coarse cell's descendants contiguous in the fine index space, so this
    is a reshape-mean -- no scatter and no index tensor.

    Args:
        tokens_bcd : ``[batch, 12 * 4**healpix_level, dim]`` latent at the source level.
        levels     : coarse levels, each ``< healpix_level``.

    Returns:
        One ``[batch, 12 * 4**level, dim]`` tensor per level, in the given order.
    """
    pooled = []
    for level in levels:
        ratio = 4 ** (healpix_level - level)
        b, c, d = tokens_bcd.shape
        pooled.append(tokens_bcd.reshape(b, c // ratio, ratio, d).mean(2))
    return pooled


def gather_context_latent(
    pooled: list[torch.Tensor],
    ctx_maps: list[torch.Tensor],
    target_lens: torch.Tensor,
    batch_size: int,
    num_cells: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build the coarse-context KV for the cells that have targets this step.

    For every active cell, concatenates the 1-ring of its ancestor at each coarse level -- 9
    pooled tokens per level. Restricting to active cells matters: for a regional stream only a few
    hundred of the 12,288 L5 cells carry targets, so a full-grid tensor would be ~900 MB of which
    almost all is unused (same reasoning as ``LatentUpsamplingEngine``). Inactive cells get
    zero-length KV, pairing with their zero-length query group in the varlen attention.

    Args:
        pooled      : per-level ``[batch, num_cells_level, dim]`` from ``pool_latent_levels``.
        ctx_maps    : per-level ``[num_cells, 9]`` from ``build_context_map``, same order.
        target_lens : ``[batch * num_cells + 1]`` per-cell target counts (``tcs_lens``).

    Returns:
        ``(ctx, ctx_lens)`` with ctx ``[num_active * 9 * num_levels, dim]`` grouped per cell and
        levels concatenated, or ``(None, None)`` when no cell is active.
    """
    idx_active = (target_lens[1:] > 0).nonzero(as_tuple=True)[0]
    if idx_active.numel() == 0:
        return None, None

    # active indices run over the flattened (sample, cell) axis
    b_idx = torch.div(idx_active, num_cells, rounding_mode="floor")
    c_idx = idx_active % num_cells

    parts = []
    for pooled_level, ctx_map in zip(pooled, ctx_maps, strict=True):
        ring = ctx_map[c_idx].long()  # [num_active, 9]
        parts.append(pooled_level[b_idx.unsqueeze(1), ring])  # [num_active, 9, dim]
    ctx = torch.cat(parts, dim=1).flatten(0, 1)

    ctx_lens = torch.zeros(
        (batch_size * num_cells + 1,), dtype=torch.int32, device=target_lens.device
    )
    ctx_lens[1:][idx_active] = 9 * len(pooled)

    return ctx, ctx_lens


def get_num_parameters(block):
    nps = filter(lambda p: p.requires_grad, block.parameters())
    return sum([torch.prod(torch.tensor(p.size())) for p in nps])


def freeze_weights(block):
    if hasattr(block, "name"):
        logger.info(f"Freeze block {block.name}")
    for p in block.parameters():
        p.requires_grad = False


def set_to_eval(block):
    if hasattr(block, "name"):
        logger.info(f"Set block {block.name} to eval mode")
    block.eval()


def apply_fct_to_blocks(model, blocks, fct):
    """
    Apply a function to specific blocks of a model.
    Args:
        model : model instance with attribute named_modules
        blocks : regex pattern to match block names
        fct : function to apply to matching blocks
    """

    for name, module in model.named_modules():
        name = module.name if hasattr(module, "name") else name
        # avoid the whole model element which has name ''
        if (re.fullmatch(blocks, name) is not None) and (name != ""):
            fct(module)


class DrizzleGate(nn.Module):
    """
    Learnable gating activation that suppresses light drizzle in precipitation predictions.

    Combines a soft-thresholding mechanism with Softplus to:
    1. Push near-zero values toward zero (no-rain regions stay dry)
    2. Preserve moderate and heavy precipitation magnitudes
    3. Ensure non-negative output

    The gate learns a threshold and sharpness during training:
        output = softplus(x) * sigmoid((x - threshold) / sharpness)

    When x << threshold: sigmoid ≈ 0, output ≈ 0 (drizzle suppressed)
    When x >> threshold: sigmoid ≈ 1, output ≈ softplus(x) (precipitation preserved)
    """

    def __init__(self, init_threshold: float = 0.1, init_sharpness: float = 0.05):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        self.sharpness = nn.Parameter(torch.tensor(init_sharpness))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp sharpness to avoid division by zero or negative values
        sharpness = self.sharpness.clamp(min=1e-4)
        gate = torch.sigmoid((x - self.threshold) / sharpness)
        return torch.nn.functional.softplus(x) * gate


class ActivationFactory:
    _registry = {
        "identity": nn.Identity,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "prelu": nn.PReLU,
        "softplus": nn.Softplus,
        "linear": nn.Linear,
        "logsoftmax": nn.LogSoftmax,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "drizzlegate": DrizzleGate,
    }

    @classmethod
    def get(cls, name: str, **kwargs):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unsupported activation type: '{name}'")
        fn = cls._registry[name]
        return fn(**kwargs) if callable(fn) else fn
