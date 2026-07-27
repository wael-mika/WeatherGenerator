# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the coarse-scale decode context (decoder_type: MultiScaleContext).

These cover the index/pooling arithmetic behind ``engines.MultiScaleContextDecoder``. They import
only ``weathergen.model.utils`` on purpose: ``weathergen.model.model`` pulls in ``flash_attn``
transitively, which is absent from the CPU environment the unit-test suite runs in. The attention
itself is exercised by the GPU integration tests.
"""

import astropy_healpix as hp
import numpy as np
import pytest
import torch

from weathergen.model.utils import (
    build_context_map,
    gather_context_latent,
    pool_latent_levels,
)

HEALPIX_LEVEL = 3
NUM_CELLS = 12 * 4**HEALPIX_LEVEL
DIM = 8


def test_context_map_matches_ancestor_one_ring():
    """ctx_map[c] must be the 1-ring (self first) of c's ancestor at the coarse level."""
    level = 2
    ctx_map = build_context_map(HEALPIX_LEVEL, level).numpy()

    assert ctx_map.shape == (NUM_CELLS, 9)
    assert ctx_map.dtype == np.int32

    ratio = 4 ** (HEALPIX_LEVEL - level)
    num_coarse = 12 * 4**level
    nbrs = hp.neighbours(np.arange(num_coarse), 2**level, order="nested").transpose()

    rng = np.random.default_rng(0)
    for c in rng.choice(NUM_CELLS, size=32, replace=False):
        ancestor = c // ratio
        assert ctx_map[c][0] == ancestor, "slot 0 must be the ancestor cell itself"
        # slots 1..8 are the ancestor's neighbours, with -1 (polar corners) filled by self
        expected = set(nbrs[ancestor][nbrs[ancestor] != -1].tolist()) | {ancestor}
        assert set(ctx_map[c][1:].tolist()) <= expected
        assert (ctx_map[c] < num_coarse).all() and (ctx_map[c] >= 0).all()


def test_context_map_is_constant_within_a_coarse_cell():
    """All fine cells sharing an ancestor must get the same context ring."""
    level = 1
    ratio = 4 ** (HEALPIX_LEVEL - level)
    ctx_map = build_context_map(HEALPIX_LEVEL, level)

    for coarse in (0, 3, 12 * 4**level - 1):
        block = ctx_map[coarse * ratio : (coarse + 1) * ratio]
        assert torch.all(block == block[0]), "ring must depend only on the ancestor"


def test_context_levels_must_be_coarser_than_the_latent():
    for bad in (HEALPIX_LEVEL, HEALPIX_LEVEL + 1, -1):
        with pytest.raises(AssertionError, match="coarser than the"):
            build_context_map(HEALPIX_LEVEL, bad)


def test_pooling_equals_mean_of_nested_children():
    """Nested ordering makes descendants contiguous, so pooling must be a plain child mean."""
    levels = [2, 1]
    tokens = torch.randn(2, NUM_CELLS, DIM)
    pooled = pool_latent_levels(tokens, HEALPIX_LEVEL, levels)

    assert len(pooled) == len(levels)
    for level, p in zip(levels, pooled, strict=True):
        ratio = 4 ** (HEALPIX_LEVEL - level)
        assert p.shape == (2, 12 * 4**level, DIM)
        for j in (0, 5, p.shape[1] - 1):
            expected = tokens[:, j * ratio : (j + 1) * ratio].mean(1)
            torch.testing.assert_close(p[:, j], expected)


def test_pooling_keeps_samples_separate():
    tokens = torch.zeros(2, NUM_CELLS, DIM)
    tokens[1] = 1.0
    pooled = pool_latent_levels(tokens, HEALPIX_LEVEL, [1])[0]
    assert torch.all(pooled[0] == 0.0) and torch.all(pooled[1] == 1.0)


def test_gather_context_restricted_to_active_cells():
    """Only cells with targets get KV; lens and row count must agree."""
    levels = [2, 1]
    batch_size = 2
    tokens = torch.randn(batch_size, NUM_CELLS, DIM)
    pooled = pool_latent_levels(tokens, HEALPIX_LEVEL, levels)
    ctx_maps = [build_context_map(HEALPIX_LEVEL, level) for level in levels]

    target_lens = torch.zeros(batch_size * NUM_CELLS + 1, dtype=torch.int32)
    active = torch.tensor([3, 7, NUM_CELLS + 11])  # two cells in sample 0, one in sample 1
    target_lens[1:][active] = torch.tensor([5, 2, 9], dtype=torch.int32)

    ctx, ctx_lens = gather_context_latent(pooled, ctx_maps, target_lens, batch_size, NUM_CELLS)

    tokens_per_cell = 9 * len(levels)
    assert ctx.shape == (len(active) * tokens_per_cell, DIM)
    assert ctx_lens.shape == (batch_size * NUM_CELLS + 1,)
    assert ctx_lens[0] == 0
    assert int(ctx_lens.sum()) == ctx.shape[0], "lens must account for every KV row"
    assert (ctx_lens[1:][active] == tokens_per_cell).all()
    mask = torch.ones(batch_size * NUM_CELLS, dtype=torch.bool)
    mask[active] = False
    assert (ctx_lens[1:][mask] == 0).all(), "inactive cells must contribute no KV"


def test_gather_context_rows_are_grouped_per_cell_levels_concatenated():
    """Row layout must be [cell0 level0 ring, cell0 level1 ring, cell1 ...] to match ctx_lens."""
    levels = [2, 1]
    tokens = torch.randn(1, NUM_CELLS, DIM)
    pooled = pool_latent_levels(tokens, HEALPIX_LEVEL, levels)
    ctx_maps = [build_context_map(HEALPIX_LEVEL, level) for level in levels]

    target_lens = torch.zeros(NUM_CELLS + 1, dtype=torch.int32)
    cells = [4, 40]
    target_lens[1:][torch.tensor(cells)] = 1

    ctx, _ = gather_context_latent(pooled, ctx_maps, target_lens, 1, NUM_CELLS)
    ctx = ctx.reshape(len(cells), 9 * len(levels), DIM)

    for i_cell, cell in enumerate(cells):
        for i_level in range(len(levels)):
            block = ctx[i_cell, i_level * 9 : (i_level + 1) * 9]
            expected = pooled[i_level][0][ctx_maps[i_level][cell].long()]
            torch.testing.assert_close(block, expected)


def test_gather_context_reads_the_right_sample():
    """Regression for the batch offset: sample 1's context must come from sample 1's latent."""
    levels = [1]
    batch_size = 2
    tokens = torch.zeros(batch_size, NUM_CELLS, DIM)
    tokens[1] = 1.0  # make the two samples trivially distinguishable
    pooled = pool_latent_levels(tokens, HEALPIX_LEVEL, levels)
    ctx_maps = [build_context_map(HEALPIX_LEVEL, level) for level in levels]

    target_lens = torch.zeros(batch_size * NUM_CELLS + 1, dtype=torch.int32)
    target_lens[1:][NUM_CELLS + 11] = 4  # active only in sample 1

    ctx, _ = gather_context_latent(pooled, ctx_maps, target_lens, batch_size, NUM_CELLS)
    assert torch.all(ctx == 1.0), "context was gathered from the wrong sample"


def test_empty_when_no_cell_is_active():
    pooled = pool_latent_levels(torch.randn(1, NUM_CELLS, DIM), HEALPIX_LEVEL, [2])
    ctx_maps = [build_context_map(HEALPIX_LEVEL, 2)]
    target_lens = torch.zeros(NUM_CELLS + 1, dtype=torch.int32)

    ctx, ctx_lens = gather_context_latent(pooled, ctx_maps, target_lens, 1, NUM_CELLS)
    assert ctx is None and ctx_lens is None
