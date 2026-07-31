# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for the decode-time 1-ring gather.

These import only ``weathergen.model.utils`` on purpose: ``weathergen.model.model`` pulls in
``flash_attn`` transitively, which is absent from the CPU environment the unit-test suite runs in.
"""

import torch

from weathergen.model.utils import neighbour_gather_idxs

NUM_CELLS = 48
DIM = 4


def _hp_nbours() -> torch.Tensor:
    """Stand-in 1-ring: self first, then 8 wrapped neighbours."""
    self_idx = torch.arange(NUM_CELLS).unsqueeze(1)
    offsets = torch.arange(1, 9).unsqueeze(0)
    return torch.cat([self_idx, (self_idx + offsets) % NUM_CELLS], dim=1)


def test_indices_stay_inside_each_samples_block():
    """Sample b must only ever index rows [b*C, (b+1)*C) of the flattened tensor."""
    batch_size = 3
    idxs = neighbour_gather_idxs(_hp_nbours(), batch_size, NUM_CELLS)

    assert idxs.shape == (batch_size * NUM_CELLS, 9)
    per_sample = idxs.reshape(batch_size, NUM_CELLS, 9)
    for b in range(batch_size):
        assert per_sample[b].min() >= b * NUM_CELLS
        assert per_sample[b].max() < (b + 1) * NUM_CELLS


def test_gather_reads_the_right_sample():
    """The bug this guards: without the offset every sample reads sample 0's latent."""
    batch_size = 4
    tokens = torch.zeros(batch_size, NUM_CELLS, DIM)
    for b in range(batch_size):
        tokens[b] = float(b)  # make each sample trivially identifiable

    idxs = neighbour_gather_idxs(_hp_nbours(), batch_size, NUM_CELLS)
    gathered = tokens.flatten(0, 1)[idxs.flatten()].reshape(batch_size, NUM_CELLS, 9, DIM)

    for b in range(batch_size):
        assert torch.all(gathered[b] == float(b)), f"sample {b} read another sample's latent"


def test_offset_is_the_only_difference_from_the_raw_ring():
    """Modulo the per-sample block offset, the ring itself must be unchanged."""
    batch_size = 3
    hp_nbours = _hp_nbours()
    idxs = neighbour_gather_idxs(hp_nbours, batch_size, NUM_CELLS)

    per_sample = idxs.reshape(batch_size, NUM_CELLS, 9)
    for b in range(batch_size):
        torch.testing.assert_close(per_sample[b] - b * NUM_CELLS, hp_nbours.long())


def test_batch_size_one_is_a_no_op():
    """Single-sample training is unaffected, which is why the bug went unnoticed."""
    hp_nbours = _hp_nbours()
    idxs = neighbour_gather_idxs(hp_nbours, 1, NUM_CELLS)
    torch.testing.assert_close(idxs, hp_nbours.long())
