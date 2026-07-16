# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests for the soft-blend decode replication (blend_replicate_targets).

The blend replaces the hard one-cell assignment of decode targets by up to k replicas in
the k nearest cells with continuous weights. The invariants tested here guarantee that the
model-side weighted scatter-add reconstructs exactly one prediction per original point:
weights per point sum to 1, every point is covered, replicas live in the own cell or one of
its HEALPix neighbours, and the host-cell-major grouping is consistent.
"""

import numpy as np
import torch

from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import blend_replicate_targets, hpy_cell_splits

HL = 3
NUM_CELLS = 12 * 4**HL


def _make_points(n=5000, seed=0):
    """Random points in cell-major (own cell) order, as the tokenizer produces them."""
    rng = np.random.default_rng(seed)
    lats = rng.uniform(-70.0, 70.0, n).astype(np.float32)
    lons = rng.uniform(-180.0, 180.0, n).astype(np.float32)
    coords = torch.from_numpy(np.stack([lats, lons], axis=-1))
    idxs_split, _, _ = hpy_cell_splits(coords, HL)
    order = torch.from_numpy(np.concatenate([i for i in idxs_split if len(i)]).astype(np.int64))
    counts = torch.tensor([len(i) for i in idxs_split], dtype=torch.int32)
    return coords[order], counts


def _blend_cfg(tok, k=3, tau=0.25, w_min=0.02):
    return {
        "ctrs": tok.hpy_ctrs_target,
        "nctrs": tok.hpy_nctrs_target,
        "nbr_ids": tok.hpy_nbr_ids_target,
        "cell_spacing": tok.hpy_cell_spacing_target,
        "k": k,
        "tau": tau,
        "w_min": w_min,
    }


def _run(n=5000, seed=0, **cfg_kwargs):
    tok = Tokenizer(HL)
    coords, counts = _make_points(n, seed)
    geoinfos = torch.randn(len(coords), 2)
    times_enc = torch.randn(len(coords), 6)
    out = blend_replicate_targets(
        coords, geoinfos, times_enc, counts, _blend_cfg(tok, **cfg_kwargs)
    )
    return tok, coords, counts, geoinfos, times_enc, out


def test_weights_partition_unity_and_cover_all_points():
    _, coords, _, _, _, (c_r, g_r, t_r, counts_r, idx, w) = _run()
    n = len(coords)
    per_point = torch.zeros(n).index_add_(0, idx, w)
    assert torch.allclose(per_point, torch.ones(n), atol=1e-6)
    assert idx.min() == 0 and idx.max() == n - 1
    assert len(torch.unique(idx)) == n
    assert int(counts_r.sum()) == len(idx) == len(c_r) == len(g_r) == len(t_r)
    assert len(idx) > n  # some boundary points were replicated


def test_reconstruction_identity():
    """Weighted scatter-add of replicated values must reproduce the original values."""
    _, coords, _, _, _, (_, _, _, _, idx, w) = _run(seed=1)
    v = torch.randn(len(coords), 4)
    out = torch.zeros_like(v).index_add_(0, idx, v[idx] * w.unsqueeze(-1))
    assert torch.allclose(out, v, atol=1e-5)


def test_replicated_arrays_match_original_rows():
    _, coords, _, geoinfos, times_enc, (c_r, g_r, t_r, _, idx, _) = _run(seed=2)
    assert torch.equal(c_r, coords[idx])
    assert torch.equal(g_r, geoinfos[idx])
    assert torch.equal(t_r, times_enc[idx])


def test_host_cells_are_own_or_neighbour():
    tok, coords, counts, _, _, (_, _, _, counts_r, idx, _) = _run(seed=3)
    own = torch.repeat_interleave(torch.arange(NUM_CELLS), counts.to(torch.int64))
    host = torch.repeat_interleave(torch.arange(NUM_CELLS), counts_r.to(torch.int64))
    own_of_replica = own[idx]
    allowed = torch.cat(
        [own_of_replica.unsqueeze(-1), tok.hpy_nbr_ids_target[own_of_replica]], dim=-1
    )
    assert (host.unsqueeze(-1) == allowed).any(-1).all()


def test_single_replica_points_have_weight_one():
    _, _, _, _, _, (_, _, _, _, idx, w) = _run(seed=4)
    counts_per_point = torch.bincount(idx)
    single = counts_per_point[idx] == 1
    assert single.any()
    assert torch.allclose(w[single], torch.ones(int(single.sum())), atol=1e-6)


def test_small_tau_collapses_to_single_assignment():
    """tau -> 0 makes the softmax one-hot on the nearest center: no effective blending
    (points lying exactly on a boundary keep an even split — a measure-zero tie)."""
    _, coords, _, _, _, (_, _, _, _, idx, w) = _run(seed=5, tau=1e-4)
    n = len(coords)
    assert len(idx) <= n * 1.005  # ties only
    counts_per_point = torch.bincount(idx)
    single = counts_per_point[idx] == 1
    assert single.float().mean() > 0.995
    assert torch.allclose(w[single], torch.ones(int(single.sum())), atol=1e-6)
