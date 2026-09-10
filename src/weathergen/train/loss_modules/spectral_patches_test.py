# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for patch recovery and the DTCWT filterbank."""

import pytest
import torch

from weathergen.train.loss_modules.spectral_patches import (
    DTCWTBands,
    check_patch_size,
    group_into_patches,
    points_to_patches,
)
from weathergen.train.loss_modules.spectral_utils import fal_fcl, wal_wcl

W = 1069  # CERRA raster width
S = 8  # small patch, keeps the tests fast


def _patch_indices(origins, size=S, width=W):
    """Flat native indices for complete patches at the given (row, col) origins."""
    out = []
    for r0, c0 in origins:
        rr = torch.arange(r0, r0 + size).repeat_interleave(size)
        cc = torch.arange(c0, c0 + size).repeat(size)
        out.append(rr * width + cc)
    return torch.cat(out)


def test_complete_patches_are_recovered():
    idx = _patch_indices([(0, 0), (40, 80), (200, 16)])
    gather, pids = group_into_patches(idx, W, S)
    assert gather.shape == (3, S, S)
    assert pids.numel() == 3
    # every point used exactly once
    assert torch.equal(gather.reshape(-1).sort().values, torch.arange(3 * S * S))


def test_patch_raster_order_is_preserved():
    """The lifted patch must reproduce the source raster exactly -- a transposed
    or rolled gather would still 'look fine' but scramble every orientation band."""
    r0, c0 = 24, 56
    idx = _patch_indices([(r0, c0)])
    gather, _ = group_into_patches(idx, W, S)
    # give each point a value encoding its native (row, col)
    rows, cols = torch.div(idx, W, rounding_mode="floor"), idx % W
    vals = (rows * 10000 + cols).float().unsqueeze(-1)
    patch = points_to_patches(vals, gather)[0, 0]
    expect = torch.tensor(
        [[(r0 + i) * 10000 + (c0 + jj) for jj in range(S)] for i in range(S)], dtype=torch.float32
    )
    assert torch.equal(patch, expect)


def test_shuffled_point_order_does_not_matter():
    idx = _patch_indices([(0, 0), (64, 64)])
    vals = idx.float().unsqueeze(-1)
    g_a, _ = group_into_patches(idx, W, S)
    perm = torch.randperm(idx.numel())
    g_b, _ = group_into_patches(idx[perm], W, S)
    assert torch.equal(points_to_patches(vals, g_a), points_to_patches(vals[perm], g_b))


def test_incomplete_patches_are_dropped_not_zero_filled():
    """A hole would inject exactly the spurious high-frequency energy the patch
    scheme exists to avoid, so a short patch must be discarded outright."""
    full = _patch_indices([(0, 0)])
    partial = _patch_indices([(40, 40)])[:-1]  # one point missing
    gather, pids = group_into_patches(torch.cat([full, partial]), W, S)
    assert gather.shape == (1, S, S)
    assert pids.numel() == 1


def test_no_complete_patches_returns_empty():
    gather, pids = group_into_patches(_patch_indices([(0, 0)])[:10], W, S)
    assert gather.shape == (0, S, S)
    assert pids.numel() == 0


def test_multiple_channels_are_lifted_independently():
    idx = _patch_indices([(0, 0), (32, 32)])
    gather, _ = group_into_patches(idx, W, S)
    vals = torch.stack([idx.float(), idx.float() * -3.0], dim=-1)
    out = points_to_patches(vals, gather)
    assert out.shape == (2, 2, S, S)
    assert torch.equal(out[:, 1], out[:, 0] * -3.0)


# --------------------------------------------------------------------------
# DTCWT filterbank
# --------------------------------------------------------------------------


def test_dtcwt_band_shapes_on_a_64_patch():
    yh = DTCWTBands(j=3)(torch.rand(12, 4, 64, 64))
    assert [tuple(b.shape) for b in yh] == [
        (12, 4, 6, 32, 32, 2),
        (12, 4, 6, 16, 16, 2),
        (12, 4, 6, 8, 8, 2),
    ]


def test_check_patch_size_rejects_silent_padding():
    check_patch_size(64, 3)
    with pytest.raises(ValueError, match="edge-replicate"):
        check_patch_size(100, 3)


def test_dtcwt_identity_gives_zero_loss():
    x = torch.rand(4, 2, 64, 64)
    bands = DTCWTBands(j=3)
    wal, wcl = wal_wcl(bands(x), bands(x))
    for a, c in zip(wal, wcl, strict=True):
        assert a.abs().max() < 1e-8
        assert c.abs().max() < 1e-5


def test_dtcwt_is_orientation_selective():
    """Horizontal and vertical stripes must excite different orientation bands.

    If they don't, o_dim is wrong and every directional claim is meaningless.
    """
    n = 64
    ramp = torch.arange(n, dtype=torch.float32)
    horiz = torch.sin(ramp * 0.8).view(1, 1, n, 1).expand(1, 1, n, n).contiguous()
    vert = torch.sin(ramp * 0.8).view(1, 1, 1, n).expand(1, 1, n, n).contiguous()
    bands = DTCWTBands(j=3)
    eh = bands(horiz)[0].pow(2).sum(dim=(-1, -2, -3))[0, 0]  # (6,)
    ev = bands(vert)[0].pow(2).sum(dim=(-1, -2, -3))[0, 0]
    assert eh.argmax() != ev.argmax(), f"no orientation selectivity: {eh.argmax()} vs {ev.argmax()}"


def test_blurring_lowers_fine_scale_amplitude():
    """The end-to-end sanity check: a blurred field must lose energy in the
    finest band, and WAL must notice."""
    torch.manual_seed(0)
    sharp = torch.rand(4, 1, 64, 64)
    k = torch.ones(1, 1, 5, 5) / 25.0
    blur = torch.nn.functional.conv2d(torch.nn.functional.pad(sharp, (2, 2, 2, 2), "reflect"), k)
    bands = DTCWTBands(j=3)
    yh_s, yh_b = bands(sharp), bands(blur)
    e_sharp = yh_s[0].pow(2).mean().item()
    e_blur = yh_b[0].pow(2).mean().item()
    assert e_blur < 0.5 * e_sharp, f"blur did not remove fine-scale energy ({e_blur} vs {e_sharp})"
    wal, _ = wal_wcl(yh_b, yh_s)
    assert wal[0].mean() > 0


def test_gradients_flow_from_loss_back_to_the_points():
    """Full chain: points -> patches -> DTCWT/FFT -> loss -> d(loss)/d(points)."""
    idx = _patch_indices([(0, 0), (64, 64)], size=64)
    gather, _ = group_into_patches(idx, W, 64)
    target_pts = torch.rand(idx.numel(), 2)
    pred_pts = torch.rand(idx.numel(), 2, requires_grad=True)
    bands = DTCWTBands(j=3)

    tp = points_to_patches(target_pts, gather)
    pp = points_to_patches(pred_pts, gather)
    fal, fcl = fal_fcl(pp, tp)
    wal, wcl = wal_wcl(bands(pp), bands(tp))
    (fal.mean() + fcl.mean() + sum(a.mean() for a in wal) + sum(c.mean() for c in wcl)).backward()

    assert pred_pts.grad is not None
    assert torch.isfinite(pred_pts.grad).all()
    assert pred_pts.grad.abs().sum() > 0


def test_sampler_output_round_trips_through_patch_recovery():
    """The contract between the two halves of the scheme.

    `ReaderData.subsample_patches` places patches; `group_into_patches` recovers
    them. If the sampler ever placed patches off the aligned lattice, every patch
    would fragment into four incomplete groups and be silently dropped -- the loss
    would just quietly see nothing. Assert the full budget survives.
    """
    import numpy as np

    from weathergen.datasets.data_reader_base import ReaderData

    h = w = 1069
    size, n_patches = 64, 12
    n = h * w
    rd = ReaderData(
        coords=np.zeros((n, 2), dtype=np.float32),
        geoinfos=np.zeros((n, 1), dtype=np.float32),
        # data[i, 0] = i, so we can prove which native points came back
        data=np.arange(n, dtype=np.float32)[:, None],
        datetimes=np.zeros((n,), dtype="datetime64[s]"),
    )
    rd = rd.subsample_patches(
        np.random.default_rng(0),
        grid_width=w,
        grid_height=h,
        patch_size=size,
        num_patches=n_patches,
    )
    assert rd.data.shape[0] == n_patches * size * size

    native_idx = torch.from_numpy(rd.data[:, 0].astype("int64"))
    gather, pids = group_into_patches(native_idx, w, size)
    assert gather.shape == (n_patches, size, size), "patches fragmented -> lattice misalignment"
    assert pids.numel() == n_patches

    # and the recovered raster really is the source raster
    patches = points_to_patches(native_idx.float().unsqueeze(-1), gather)[:, 0]
    rows = torch.div(patches, w, rounding_mode="floor")
    cols = patches % w
    assert torch.equal(rows.diff(dim=-2), torch.ones_like(rows.diff(dim=-2)))
    assert torch.equal(cols.diff(dim=-1), torch.ones_like(cols.diff(dim=-1)))


def test_sampler_handles_multi_step_time_major_output():
    """The reader concatenates one full raster PER FORECAST STEP
    (`data.transpose([0,2,1]).reshape((T*G, -1))` in data_reader_anemoi._get), so a
    2-step window arrives as 2*1069^2 points. This is what killed run p5991gew.

    The same patches must be taken in every step block, and each block must still
    recover as complete patches on its own.
    """
    import numpy as np

    from weathergen.datasets.data_reader_base import ReaderData

    h = w = 1069
    size, n_patches, n_steps = 64, 6, 2
    g = h * w
    n = g * n_steps
    # data[i,0] = i, so the step block is recoverable as i // g
    rd = ReaderData(
        coords=np.zeros((n, 2), dtype=np.float32),
        geoinfos=np.zeros((n, 1), dtype=np.float32),
        data=np.arange(n, dtype=np.float64)[:, None],
        datetimes=np.zeros((n,), dtype="datetime64[s]"),
    ).subsample_patches(
        np.random.default_rng(0),
        grid_width=w,
        grid_height=h,
        patch_size=size,
        num_patches=n_patches,
    )
    assert rd.data.shape[0] == n_steps * n_patches * size * size

    flat = rd.data[:, 0].astype("int64")
    step = flat // g
    within = flat % g
    # every step block carries the SAME patch geometry
    per_step = [np.sort(within[step == t]) for t in range(n_steps)]
    assert np.array_equal(per_step[0], per_step[1]), "patch geometry differs between steps"

    # and each block on its own recovers as complete patches
    for t in range(n_steps):
        gather, pids = group_into_patches(torch.from_numpy(per_step[t]), w, size)
        assert gather.shape == (n_patches, size, size)
        assert pids.numel() == n_patches


def test_sampler_rejects_a_partial_raster():
    """A count that is not a whole number of rasters means upstream filtering."""
    import numpy as np

    from weathergen.datasets.data_reader_base import ReaderData

    n = 1069 * 1069 + 17
    rd = ReaderData(
        coords=np.zeros((n, 2), dtype=np.float32),
        geoinfos=np.zeros((n, 1), dtype=np.float32),
        data=np.zeros((n, 1), dtype=np.float32),
        datetimes=np.zeros((n,), dtype="datetime64[s]"),
    )
    with pytest.raises(ValueError, match="rasters"):
        rd.subsample_patches(
            np.random.default_rng(0),
            grid_width=1069,
            grid_height=1069,
            patch_size=64,
            num_patches=4,
        )


def test_sampler_rejects_a_pre_filtered_raster():
    """Silently accepting a reordered point list would scramble every patch."""
    import numpy as np

    from weathergen.datasets.data_reader_base import ReaderData

    n = 1000
    rd = ReaderData(
        coords=np.zeros((n, 2), dtype=np.float32),
        geoinfos=np.zeros((n, 1), dtype=np.float32),
        data=np.zeros((n, 1), dtype=np.float32),
        datetimes=np.zeros((n,), dtype="datetime64[s]"),
    )
    with pytest.raises(ValueError, match="rasters"):
        rd.subsample_patches(
            np.random.default_rng(0),
            grid_width=1069,
            grid_height=1069,
            patch_size=64,
            num_patches=4,
        )


def test_all_dry_prediction_keeps_finite_gradients_through_the_full_chain():
    """Precipitation's real initial condition, end to end."""
    idx = _patch_indices([(0, 0)], size=64)
    gather, _ = group_into_patches(idx, W, 64)
    target_pts = torch.rand(idx.numel(), 1)
    pred_pts = torch.zeros(idx.numel(), 1, requires_grad=True)
    bands = DTCWTBands(j=3)

    tp = points_to_patches(target_pts, gather)
    pp = points_to_patches(pred_pts, gather)
    fal, fcl = fal_fcl(pp, tp)
    wal, wcl = wal_wcl(bands(pp), bands(tp))
    (fal.mean() + fcl.mean() + sum(a.mean() for a in wal) + sum(c.mean() for c in wcl)).backward()
    assert torch.isfinite(pred_pts.grad).all()


def test_repeated_snapshots_need_the_time_key():
    """`tokenize_spacetime` puts every snapshot of the target window in one array, so
    each grid cell appears once PER TIME. Measured on the real CERRA path: 56 patches
    touched, 8192 points each, 0 complete. Grouping on (time, cell) must recover
    one raster per snapshot instead."""
    n_times = 2
    idx_one = _patch_indices([(0, 0), (64, 64), (128, 128)], size=S)
    native = torch.cat([idx_one] * n_times)
    times = torch.cat([torch.full_like(idx_one, 1000 * t) for t in range(n_times)])

    # cell-only grouping sees 2*S*S points per patch and rejects every one
    gather_bad, _ = group_into_patches(native, W, S)
    assert gather_bad.shape[0] == 0, "the bug this test exists for"

    # (time, cell) grouping recovers one raster per snapshot per patch
    gather, gids = group_into_patches(native, W, S, group_key=times)
    assert gather.shape == (n_times * 3, S, S)
    assert gids.numel() == n_times * 3

    # and each raster really is a single snapshot of a single patch
    vals = points_to_patches(native.float().unsqueeze(-1), gather)[:, 0]
    tvals = points_to_patches(times.float().unsqueeze(-1), gather)[:, 0]
    for r in range(gather.shape[0]):
        assert tvals[r].unique().numel() == 1, "raster mixes timestamps"
        rows = torch.div(vals[r], W, rounding_mode="floor")
        cols = vals[r] % W
        assert torch.equal(rows.diff(dim=-2), torch.ones_like(rows.diff(dim=-2)))
        assert torch.equal(cols.diff(dim=-1), torch.ones_like(cols.diff(dim=-1)))


def test_group_key_composition_is_injective():
    """A stride taken from the observed pids rather than the possible ones would let
    a (time, patch) pair alias onto another and silently merge two rasters."""
    idx = _patch_indices([(0, 0), (1000, 1000)], size=S)  # far-apart pids
    native = torch.cat([idx, idx])
    times = torch.cat([torch.zeros_like(idx), torch.full_like(idx, 7)])
    gather, gids = group_into_patches(native, W, S, group_key=times)
    assert gather.shape[0] == 4
    assert gids.unique().numel() == 4
