# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the residual flow-matching algebra.

These import only ``weathergen.model.flow_math`` and the loss functions on purpose:
``weathergen.model.model`` and ``engines`` pull in ``flash_attn``, which is absent from the CPU
environment the unit-test suite runs in.
"""

import pytest
import torch

from weathergen.model.flow_math import (
    EARTH_RADIUS_KM,
    ResidualScale,
    correlated_source,
    dominant_replica,
    physical_to_residual,
    residual_to_physical,
    sample_flow_time_for_points,
)
from weathergen.train.loss_modules.loss_functions import mse, mse_det, mse_flow

N = 64
C = 3


def _pack(mu, y0, v, r_scale):
    """The decoder's training return: [mu, mu.detach() + r_scale*(y0+v), r_scale]."""
    return torch.stack([mu, residual_to_physical(mu, y0, v, r_scale), r_scale.expand_as(mu)], 0)


def test_packed_mse_is_the_cfm_objective_weighted_equally_per_channel():
    """``mse_flow`` must score the CFM objective with the SAME weight on every channel.

    Member 1 is ``mu + r_scale*(y0+v)``, so a plain mse against the target gives
    ``r_scale^2 * ||(r - y0) - v||^2`` -- the CFM objective, but with each channel's gradient
    scaled by ``r_scale_c^2``. Since ``r_scale`` IS the residual std, that starves exactly the
    channels the base already predicts well: on the level-5 arms it gave an 80x spread in
    effective learning weight and 20-75x over-dispersion on the small-residual channels, while
    ``tp`` (the largest ``r_scale``) trained fine and hid it.

    ``mse_flow`` therefore divides member 1 and the target by ``r_scale`` (member 2) and
    multiplies back by the rms ``r_scale``, so the per-channel weight ``r_scale_c^2`` becomes the
    CONSTANT ``mean(r_scale^2)`` and the overall loss magnitude is unchanged.

    If anyone reintroduces a clamp on the residual, drops the detach, or drops member 2, this
    fails.
    """
    torch.manual_seed(0)
    mu = torch.randn(N, C)
    y0 = torch.randn(N, C)
    v = torch.randn(N, C)
    y1 = torch.randn(N, C)
    # per-channel scale, the configuration that actually ships; deliberately a wide spread
    r_scale = torch.tensor([0.851, 0.348, 0.563])

    packed = _pack(mu, y0, v, r_scale)
    got, per_channel_got = mse_flow(y1, packed, None, None)

    r = physical_to_residual(y1, mu, r_scale)
    cfm = ((v - (r - y0)) ** 2).mean(0)  # per channel, UNWEIGHTED
    want_per_channel = cfm * r_scale.pow(2).mean()  # one constant weight for every channel

    assert torch.allclose(per_channel_got, want_per_channel, atol=1e-5), (
        f"{per_channel_got} != {want_per_channel}"
    )
    assert torch.allclose(got, want_per_channel.mean(), atol=1e-6)


def test_flow_loss_no_longer_favours_large_residual_channels():
    """The regression that cost the level-5 arms: equal CFM error must cost the same everywhere.

    Two channels with an 8x difference in ``r_scale`` but an identical velocity error must
    contribute identically. Under the old ``r_scale^2`` weighting they differed by 64x.
    """
    n = 4096
    r_scale = torch.tensor([0.80, 0.10])
    mu = torch.zeros(n, 2)
    y0 = torch.zeros(n, 2)
    v = torch.zeros(n, 2)
    # same CFM error (r - y0 - v = 1) in both channels
    y1 = r_scale.expand(n, 2) * 1.0

    _, per_channel = mse_flow(y1, _pack(mu, y0, v, r_scale), None, None)
    assert per_channel[0].item() == pytest.approx(per_channel[1].item(), rel=1e-5), (
        f"channels weighted unequally: {per_channel.tolist()}"
    )


def test_mse_det_and_flow_slice_the_right_member():
    """Guards against lp_loss's pred.mean(0) creeping back in and averaging the pack."""
    torch.manual_seed(1)
    y1 = torch.randn(N, C)
    other = y1 + 3.0

    ones = torch.ones_like(y1)
    exact_mu = torch.stack([y1, other, ones], 0)
    det, _ = mse_det(y1, exact_mu, None, None)
    flow, _ = mse_flow(y1, exact_mu, None, None)
    assert det.item() == pytest.approx(0.0, abs=1e-12)
    assert flow.item() == pytest.approx(9.0, rel=1e-6)

    exact_flow = torch.stack([other, y1, ones], 0)
    det, _ = mse_det(y1, exact_flow, None, None)
    flow, _ = mse_flow(y1, exact_flow, None, None)
    assert det.item() == pytest.approx(9.0, rel=1e-6)
    assert flow.item() == pytest.approx(0.0, abs=1e-12)

    # plain mse on the pack averages ALL THREE members -- exactly what must never be configured.
    # Assert against the explicit mean rather than a literal, so the check keeps its meaning if
    # the pack ever grows another slot.
    both, _ = mse(y1, exact_flow, None, None)
    nonsense = ((y1 - exact_flow.mean(0)) ** 2).mean()
    assert both.item() == pytest.approx(nonsense.item(), rel=1e-6)
    assert both.item() > 1.0, "averaging the pack must not accidentally look like a good loss"


def test_pack_functions_reject_a_non_pack():
    """A misconfigured `mse_det`/`mse_flow` must fail loudly, not train on garbage."""
    y1 = torch.randn(N, C)
    for fn in (mse_det, mse_flow):
        with pytest.raises(AssertionError, match="training pack"):
            fn(y1, torch.randn(1, N, C), None, None)
        with pytest.raises(AssertionError, match="training pack"):
            fn(y1, torch.randn(2, N, C), None, None)


def test_nan_targets_stay_finite():
    """Masked/spoofed targets are NaN; they must not poison the loss or the gradients."""
    torch.manual_seed(2)
    mu = torch.randn(N, C)
    y0 = torch.randn(N, C)
    v = torch.randn(N, C, requires_grad=True)
    r_scale = torch.full((C,), 0.5)

    y1 = torch.randn(N, C)
    y1[::3, 0] = float("nan")
    y1[1, :] = float("nan")

    packed = _pack(mu, y0, v, r_scale)
    loss, per_ch = mse_flow(y1, packed, None, None)
    assert torch.isfinite(loss)
    assert torch.isfinite(per_ch).all()

    loss.backward()
    assert torch.isfinite(v.grad).all()


def test_residual_substitution_keeps_the_path_finite():
    """The decoder substitutes y0 where the residual is NaN, giving those points zero velocity."""
    mu = torch.zeros(4, 2)
    y1 = torch.tensor([[1.0, float("nan")], [float("nan"), 2.0], [3.0, 4.0], [5.0, 6.0]])
    r_scale = torch.ones(2)

    r = physical_to_residual(y1, mu, r_scale)
    finite = torch.isfinite(r)
    y0 = torch.full_like(r, 7.0)
    r = torch.where(finite, r, y0)

    assert torch.isfinite(r).all()
    # substituted points sit exactly on y0, so the target velocity r - y0 is zero there
    assert (r - y0)[~finite].abs().max().item() == 0.0


def test_residual_scale_recovers_per_channel_std():
    """The EMA buffer must converge on the per-channel residual std, not a global average."""
    torch.manual_seed(3)
    stds = torch.tensor([0.851, 0.348, 0.563])
    rs = ResidualScale(C, ema=0.9, freeze_after=10_000)

    for _ in range(400):
        resid = torch.randn(4096, C) * stds
        rs.update(resid, torch.ones_like(resid))

    assert torch.allclose(rs.value(), stds, rtol=0.05), f"{rs.value()} vs {stds}"

    rs.reset_parameters()
    assert torch.allclose(rs.value(), torch.ones(C))
    assert int(rs.count) == 0


def test_residual_scale_freezes_and_honours_fixed():
    torch.manual_seed(4)
    rs = ResidualScale(C, ema=0.5, freeze_after=3)
    for _ in range(3):
        rs.update(torch.randn(256, C) * 5.0, torch.ones(256, C))
    frozen = rs.value().clone()
    for _ in range(50):
        rs.update(torch.randn(256, C) * 0.01, torch.ones(256, C))
    assert torch.equal(rs.value(), frozen), "scale kept moving after freeze_after"

    pinned = ResidualScale(C, fixed=0.25)
    pinned.update(torch.randn(256, C) * 9.0, torch.ones(256, C))
    assert torch.allclose(pinned.value(), torch.full((C,), 0.25))


def test_residual_scale_ignores_non_finite_points():
    """Channels with no finite points this step must keep their previous scale."""
    rs = ResidualScale(2, ema=0.0, freeze_after=100)
    resid = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    rs.update(resid, mask)
    assert rs.value()[0].item() == pytest.approx(2.0, rel=1e-5)
    assert rs.value()[1].item() == pytest.approx(1.0, rel=1e-5)


def test_time_per_cell_is_constant_within_group():
    """flow_time_per_cell must give every point of a decode cell the same t."""
    torch.manual_seed(5)
    counts = torch.tensor([3, 5, 1, 4], dtype=torch.int32)
    lens = torch.cat([torch.zeros(1, dtype=torch.int32), counts])
    n = int(counts.sum())

    t = sample_flow_time_for_points(n, lens, torch.device("cpu"), torch.float32, per_cell=True)
    assert t.shape == (n, 1)

    off = 0
    seen = []
    for c in counts.tolist():
        group = t[off : off + c]
        assert torch.allclose(group, group[0].expand_as(group))
        seen.append(group[0].item())
        off += c
    # cells stay independent, so the draws differ
    assert len(set(seen)) > 1

    # per-point mode gives (almost surely) distinct times
    t_pt = sample_flow_time_for_points(n, lens, torch.device("cpu"), torch.float32, per_cell=False)
    assert t_pt.unique().numel() == n


def test_time_per_cell_asserts_on_length_mismatch():
    counts = torch.tensor([3, 5], dtype=torch.int32)
    lens = torch.cat([torch.zeros(1, dtype=torch.int32), counts])
    with pytest.raises(AssertionError, match="decode-group counts"):
        sample_flow_time_for_points(99, lens, torch.device("cpu"), torch.float32, per_cell=True)


def test_dominant_replica_picks_the_heaviest_row_per_point():
    """The selector soft blend needs when it must NOT average.

    Three original points; point 1 is replicated into three host cells, point 2 into two, point 0
    stays single. The winner is the largest weight in each group, whatever order the rows arrive
    in -- `blend_replicate_targets` sorts its output by host cell, so the rows of one point are
    NOT contiguous.
    """
    idx = torch.tensor([1, 0, 2, 1, 2, 1])
    w = torch.tensor([0.2, 1.0, 0.4, 0.7, 0.6, 0.1])
    pick = dominant_replica(idx, w, n_state=3)

    assert pick.tolist() == [1, 3, 4]
    assert torch.equal(w[pick], torch.tensor([1.0, 0.7, 0.6]))


def test_dominant_replica_is_the_identity_when_nothing_replicates():
    """With k=1 every point is its own single host, so selection must change nothing."""
    n = 7
    idx = torch.arange(n)
    w = torch.ones(n)
    assert torch.equal(dominant_replica(idx, w, n_state=n), torch.arange(n))


def test_dominant_replica_selects_rather_than_averages():
    """The whole point: the returned velocity is one replica's, never a blend of several.

    Averaging is what costs 31-69% of the small-scale amplitude, so a gather that happened to
    coincide with the weighted mean would be the bug this guards against.
    """
    idx = torch.tensor([0, 0, 0])
    w = torch.tensor([0.5, 0.3, 0.2])
    v = torch.tensor([[10.0], [20.0], [30.0]])

    pick = dominant_replica(idx, w, n_state=1)
    selected = v[pick]
    blended = torch.zeros(1, 1).index_add_(0, idx, v * w.view(-1, 1))

    assert selected.item() == 10.0, "must take the heaviest replica verbatim"
    assert not torch.allclose(selected, blended), "must not reproduce the weighted mean"


# ---------------------------------------------------------------------------------------------
# Correlated source. Every one of these guards a property the seam fix rests on; a source that
# is coherent but not unit-variance would silently rescale the residual, and one that is unit
# variance but not coherent would be the old behaviour under a new name.
# ---------------------------------------------------------------------------------------------


def _ring(n, lat_deg=45.0, spacing_km=5.0):
    """``n`` points along a parallel, ``spacing_km`` apart -> ``[n, 3]`` unit vectors."""
    lat = torch.full((n,), lat_deg * torch.pi / 180.0, dtype=torch.float64)
    dlon = spacing_km / (EARTH_RADIUS_KM * torch.cos(lat[0]))
    lon = torch.arange(n, dtype=torch.float64) * dlon
    return torch.stack(
        [torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], -1
    ).float()


@pytest.mark.parametrize("modes", [8, 256])
def test_correlated_source_has_a_standard_normal_marginal(modes):
    """Unit variance is EXACT for any M, and the rest of the algebra depends on it.

    ``ResidualScale`` normalises the residual to unit std and the ``y0 + v`` identity assumes the
    source matches it. A source at, say, 0.8 std would rescale every sample without touching a
    single metric that is currently logged.
    """
    g = torch.Generator().manual_seed(0)
    y0 = correlated_source(
        _ring(20000, spacing_km=3.0), 4, length_km=15.0, num_modes=modes, generator=g
    )

    assert y0.shape == (20000, 4)
    assert y0.mean().abs() < 0.05
    assert torch.allclose(y0.std(0), torch.ones(4), atol=0.05), y0.std(0)


def test_correlated_source_matches_the_gaussian_covariance_it_promises():
    """cov(d) = exp(-d^2 / 2l^2). This IS the mechanism -- without it nothing is shared."""
    g = torch.Generator().manual_seed(1)
    length_km, n = 40.0, 4000
    # one long ring so every separation is sampled many times, averaged over channels
    x = _ring(n, spacing_km=2.0)
    y0 = correlated_source(x, 64, length_km=length_km, num_modes=4096, generator=g)

    for lag, d_km in [(0, 0.0), (5, 10.0), (10, 20.0), (25, 50.0)]:
        emp = (y0[: n - lag] * y0[lag:]).mean().item()
        want = torch.exp(torch.tensor(-(d_km**2) / (2 * length_km**2))).item()
        assert abs(emp - want) < 0.06, f"lag {d_km} km: {emp:.3f} vs {want:.3f}"


def test_correlated_source_couples_near_points_and_not_far_ones():
    """The property in the units that matter: neighbours across a cell border share noise.

    A HEALPix level-5 cell is ~200 km, CERRA points are ~5-8 km apart, so two points straddling a
    boundary must be strongly correlated while two points in genuinely different weather must not.
    """
    g = torch.Generator().manual_seed(2)
    near, far = _ring(2, spacing_km=5.0), _ring(2, spacing_km=500.0)
    x = torch.cat([near, far])

    c_near, c_far = [], []
    for _ in range(400):
        y0 = correlated_source(x, 8, length_km=15.0, num_modes=256, generator=g)
        c_near.append((y0[0] * y0[1]).mean())
        c_far.append((y0[2] * y0[3]).mean())

    assert torch.stack(c_near).mean() > 0.6, "5 km apart must share noise"
    assert torch.stack(c_far).mean().abs() < 0.1, "500 km apart must not"


def test_correlated_source_with_mix_zero_is_exactly_the_old_iid_draw():
    """The default-off guarantee: mix=0 must reproduce ``torch.randn`` bit-for-bit.

    This is what makes every existing run bit-identical, and it is cheap to keep true.
    """
    x = _ring(500)
    a = correlated_source(
        x, 6, length_km=15.0, num_modes=64, generator=torch.Generator().manual_seed(7), mix=0.0
    )
    b = torch.randn((500, 6), generator=torch.Generator().manual_seed(7), dtype=torch.float32)
    assert torch.equal(a, b)


def test_correlated_source_is_a_function_of_position_only():
    """Two rows with the SAME coordinate must get the SAME noise, wherever they sit in the batch.

    Soft blend replicates a point into several host cells; the replicas carry copies of the raw
    coordinate row. This property is what gives every replica one shared ``y0`` -- and it removes
    an existing train/eval asymmetry, since training draws ``randn_like`` on the replicated axis
    while ``sample`` shares one draw through ``_expand``.
    """
    x = _ring(64)
    dup = torch.cat([x, x[:8]])  # 8 "replicas" appended out of order
    y0 = correlated_source(
        dup, 5, length_km=20.0, num_modes=32, generator=torch.Generator().manual_seed(3)
    )
    assert torch.allclose(y0[:8], y0[64:], atol=1e-5)


def test_flow_time_global_scope_gives_every_point_one_time():
    """Required by a correlated source: y_t is continuous across a boundary only if t is."""
    lens = torch.tensor([0, 3, 5, 2], dtype=torch.int32)
    t = sample_flow_time_for_points(10, lens, torch.device("cpu"), torch.float32, scope="global")

    assert t.shape == (10, 1)
    assert t.unique().numel() == 1, "global scope must be ONE draw for the whole batch"
    assert 0.0 < float(t[0]) < 1.0


def test_flow_time_scope_matches_the_legacy_boolean():
    """`per_cell` is the old spelling; it must keep meaning exactly what it meant."""
    lens = torch.tensor([0, 4, 6], dtype=torch.int32)
    kw = dict(output_lens=lens, device=torch.device("cpu"), dtype=torch.float32)

    torch.manual_seed(0)
    a = sample_flow_time_for_points(10, per_cell=True, **kw)
    torch.manual_seed(0)
    b = sample_flow_time_for_points(10, scope="cell", **kw)
    assert torch.equal(a, b)
    assert a[:4].unique().numel() == 1 and a[4:].unique().numel() == 1

    with pytest.raises(AssertionError, match="unknown flow_time_scope"):
        sample_flow_time_for_points(10, scope="per_cell", **kw)
