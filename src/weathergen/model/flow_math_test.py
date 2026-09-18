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
    PACK_CFM,
    PACK_MU,
    PACK_RSCALE,
    PACK_WIDTH,
    PACK_X1,
    ResidualScale,
    correlated_source,
    dominant_replica,
    physical_to_residual,
    residual_to_physical,
    residual_to_physical_x1,
    sample_flow_time_for_points,
)
from weathergen.train.loss_modules.loss_functions import mse, mse_det, mse_flow

N = 64
C = 3


def _pack(mu, y0, v, r_scale, t=None, y_t=None):
    """The decoder's training return -- the PACK_* slots in order.

    ``t`` / ``y_t`` default to a mid-path state; the CFM slot does not depend on them.
    """
    if t is None:
        t = torch.full((mu.shape[0], 1), 0.5)
    if y_t is None:
        r = physical_to_residual(mu + r_scale * (y0 + v), mu, r_scale)
        y_t = (1.0 - t) * y0 + t * r
    slots = [None] * PACK_WIDTH
    slots[PACK_MU] = mu
    slots[PACK_CFM] = residual_to_physical(mu, y0, v, r_scale)
    slots[PACK_RSCALE] = r_scale.expand_as(mu)
    slots[PACK_X1] = residual_to_physical_x1(mu, y_t, t, v, r_scale)
    return torch.stack(slots, 0)


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

    # slots addressed by name: only PACK_MU / PACK_CFM are read by these two losses, so the
    # other slots are filled with a distinguishable value that must never leak into either.
    def _slots(mu, cfm):
        s = [other + 7.0] * PACK_WIDTH
        s[PACK_MU], s[PACK_CFM], s[PACK_RSCALE] = mu, cfm, ones
        return torch.stack(s, 0)

    exact_mu = _slots(y1, other)
    det, _ = mse_det(y1, exact_mu, None, None)
    flow, _ = mse_flow(y1, exact_mu, None, None)
    assert det.item() == pytest.approx(0.0, abs=1e-12)
    assert flow.item() == pytest.approx(9.0, rel=1e-6)

    exact_flow = _slots(other, y1)
    det, _ = mse_det(y1, exact_flow, None, None)
    flow, _ = mse_flow(y1, exact_flow, None, None)
    assert det.item() == pytest.approx(9.0, rel=1e-6)
    assert flow.item() == pytest.approx(0.0, abs=1e-12)

    # plain mse on the pack averages ALL the slots -- exactly what must never be configured.
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


# =================================================================================================
# The x1 slot, and why a spectral loss must score it rather than the CFM slot.
# Regression cover for run `prr1iy4u`; see the PACK_* comment in flow_math.
# =================================================================================================


def test_x1_and_cfm_slots_agree_exactly_at_a_perfect_velocity():
    """Both parametrisations recover y1 when v == u, so they share a minimiser."""
    torch.manual_seed(3)
    mu = torch.randn(N, C)
    r_scale = torch.tensor([0.851, 0.348, 0.563])
    y1 = mu + r_scale * torch.randn(N, C)
    y0 = torch.randn(N, C)

    r = physical_to_residual(y1, mu, r_scale)
    u = r - y0  # the exact CFM target velocity on the linear path
    for t_val in (0.05, 0.5, 0.95):
        t = torch.full((N, 1), t_val)
        y_t = (1.0 - t) * y0 + t * r
        cfm = residual_to_physical(mu, y0, u, r_scale)
        x1 = residual_to_physical_x1(mu, y_t, t, u, r_scale)
        assert torch.allclose(cfm, y1, atol=1e-5)
        assert torch.allclose(x1, y1, atol=1e-5)


def test_x1_slot_carries_less_of_the_raw_source_than_the_cfm_slot():
    """The property the spectral loss needs: contamination scales as (1 - t), not as 1.

    The CFM slot carries y0 at full amplitude at every t, so its fine-scale spectrum is set by
    the injected noise rather than by the model. That is the second, independent reason WFCL
    could not work on the flow arm even once the pointwise-sort bug is fixed.
    """
    torch.manual_seed(4)
    mu = torch.randn(N, C)
    r_scale = torch.tensor([0.851, 0.348, 0.563])
    y1 = mu + r_scale * torch.randn(N, C)
    y0 = torch.randn(N, C)
    r = physical_to_residual(y1, mu, r_scale)
    u = r - y0
    v = 0.9 * u + 0.1 * torch.randn(N, C)  # a corrector that has mostly learned the velocity

    errs = []
    for t_val in (0.1, 0.5, 0.9):
        t = torch.full((N, 1), t_val)
        y_t = (1.0 - t) * y0 + t * r
        cfm_err = (residual_to_physical(mu, y0, v, r_scale) - y1).abs().mean()
        x1_err = (residual_to_physical_x1(mu, y_t, t, v, r_scale) - y1).abs().mean()
        assert x1_err <= cfm_err + 1e-6, f"x1 must never be dirtier than cfm (t={t_val})"
        errs.append((cfm_err.item(), x1_err.item()))

    # cfm is flat in t; x1 falls off with it
    assert errs[0][0] == pytest.approx(errs[2][0], rel=1e-6), "cfm error must not depend on t"
    assert errs[2][1] < 0.5 * errs[0][1], "x1 error must shrink substantially as t -> 1"


def test_pointwise_sort_of_a_pack_is_not_any_slot():
    """`reduce: members` on a pack returns per-pixel chimeras -- the prr1iy4u bug, pinned.

    This is the check that would have caught it: no order statistic of the pack equals any slot,
    and the sorted fields draw most of their pixels from the two slots that carry NO gradient
    (PACK_MU under flow_freeze_base, and the PACK_RSCALE buffer).
    """
    from weathergen.train.loss_modules.loss_module_structure import LossStructureFunction

    torch.manual_seed(5)
    mu = torch.randn(N, C)
    r_scale = torch.tensor([0.851, 0.348, 0.563])
    y0, v = torch.randn(N, C), torch.randn(N, C)
    pack = _pack(mu, y0, v, r_scale)

    sorted_fields = LossStructureFunction.ensemble_fields(pack, "members")
    assert len(sorted_fields) == PACK_WIDTH
    for f in sorted_fields:
        for slot in range(PACK_WIDTH):
            assert not torch.allclose(f, pack[slot]), (
                "a sorted 'member' coincided with a real slot -- the test data is degenerate "
                "and no longer demonstrates the failure"
            )

    # and the explicit escape hatch returns exactly the slot asked for
    for slot in range(PACK_WIDTH):
        (only,) = LossStructureFunction.ensemble_fields(pack, "members", pack_member=slot)
        assert torch.equal(only, pack[slot])


def test_wfcl_refuses_to_score_a_residualflow_pack_without_pack_member():
    """The constructor guard: configuring the prr1iy4u bug again must fail loudly."""
    from omegaconf import OmegaConf

    from weathergen.train.loss_modules.loss_module_spectral import LossSpectralWFCL
    from weathergen.train.utils import TRAIN, VAL

    cfg = {
        "target_stream": "CERRA",
        "grid_width": 1069,
        "grid_height": 1069,
        "patch_size": 64,
        "coords_zarr": "/nonexistent.zarr",  # never opened: __init__ fails before any IO
        "total_steps": 1000,
        "J": 3,
        "reduce": "members",
    }
    cf = OmegaConf.create({"decoder_type": "ResidualFlow", "streams": {}})

    with pytest.raises(ValueError, match="must set `pack_member`"):
        LossSpectralWFCL(cf, OmegaConf.create({}), TRAIN, "cpu", wfcl=cfg)

    # validation sees real samples from the same decoder, so `reduce` is legitimate there
    LossSpectralWFCL(cf, OmegaConf.create({}), VAL, "cpu", wfcl=cfg)

    # and `pack_member` on a non-pack decoder is equally a misconfiguration
    cf_det = OmegaConf.create({"decoder_type": "PerceiverIO", "streams": {}})
    with pytest.raises(ValueError, match="only meaningful for a ResidualFlow"):
        LossSpectralWFCL(
            cf_det, OmegaConf.create({}), TRAIN, "cpu", wfcl={**cfg, "pack_member": "x1"}
        )


def test_pack_member_is_inert_at_validation():
    """kc5oigof attempt 1: at VAL the decoder returns real samples, so the slot index must
    not be applied. Died with `index 3 is out of bounds for dimension 0 with size 2`."""
    from omegaconf import OmegaConf

    from weathergen.train.loss_modules.loss_module_spectral import LossSpectralWFCL
    from weathergen.train.loss_modules.loss_module_structure import LossStructureFunction
    from weathergen.train.utils import TRAIN, VAL

    cfg = {
        "target_stream": "CERRA",
        "grid_width": 1069,
        "grid_height": 1069,
        "patch_size": 64,
        "coords_zarr": "/nonexistent.zarr",
        "total_steps": 1000,
        "J": 3,
        "reduce": "members",
        "pack_member": "x1",
    }
    cf = OmegaConf.create({"decoder_type": "ResidualFlow", "streams": {}})
    at_train = LossSpectralWFCL(cf, OmegaConf.create({}), TRAIN, "cpu", wfcl=cfg)
    at_val = LossSpectralWFCL(cf, OmegaConf.create({}), VAL, "cpu", wfcl=cfg)
    assert at_train.pack_member == PACK_X1
    assert at_val.pack_member is None

    # a real 2-member validation ensemble must resolve through `reduce`, not the slot index
    ens2 = torch.randn(2, N, C)
    fields = LossStructureFunction.ensemble_fields(ens2, at_val.reduce, at_val.pack_member)
    assert len(fields) == 2
    with pytest.raises(IndexError):
        LossStructureFunction.ensemble_fields(ens2, "members", pack_member=PACK_X1)
