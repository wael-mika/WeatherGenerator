# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests for the ensemble loss functions (quantile_pinball level handling)."""

import pytest
import torch

from weathergen.train.loss_modules.loss_functions import pop_bce, quantile_pinball


def _pinball_reference(target, pred, taus):
    """Straightforward per-quantile pinball, sorted preds, no weights."""
    p = torch.sort(pred, dim=0).values
    total = 0.0
    for k, tau in enumerate(taus):
        r = target - p[k]
        total = total + torch.maximum(tau * r, (tau - 1.0) * r)
    return (total / len(taus)).mean(0)


def test_levels_none_matches_equispaced_grid():
    """Default behaviour is unchanged: levels=None == (k + 0.5) / K."""
    g = torch.Generator().manual_seed(0)
    target = torch.randn(64, 2, generator=g)
    pred = torch.randn(8, 64, 2, generator=g)

    loss_default, chs_default = quantile_pinball(target, pred, None, None)
    equispaced = [(k + 0.5) / 8 for k in range(8)]
    loss_explicit, chs_explicit = quantile_pinball(target, pred, None, None, levels=equispaced)

    assert torch.allclose(loss_default, loss_explicit)
    assert torch.allclose(chs_default, chs_explicit)


def test_custom_levels_match_reference_implementation():
    g = torch.Generator().manual_seed(1)
    target = torch.randn(32, 3, generator=g)
    pred = torch.randn(4, 32, 3, generator=g)
    levels = [0.3, 0.75, 0.9, 0.99]

    _, chs = quantile_pinball(target, pred, None, None, levels=levels)
    expected = _pinball_reference(target, pred, levels)

    assert torch.allclose(chs, expected, atol=1e-6)


def test_tail_levels_penalise_a_capped_prediction():
    """A prediction that cannot reach the wet tail is penalised more under tail-heavy levels.

    This is the IMERG motivation: with the equispaced grid most heads sit on the zero atom,
    so a model that never predicts heavy rain is barely charged for it.
    """
    target = torch.zeros(1000, 1)
    target[950:] = 10.0  # 5% heavy events
    pred = torch.zeros(16, 1000, 1)  # capped: predicts dry everywhere

    equispaced, _ = quantile_pinball(target, pred, None, None)
    tail_heavy, _ = quantile_pinball(
        target,
        pred,
        None,
        None,
        levels=[
            0.3,
            0.55,
            0.68,
            0.75,
            0.8,
            0.85,
            0.89,
            0.92,
            0.94,
            0.96,
            0.97,
            0.98,
            0.99,
            0.995,
            0.998,
            0.999,
        ],
    )

    assert tail_heavy > equispaced


def test_levels_are_validated():
    target = torch.zeros(10, 1)
    pred = torch.zeros(4, 10, 1)

    with pytest.raises(ValueError, match="must equal ens_size"):
        quantile_pinball(target, pred, None, None, levels=[0.25, 0.5, 0.75])
    with pytest.raises(ValueError, match="strictly in"):
        quantile_pinball(target, pred, None, None, levels=[0.0, 0.25, 0.5, 0.75])
    with pytest.raises(ValueError, match="strictly ascending"):
        quantile_pinball(target, pred, None, None, levels=[0.25, 0.75, 0.5, 0.9])


def test_custom_levels_are_differentiable_and_weightable():
    g = torch.Generator().manual_seed(2)
    target = torch.randn(20, 2, generator=g)
    target[0, 0] = float("nan")  # NaN targets must stay masked out
    pred = torch.randn(3, 20, 2, generator=g).requires_grad_(True)

    loss, chs = quantile_pinball(
        target,
        pred,
        torch.tensor([1.0, 0.5]),
        torch.rand(20, generator=g),
        levels=[0.5, 0.9, 0.99],
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert chs.shape == (2,)
    assert torch.isfinite(pred.grad).all()


# --------------------------------------------------------------------------- pop_bce


def test_pop_bce_matches_manual_bce_with_logits():
    """pop_bce equals plain BCE-with-logits on the (target>threshold) label, no weights."""
    g = torch.Generator().manual_seed(10)
    target = torch.randn(200, 2, generator=g)
    logit = torch.randn(200, 2, generator=g)
    pred = logit.unsqueeze(0)  # [1, N, C]

    thr = 0.3
    loss, chs = pop_bce(target, pred, None, None, threshold=thr)

    y = (target > thr).to(logit.dtype)
    manual = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    expected_chs = manual.mean(0)

    assert torch.allclose(chs, expected_chs, atol=1e-6)
    assert torch.allclose(loss, expected_chs.mean(), atol=1e-6)


def test_pop_bce_lower_when_logits_agree_with_label():
    """A logit that points the right way beats the opposite sign."""
    target = torch.zeros(1000, 1)
    target[800:] = 5.0  # 20% wet at threshold 0.1
    thr = 0.1
    y = (target > thr).float()

    good = torch.where(y > 0, 4.0, -4.0).unsqueeze(0)  # confident & correct
    bad = torch.where(y > 0, -4.0, 4.0).unsqueeze(0)  # confident & wrong
    loss_good, _ = pop_bce(target, good, None, None, threshold=thr)
    loss_bad, _ = pop_bce(target, bad, None, None, threshold=thr)

    assert loss_good < 0.1
    assert loss_bad > 1.0
    assert loss_good < loss_bad


def test_pop_bce_calibrated_minimiser_recovers_base_rate():
    """With no per-point information, the BCE-optimal constant logit is logit(base rate).

    This is the 'calibrated by construction' property: the minimiser is the true P(wet).
    """
    torch.manual_seed(0)
    base = 0.2
    target = (torch.rand(20000, 1) < base).float()  # already 0/1; threshold 0.5 splits it
    thr = 0.5

    logit = torch.zeros(1, 20000, 1, requires_grad=True)
    opt = torch.optim.Adam([logit], lr=0.2)
    for _ in range(300):
        opt.zero_grad()
        loss, _ = pop_bce(target, logit, None, None, threshold=thr)
        loss.backward()
        opt.step()

    prob = torch.sigmoid(logit.detach()).mean().item()
    assert abs(prob - base) < 0.02  # recovered the base rate => calibrated


def test_pop_bce_masks_nan_and_is_differentiable():
    g = torch.Generator().manual_seed(3)
    target = torch.randn(50, 2, generator=g)
    target[0, 0] = float("nan")  # must be masked out
    pred = torch.randn(1, 50, 2, generator=g).requires_grad_(True)

    loss, chs = pop_bce(
        target, pred, torch.tensor([1.0, 0.5]), torch.rand(50, generator=g), threshold=-0.1768
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert chs.shape == (2,)
    assert torch.isfinite(pred.grad).all()
