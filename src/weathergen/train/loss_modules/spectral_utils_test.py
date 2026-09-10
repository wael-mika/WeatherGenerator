# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Correctness tests for the WFCL spectral primitives.

The wavelet tests here feed synthetic sub-band tensors in ``DTCWTForward``'s
layout, so the band *math* is covered without ``pytorch_wavelets`` installed.
The filterbank itself is exercised in ``spectral_patches_test.py``.
"""

import pytest
import torch

from weathergen.train.loss_modules.spectral_utils import (
    amplitude,
    fal_fcl,
    phase_weight,
    wal_wcl,
)


def _bands(n=2, c=3, j=3, h=32, w=32, seed=0):
    """Synthetic DTCWT-shaped high-pass bands: list[j] of (N, C, 6, H_l, W_l, 2)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, c, 6, h >> lv, w >> lv, 2, generator=g) for lv in range(j)]


# --------------------------------------------------------------------------
# Fourier terms
# --------------------------------------------------------------------------


def test_fcl_equals_image_cosine_distance():
    """Parseval: FCL is exactly 1 - cosine similarity of the flattened fields.

    This is the strongest single check on FCL -- and it is also precisely why FCL
    cannot constrain *local* phase, which is the paper's whole motivation for
    adding the wavelet term.
    """
    x = torch.rand(2, 3, 64, 96)
    y = torch.rand(2, 3, 64, 96)
    _, fcl = fal_fcl(y, x)
    cos = 1 - (x * y).sum((-2, -1)) / (x.norm(dim=(-2, -1)) * y.norm(dim=(-2, -1)))
    assert torch.allclose(fcl, cos, atol=1e-5)


def test_fcl_is_scale_invariant():
    """FCL is blind to a global rescaling -- the reason a mean-bias guard is needed."""
    x, y = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
    assert torch.allclose(fal_fcl(y, x)[1], fal_fcl(3.7 * y, x)[1], atol=1e-6)


def test_fal_is_translation_invariant_but_mse_is_not():
    x = torch.rand(1, 1, 64, 64)
    y = torch.roll(x, (5, 7), dims=(-2, -1))
    assert fal_fcl(y, x)[0].item() < 1e-10
    assert torch.mean((x - y) ** 2).item() > 1e-2


def test_fal_is_ortho_normalised():
    """FAL must be O(field^2), not O(H*W*field^2).

    A non-ortho FFT inflates FAL by H*W and silently reduces the composite to
    FAL-only. Bound it well below the H*W = 4096 that the unnormalised form gives.
    """
    x, y = torch.rand(1, 1, 64, 64), torch.zeros(1, 1, 64, 64)
    fal, _ = fal_fcl(y, x)
    assert fal.item() < 1.0


def test_identity_gives_zero_for_every_fourier_term():
    x = torch.rand(2, 3, 64, 64)
    fal, fcl = fal_fcl(x, x)
    assert fal.abs().max() < 1e-8
    assert fcl.abs().max() < 1e-6


# --------------------------------------------------------------------------
# Wavelet terms
# --------------------------------------------------------------------------


def test_wavelet_identity_and_shapes():
    yh = _bands()
    wal, wcl = wal_wcl(yh, yh)
    assert [tuple(t.shape) for t in wal] == [(2, 3, 6)] * 3
    for a, c in zip(wal, wcl, strict=True):
        assert a.abs().max() < 1e-8
        assert c.abs().max() < 1e-6


def test_wcl_is_per_band_scale_invariant():
    """Each (level, orientation) is normalised on its own, so a per-band rescale
    must not move WCL. Catches a reduction that collapses the band axis."""
    yh_t = _bands(seed=1)
    yh_p = _bands(seed=2)
    _, wcl_a = wal_wcl(yh_p, yh_t)
    scale = torch.tensor([0.1, 1.0, 5.0, 2.0, 0.5, 3.0]).view(1, 1, 6, 1, 1, 1)
    _, wcl_b = wal_wcl([h * scale for h in yh_p], yh_t)
    for a, b in zip(wcl_a, wcl_b, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_wcl_detects_a_band_local_phase_error_that_fcl_misses():
    """The paper's core claim, as a test.

    Flipping the sign of one orientation at one level leaves the *global* field
    correlation nearly untouched but must show up sharply in that band's WCL.
    """
    yh_t = _bands(seed=3)
    yh_p = [h.clone() for h in yh_t]
    yh_p[0][:, :, 2] *= -1.0  # invert phase of one orientation at the finest level
    _, wcl = wal_wcl(yh_p, yh_t)
    assert wcl[0][:, :, 2].min() > 1.9  # anti-correlated -> near 2
    others = torch.cat([wcl[0][:, :, :2].flatten(), wcl[0][:, :, 3:].flatten()])
    assert others.abs().max() < 1e-5  # every other orientation untouched


# --------------------------------------------------------------------------
# Numerical safety -- the failure mode that kills a training run
# --------------------------------------------------------------------------


def test_amplitude_gradient_is_finite_at_the_origin():
    z = torch.zeros(64, requires_grad=True)
    amplitude(z, z).sum().backward()
    assert torch.isfinite(z.grad).all()


def test_no_nan_gradients_on_all_zero_prediction():
    """Precipitation predictions are mostly exact zeros; an all-dry field is not
    a corner case, it is the initial condition."""
    target = torch.rand(1, 1, 64, 64)
    pred = torch.zeros(1, 1, 64, 64, requires_grad=True)
    fal, fcl = fal_fcl(pred, target)
    (fal.mean() + fcl.mean()).backward()
    assert torch.isfinite(pred.grad).all(), "NaN gradient from an all-zero prediction"


def test_no_nan_gradients_on_all_zero_wavelet_band():
    yh_t = _bands(seed=4)
    yh_p = [torch.zeros_like(h).requires_grad_(True) for h in yh_t]
    wal, wcl = wal_wcl(yh_p, yh_t)
    (sum(a.mean() for a in wal) + sum(c.mean() for c in wcl)).backward()
    for h in yh_p:
        assert torch.isfinite(h.grad).all(), "NaN gradient from an all-zero sub-band"


def test_unclamped_denominator_would_have_nan_gradients():
    """Guards the guard: shows the naive `sqrt(sum) + eps` form really does fail,
    so the clamp_min in _corr is not cargo cult."""
    target = torch.rand(1, 1, 32, 32)
    pred = torch.zeros(1, 1, 32, 32, requires_grad=True)
    f_t = torch.fft.fft2(target, norm="ortho")
    f_p = torch.fft.fft2(pred, norm="ortho")
    num = (f_t.real * f_p.real + f_t.imag * f_p.imag).sum()
    bad = (f_t.abs().pow(2).sum().sqrt() * f_p.abs().pow(2).sum().sqrt()) + 1e-8
    (1.0 - num / bad).backward()
    assert torch.isnan(pred.grad).any()


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 1.0), (450, 0.5), (900, 0.0), (1000, 0.0), (5000, 0.0)],
)
def test_schedule_endpoints(step, expected):
    assert phase_weight(step, 1000, alpha=0.1) == pytest.approx(expected)


def test_schedule_is_monotone_and_bounded():
    vals = [phase_weight(s, 1000) for s in range(0, 1001, 10)]
    assert all(1.0 >= a >= b >= 0.0 for a, b in zip(vals, vals[1:], strict=False))


def test_schedule_degenerates_safely():
    assert phase_weight(0, 0) == 0.0


def test_schedule_never_exceeds_one_before_its_start():
    """A negative `step` (schedule_start_step set above the parent's real final istep)
    must not push P_t over 1, which would give WAL a negative weight."""
    assert phase_weight(-12, 16500) == 1.0
    assert phase_weight(-100000, 16500) == 1.0
