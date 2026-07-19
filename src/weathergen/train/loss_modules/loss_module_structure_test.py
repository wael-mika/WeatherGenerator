# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests for LossStructureFunction ensemble handling and the SF estimator."""

import torch

from weathergen.train.loss_modules.loss_module_structure import (
    LossStructureFunction,
    structure_function_loss,
)

ensemble_fields = LossStructureFunction.ensemble_fields


def _synthetic_ens(k=3, n=50, c=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, n, c, generator=g)


def test_ensemble_fields_ens1_all_modes_identical():
    pred = _synthetic_ens(k=1)
    for mode in ["mean", "median", "members", 0]:
        fields = ensemble_fields(pred, mode)
        assert len(fields) == 1
        assert torch.equal(fields[0], pred[0])


def test_ensemble_fields_mean():
    pred = _synthetic_ens(k=3)
    (field,) = ensemble_fields(pred, "mean")
    assert torch.allclose(field, pred.mean(0))


def test_ensemble_fields_median_odd_and_even():
    pred = _synthetic_ens(k=3)
    (field,) = ensemble_fields(pred, "median")
    assert torch.allclose(field, pred.sort(0).values[1])

    pred = _synthetic_ens(k=4)
    (field,) = ensemble_fields(pred, "median")
    srt = pred.sort(0).values
    assert torch.allclose(field, srt[1:3].mean(0))


def test_ensemble_fields_members_sorted_per_point():
    pred = _synthetic_ens(k=4)
    fields = ensemble_fields(pred, "members")
    assert len(fields) == 4
    srt = pred.sort(0).values
    for m, field in enumerate(fields):
        assert torch.equal(field, srt[m])
    # per-point monotone: field m <= field m+1 everywhere
    for m in range(3):
        assert (fields[m] <= fields[m + 1]).all()


def test_ensemble_fields_int_is_raw_member():
    pred = _synthetic_ens(k=4)
    (field,) = ensemble_fields(pred, 2)
    assert torch.equal(field, pred[2])


def _grid_coords(n_side=24, extent_deg=3.0):
    """Small lat/lon grid around (45N, 10E); ~extent_deg span => pairs in the 10-200 km bins."""
    lats = torch.linspace(45.0, 45.0 + extent_deg, n_side)
    lons = torch.linspace(10.0, 10.0 + extent_deg, n_side)
    grid = torch.cartesian_prod(lats, lons)
    return grid  # (n_side^2, 2) as (lat, lon)


def test_sf_loss_zero_for_identical_fields():
    coords = _grid_coords()
    g = torch.Generator().manual_seed(1)
    y = torch.randn(coords.shape[0], 2, generator=g)
    loss = structure_function_loss(
        y, y.clone(), coords, bin_edges_km=[10, 25, 50, 100, 200], num_pairs=20000
    )
    assert loss is not None
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_sf_loss_penalizes_damped_field_and_has_grad():
    coords = _grid_coords()
    g = torch.Generator().manual_seed(2)
    y = torch.randn(coords.shape[0], 1, generator=g)
    pred = (0.3 * y).requires_grad_(True)
    loss = structure_function_loss(
        y, pred, coords, bin_edges_km=[10, 25, 50, 100, 200], num_pairs=20000
    )
    assert loss is not None
    total = loss.mean()
    assert total.item() > 0.5  # log(0.09)^2 ~ 5.8 expected per bin
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def _intermittent_pair(seed=3, dry_frac=0.7, noise_amp=None):
    """Zero-inflated target (precip-like) + smooth pred with uniform noise calibrated so the
    mean-|increment| statistic matches the target's (the ymlx3c4r drizzle solution)."""
    coords = _grid_coords()
    n = coords.shape[0]
    g = torch.Generator().manual_seed(seed)
    wet = torch.rand(n, 1, generator=g) > dry_frac
    y = torch.where(wet, torch.randn(n, 1, generator=g).abs() * 2.0, torch.zeros(n, 1))
    if noise_amp is None:
        # mean |increment| of the target over random pairs ~ E|y_i - y_j|; iid uniform noise
        # U(-a, a) has E|n_i - n_j| = 2a/3 -> a = 1.5 * E|dy| reproduces the bin means
        gi = torch.randint(0, n, (20000,), generator=g)
        gj = torch.randint(0, n, (20000,), generator=g)
        noise_amp = 1.5 * (y[gi] - y[gj]).abs().mean().item()
    noise = (torch.rand(n, 1, generator=g) * 2.0 - 1.0) * noise_amp
    pred_noise = y.mean().expand_as(y) + noise  # structureless: flat field + uniform noise
    return y, pred_noise, coords


def test_sf_quantile_loss_catches_uniform_noise_gaming():
    """Regression test of the IMERG failure: uniform drizzle noise matches the per-bin MEAN
    increment (mean-only loss small) but not the increment DISTRIBUTION (quantile loss big)."""
    y, pred_noise, coords = _intermittent_pair()
    bins = [10, 25, 50, 100, 200]
    kw = dict(bin_edges_km=bins, num_pairs=50000, increment_power=1.0)
    loss_mean = structure_function_loss(y, pred_noise, coords, **kw)
    loss_quant = structure_function_loss(
        y,
        pred_noise,
        coords,
        quantile_levels=[0.5, 0.9],
        quantile_eps=1e-3,
        match_mean=False,
        **kw,
    )
    assert loss_mean is not None and loss_quant is not None
    # the noise field games the mean statistic ...
    assert loss_mean.mean().item() < 0.15
    # ... but its q50 sits at the noise amplitude while the target's q50 is ~0 (dry pairs)
    assert loss_quant.mean().item() > 5.0
    assert loss_quant.mean().item() > 20 * loss_mean.mean().item()


def test_sf_quantile_loss_zero_for_identical_and_has_grad():
    y, _, coords = _intermittent_pair(seed=4)
    kw = dict(
        bin_edges_km=[10, 25, 50, 100, 200],
        num_pairs=50000,
        increment_power=1.0,
        quantile_levels=[0.5, 0.9, 0.99],
    )
    loss = structure_function_loss(y, y.clone(), coords, **kw)
    assert loss is not None
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)

    pred = (0.3 * y).clone().requires_grad_(True)
    loss = structure_function_loss(y, pred, coords, **kw)
    assert loss is not None
    total = loss.mean()
    assert total.item() > 0.1
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_sf_loss_quantile_none_matches_legacy():
    """quantile_levels=None + match_mean=True must reproduce the original loss bit-for-bit."""
    coords = _grid_coords()
    g = torch.Generator().manual_seed(5)
    y = torch.randn(coords.shape[0], 2, generator=g)
    pred = 0.5 * y + 0.1
    kw = dict(bin_edges_km=[10, 25, 50, 100, 200], num_pairs=20000)
    legacy = structure_function_loss(
        y, pred, coords, generator=torch.Generator().manual_seed(7), **kw
    )
    new = structure_function_loss(
        y,
        pred,
        coords,
        generator=torch.Generator().manual_seed(7),
        quantile_levels=None,
        match_mean=True,
        **kw,
    )
    assert legacy is not None and new is not None
    assert torch.equal(legacy, new)


def test_sf_loss_match_mean_false_requires_quantiles():
    y, _, coords = _intermittent_pair(seed=6)
    try:
        structure_function_loss(
            y, y, coords, bin_edges_km=[10, 200], num_pairs=1000, match_mean=False
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
