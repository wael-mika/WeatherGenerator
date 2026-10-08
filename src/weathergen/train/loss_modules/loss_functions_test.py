# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import weathergen.train.loss_modules.loss_functions as lf
from weathergen.train.loss_modules.loss_module_physical import build_loss_fcts


def _data(n=4000, c=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.distributions.Gamma(0.5, 1.0).sample((n, c))  # skewed, like precipitation
    pred = torch.randn((1, n, c), generator=g)
    w_pts = torch.rand(n, generator=g)
    w_chs = torch.tensor([1.0, 3.0])[:c]
    return target, pred, w_pts, w_chs


def test_pinball_median_is_half_mae():
    target, pred, w_pts, w_chs = _data()
    target[::7, 0] = torch.nan
    loss, chs = lf.quantile_pinball(target, pred, w_chs, w_pts, quantiles=[0.5])
    ref, ref_chs = lf.mae(target, pred, w_chs, w_pts)
    torch.testing.assert_close(loss, 0.5 * ref)
    torch.testing.assert_close(chs, 0.5 * ref_chs)


def test_expectile_half_is_half_mse():
    target, pred, w_pts, w_chs = _data()
    target[::5, 1] = torch.nan
    loss, chs = lf.expectile(target, pred, w_chs, w_pts, tau=0.5)
    ref, ref_chs = lf.mse(target, pred, w_chs, w_pts)
    torch.testing.assert_close(loss, 0.5 * ref)
    torch.testing.assert_close(chs, 0.5 * ref_chs)


def _fit_constants(loss_fn, target, n_members, steps=3000, lr=0.05, **kw):
    """Minimise the loss over one constant per member (and channel) by gradient descent."""
    c = torch.zeros((n_members, 1, target.shape[1]), requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss, _ = loss_fn(target, c.expand(n_members, *target.shape), None, None, **kw)
        loss.backward()
        opt.step()
    return c.detach()[:, 0, :]


def test_pinball_members_converge_to_their_quantiles():
    target, _, _, _ = _data(n=20000, c=1)
    qs = [0.5, 0.9, 0.99]
    c = _fit_constants(lf.quantile_pinball, target, len(qs), quantiles=qs)
    # gradient steps are scaled by 1/len(qs): allow a small tolerance
    for k, q in enumerate(qs):
        emp = float(torch.quantile(target[:, 0], q))
        assert abs(float(c[k, 0]) - emp) < 0.03 * max(1.0, emp), (q, float(c[k, 0]), emp)


def test_expectile_converges_to_empirical_expectile():
    target, _, _, _ = _data(n=20000, c=1)
    y = target[:, 0].double().numpy()
    tau = 0.9
    e = y.mean()
    for _ in range(200):  # fixed-point iteration for the tau-expectile
        w = np.where(y > e, tau, 1 - tau)
        e = (w * y).sum() / w.sum()
    c = _fit_constants(lf.expectile, target, 1, tau=tau)
    assert abs(float(c[0, 0]) - e) < 0.01, (float(c[0, 0]), e)
    assert e > y.mean()  # tau > 0.5 sits above the mean


def test_pinball_requires_one_member_per_quantile():
    target, pred, _, _ = _data()
    with pytest.raises(AssertionError, match="one ensemble member per quantile"):
        lf.quantile_pinball(target, pred, quantiles=[0.5, 0.9])


def test_pinball_penalises_underprediction_more_at_high_quantile():
    target = torch.ones((10, 1))
    low, high = torch.zeros((1, 10, 1)), 2 * torch.ones((1, 10, 1))
    under, _ = lf.quantile_pinball(target, low, quantiles=[0.9])
    over, _ = lf.quantile_pinball(target, high, quantiles=[0.9])
    torch.testing.assert_close(under, torch.tensor(0.9))
    torch.testing.assert_close(over, torch.tensor(0.1))


def test_build_loss_fcts_binds_parameters(caplog):
    cfg = OmegaConf.create(
        {
            "quantile_pinball": {"weight": 2.0, "quantiles": [0.5, 0.9, 0.99]},
            "expectile": {"tau": 0.9},
            "mse": {"target_source_correspondence": {"x": 1}},  # framework key: silently ignored
            "mae": {"quantile": 0.9},  # misspelt / unknown: ignored with a warning
        }
    )
    with caplog.at_level(logging.WARNING):
        fcts = {name: (fct, w) for fct, w, name in build_loss_fcts(cfg)}
    assert fcts["quantile_pinball"][1] == 2.0
    assert fcts["quantile_pinball"][0].keywords == {"quantiles": [0.5, 0.9, 0.99]}
    assert fcts["expectile"][0].keywords == {"tau": 0.9}
    assert fcts["mse"][0] is lf.mse and fcts["mae"][0] is lf.mae
    assert "mae" in caplog.text and "quantile" in caplog.text
    assert "target_source_correspondence" not in caplog.text
    # bound functions keep the positional loss-function interface
    target, pred, w_pts, w_chs = _data(c=1)
    loss, chs = fcts["expectile"][0](target, pred, w_chs[:1], w_pts)
    assert loss.ndim == 0 and chs.shape == (1,)
