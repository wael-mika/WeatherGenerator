# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for per-stage loss-function selection in LossPhysical."""

from omegaconf import OmegaConf

from weathergen.train.loss_modules.loss_module_physical import LossPhysical
from weathergen.train.utils import TRAIN, VAL


def _cf():
    return OmegaConf.create(
        {"streams": {"CERRA": {"train_target_channels": ["tp", "2t"]}}, "loss_chs_weights": None}
    )


def _mode_cfg():
    return OmegaConf.create({"forecast": {"offset": 0}})


def _names(stage, **loss_fcts):
    lp = LossPhysical(_cf(), _mode_cfg(), stage, "cpu", **loss_fcts)
    return [name for _, _, name in lp.loss_fcts]


def test_enabled_false_drops_a_loss_function():
    """validation_config is an OmegaConf merge *union* of the training config, so a loss function
    cannot be removed by overriding. `enabled: False` is the only way to score the two stages with
    different functions -- which the residual-flow decoder requires."""
    assert _names(VAL, mse_det={"enabled": False}, mse_flow={"enabled": False}, mse={}) == ["mse"]


def test_enabled_defaults_to_true_and_weights_survive():
    got = _names(TRAIN, mse_det={"weight": 1.0}, mse_flow={"weight": 0.5})
    assert got == ["mse_det", "mse_flow"]

    lp = LossPhysical(
        _cf(), _mode_cfg(), TRAIN, "cpu", mse_det={"weight": 1.0}, mse_flow={"weight": 0.5}
    )
    assert [w for _, w, _ in lp.loss_fcts] == [1.0, 0.5]


def test_null_loss_fct_is_dropped_not_crashed():
    """`mse: null` appears in configs that disable a term by nulling it."""
    assert _names(TRAIN, mse={}, mse_flow=None) == ["mse"]


def test_dynamic_loss_is_still_excluded():
    assert _names(TRAIN, mse={}, dynamic_loss={"window": 100}) == ["mse"]
