# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests for the per-loss-block weight schedule (effective_weight)."""

import pytest

from weathergen.train.loss_calculator import effective_weight

W = 0.05
SCHED = {"start_step": 1000, "end_step": 5000}


def test_no_schedule_returns_static_weight():
    assert effective_weight(W, None, 0) == W
    assert effective_weight(W, None, 10**9) == W


def test_zero_before_start():
    assert effective_weight(W, SCHED, 0) == 0.0
    assert effective_weight(W, SCHED, 999) == 0.0


def test_linear_ramp():
    assert effective_weight(W, SCHED, 1000) == 0.0
    assert effective_weight(W, SCHED, 3000) == pytest.approx(W / 2)
    assert effective_weight(W, SCHED, 4000) == pytest.approx(0.75 * W)


def test_full_weight_from_end_step():
    assert effective_weight(W, SCHED, 5000) == pytest.approx(W)
    assert effective_weight(W, SCHED, 10**9) == pytest.approx(W)


def test_degenerate_schedule_is_step_function():
    sched = {"start_step": 1000, "end_step": 1000}
    assert effective_weight(W, sched, 999) == 0.0
    assert effective_weight(W, sched, 1000) == W


def test_monotone_non_decreasing():
    prev = -1.0
    for istep in range(0, 8000, 250):
        w = effective_weight(W, SCHED, istep)
        assert w >= prev
        prev = w
