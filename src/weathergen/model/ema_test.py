# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for EMAModel, focused on the buffer-tracking contract.

Parameters are averaged; buffers are calibration STATE and must be copied. Getting that wrong is
invisible in training curves and silently corrupts both checkpoints and EMA validation, because
``EMAModel.state_dict`` returns the EMA model.
"""

import copy

import pytest
import torch

from weathergen.model.ema import EMAModel

# EMAModel.reset() hardcodes `to_empty(device="cuda")`, so this cannot run on CPU.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="EMAModel requires CUDA")


class _Net(torch.nn.Module):
    """A weight to average and a running-statistic buffer to track."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(3))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))


def _ema(model):
    return EMAModel(
        model,
        copy.deepcopy(model),
        halflife_steps=1000.0,
        rampup_ratio=None,  # 0.0 would collapse halflife to 0, i.e. a full copy
        is_model_sharded=False,
    )


def test_buffers_track_the_source_model():
    """The regression that cost the residual-flow campaign.

    ``ResidualScale`` calibrated correctly on the live model (count reached 1000 with per-channel
    values), but every checkpoint stored count=0 and scale=[1, 1, ...] because the EMA model's
    buffers were copied once at construction and never resynced. Validation ran with
    ``validate_with_ema``, so the corrector was trained in units of ``r_scale`` and sampled with
    ``r_scale = 1.0`` -- 1.6x to 14x too large depending on channel.
    """
    m = _Net().cuda()
    ema = _ema(m)

    # the live model calibrates, as ResidualScale does during training
    m.scale.copy_(torch.tensor([0.31, 0.096, 0.62]))
    m.count.fill_(1000)
    ema.update(cur_step=1, batch_size=1)

    assert torch.allclose(ema.ema_model.scale, m.scale), (
        f"EMA buffer went stale: {ema.ema_model.scale.tolist()} != {m.scale.tolist()}"
    )
    assert int(ema.ema_model.count) == 1000

    # and it is the EMA state_dict that gets checkpointed, so check that too
    sd = ema.state_dict()
    assert torch.allclose(sd["scale"], m.scale)
    assert int(sd["count"]) == 1000


def test_parameters_are_averaged_not_copied():
    """Buffers are copied; parameters must still be interpolated, or EMA is pointless."""
    m = _Net().cuda()
    with torch.no_grad():
        m.lin.weight.fill_(0.0)
    ema = _ema(m)
    with torch.no_grad():
        m.lin.weight.fill_(1.0)

    ema.update(cur_step=1, batch_size=1)
    w = ema.ema_model.lin.weight
    assert not torch.allclose(w, m.lin.weight), "parameters must not be hard-copied"
    assert (w.abs() > 0).any(), "parameters must move toward the source"
