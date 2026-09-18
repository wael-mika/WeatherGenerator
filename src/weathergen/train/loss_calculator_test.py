# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Term-level `enabled: False` drops an inherited loss term before it is constructed."""

import pytest
import torch
from omegaconf import OmegaConf

# loss_calculator imports model.model, which imports flash_attn -- GPU only, like
# decode_residual_flow_test. The import is inside the test so collection succeeds on CPU.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="flash_attn needs a GPU")


def test_disabled_term_is_not_constructed():
    """The WFCL term is deliberately unconstructable here (no coords store, and its pack guard
    would raise on a ResidualFlow decoder) -- proving `enabled: False` never reaches __init__."""
    from weathergen.train.loss_calculator import LossCalculator
    from weathergen.train.utils import TRAIN

    cf = OmegaConf.create({"decoder_type": "ResidualFlow", "streams": {}})
    mode_cfg = OmegaConf.create(
        {
            "losses": {
                "physical": {
                    "type": "LossPhysical",
                    "loss_fcts": {"mse": {"weight": 1.0}},
                },
                "wfcl": {
                    "type": "LossSpectralWFCL",
                    "enabled": False,
                    "loss_fcts": {"wfcl": {"target_stream": "CERRA"}},
                },
            }
        }
    )
    lc = LossCalculator(cf, mode_cfg, TRAIN, device="cpu")
    assert set(lc.loss_calculators) == {"physical"}
