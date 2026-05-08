# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import pytest

# Skip this module when flash_attn is unavailable (CPU-only unit-test environment).
# The variable_groups_test.py file covers the CPU-compatible _resolve_variable_groups tests.
pytest.importorskip("flash_attn", exc_type=ImportError)

import torch

from weathergen.model.engines import EnsPredictionHead
from weathergen.model.utils import _resolve_variable_groups


class TestEnsPredictionHead:
    def test_single_head_output_shape(self):
        head = EnsPredictionHead(
            dim_embed=64, dim_out=10, ens_num_layers=1, ens_size=3, stream_name="test"
        )
        x = torch.randn(100, 64)
        out = head(x)
        assert out.shape == (3, 100, 10)

    def test_identity_activation_output_shape(self):
        head = EnsPredictionHead(
            dim_embed=32,
            dim_out=5,
            ens_num_layers=1,
            ens_size=1,
            stream_name="test",
            final_activation="Identity",
        )
        x = torch.randn(50, 32)
        out = head(x)
        assert out.shape == (1, 50, 5)

    def test_softplus_output_nonnegative(self):
        head = EnsPredictionHead(
            dim_embed=32,
            dim_out=4,
            ens_num_layers=1,
            ens_size=1,
            stream_name="test",
            final_activation="Softplus",
        )
        x = torch.randn(200, 32)
        out = head(x)
        assert out.shape == (1, 200, 4)
        assert (out >= 0).all(), "Softplus output must be non-negative"

    def test_multi_layer_head(self):
        head = EnsPredictionHead(
            dim_embed=64, dim_out=8, ens_num_layers=3, ens_size=2, stream_name="test"
        )
        x = torch.randn(50, 64)
        out = head(x)
        assert out.shape == (2, 50, 8)


class TestGroupedHeadScatter:
    """Verifies that per-group heads + scatter reassembly preserves shape and activation routing."""

    def test_scatter_roundtrip_shape_and_softplus(self):
        channels = ["u_850", "v_850", "tp", "2t", "sp"]
        cfg = {
            "dynamics": {
                "variables": [r"u_\d+", r"v_\d+"],
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Identity"},
            },
            "_default": {
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Softplus"},
            },
        }
        groups = _resolve_variable_groups(cfg, channels)

        dim_embed = 32
        n_toks = 60
        total_ch = len(channels)
        tc_tokens = torch.randn(n_toks, dim_embed)

        heads = {
            group_name: EnsPredictionHead(
                dim_embed=dim_embed,
                dim_out=len(ch_indices),
                ens_num_layers=1,
                ens_size=1,
                stream_name=group_name,
                final_activation=group_cfg.get("pred_head", {}).get("final_activation", "Identity"),
            )
            for group_name, ch_indices, group_cfg in groups
        }

        ens_size = 1
        pred = torch.zeros(ens_size, n_toks, total_ch)
        for group_name, ch_indices, _ in groups:
            grp_pred = heads[group_name](tc_tokens)
            ch_idx = torch.tensor(ch_indices, dtype=torch.long)
            pred[:, :, ch_idx] = grp_pred

        assert pred.shape == (1, 60, 5)

        default_indices = next(ch for name, ch, _ in groups if name == "_default")
        assert (pred[:, :, default_indices] >= 0).all()
