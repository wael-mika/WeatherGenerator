# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU unit tests for the MoE implementation.

Covers the redesign that addresses "MoE == dense": the router-init fix
(ForecastingEngine no longer zeroes the MoE router), the shared always-on
expert (>= dense guarantee), auxiliary-loss-free bias balancing, and routing
diagnostics.  All tests run on CPU via ``uv run pytest``.
"""

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.model.layers import MLP, MoEBlock, _select_top_k
from weathergen.model.moe_diagnostics import compute_moe_block_diagnostics


def _make_expert_fn(dim, hidden_factor=1.0, dim_aux=None):
    return lambda: MLP(
        dim,
        dim,
        with_residual=False,
        hidden_factor=hidden_factor,
        dropout_rate=0.0,
        norm_type="LayerNorm",
        dim_aux=dim_aux,
    )


def _make_block(dim=16, num_experts=4, top_k=2, **kwargs):
    return MoEBlock(
        expert_fn=_make_expert_fn(dim),
        dim_in=dim,
        num_experts=num_experts,
        top_k=top_k,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# B. Shared always-on expert                                                   #
# --------------------------------------------------------------------------- #
def test_shared_expert_recovers_dense_path():
    """With routed experts zeroed, the block output equals the shared expert
    plus the residual — proving the shared path makes MoE >= dense."""
    torch.manual_seed(0)
    dim = 16
    block = _make_block(dim=dim, num_experts=2, top_k=2, num_shared_experts=1)
    block.eval()

    # Zero the routed experts' output projection so they contribute nothing.
    for expert in block.experts:
        last = expert.layers[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    x = torch.randn(8, dim)
    out, aux = block(x)

    assert aux is None  # eval mode
    expected = block.shared_expert[0](x) + x  # shared output + residual
    assert torch.allclose(out, expected, atol=1e-5)


def test_no_shared_expert_by_default():
    block = _make_block()
    assert block.shared_expert is None


# --------------------------------------------------------------------------- #
# C. Auxiliary-loss-free bias balancing                                        #
# --------------------------------------------------------------------------- #
def test_select_top_k_bias_changes_selection():
    """A large negative bias on the otherwise-winning expert steers selection
    away from it, while the (unbiased) gate weights still come from softmax."""
    logits = torch.tensor([[3.0, 1.0, 0.0, -1.0]])
    probs = torch.softmax(logits, dim=-1)

    idx_nobias, _ = _select_top_k(probs, logits, top_k=1, expert_bias=None)
    assert idx_nobias.item() == 0  # expert 0 wins unbiased

    bias = torch.tensor([-10.0, 0.0, 0.0, 0.0])
    idx_bias, w_bias = _select_top_k(probs, logits, top_k=1, expert_bias=bias)
    assert idx_bias.item() == 1  # bias steers away from expert 0
    # Gate weight is the unbiased softmax prob of the selected expert (renormed).
    assert torch.allclose(w_bias, torch.ones_like(w_bias))


def test_bias_update_rule_penalizes_overloaded():
    """update_expert_bias nudges overloaded experts down, underloaded up."""
    block = _make_block(num_experts=4, balance_mode="bias", bias_update_rate=0.1)
    block.last_expert_load = torch.tensor([10.0, 0.0, 0.0, 0.0])
    block.update_expert_bias()
    b = block.expert_bias
    assert b[0] < 0  # overloaded expert 0 penalized
    assert b[1] > 0 and b[2] > 0 and b[3] > 0  # underloaded experts boosted


def test_aux_mode_leaves_bias_untouched():
    """In the default 'aux' mode the selection bias never moves."""
    block = _make_block(num_experts=4, balance_mode="aux")
    block.train()
    x = torch.randn(12, 16)
    block(x)
    block.update_expert_bias()
    assert torch.count_nonzero(block.expert_bias) == 0


def test_forward_populates_load_and_aux():
    block = _make_block(num_experts=4, top_k=2, balance_mode="bias")
    block.train()
    n = 20
    x = torch.randn(n, 16)
    _, aux = block(x)
    assert aux is not None  # training aux loss present
    assert block.last_expert_load is not None
    # No mask, pre-capacity: every token contributes top_k assignments.
    assert int(block.last_expert_load.sum().item()) == n * block.top_k


def test_bias_balancing_drives_overloaded_expert_down():
    """With a static router that always prefers expert 0 on a constant input,
    repeated bias updates make expert 0 the most-penalized expert."""
    torch.manual_seed(0)
    dim = 8
    block = _make_block(dim=dim, num_experts=3, top_k=1, balance_mode="bias", bias_update_rate=0.5)
    block.train()
    # Force expert 0 to win: large positive logit for expert 0 on positive input.
    with torch.no_grad():
        block.router.router_weights.weight.zero_()
        block.router.router_weights.weight[0] = 5.0
    x = torch.ones(16, dim)

    for _ in range(50):
        block(x)
        block.update_expert_bias()

    assert block.expert_bias.argmin().item() == 0
    assert block.expert_bias[0] < 0


# --------------------------------------------------------------------------- #
# D. Diagnostics                                                               #
# --------------------------------------------------------------------------- #
def test_diagnostics_keys_and_ranges():
    block = _make_block(num_experts=4, top_k=2, balance_mode="bias")
    block.train()
    block(torch.randn(32, 16))

    diag = compute_moe_block_diagnostics(block)
    for key in ("entropy", "load_cv", "dead_expert_frac", "mean_top1_gate", "bias_spread"):
        assert key in diag
    assert 0.0 <= diag["entropy"] <= 1.0 + 1e-6
    assert 0.0 <= diag["dead_expert_frac"] <= 1.0
    assert diag["load_cv"] >= 0.0


def test_diagnostics_empty_without_forward():
    block = _make_block()
    assert compute_moe_block_diagnostics(block) == {}


# --------------------------------------------------------------------------- #
# A. ForecastingEngine router-init fix                                         #
# --------------------------------------------------------------------------- #
def _fe_config(dim=32):
    return OmegaConf.create(
        {
            "forecast_att_dense_rate": 1.0,
            "fe_num_blocks": 2,
            "ae_global_dim_embed": dim,
            "fe_num_heads": 4,
            "fe_dropout_rate": 0.0,
            "fe_with_qk_lnorm": True,
            "with_flash_attention": False,
            "norm_type": "LayerNorm",
            "qk_norm_type": "LayerNorm",
            "norm_eps": 1e-5,
            "mlp_norm_eps": 1e-5,
            "attention_dtype": "float32",
            "rope_2D": False,
            "ae_global_block_factor": 4,
            "ae_local_num_queries": 1,
            "num_register_tokens": 0,
            "num_class_tokens": 0,
            "fe_impute_latent_noise_std": 0.0,
            "fe_layer_norm_after_blocks": [],
            # MoE on block 0 only, token routing.
            "fe_use_moe": True,
            "fe_moe_blocks": [0],
            "fe_moe_num_experts": 4,
            "fe_moe_top_k": 2,
            "fe_moe_capacity_factor": None,
            "fe_moe_load_balance_weight": 0.0005,
            "fe_moe_jitter_noise": 0.0,
            "fe_moe_router_bias": False,
            "fe_moe_renormalize_gates": True,
            "fe_moe_expert_hidden_factor": 1.0,
            "fe_moe_use_spatial_routing": False,
            "fe_moe_position_embed_dim": 16,
            "fe_moe_router_hidden_dim": 0,
            "fe_moe_router_z_loss_weight": 0.0,
            "fe_moe_num_shared_experts": 1,
            "fe_moe_balance_mode": "aux",
            "fe_moe_bias_update_rate": 0.001,
            "fe_moe_debug": False,
            "fe_moe_debug_interval": 100,
            "fe_moe_debug_top_experts": 3,
        }
    )


def _build_fe(cf):
    # ForecastingEngine pulls in attention.py, which imports the GPU-only
    # flash_attn at module load; skip these engine-level tests on CPU-only envs.
    pytest.importorskip("flash_attn", reason="ForecastingEngine requires flash_attn")
    from weathergen.model.engines import ForecastingEngine

    mode_cfg = OmegaConf.create({"forecast": {"policy": "fixed"}})
    return ForecastingEngine(cf, mode_cfg, num_healpix_cells=4)


def _find_moe_block(fe):
    return next(b for b in fe.fe_blocks if isinstance(b, MoEBlock))


def test_fe_init_leaves_router_discriminative_but_experts_near_zero():
    """The core bug fix: the FE near-zero init (std=0.001) must NOT be applied
    to the MoE router (it needs to route), while experts still start near-zero
    so the block starts ~identity."""
    fe = _build_fe(_fe_config())
    moe = _find_moe_block(fe)

    router_std = moe.router.router_weights.weight.std().item()
    expert_out_std = moe.experts[0].layers[-1].weight.std().item()

    # Router keeps its default init — clearly larger than the 0.001 near-zero init.
    assert router_std > 0.01, router_std
    # Experts (and shared expert) keep the near-identity init.
    assert expert_out_std < 0.005, expert_out_std
    assert moe.shared_expert[0].layers[-1].weight.std().item() < 0.005


def test_fe_moe_forward_backward_trains_router():
    """End-to-end on CPU: a forward+backward through the FE produces a
    non-trivial gradient on the router weights (routing is actually learned)."""
    fe = _build_fe(_fe_config())
    fe.train()
    moe = _find_moe_block(fe)

    tokens = torch.randn(4, 32, requires_grad=False)
    out = fe(tokens, fstep=0)
    # Emulate the trainer: main loss + collected MoE aux loss.
    loss = out.square().mean()
    if moe.get_aux_loss() is not None:
        loss = loss + moe.get_aux_loss()
    loss.backward()

    grad = moe.router.router_weights.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0
