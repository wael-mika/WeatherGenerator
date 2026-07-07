# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Lightweight, scalar routing diagnostics for :class:`MoEBlock`.

These turn the per-block routing state that ``MoEBlock`` already records in its
forward pass (``last_expert_load``, ``last_expert_weights``, ``expert_bias``)
into a handful of scalars that can be logged over training so routing health is
observable — the key question of whether MoE is doing anything a dense FFN
cannot.  Nothing here computes gradients; it reads detached state only.

Metric definitions (per MoE block):

* ``entropy`` — Shannon entropy of the per-expert load distribution, normalised
  by ``log(num_experts)`` so ``1.0`` is perfectly uniform routing and values
  near ``0`` mean the router collapsed onto a few experts.
* ``load_cv`` — coefficient of variation (std / mean) of the per-expert load;
  ``0`` is perfectly balanced, larger is more imbalanced.
* ``dead_expert_frac`` — fraction of experts that received (almost) no tokens.
* ``mean_top1_gate`` — mean gate weight of each token's top-1 expert; near
  ``1/top_k`` means the router is indecisive, near ``1`` means confident.
* ``bias_spread`` — range of the auxiliary-loss-free selection bias
  (``max - min``); only meaningful in ``balance_mode == "bias"``.
"""

from __future__ import annotations

import torch


def compute_moe_block_diagnostics(block, dead_expert_threshold: float = 1e-6) -> dict[str, float]:
    """Return scalar routing diagnostics for a single :class:`MoEBlock`.

    Args:
        block: A ``MoEBlock`` instance whose most recent forward stored routing
            state.  Any block missing that state (e.g. never run, or run in eval
            mode) yields an empty dict.
        dead_expert_threshold: Load fraction below which an expert counts as
            dead.

    Returns:
        Mapping of metric name to float.  Empty when no routing state is
        available.
    """
    load = getattr(block, "last_expert_load", None)
    if load is None:
        return {}

    load = load.detach().to(torch.float32)
    total = load.sum().clamp_min(1.0)
    probs = load / total

    num_experts = load.numel()
    # Normalised entropy in [0, 1]: 1.0 == uniform routing.
    nonzero = probs > 0
    entropy = float(-(probs[nonzero] * probs[nonzero].log()).sum().item())
    max_entropy = float(torch.log(torch.tensor(float(max(num_experts, 1)))).item())
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    mean_load = load.mean()
    load_cv = float((load.std(unbiased=False) / mean_load.clamp_min(1e-9)).item())
    dead_frac = float((probs < dead_expert_threshold).to(torch.float32).mean().item())

    metrics = {
        "entropy": norm_entropy,
        "load_cv": load_cv,
        "dead_expert_frac": dead_frac,
    }

    weights = getattr(block, "last_expert_weights", None)
    if weights is not None and weights.numel() > 0:
        top1 = weights.detach().to(torch.float32).reshape(-1, weights.shape[-1])[:, 0]
        metrics["mean_top1_gate"] = float(top1.mean().item())

    if getattr(block, "balance_mode", "aux") == "bias":
        bias = block.expert_bias.detach().to(torch.float32)
        metrics["bias_spread"] = float((bias.max() - bias.min()).item())

    return metrics


def collect_moe_diagnostics(model, prefix: str = "moe") -> dict[str, float]:
    """Walk every :class:`MoEBlock` in *model* and collect flattened, prefixed
    diagnostics plus per-metric averages across blocks.

    Keys are ``{prefix}.{block_name}.{metric}`` for each block, and
    ``{prefix}.mean.{metric}`` for the cross-block average.  Returns an empty
    dict when the model has no MoE blocks with routing state.
    """
    # Local import to avoid a circular import at module load time.
    from weathergen.model.layers import MoEBlock

    per_metric: dict[str, list[float]] = {}
    out: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, MoEBlock):
            continue
        stats = compute_moe_block_diagnostics(module)
        if not stats:
            continue
        block_name = getattr(module, "debug_name", None) or name or "block"
        for metric, value in stats.items():
            out[f"{prefix}.{block_name}.{metric}"] = value
            per_metric.setdefault(metric, []).append(value)

    for metric, values in per_metric.items():
        out[f"{prefix}.mean.{metric}"] = sum(values) / len(values)

    return out
