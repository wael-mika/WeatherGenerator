# MoE Parameters and Debugging Guide

## Purpose
This document explains:
- What each MoE configuration parameter does.
- How to interpret MoE debug logs during training.
- How to reason about whether spatial routing is helping.

It applies to both engines currently using MoE:
- AE-global (`ae_global_moe_*`)
- Forecasting engine (`fe_moe_*`)

## Conceptual Model
### Terminology
- Token: one latent vector routed by MoE.
- Expert: one MLP branch inside an MoE block.
- Router: module that scores experts per token.
- Gate weight: mixing weight for an expert output.
- Assignment edge: one token-to-expert routing link.
- `top_k`: number of assignment edges created per token.

### Assignment Edge Intuition
You can think of routing as a bipartite graph:
- Left side: tokens.
- Right side: experts.
- Router creates `top_k` weighted edges from each token to experts.

For each edge we have:
- `token_idx`
- `expert_idx`
- `gate_weight`

If capacity is exceeded for an expert, some edges to that expert are dropped.

### End-to-End MoE Flow (Current Implementation)
1. Flatten tokens to `[N, D]`.
2. Router computes expert probabilities and selects top-k experts per token.
3. Build assignment edge lists (`token_indices`, `expert_indices`, `expert_weights`).
4. Remove masked-token edges (if token mask is provided).
5. Compute per-expert capacity and drop overflow edges.
6. Optionally re-normalize gate weights after dropping.
7. Dispatch routed tokens to each expert.
8. Combine weighted expert outputs back to token order.
9. Apply residual connection (if enabled).
10. In training, compute load-balance auxiliary loss.

## Parameter Reference
Use the same meaning for both prefixes (`ae_global_moe_*` and `fe_moe_*`).

| Parameter | Meaning | Typical values | Practical effect |
|---|---|---|---|
| `*_use_moe` | Enable MoE in that engine. | `True` / `False` | Switches MLP blocks to MoE blocks. |
| `*_moe_blocks` | Which block indices use MoE. | `"all"` or list like `[0, 2]` | Controls where sparse experts are applied. |
| `*_moe_num_experts` | Number of experts per MoE block. | `4`, `8` | More experts increase specialization and compute/memory overhead. |
| `*_moe_top_k` | Experts selected per token. | `1`, `2` | Higher `top_k` improves expressivity but increases routing traffic. |
| `*_moe_capacity_factor` | Per-expert capacity multiplier. | `1.0` to `2.0` | Lower value drops more assignments; higher value reduces drops. |
| `*_moe_load_balance_weight` | Weight of router aux loss. | `1e-3`, `1e-2` | Too low can collapse routing; too high can hurt task loss. |
| `*_moe_jitter_noise` | Noise added to router logits in training. | `0.0` to `0.05` | Encourages exploration and avoids early hard collapse. |
| `*_moe_router_bias` | Bias term in router projection. | `False` / `True` | Can help fit but may increase routing skew. |
| `*_moe_renormalize_gates` | Re-normalize gate weights after capacity drops. | `True` | Keeps per-token gate mass consistent after dropped assignments. |
| `*_moe_expert_hidden_factor` | Hidden expansion factor inside each expert MLP. | `1.0`, `2.0` | Controls expert capacity and cost. |
| `*_moe_use_spatial_routing` | Use spatial router with position IDs. | `True` / `False` | Adds position embedding signal to router decisions. |
| `*_moe_position_embed_dim` | Dimension of spatial position embedding. | `64`, `128` | Larger values increase spatial signal capacity and parameters. |
| `*_moe_debug` | Enable periodic debug logs. | `True` / `False` | Emits MoE diagnostic line during training. |
| `*_moe_debug_interval` | Log every N MoE forwards. | `50`, `100` | Lower values give more visibility with more log volume. |
| `*_moe_debug_top_experts` | How many experts to print in distributions. | `3`, `4` | Controls log compactness; does not affect training. |

Assignment edge definition:
- One assignment edge = one token-to-expert route selected by top-k routing.
- Example: with `valid_tokens=12296` and `top_k=2`, requested edges are `12296 * 2 = 24592`.

## Debug Log Fields
Current log format:

```text
MoE[<block>] fwd=<n> valid_tokens=<v>/<n_tokens> cap=<c>
dropped=<d>(<p>%) mean_top1_gate=<g>
experts_pre=<...> experts_post=<...> aux=<a>
```

Field meaning:
- `MoE[ae_global.block_0]`: block identifier.
- `fwd`: local forward counter for that MoE block.
- `valid_tokens`: tokens considered for routing after masking.
- `cap`: per-expert capacity used in that forward pass.
- `dropped`: number and percent of assignment edges dropped by capacity.
  Example: `dropped=9220(37.49%)` means 9,220 token->expert routes were removed out of 24,592 requested edges.
- `mean_top1_gate`: average gate weight of the top-1 selected expert.
  Example: `mean_top1_gate=0.5895` means the first-choice expert carries about 58.95% of per-token routed weight on average.
- `experts_pre`: top experts by assignment fraction before capacity clipping.
  Example: `experts_pre=1:0.409, 0:0.267, 2:0.239` means 40.9% of requested edges targeted expert 1 before clipping.
- `experts_post`: top experts by assignment fraction after capacity clipping.
  Example: `experts_post=2:0.282, 1:0.282, 0:0.282` means after clipping, kept edges are much more balanced across experts.
- `aux`: load-balance auxiliary loss for that block.
  Example: `aux=1.1239e-03` is the per-block router balancing penalty (already scaled by `*_moe_load_balance_weight`).

## How to Interpret Your Current Numbers
Given your config:
- `num_experts = 4`
- `top_k = 2`
- `capacity_factor = 1.25`
- `valid_tokens = 12296`

Capacity per expert:

```text
cap = ceil(1.25 * 12296 / 4) = 3843
```

Total routing assignments requested:

```text
requested = valid_tokens * top_k = 12296 * 2 = 24592
```

Maximum assignments that can be kept:

```text
kept_max = num_experts * cap = 4 * 3843 = 15372
```

Theoretical minimum dropped assignments:

```text
dropped_min = requested - kept_max = 24592 - 15372 = 9220
dropped_min_ratio = 9220 / 24592 = 37.49%
```

Interpretation of what you saw:
- FE blocks at `37.49%` dropped are exactly at the theoretical floor.
- AE-global blocks at `~39.6%` to `46.9%` are above floor, which indicates extra skew/specialization before clipping.
- `experts_post` near uniform (`~0.25` each with 4 experts) means capacity clipping is strongly flattening final assignment distribution.
- `mean_top1_gate` around `0.50` to `0.61` is expected for `top_k=2`; values are not fully sharp/hard-routed.
- `aux` around `1e-3` per block is consistent with `*_moe_load_balance_weight: 1e-3`.
- `moe_router.loss_avg` around `1.3e-2` is the sum across all active MoE blocks, not one block.

## Does This Prove Spatial Routing Works?
Not by itself. These logs prove MoE routing and clipping behavior, but they do not isolate spatial contribution.

What would increase confidence that spatial routing is active and useful:
1. `use_spatial_routing=True` beats `False` on validation loss with similar compute.
2. Shuffling `position_ids` degrades performance or changes routing distributions materially.
3. Per-cell routing patterns are stable over time and non-random.

## Quick Tuning Rules
1. If `dropped` is very high and you care about expert specialization, raise `capacity_factor` first.
2. If experts collapse to a subset in `experts_pre`, increase `load_balance_weight` slightly or add small `jitter_noise`.
3. If routing is too uniform and not specializing, reduce `load_balance_weight` slightly.
4. If logs are too noisy, increase `*_moe_debug_interval`.
5. For diagnosis runs, use `top_k=1` or `capacity_factor=2.0` to reduce clipping confounds.
