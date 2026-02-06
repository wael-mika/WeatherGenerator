# MoE Implementation Ideas (with FE)

## Scope
This note documents implementation ideas for making Mixture-of-Experts (MoE) robust, efficient, and reusable across WeatherGenerator engines, including Forecasting Engine (FE).

For runtime parameter meaning and log interpretation, see `docs/moe_parameters_and_debugging.md`.

The focus is:
- Better routing quality (including spatial routing)
- Lower dispatch overhead
- Correct aux conditioning behavior
- Cleaner config structure
- Engine-agnostic MoE integration

## Design Goals
1. Make MoE usable in any engine that currently uses an MLP block.
2. Support both token-only and token+aux experts (AdaLayerNorm-compatible).
3. Support explicit per-token spatial IDs instead of relying on sequence order.
4. Keep runtime predictable via capacity control.
5. Keep config simple enough for training sweeps.

## Core MoE Building Blocks
### Router
- Inputs:
  - Token embeddings `x`
  - Optional `position_ids` (for spatial router)
  - Optional `token_mask` (to ignore padded/invalid tokens)
- Outputs:
  - `router_probs` over experts
  - Top-k `expert_indices`
  - Top-k `expert_weights`

Implementation idea:
- Keep one dense linear projection for logits.
- Apply optional jitter noise in training.
- Compute top-k routing from probabilities.
- Use mask-aware softmax so masked tokens do not affect routing or aux loss.

### Spatial Router
Implementation idea:
- Append learned position embeddings before router projection.
- Initialize position embeddings from HEALPix coordinates (`theta`, `phi`) using sinusoidal features.
- Require explicit `position_ids` when available.
- Reserve special tokens with `position_id = -1` and map them to zero spatial embedding.

Reason:
- Sequence-order fallback is fragile once tokens are reordered or include special tokens.
- Explicit IDs make routing semantics stable across engines.

### Dispatch and Combine
Implementation idea:
- Flatten to token dimension once.
- Build assignment lists from `(token_idx, expert_idx, gate_weight)`.
- Apply optional capacity per expert.
- Dispatch expert inputs by indexing.
- Combine outputs with weighted `index_add_`.

Reason:
- Avoid repeated `isin`-style set membership checks in nested loops.
- Keep dispatch complexity near linear in number of active assignments.

### Aux Conditioning
Implementation idea:
- For routed token subset per expert, gather matching `aux[token_idx]`.
- Call expert with `(expert_input, expert_aux)` only for that subset.
- Keep no-aux path fast and unchanged.

Reason:
- Prevent shape mismatch and misaligned conditioning when MoE is enabled with AdaLayerNorm.

### Load-Balance Loss
Implementation idea:
- Use weighted per-token assignments (not only binary top-k mask).
- Mask out invalid tokens.
- Keep one configurable weight per engine.

Reason:
- Weighted assignments better reflect top-k gating behavior.

## Capacity Behavior
### Current Recommendation
- Add `capacity_factor` support and enforce per-expert capacity:
  - `capacity = ceil(capacity_factor * active_tokens / num_experts)`
- If assignments exceed capacity:
  - Keep highest-gate assignments first
  - Drop overflow assignments
- Optional `renormalize_gates` after dropping

### Why
- Prevent expert overload under skewed routing.
- Improve throughput predictability in long sequences.

## Engine-Agnostic Integration
Use a shared factory/helper so any engine can switch between MLP and MoE without duplicated logic.

Suggested helper responsibilities:
- Read MoE config block
- Instantiate expert factory
- Instantiate MoE block with router/capacity/loss options
- Register optional static `position_ids` for engines with known token layout

Target engines:
- Global Assimilation Engine (already natural fit)
- Forecasting Engine (already natural fit)
- Future: local/adaptor/aggregation blocks where MLP replacement is valid

## FE-Specific Guidance
FE should keep MoE enabled in forecasting training mode.

Recommended initial FE settings:
- `fe_use_moe: True`
- `fe_moe_blocks: "all"`
- `fe_moe_num_experts: 4` or `8`
- `fe_moe_top_k: 2`
- `fe_moe_capacity_factor: 1.25`
- `fe_moe_load_balance_weight: 1e-3`
- `fe_moe_use_spatial_routing: True` (if token-cell mapping is available)

## Config Structure Ideas
Current config uses separate per-engine keys (`ae_global_moe_*`, `fe_moe_*`), which is practical for immediate use.

Longer-term option:
- Keep engine overrides but add a shared global template.

Example:
```yaml
moe_defaults:
  top_k: 2
  capacity_factor: 1.25
  router_bias: false
  renormalize_gates: true
  load_balance_weight: 0.001

ae_global_moe:
  enabled: true
  num_experts: 4
  use_spatial_routing: true

fe_moe:
  enabled: true
  num_experts: 4
  use_spatial_routing: true
```

## Rollout Plan
1. Correctness pass:
   - Explicit position IDs
   - Aux gather per routed token subset
   - Mask-aware routing/loss
2. Efficiency pass:
   - Replace expensive assignment matching with index-based dispatch/combine
   - Capacity enforcement and gate renormalization
3. Generalization pass:
   - Shared factory for engine-agnostic MoE integration
   - Optional extension to more engine blocks
4. Validation pass:
   - Unit tests for routing, aux path, spatial IDs, and capacity behavior
   - Small profiling run for FE throughput and memory

## Minimum Test Matrix
1. Routing determinism:
   - Controlled router weights produce expected expert assignment.
2. Aux conditioning:
   - Routed tokens receive matching aux rows.
3. Spatial routing:
   - Explicit `position_ids` change routing as expected.
4. Capacity:
   - Overflow assignments are dropped consistently.
5. Mask behavior:
   - Invalid tokens do not contribute to dispatch or load-balance loss.

## Practical Training Notes
- Start with fewer experts and short forecast horizon.
- Keep load-balance weight non-zero but small (`1e-3` scale).
- Monitor:
  - Expert utilization histogram
  - Router aux loss trend
  - Tokens dropped by capacity
  - Throughput vs dense baseline
