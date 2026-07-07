# Porting the WeatherGenerator MoE into another branch

**Audience:** an engineer (e.g. Opus) importing the Mixture-of-Experts (MoE) implementation from
`wm/dev/MoE_v2` into a different WeatherGenerator branch, and configuring MoE runs.

This guide documents the *modern* FE-MoE implementation: standard sparse top-k MoE with the fixes
that make it actually train (router-init fix, shared expert, aux-loss-free balancing, routing
diagnostics). It is written to survive a divergent target branch — the shared model/trainer files
have moved around across history, so the instructions are **anchored on symbol names, not line
numbers**.

---

## 1. What the implementation consists of

MoE is a standard transformer construct here: it **replaces the FFN (MLP) block** inside a
transformer engine with a sparse mixture of expert MLPs plus a router. It is wired into two engines
— the forecasting engine (`fe_moe_*`, the one actually used) and, optionally, the global
assimilation engine (`ae_global_moe_*`). Everything is gated off by default.

### Files that make up MoE

| File | MoE contribution |
|------|------------------|
| `src/weathergen/model/layers.py` | **Core.** `MoERouter`, `SpatialMoERouter`, `MoEBlock`, `LoadBalancingLoss`, `RouterZLoss`, the `_select_top_k` helper. (`MLP` lives here too and is the expert body.) |
| `src/weathergen/model/engines.py` | Wiring: `MoEBlockConfig`, `_get_moe_block_config`, `_moe_block_enabled`, `_moe_forward_wrapper_with_aux` / `_no_aux`, `_build_global_position_ids`; MoE branches inside `ForecastingEngine` and `GlobalAssimilationEngine` (`__init__` + `forward`); **the router-init fix**; `initialize_moe_position_ids` / `initialize_spatial_routers` per engine. |
| `src/weathergen/model/model.py` | Model-level `initialize_spatial_routers` (HEALPix `pix2ang` coord injection) and `initialize_moe_position_ids` that fan out to the engines. |
| `src/weathergen/model/model_interface.py` | FSDP2 sharding of `MoEBlock` (with the double-shard guard) and **re-initialising the non-persistent position-id buffers after every checkpoint load**. |
| `src/weathergen/train/trainer.py` | Per-step glue: `reset_aux_loss` before forward, `_collect_moe_aux_losses` + `_record_moe_aux_loss` after loss, `_update_moe_expert_bias` after backward, `_log_moe_diagnostics` at metrics cadence. |
| `src/weathergen/model/moe_diagnostics.py` | **New file.** Scalar routing diagnostics (`compute_moe_block_diagnostics`, `collect_moe_diagnostics`). |
| `src/weathergen/model/moe_test.py` | **New file.** CPU unit tests. |
| `config/default_config.yml` | The `fe_moe_*` (and `ae_global_moe_*`) key block. |

---

## 2. Two ways to import

### Option A — targeted symbol port (recommended for a divergent branch)

The shared files (`layers.py`, `engines.py`, `model.py`, `model_interface.py`, `trainer.py`) are
entangled with the rest of the model, so **do not copy whole files**. Instead port symbol-by-symbol
using §3. Copy the two new files (`moe_diagnostics.py`, `moe_test.py`) wholesale. This is the most
reliable path when the target branch has diverged.

### Option B — cherry-pick commits

The MoE work landed across many commits on `wm/dev/MoE_v2` (interleaved with config/eval commits and
rebase artifacts), so a clean cherry-pick is unlikely without conflicts. If you try it, the
implementation-bearing commits are, oldest→newest:

```
ebccdd1f  Add MoE with spatial routing and usage docs/example
baef1c91  Improving and documenting MoE in better way
7d1529af  applying the first fix to fsdp2 issue
0a339cc6  hotfix: spatial routing
2d9a1f46  Fix embedding sharding issue
e2555cb6  fix the checkpoint deadlock
8ce08ce2  importing and updating MoE implementations
9693ebad  fixing sharding issues with spatial weights
e07b764a  applying old MoE implementation
# + the uncommitted "modern redesign" (init fix, shared expert, bias balancing, diagnostics)
```

Better than cherry-picking: generate one patch of just the MoE-relevant files against the target's
merge base and apply it, then reconcile the shared files by hand using §3 as the checklist:

```bash
# on wm/dev/MoE_v2, produce a reviewable patch of the MoE files
git diff <merge-base-with-target> -- \
  src/weathergen/model/layers.py src/weathergen/model/engines.py \
  src/weathergen/model/model.py src/weathergen/model/model_interface.py \
  src/weathergen/train/trainer.py config/default_config.yml > moe.patch
# copy the two standalone files directly
git show wm/dev/MoE_v2:src/weathergen/model/moe_diagnostics.py > <target>/src/weathergen/model/moe_diagnostics.py
git show wm/dev/MoE_v2:src/weathergen/model/moe_test.py       > <target>/src/weathergen/model/moe_test.py
```

Whichever option you use, §4 is the correctness checklist you must satisfy, and §5–§6 are how to
configure and verify.

---

## 3. Symbol-by-symbol port (the shared files)

### 3.1 `layers.py` — copy these classes/functions verbatim
- `LoadBalancingLoss`, `RouterZLoss` — auxiliary losses.
- `_select_top_k(router_probs, router_logits, top_k, expert_bias)` — top-k selection; when
  `expert_bias` is given, selection uses `logits + bias` while gate weights come from the *unbiased*
  softmax. Both routers call it.
- `MoERouter`, `SpatialMoERouter` — routers. Both `forward`s take
  `(x, position_ids=None, token_mask=None, expert_bias=None)` and return
  `(router_probs, expert_indices, expert_weights, router_logits)`.
- `MoEBlock` — the block. Prerequisite: an `MLP` (or any FFN) whose `forward` accepts an optional
  per-token `aux` tensor if you use AdaLN conditioning, and that supports `with_residual=False`
  (the block owns the residual). Key `MoEBlock` behaviours to preserve:
  - **Output buffer is `torch.zeros_like(x_flat)` ⇒ expert `dim_out` must equal `dim_in`.**
  - Shared expert(s): `self.shared_expert` (an `nn.ModuleList` or `None`), applied densely to every
    token and added to the routed output *before* the single residual.
  - Balancing: `self.expert_bias` persistent buffer; forward records `self.last_expert_load`
    (detached) but does **not** mutate the bias; `update_expert_bias()` does the mutation.
  - Idle experts are called with a zero-weight dummy token (DDP autograd-graph safety).

### 3.2 `engines.py`
- Copy `MoEBlockConfig` (dataclass), `_get_moe_block_config`, `_moe_block_enabled`,
  `_moe_forward_wrapper_with_aux` / `_moe_forward_wrapper_no_aux`, `_build_global_position_ids`.
- In the engine that owns the FFN (`ForecastingEngine`, and optionally `GlobalAssimilationEngine`):
  - **`__init__`:** where the code appends an `MLP` for block `i`, branch on
    `_moe_block_enabled(cf, "fe_use_moe", "fe_moe_blocks", i)`; if true append a `MoEBlock` built
    from `_get_moe_block_config(cf, "fe_moe", <mlp_hidden_factor>)`, passing all fields through
    (including `num_shared_experts`, `balance_mode`, `bias_update_rate`).
  - **`forward`:** where the code runs the MLP under `checkpoint(...)`, add a branch
    `elif isinstance(block, MoEBlock):` that calls
    `checkpoint(_moe_forward_wrapper_with_aux, block, tokens, aux, None, None, use_reentrant=False)`.
    The aux loss is *not* returned through the checkpoint — it is read off the block later.
  - Add `initialize_moe_position_ids` and `initialize_spatial_routers` methods to the engine.
- **The router-init fix (critical).** If the target engine applies a near-identity final init to its
  blocks (`ForecastingEngine` does: `for block in self.fe_blocks: block.apply(init_weights_final)`
  with `std=0.001`), you **must** skip the `MoEBlock`'s router:

  ```python
  for block in self.fe_blocks:
      if isinstance(block, MoEBlock):
          for expert in block.experts:
              expert.apply(init_weights_final)
          if getattr(block, "shared_expert", None) is not None:
              block.shared_expert.apply(init_weights_final)
      else:
          block.apply(init_weights_final)
  ```

  Applying `std=0.001` to the router zeroes it → near-uniform softmax → random routing → experts
  never specialise → **MoE behaves identically to dense.** This was the primary reason MoE showed
  no gain. (An engine with no such init loop, e.g. `GlobalAssimilationEngine`, needs no change.)

### 3.3 `model.py`
- Add `initialize_spatial_routers(self)`: if any engine has `*_moe_use_spatial_routing`, compute
  HEALPix `theta, phi` via `pix2ang(nside, arange(num_cells), nest=True)` and pass to each engine's
  `initialize_spatial_routers`; always finish by calling `self.initialize_moe_position_ids()`.
- Add `initialize_moe_position_ids(self)`: fan out to each engine's `initialize_moe_position_ids`.

### 3.4 `model_interface.py`
- Add `MoEBlock` to `modules_to_shard` and keep the **descendant-based double-shard guard** (shard a
  `MoEBlock` as a unit; skip its descendant experts, incl. `shared_expert`, so they are not
  double-sharded).
- After building/loading, call `initialize_spatial_routers()` (fresh run) or
  `initialize_moe_position_ids()` (continue/load) — the position-id buffers are **non-persistent**
  and must be recomputed after every checkpoint load, or the spatial router reads a stale/empty
  buffer.

### 3.5 `trainer.py` — the per-step lifecycle (order matters)
1. **Before `model(...)`:** walk `model.modules()`, call `block.reset_aux_loss()` on each `MoEBlock`.
2. **After `compute_loss`, before `backward`:** `loss = loss + self._collect_moe_aux_losses()` and
   `self._record_moe_aux_loss(...)` (adds the `moe_router` term to the loss breakdown).
3. **After `backward()`:** `self._update_moe_expert_bias()` — walk blocks, call
   `update_expert_bias()`. **This must be after backward**, never inside the forward (see §4).
4. **At metrics cadence, rank 0:** `self._log_moe_diagnostics(TRAIN)` →
   `collect_moe_diagnostics(model)`.

Import `MoEBlock` and `collect_moe_diagnostics`. Gate the diagnostics with a flag such as
`fe_moe_diag` (default on when `fe_use_moe`).

---

## 4. Correctness invariants (do not break these)

1. **Router init must NOT be near-zero.** See §3.2. This is the single highest-leverage detail.
2. **Aux loss travels via a side-effect, not the checkpoint return.** `MoEBlock` stores
   `last_aux_loss`; the trainer reads it after the forward. Non-reentrant activation checkpointing
   runs the forward with grad enabled, so `last_aux_loss` keeps its `grad_fn` and the router is
   trained. Never try to return the aux loss through `torch.utils.checkpoint`.
3. **The bias update runs AFTER `backward()`, outside the checkpointed forward.** Non-reentrant
   checkpoint re-executes the forward during backward. If you mutate `expert_bias` inside `forward`,
   it is applied twice per step *and* the recomputed routing diverges from the original forward
   (corrupting gradients). The block only records `last_expert_load` in forward; the trainer applies
   the nudge once, post-backward.
4. **`dim_out == dim_in` for experts.** The block's output buffer is `zeros_like` the input. For a
   tapered-dim stack, guard with an assert.
5. **FSDP: shard `MoEBlock` as a unit, not its experts.** The descendant guard handles both
   `experts` and `shared_expert`.
6. **Position-id buffers are non-persistent** → recompute after every checkpoint load.
7. **Idle experts get a zero-weight dummy call** in training so their params stay in the autograd
   graph (DDP `find_unused_parameters` / FSDP correctness).
8. **`reset_aux_loss` once per step; aux accumulates across rollout steps.** With multi-step
   forecasting the aux loss sums over steps, so its effective weight scales with `num_steps` — keep
   `load_balance_weight` modest.

---

## 5. Configuration reference

All keys are flat and prefixed per engine: `fe_moe_*` (forecasting engine) and `ae_global_moe_*`
(global assimilation engine). `_get_moe_block_config(cf, prefix, default_hidden_factor)` reads them,
so the two engines share the exact same schema. Defaults below are the code defaults (used when a
key is absent).

| Key (suffix) | Default | Meaning |
|---|---|---|
| `use_moe` (`fe_use_moe`) | `False` | Master on/off for the engine. |
| `moe_blocks` | `"all"` | `"all"` or an explicit list of block indices to make MoE, e.g. `[0, 5, 10, 15]`. |
| `moe_num_experts` | `8` | Number of routed experts. |
| `moe_top_k` | `2` | Experts each token is routed to. |
| `moe_capacity_factor` | `1.25` | Per-expert capacity as a fraction of the balanced load; `null`/`0` disables token dropping. |
| `moe_load_balance_weight` | `0.01` | Switch-transformer aux-loss weight. **`0.01` is usually too high (kills specialisation); `0.0005` is a good starting point.** |
| `moe_jitter_noise` | `0.0` | Std of additive Gaussian noise on router logits (training only). |
| `moe_expert_hidden_factor` | engine MLP factor | Expert hidden width multiplier. Set so `top_k * this == dense_hidden_factor` to match dense active params. |
| `moe_router_bias` | `False` | Bias on the router projection. |
| `moe_renormalize_gates` | `True` | Renormalise gate weights per token after capacity dropping. |
| `moe_use_spatial_routing` | `False` | Use `SpatialMoERouter` (HEALPix-aware). Only valid for HEALPix-cell tokens (encoder/forecast latents), **not** varlen decoder points. |
| `moe_position_embed_dim` | `128` | Spatial embedding dim (spatial routing only). |
| `moe_router_hidden_dim` | `0` | `0` = linear router; `>0` = 2-layer MLP router with this hidden size. |
| `moe_router_z_loss_weight` | `0.0` | Router z-loss weight (0 = off). |
| `moe_num_shared_experts` | `0` | **Modern.** Always-on dense expert(s) added to every token; `>= 1` guarantees MoE ≥ dense. |
| `moe_balance_mode` | `"aux"` | **Modern.** `"aux"` = load-balancing loss; `"bias"` = aux-loss-free bias balancing. |
| `moe_bias_update_rate` | `0.001` | **Modern.** Per-step selection-bias nudge (only in `"bias"` mode). |
| `moe_diag` | `True`* | **Modern.** Log routing diagnostics. (*Trainer gates on this; add the key.) |
| `moe_debug` | `False` | Periodic `MoE[...]` stderr diagnostic line. |
| `moe_debug_interval` | `100` | Log every N-th forward (rank 0). |
| `moe_debug_top_experts` | `3` | Experts shown in the debug line. |

### Reading the diagnostics
Logged per block and averaged as `moe.<block>.<metric>` / `moe.mean.<metric>`:
- `entropy` ∈ [0,1] — normalised entropy of per-expert load; **1 = uniform, →0 = collapsed**.
- `load_cv` — coefficient of variation of load; 0 = perfectly balanced.
- `dead_expert_frac` — fraction of experts receiving ~no tokens (want ~0).
- `mean_top1_gate` — top-1 gate confidence (near `1/top_k` = indecisive).
- `bias_spread` — range of `expert_bias` (bias mode only).

A healthy MoE run shows `entropy` clearly below 1 but not collapsed, `dead_expert_frac ≈ 0`, and
bounded `load_cv`. If `entropy ≈ 1` and nothing specialises, suspect the router-init bug (§3.2) or
too-high `load_balance_weight`.

---

## 6. Writing an MoE config (worked example)

MoE keys layer on top of any base config. A minimal FE-MoE overlay:

```yaml
fe_use_moe: True
fe_moe_blocks: [0, 5, 10, 15]      # subset of the fe_num_blocks stack
fe_moe_num_experts: 8
fe_moe_top_k: 2
fe_moe_expert_hidden_factor: 1.0    # top_k*1.0 == dense hidden factor 2 => active params match dense
fe_moe_capacity_factor: null        # no token dropping
fe_moe_load_balance_weight: 0.0005  # weak; NOT 0.01
fe_moe_router_hidden_dim: 0         # linear router
fe_moe_num_shared_experts: 1        # >= dense guarantee
fe_moe_balance_mode: "aux"          # or "bias" (then set fe_moe_load_balance_weight: 0)
fe_moe_diag: True
```

Guidance:
- **Match active params to dense** for a clean A/B: `top_k * expert_hidden_factor == mlp_hidden_factor`.
- **Start with `balance_mode: "aux"`, `load_balance_weight: 0.0005`, `router_hidden_dim: 0`,
  `capacity_factor: null`.** These are the settings that first beat the dense baseline.
- **Add a shared expert** (`num_shared_experts: 1`) so the block cannot underperform dense.
- **Content vs spatial routing:** use content routing (`use_spatial_routing: False`) unless the
  tokens are HEALPix-cell-indexed and you want a geographic prior. Spatial routing needs the
  model-level `initialize_spatial_routers` (HEALPix coords) to have run.
- **Multi-node scaling:** LR and AdamW betas auto-scale by `world_size` (LR ∝ `sqrt(bs*world_size)`;
  `beta_eff = 1 - world_size*bs*(1-beta_cfg)`), so the *base* config values are node-count-agnostic —
  do not hand-inflate them. Retune the base betas only if you want a specific effective momentum at a
  given node count (see `config/config_moe/forecasting_moe_1node.yml` for a single-node example).

A complete, runnable example is `config/config_moe/forecasting_moe_1node.yml` (an A/B against
`config/config_forecasting.yml`).

---

## 7. Verification after porting

1. `uv run pytest src/weathergen/model/moe_test.py` — the block/router/diagnostics tests run on CPU;
   the two `ForecastingEngine`-level tests self-skip without `flash_attn` (run them on GPU).
2. `./scripts/actions.sh lint-check` and `type-check` — expect only the target branch's pre-existing
   findings in the changed files.
3. One training step with an MoE config: confirm a `moe_router` term appears in the loss breakdown
   (proves aux collection) and `moe.*` diagnostics appear at the metrics cadence.
4. A short GPU run with `with_fsdp: True`: confirm `MoEBlock`s shard, forward/backward is clean, and
   diagnostics show non-uniform, non-collapsed routing (entropy < 1, no dead experts). If routing is
   uniform and metrics match dense, re-check §3.2 (router init) and `load_balance_weight`.
