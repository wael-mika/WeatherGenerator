# MoE Implementation Overview

## Architecture at a glance

MoE replaces individual **MLP blocks** inside the Global Assimilation Engine (`ae_global_blocks`) and/or the Forecasting Engine (`fe_blocks`) with sparse `MoEBlock` modules. Attention blocks and LayerNorm are never touched. Everything else in the model is unchanged.

---

## Key files and their roles

| File | Role |
|---|---|
| `src/weathergen/model/layers.py` | `MoERouter`, `SpatialMoERouter`, `MoEBlock`, `LoadBalancingLoss` |
| `src/weathergen/model/engines.py` | Config helpers + forward integration |
| `src/weathergen/model/model.py` | Spatial init, position ID reset after checkpointing |
| `src/weathergen/model/model_interface.py` | FSDP2-aware sharding, post-load init |
| `src/weathergen/train/trainer.py` | Aux loss collection and recording |

---

## `MoEBlock` (layers.py)

The core module. Replaces a single MLP in the transformer stack.

**Forward pass:**
1. Flatten input to `[N, D]` (N = total tokens across the batch)
2. Router scores all N tokens over E experts → `router_probs [N, E]`, picks top-k → `expert_indices [N, K]`, `expert_weights [N, K]`
3. Build assignment edges `(token_idx, expert_idx, gate_weight)`
4. Enforce **per-expert capacity** — drop lowest-gate edges if an expert is over-subscribed (`capacity_factor * N / E` slots per expert)
5. Optionally re-normalise the gate weights per token after dropping
6. Dispatch each surviving edge to its expert; accumulate weighted outputs with `index_add_`
7. Add residual (`output += x` if `with_residual=True`)
8. During training, compute and store `last_aux_loss` (load-balancing loss) — **not returned**, stored on the block

The block stores `last_aux_loss`, `last_expert_indices`, `last_expert_weights` for the trainer and debugging.

---

## Two router variants

**`MoERouter`** — content-only routing. Linear `[D → E]` → softmax → topk.

**`SpatialMoERouter`** — spatial-aware routing. Concatenates a learned position embedding `[D + P → E]` before the linear projection. The embedding table is a **`register_buffer`** (not `nn.Embedding`/`nn.Parameter`) to avoid an FSDP2 DTensor all-gather on every forward call. It is initialised with sinusoidal `(theta, phi)` HEALPix coordinates via `initialize_from_coordinates()`.

Each `MoEBlock` selects one router type at construction time via `use_spatial_router`.

---

## Config helpers (engines.py)

Three helpers keep MoE wiring out of `__init__`:

```python
_moe_block_enabled(cf, use_key, blocks_key, i)
# Returns True if block i should be MoE.
# use_key = "fe_use_moe" | "ae_global_use_moe"
# blocks_key value = "all" or [0, 7, 15]

_get_moe_block_config(cf, prefix, default_hidden_factor)
# Reads all fe_moe_* or ae_global_moe_* keys into a MoEBlockConfig dataclass.

_build_global_position_ids(num_cells, num_queries, ...)
# Builds [T] position ID tensor: -1 for special tokens, cell_idx for query tokens.
```

---

## How MoE blocks are inserted

In both `GlobalAssimilationEngine.__init__` and `ForecastingEngine.__init__`, after each attention block, the engine adds either an `MoEBlock` or a plain `MLP` depending on `_moe_block_enabled()`:

```python
if _moe_block_enabled(cf, "fe_use_moe", "fe_moe_blocks", i):
    moe_cfg = _get_moe_block_config(cf, "fe_moe", default_hidden_factor=2.0)
    self.fe_blocks.append(MoEBlock(
        expert_fn=lambda: MLP(..., with_residual=False, ...),  # experts have NO residual
        with_residual=True,  # residual lives on MoEBlock itself
        use_spatial_router=moe_cfg.use_spatial_routing,
        ...
    ))
else:
    self.fe_blocks.append(MLP(..., with_residual=True, ...))
```

Note: expert MLPs have `with_residual=False`; the `MoEBlock` wrapper owns the residual.

---

## Forward pass integration (engines.py)

Both `GlobalAssimilationEngine.forward` and `ForecastingEngine.forward` follow the same pattern — the only difference from the dense (non-MoE) version is the `MoEBlock` branch:

```python
def forward(self, tokens, [fstep,] coords=None):
    aux_info = None
    for block in self.[ae_global/fe]_blocks:
        if isinstance(block, MoEBlock):
            tokens = checkpoint(_moe_forward_wrapper_with_aux,
                                block, tokens, aux_info, None, None,
                                use_reentrant=False)
        elif isinstance(block, LayerNorm):
            tokens = checkpoint(block, tokens, use_reentrant=False)
        else:  # attention or MLP
            tokens = checkpoint(block, tokens, coords, aux_info, use_reentrant=False)
    return tokens
```

**Important**: `coords` must always be accepted and passed through to the non-MoE blocks (attention heads use it for 2D RoPE). MoE blocks do not use `coords`.

The `_moe_forward_wrapper_with_aux` wrapper is needed because `torch.utils.checkpoint` expects a function returning a single tensor, but `MoEBlock.forward` returns `(output, aux_loss)`. The wrapper discards the aux loss (it is already stored on the block as `last_aux_loss`).

---

## Aux loss lifecycle (trainer.py)

Every training step:
1. **Before forward**: `module.reset_aux_loss()` on every `MoEBlock` — clears `last_aux_loss`
2. **Forward runs**: each `MoEBlock` sets `self.last_aux_loss` internally
3. **After loss computation**: `_collect_moe_aux_losses()` walks all `MoEBlock`s and sums their `last_aux_loss` values → added to the total loss
4. **Recorded** under `"moe_router"` key in the loss history for monitoring

Validation: aux loss is not applied (blocks are in eval mode → `last_aux_loss` is `None`).

---

## Spatial router initialisation lifecycle (model.py / model_interface.py)

This is the trickiest part. Two things must happen after construction and after every checkpoint load:

| Event | Method | What it does |
|---|---|---|
| Fresh run only | `model.initialize_spatial_routers()` | Calls `router.initialize_from_coordinates(theta, phi)` on every spatial `MoEBlock`; also calls `initialize_moe_position_ids()` |
| Fresh run + every checkpoint load | `model.initialize_moe_position_ids()` | Re-registers `position_ids` buffer on every `MoEBlock` (non-persistent buffer, not saved in checkpoint) |

`model_interface.py` calls these in the right places after FSDP sharding and checkpoint loading.

---

## FSDP2 sharding (model_interface.py)

A `_shard_modules(root, kwargs)` helper prevents **double-sharding**: since `MoEBlock` contains `MLP` experts, naive code would shard the experts first, then try to shard the `MoEBlock` again. The helper tracks already-sharded module IDs and skips children of already-sharded modules.

---

## Known pitfalls

- **`nn.Embedding` + FSDP2 = deadlock**: FSDP2 does not unwrap `F.embedding` output the way it does for `F.linear`. Using `nn.Embedding` inside an FSDP2-sharded module causes an implicit `full_tensor()` all-gather on every forward. Fix: use `register_buffer()` for the embedding table.
- **`tensor.any()` in forward hot-path**: triggers GPU-CPU sync. Use elementwise multiply instead.
- **Position IDs are non-persistent buffers**: they are NOT saved in checkpoints and must be recomputed after every load via `initialize_moe_position_ids()`.
- **`@dataclasses.dataclass(init=False)` on `ModelOutput`**: PyTorch DDP's `_find_tensors()` only recurses into `Tensor`, `list/tuple`, `dict`, and `dataclass`. Without this decorator, DDP can't discover output tensors and `find_unused_parameters=True` breaks.

---

## Config keys reference

```yaml
# FE MoE (prefix: fe_moe_*)
fe_use_moe: True                   # master switch
fe_moe_blocks: [0, 7, 15]          # or "all"
fe_moe_num_experts: 4
fe_moe_top_k: 2
fe_moe_capacity_factor: 1.25
fe_moe_load_balance_weight: 0.001
fe_moe_jitter_noise: 0.01
fe_moe_expert_hidden_factor: 1.0   # per-expert MLP hidden size factor
fe_moe_use_spatial_routing: True
fe_moe_position_embed_dim: 64
fe_moe_debug: False                # periodic routing stats to log
fe_moe_debug_interval: 100
fe_moe_debug_top_experts: 3

# AE-global MoE (same pattern, prefix: ae_global_moe_*)
ae_global_use_moe: False
ae_global_moe_blocks: "all"
ae_global_moe_num_experts: 4
# ... same keys with ae_global_moe_ prefix
```

### Block index mapping

Config `fe_moe_blocks: [0, 3, 7]` maps to `fe_blocks.[1, 7, 16]` in module names because `fe_blocks` interleaves attention + MLP/MoE + optional LayerNorm blocks:
- Block 0: attention[0], **MoE[0]**
- Block 1: attention[1], MLP[1]
- ...

### Parameter fairness note

With `top_k=2`, `num_experts=4`, `expert_hidden_factor=1.0`, the 2 active experts together have 2x the hidden size of a single expert — matching a dense MLP with `hidden_factor=2`. Parameter count is 4x larger but FLOPs per token are the same.
