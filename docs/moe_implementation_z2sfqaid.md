# Mixture of Experts (MoE) Implementation — Branch `z2sfqaid`

> This document fully describes the MoE implementation from the `z2sfqaid` experiment run, which produced the best MoE results. It is intended as a complete specification for rebuilding this implementation.

---

## 1. File Map

| File | Role |
|------|------|
| `src/weathergen/model/layers.py` | **Core**: `MoEBlock`, `MoERouter`, `SpatialMoERouter`, `LoadBalancingLoss`, `MLP` |
| `src/weathergen/model/engines.py` | MoE blocks inserted into `GlobalAssimilationEngine` and `ForecastingEngine` |
| `src/weathergen/model/model.py` | Model-level HEALPix coordinate injection into spatial routers |
| `src/weathergen/model/moe_diagnostics.py` | Spatial coherence analysis (routing quality metrics) |
| `src/weathergen/model/router_diagnostics.py` | Router internals diagnostics (entropy, balance, position contribution) |
| `src/weathergen/train/trainer.py` | Training: aux loss collection, FSDP sharding strategy |
| `config/default_config.yml` | All MoE hyperparameters (lines 37–84) |

---

## 2. Architecture Overview

**Type**: Standard sparse top-k MoE replacing the FFN block in transformer layers.

| Property | Value |
|----------|-------|
| Routing | Linear projection → Softmax → Top-k |
| Expert type | 2-layer MLP (GELU activation) |
| Expert hidden factor | 1.0 (vs 2.0 for baseline MLP — experts are half-size) |
| Top-k | 2 (each token routed to 2 experts) |
| Num experts (global engine) | 4 |
| Num experts (forecast engine) | 8 |
| Load balancing | Switch Transformers auxiliary loss (Fedus et al., 2021) |
| Jitter noise | 0.01 (global engine), 0.0 (forecast engine) |
| Residual | Managed by `MoEBlock` (not inside each expert) |
| Capacity factor | Defined (1.25) but not enforced — no token dropping |

---

## 3. Core Classes

### 3.1 `LoadBalancingLoss` (`layers.py:101–147`)

Implements the Switch Transformers load balancing loss. Encourages uniform routing across experts.

**Formula**: `L = num_experts * Σ_i (f_i * P_i)`

Where:
- `f_i` = fraction of tokens actually routed to expert `i`
- `P_i` = mean softmax probability assigned to expert `i` across all tokens

```python
class LoadBalancingLoss(torch.nn.Module):
    def __init__(self, num_experts: int, weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight

    def forward(
        self,
        router_probs: torch.Tensor,   # [batch, seq_len, num_experts]
        expert_mask: torch.Tensor,    # [batch, seq_len, num_experts]  (binary)
    ) -> torch.Tensor:
        mean_prob = router_probs.mean(dim=[0, 1])         # [num_experts]
        mean_assignment = expert_mask.float().mean(dim=[0, 1])  # [num_experts]
        loss = self.num_experts * (mean_prob * mean_assignment).sum()
        return self.weight * loss
```

**Config keys**:
```yaml
ae_global_moe_load_balance_weight: 0        # disabled in this run
fe_moe_load_balance_weight: 0.0005          # very weak in forecasting
```

---

### 3.2 `MoERouter` (`layers.py:150–212`)

Basic linear router with top-k selection and optional exploration noise.

```python
class MoERouter(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        router_bias: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.router_weights = nn.Linear(dim_in, num_experts, bias=router_bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, dim_in]
        router_logits = self.router_weights(x)   # [batch, seq_len, num_experts]

        # Exploration noise (training only)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

        router_probs = F.softmax(router_logits, dim=-1)   # [batch, seq_len, num_experts]

        # Top-k selection
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # [batch, seq_len, top_k]

        # Re-normalize selected weights to sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return router_probs, expert_indices, expert_weights
```

**Returns**:
- `router_probs`: `[batch, seq_len, num_experts]` — full softmax distribution (used for aux loss)
- `expert_indices`: `[batch, seq_len, top_k]` — which experts each token uses
- `expert_weights`: `[batch, seq_len, top_k]` — re-normalized combination weights

---

### 3.3 `SpatialMoERouter` (`layers.py:215–362`)

Extended router that concatenates per-cell position embeddings to the token features before routing. Enables geographic coherence: nearby HEALPix cells tend to route to the same expert.

#### Constructor

```python
class SpatialMoERouter(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        num_experts: int,
        num_positions: int,           # Total HEALPix cells
        position_embed_dim: int = 128,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        router_bias: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.position_embed = nn.Embedding(num_positions, position_embed_dim)

        # Router sees features + position embedding
        self.router_weights = nn.Linear(
            dim_in + position_embed_dim, num_experts, bias=router_bias
        )
```

#### Coordinate Initialization

Called once at model init with HEALPix `theta` (colatitude) and `phi` (azimuth):

```python
def initialize_from_coordinates(self, theta: torch.Tensor, phi: torch.Tensor):
    # theta: [num_positions]  colatitude  0=North Pole, π=South Pole
    # phi:   [num_positions]  azimuth     0 to 2π

    half_dim = self.position_embed.embedding_dim // 2
    embeddings = torch.zeros(len(theta), self.position_embed.embedding_dim)

    # Frequency bands for theta (latitude)
    freq_theta = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
    # Frequency bands for phi (longitude)
    freq_phi   = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))

    # Sinusoidal encoding
    embeddings[:, 0:half_dim:2] = torch.sin(theta.unsqueeze(1) * freq_theta)
    embeddings[:, 1:half_dim:2] = torch.cos(theta.unsqueeze(1) * freq_theta)
    embeddings[:, half_dim+0::2] = torch.sin(phi.unsqueeze(1) * freq_phi)
    embeddings[:, half_dim+1::2] = torch.cos(phi.unsqueeze(1) * freq_phi)

    with torch.no_grad():
        self.position_embed.weight.copy_(embeddings)
```

#### Forward Pass

```python
def forward(
    self, x: torch.Tensor, position_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x:            [batch, seq_len, dim_in]
    # position_ids: [seq_len]  integer indices into position_embed

    pos_embed = self.position_embed(position_ids)        # [seq_len, pos_dim]
    pos_embed = pos_embed.unsqueeze(0).expand(x.shape[0], -1, -1)  # [batch, seq_len, pos_dim]

    x_with_pos = torch.cat([x, pos_embed], dim=-1)      # [batch, seq_len, dim_in + pos_dim]
    router_logits = self.router_weights(x_with_pos)

    if self.training and self.jitter_noise > 0:
        router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

    router_probs = F.softmax(router_logits, dim=-1)
    expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

    return router_probs, expert_indices, expert_weights
```

---

### 3.4 `MoEBlock` (`layers.py:365–627`)

The top-level MoE module: creates all experts, owns the router and load-balancing loss, handles the dispatch/combine loop, and manages the residual connection.

#### Constructor

```python
class MoEBlock(torch.nn.Module):
    def __init__(
        self,
        expert_fn: Callable[[], nn.Module],   # Factory function, called once per expert
        dim_in: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,        # Defined but not enforced (no token dropping)
        jitter_noise: float = 0.0,
        router_bias: bool = False,
        load_balance_weight: float = 0.01,
        with_residual: bool = True,
        use_spatial_router: bool = False,
        num_positions: int = None,
        position_embed_dim: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.with_residual = with_residual

        # Instantiate experts independently
        self.experts = nn.ModuleList([expert_fn() for _ in range(num_experts)])

        if use_spatial_router:
            self.router = SpatialMoERouter(
                dim_in=dim_in,
                num_experts=num_experts,
                num_positions=num_positions,
                position_embed_dim=position_embed_dim,
                top_k=top_k,
                jitter_noise=jitter_noise,
                router_bias=router_bias,
            )
        else:
            self.router = MoERouter(
                dim_in=dim_in,
                num_experts=num_experts,
                top_k=top_k,
                jitter_noise=jitter_noise,
                router_bias=router_bias,
            )

        self.load_balance_loss = LoadBalancingLoss(
            num_experts=num_experts,
            weight=load_balance_weight,
        )
        self.last_aux_loss: Optional[torch.Tensor] = None

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self.last_aux_loss
```

#### Forward Pass

```python
def forward(self, *args) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    x = args[0]                                    # [batch, seq_len, dim]
    x_in = x                                       # save for residual
    batch_size, seq_len, dim = x.shape

    # Route tokens
    router_probs, expert_indices, expert_weights = self.router(x)
    # router_probs:    [batch, seq_len, num_experts]
    # expert_indices:  [batch, seq_len, top_k]
    # expert_weights:  [batch, seq_len, top_k]

    # Flatten batch and seq dims for dispatch
    x_flat             = x.view(-1, dim)                         # [N, dim]  N = batch*seq_len
    expert_indices_flat = expert_indices.view(-1, self.top_k)    # [N, top_k]
    expert_weights_flat = expert_weights.view(-1, self.top_k)    # [N, top_k]

    output = torch.zeros_like(x_flat)

    for expert_idx in range(self.num_experts):
        # Which tokens use this expert (in any of the top_k slots)?
        expert_mask   = (expert_indices_flat == expert_idx)      # [N, top_k]
        token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]  # [m]

        if len(token_indices) == 0:
            continue

        expert_input = x_flat[token_indices]  # [m, dim]

        # Run expert (pass auxiliary args if present, e.g. timestep)
        if len(args) > 1:
            expert_output = self.experts[expert_idx](*[expert_input] + list(args[1:]))
        else:
            expert_output = self.experts[expert_idx](expert_input)

        # Weighted accumulation across top_k positions
        for k in range(self.top_k):
            k_mask         = expert_mask[:, k]                           # [N]
            k_token_indices = k_mask.nonzero(as_tuple=True)[0]           # [p]

            if len(k_token_indices) == 0:
                continue

            # Map k_token_indices → rows inside expert_output
            expert_output_indices = torch.isin(
                token_indices, k_token_indices
            ).nonzero(as_tuple=True)[0]

            weights = expert_weights_flat[k_token_indices, k].unsqueeze(-1)  # [p, 1]
            output[k_token_indices] += weights * expert_output[expert_output_indices]

    output = output.view(batch_size, seq_len, dim)

    if self.with_residual:
        output = output + x_in

    # Auxiliary loss (training only)
    aux_loss = None
    if self.training:
        expert_mask_full = torch.zeros(
            batch_size, seq_len, self.num_experts, device=x.device
        )
        expert_mask_full.scatter_(2, expert_indices, 1.0)
        aux_loss = self.load_balance_loss(router_probs, expert_mask_full)
        self.last_aux_loss = aux_loss

    return output, aux_loss
```

---

### 3.5 Expert MLP Factory

Each expert is created by a factory lambda. The critical detail is the **50% hidden-size reduction** (`hidden_factor=1.0` instead of 2.0):

```python
# From GlobalAssimilationEngine (engines.py:328–336)
expert_fn = lambda: MLP(
    dim_in          = cf.ae_global_dim_embed,
    dim_out         = cf.ae_global_dim_embed,
    with_residual   = False,            # MoEBlock owns the residual
    dropout_rate    = cf.ae_global_dropout_rate,
    hidden_factor   = expert_hidden_factor,   # 1.0  (baseline is 2.0)
    norm_type       = cf.norm_type,
    norm_eps        = cf.mlp_norm_eps,
)
```

**Why `hidden_factor=1.0`**: With 4 experts each at 1.0× hidden size, the total FFN parameter budget is `4 × 1.0 = 4.0×` the embedding dim — comparable to `1 × 2.0 = 2.0×` but distributed. The intent is to double parameter count while keeping per-expert compute manageable.

---

## 4. Integration into Transformer Engines

### 4.1 GlobalAssimilationEngine (`engines.py:241–455`)

MoE blocks are inserted alternating with attention blocks. The decision is per-block-index:

```python
# In __init__, for each block index i:
use_moe    = getattr(cf, "ae_global_use_moe", False)
moe_blocks = getattr(cf, "ae_global_moe_blocks", "all")
should_use_moe = use_moe and (moe_blocks == "all" or i in moe_blocks)

if should_use_moe:
    self.ae_global_blocks.append(
        MoEBlock(
            expert_fn            = lambda: MLP(..., hidden_factor=expert_hidden_factor),
            dim_in               = cf.ae_global_dim_embed,
            num_experts          = cf.ae_global_moe_num_experts,    # 4
            top_k                = cf.ae_global_moe_top_k,          # 2
            load_balance_weight  = cf.ae_global_moe_load_balance_weight,  # 0
            jitter_noise         = cf.ae_global_moe_jitter_noise,   # 0.01
            with_residual        = True,
            use_spatial_router   = cf.ae_global_moe_use_spatial_routing,  # False
            num_positions        = num_healpix_cells,
            position_embed_dim   = cf.ae_global_moe_position_embed_dim,
        )
    )
else:
    self.ae_global_blocks.append(MLP(..., hidden_factor=2.0, with_residual=True))
```

**Forward** (inside activation checkpoint):
```python
for block in self.ae_global_blocks:
    if isinstance(block, MoEBlock):
        tokens, aux_loss = block(tokens)   # aux_loss stored in block.last_aux_loss
    elif isinstance(block, MLP):
        tokens = checkpoint(block, tokens, ...)
    else:  # Attention block
        tokens = checkpoint(block, tokens, ...)
```

### 4.2 ForecastingEngine (`engines.py:458–682`)

Identical logic but with **timestep conditioning**: both attention blocks and expert MLPs receive an auxiliary `fstep` scalar via `AdaLayerNorm`.

```python
# In __init__, expert factory passes dim_aux=1 to MLP
expert_fn = lambda: MLP(..., dim_aux=1)

# In forward:
fstep_tensor = torch.tensor([fstep], dtype=torch.float32, device=tokens.device)
for block in self.fe_blocks:
    if isinstance(block, MoEBlock):
        tokens, aux_loss = block(tokens, fstep_tensor)  # extra arg forwarded to experts
    elif isinstance(block, MLP):
        tokens = checkpoint(block, tokens, fstep_tensor, ...)
```

---

## 5. Model-Level Spatial Router Initialization (`model.py:333–350`)

After building all engines, if spatial routing is enabled, HEALPix coordinates are injected once:

```python
import healpy
import numpy as np

if getattr(cf, "ae_global_moe_use_spatial_routing", False) or \
   getattr(cf, "fe_moe_use_spatial_routing", False):

    nside = 2 ** self.healpix_level
    ipix  = np.arange(self.num_healpix_cells)
    theta, phi = healpy.pix2ang(nside, ipix, nest=True)

    theta_t = torch.from_numpy(np.array(theta)).float()
    phi_t   = torch.from_numpy(np.array(phi)).float()

    if getattr(cf, "ae_global_moe_use_spatial_routing", False):
        self.ae_global_engine.initialize_spatial_routers(theta_t, phi_t)

    if getattr(cf, "fe_moe_use_spatial_routing", False):
        self.forecast_engine.initialize_spatial_routers(theta_t, phi_t)
```

Each engine's `initialize_spatial_routers` method loops over its blocks and calls `router.initialize_from_coordinates(theta, phi)` on every `SpatialMoERouter`.

---

## 6. Distributed Training: FSDP Sharding (`trainer.py:137–183`)

### Sharding Strategy

```python
modules_to_shard = (
    MLP,
    MoEBlock,                           # sharded as a unit
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHead,
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHeadVarlen,
)

def should_shard_module(module, parent_path):
    if not isinstance(module, modules_to_shard):
        return False
    # Expert MLPs live inside MoEBlock — don't double-shard them
    if isinstance(module, MLP) and '.experts.' in parent_path:
        return False
    return True
```

**Key insight**: `MoEBlock` itself is sharded as one unit. The individual expert `MLP` instances nested inside it are **not** individually sharded to avoid FSDP nesting overhead.

### Auxiliary Loss Collection (`trainer.py:1222–1248`)

```python
def _collect_moe_aux_losses(self) -> Optional[torch.Tensor]:
    moe_losses = []
    model = self.model.module if hasattr(self.model, 'module') else self.model

    for module in model.modules():
        if isinstance(module, MoEBlock):
            aux_loss = module.get_aux_loss()
            if aux_loss is not None:
                moe_losses.append(aux_loss)

    return sum(moe_losses) if moe_losses else None
```

**Called in training loop** (`trainer.py:650–654`):
```python
moe_aux_loss = self._collect_moe_aux_losses()
if moe_aux_loss is not None:
    loss_values.loss += moe_aux_loss
    self.moe_aux_loss_hist.append(moe_aux_loss.item())
```

Load balancing loss is computed locally per rank using local token batches. FSDP handles gradient synchronization in the backward pass automatically — no explicit `all_reduce` needed for the aux loss.

---

## 7. Configuration Reference (`config/default_config.yml`, lines 37–84)

### Global Assimilation Engine

```yaml
ae_global_use_moe: False                      # Master on/off switch
ae_global_moe_blocks: [0, 3]                  # Which block indices use MoE
ae_global_moe_num_experts: 4                  # Number of experts
ae_global_moe_top_k: 2                        # Tokens routed to k experts
ae_global_moe_load_balance_weight: 0          # Load balance loss weight (0 = disabled)
ae_global_moe_jitter_noise: 0.01              # Noise std for router exploration
ae_global_moe_expert_hidden_factor: 1.0       # Expert hidden size (1.0 vs 2.0 baseline)
ae_global_moe_use_spatial_routing: False      # HEALPix-aware routing
ae_global_moe_position_embed_dim: 128         # Spatial embedding dimension
```

### Forecasting Engine

```yaml
fe_use_moe: True                              # Master on/off switch
fe_moe_blocks: [0, 3]                         # Which block indices use MoE
fe_moe_num_experts: 8                         # Number of experts (2× global engine)
fe_moe_top_k: 2
fe_moe_load_balance_weight: 0.0005            # Very weak (0.0005)
fe_moe_jitter_noise: 0.0                      # No exploration noise
fe_moe_expert_hidden_factor: 1.0
fe_moe_use_spatial_routing: False
fe_moe_position_embed_dim: 128
```

---

## 8. Special Techniques

### 8.1 Jitter Noise (Router Exploration)

```python
if self.training and self.jitter_noise > 0:
    router_logits += torch.randn_like(router_logits) * self.jitter_noise
```

- Applied **only during training**
- Prevents routing collapse (all tokens to one expert)
- Used in global engine (`0.01`), disabled in forecasting engine (`0.0`)

### 8.2 Weight Re-normalization After Top-k

```python
expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
```

After selecting top-k experts, their weights are re-normalized to sum to 1. This means the expert combination is a proper convex combination regardless of how many experts are dropped.

### 8.3 Expert Residual Ownership

**Critical design choice**: `with_residual=False` is passed to each expert `MLP`; instead `MoEBlock` itself has `with_residual=True` and adds the residual after combining expert outputs:

```python
output = output + x_in   # x_in saved before routing
```

This ensures the residual is applied once over the combined output, not separately per expert.

### 8.4 Auxiliary Conditioning (AdaLayerNorm)

In the forecasting engine, each expert `MLP` is built with `dim_aux=1`. Inside `MLP`, this triggers `AdaLayerNorm`:

```python
# AdaLayerNorm (norms.py:64–89)
def forward(self, x, aux):
    scale, shift = self.mlp(aux).chunk(2, dim=-1)
    return LayerNorm(x) * (1 + scale) + shift
```

The timestep scalar `fstep` flows from `ForecastingEngine.forward()` through `MoEBlock.forward(*args)` (via `args[1:]`) into each expert.

### 8.5 Capacity Factor (Not Enforced)

`capacity_factor=1.25` is stored but no token-dropping or overflow logic is implemented. Every token is processed regardless of load. This simplifies implementation at the cost of variable per-expert batch sizes.

---

## 9. Diagnostics

### `moe_diagnostics.py` — Spatial Coherence

```python
def analyze_routing_spatial_coherence(expert_indices, neighbor_structure=None):
    # Returns:
    # - sequential_transition_rate: fraction of consecutive cells using different experts
    # - avg_run_length:  mean contiguous region using same expert
    # - neighbor_agreement_rate: fraction of neighbors using same expert
```

Interpretation: lower `sequential_transition_rate` = better geographic coherence.

### `router_diagnostics.py` — Router Internals

```python
def analyze_router_internals(moe_block, sample_input):
    # Returns:
    # - position_contribution (KL divergence)   — is pos embedding helping?
    # - position_embedding_variance             — are embeddings diverse?
    # - router_weight_norms                     — feature vs position importance
    # - router_entropy                          — diversity of routing decisions
    # - routing_confidence                      — certainty in top expert
    # - load_balance_loss                       — deviation from uniform usage
```

---

## 10. Summary of Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Expert size | `hidden_factor=1.0` | 50% smaller per expert; budget shared across 4–8 experts |
| Top-k | 2 | Standard; each token gets blended output from 2 experts |
| Load balancing weight (global) | 0 | Disabled — was causing instability (high router loss) |
| Load balancing weight (forecast) | 0.0005 | Very weak signal to encourage diversity without instability |
| Jitter noise (global) | 0.01 | Small exploration; absent in forecasting |
| Residual placement | In `MoEBlock` | One residual over combined expert output |
| Expert conditioning | `dim_aux=1` in forecast engine | Timestep modulation via AdaLayerNorm |
| Spatial routing | Off (both engines) | Available but not used in this run |
| Capacity enforcement | Off | No token dropping; variable per-expert batch size |
| FSDP sharding | `MoEBlock` as unit | Expert MLPs not individually sharded (avoids double-wrapping) |

---

*Source: `/capstor/store/cscs/userlab/ch17/slurm/slurm_weathergen_z2sfqaid_dir/WeatherGenerator/` — experiment run `z2sfqaid`.*
