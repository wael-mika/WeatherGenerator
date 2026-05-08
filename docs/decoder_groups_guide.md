# Decoder Groups Configuration Guide

This guide explains how to configure per-group decoders using the `variable_groups` feature.
It covers the full configuration surface across both the stream config and the model config,
with annotated examples for every supported combination.

---

## Table of Contents

1. [Core concept](#1-core-concept)
2. [How it fits into the decoder pipeline](#2-how-it-fits-into-the-decoder-pipeline)
3. [Option A — shared decoder, per-group heads](#3-option-a--shared-decoder-per-group-heads)
4. [Option B — per-group decoder and head](#4-option-b--per-group-decoder-and-head)
5. [Mixing Option A and Option B in one stream](#5-mixing-option-a-and-option-b-in-one-stream)
6. [Stream config reference](#6-stream-config-reference)
7. [Model config setup](#7-model-config-setup)
8. [Ready-made configs](#8-ready-made-configs)
9. [Choosing decoder depth and width](#9-choosing-decoder-depth-and-width)
10. [Checkpoint compatibility](#10-checkpoint-compatibility)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Core concept

By default, every channel in a stream shares one reconstruction pipeline:

```
coord_embed(t_coords)
    → TargetPredictionEngine  (cross-attention decoder — the expensive part)
        → EnsPredictionHead   (final MLP + linear projection)
            → [ens_size, N, total_channels]
```

`variable_groups` lets you break the channel set into named physical groups and configure
the tail of this pipeline independently per group.

Two modes are available:

| Mode | What is per-group | Decoder TTE | Head |
|---|---|---|---|
| **Option A** | Head only | Shared (one TTE for the stream) | ✅ per-group |
| **Option B** | Decoder + head | ✅ per-group | ✅ per-group |

The modes can be **mixed within one stream**: some groups can own their decoder (B) while
others share the stream-level one (A).

---

## 2. How it fits into the decoder pipeline

At inference time `predict_decoders` runs the following logic when `variable_groups` is active:

```
coord_embed(t_coords)                          ← always shared: one embedding per stream
    │
    ├─ group "precipitation"  has_own_tte=True
    │       └─ own TTE (1-layer, 4-head)       ← Option B: group runs its own cross-attention
    │               └─ own Head  (Softplus)
    │
    ├─ group "upper_air_wind"  has_own_tte=False
    │       └─ shared TTE (cached)             ← Option A: result cached across A-groups
    │               └─ own Head  (Identity)
    │
    └─ group "_default"  has_own_tte=False
            └─ shared TTE (reuse cache)
                    └─ own Head  (Identity)

scatter all group predictions → pred[ens_size, N, total_channels]
```

Key properties:
- Coordinate embedding is **always shared** — spatial location has no channel identity.
- The shared TTE is computed **once** and reused for all Option-A groups; it is **not
  instantiated at all** if every group has its own TTE.
- Output tensor shape `[ens_size, N, total_channels]` is unchanged — the loss module
  requires no modifications.
- `pred_heads` keys: `"{stream_name}/{group_name}"` (grouped) vs `"{stream_name}"` (plain).
- `target_token_engines` keys: `"{stream_name}/{group_name}"` (Option B) and/or
  `"{stream_name}"` (shared / Option A).

---

## 3. Option A — shared decoder, per-group heads

### When to use

Use Option A when you want to apply a physics-motivated final activation to a subset of
channels (most commonly `Softplus` for precipitation or humidity) without adding decoder
parameters. The overhead vs. baseline is negligible — only a few extra linear layers.

### Stream config skeleton

```yaml
ERA5:
  # ... dataset, embed, source/target_exclude ...

  # Stream-level decoder — shared by ALL groups (no target_readout inside groups below)
  target_readout:
    num_layers: 2        # number of cross-attention + MLP pairs
    num_heads: 4         # attention heads in the cross-attention

  variable_groups:

    group_name_1:
      variables: ["<regex1>", "<regex2>"]   # Python re.fullmatch patterns
      pred_head:
        ens_size: 1          # number of ensemble members
        num_layers: 1        # MLP layers inside the head (≥1)
        final_activation: Softplus   # activation applied after the last linear layer

    group_name_2:
      variables: ["<regex3>"]
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    _default:            # catches every channel not matched above (REQUIRED if any remain)
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity
```

**Rules for `variable_groups` keys:**
- Group names are arbitrary strings; `_default` is the reserved catch-all.
- Patterns use `re.fullmatch` — `"u_\\d+"` matches `u_50`, `u_850`, etc. but NOT `u_extra`.
- Groups are mutually exclusive — a channel matched by two groups raises `ValueError` at startup.
- `_default` is **required** when any channel is unmatched; omit it only if all patterns are exhaustive.

### Minimal Option A example — Softplus for precipitation

```yaml
# config/streams/era5_1deg/era5.yml

ERA5:
  type: anemoi
  filenames: ['aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr']
  stream_id: 0
  source_exclude: ['w_', 'skt', 'tcw', 'cp', 'tp']
  target_exclude: ['w_', 'slor', 'sdor', 'tcw']
  loss_weight: 1.0
  location_weight: cosine_latitude
  masking_rate: 0.6
  masking_rate_none: 0.05
  token_size: 8
  tokenize_spacetime: True
  max_num_targets: -1

  embed:
    net: transformer
    num_tokens: 1
    num_heads: 8
    dim_embed: 256
    num_blocks: 2

  embed_target_coords:
    net: linear
    dim_embed: 256

  target_readout:          # shared by all groups (none have their own)
    num_layers: 2
    num_heads: 4

  variable_groups:
    precipitation:
      variables: ["tp", "cp"]
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus    # tp, cp ≥ 0

    _default:
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity
```

### Full physics grouping with Option A

```yaml
  variable_groups:

    precipitation:
      variables: ["tp", "cp"]
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus

    upper_air_wind:
      variables:
        - "u_\\d+"           # zonal wind at pressure levels
        - "v_\\d+"           # meridional wind
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    upper_air_mass:
      variables:
        - "z_\\d+"           # geopotential
        - "t_\\d+"           # temperature
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    upper_air_moisture:
      variables:
        - "q_\\d+"           # specific humidity
        - "r_\\d+"           # relative humidity (if present)
        - "o3_\\d+"          # ozone (if present)
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus    # mixing ratios ≥ 0

    _default:                # surface + all unmatched single-level fields
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity
```

---

## 4. Option B — per-group decoder and head

### When to use

Use Option B when different physical groups benefit from different decoder depths or widths.
The cross-attention decoder conditions target tokens on the latent state — giving each group
its own decoder allows the model to learn group-specific latent-to-output mappings.

Practical motivations:
- **Precipitation** is locally forced at convective scales → shallow 1-layer decoder.
- **Upper-air dynamics** has large-scale coherent structures → deeper 2-layer, wider decoder.
- **Moisture** is intermediate in spatial scale → moderate depth.

**Parameters added per Option-B group:**
One `TargetPredictionEngineClassic` with `num_layers` cross-attention + MLP pairs, each
of width `dim_embed` (set by `embed_target_coords.dim_embed`). The stream-level TTE is
**not instantiated** when every group has its own `target_readout`.

### Stream config skeleton

```yaml
ERA5:
  # ... dataset, embed ...

  embed_target_coords:
    net: linear
    dim_embed: 256        # width of ALL decoders (shared and per-group alike)

  # Stream-level target_readout: kept for schema completeness.
  # It is NOT used when every group carries its own target_readout.
  target_readout:
    num_layers: 2
    num_heads: 4

  variable_groups:

    group_name_1:
      variables: ["<regex>"]
      target_readout:         # ← presence of this key triggers Option B for this group
        num_layers: 2         # cross-attention layers in this group's decoder
        num_heads: 8          # attention heads
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    group_name_2:
      variables: ["<regex>"]
      target_readout:
        num_layers: 1
        num_heads: 4
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus

    _default:
      target_readout:
        num_layers: 1
        num_heads: 4
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity
```

### Full physics grouping with Option B

```yaml
  variable_groups:

    upper_air_wind:
      variables:
        - "u_\\d+"
        - "v_\\d+"
      target_readout:
        num_layers: 2
        num_heads: 8     # wider: large-scale dynamics benefit from more cross-attention capacity
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    upper_air_mass:
      variables:
        - "z_\\d+"
        - "t_\\d+"
      target_readout:
        num_layers: 2
        num_heads: 8     # thermodynamics is tightly coupled to dynamics
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    upper_air_moisture:
      variables:
        - "q_\\d+"
        - "r_\\d+"
        - "o3_\\d+"
      target_readout:
        num_layers: 2
        num_heads: 4     # moisture transport is more local than wind/mass
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus

    precipitation:
      variables: ["tp", "cp"]
      target_readout:
        num_layers: 1
        num_heads: 4     # shallow: convective-scale, locally determined
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus

    _default:            # surface + single-level fields
      target_readout:
        num_layers: 1
        num_heads: 4     # boundary-layer fields are locally determined
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity
```

---

## 5. Mixing Option A and Option B in one stream

Groups without `target_readout` share the stream-level TTE (Option A behaviour).
Groups with `target_readout` get their own TTE (Option B).

The stream-level TTE is instantiated **only** when at least one group lacks `target_readout`.

```yaml
  target_readout:
    num_layers: 2         # used only by the groups that have no target_readout below
    num_heads: 4

  variable_groups:

    precipitation:
      variables: ["tp", "cp"]
      target_readout:           # Option B: own 1-layer decoder
        num_layers: 1
        num_heads: 4
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus

    upper_air_wind:             # Option A: shares the 2-layer stream decoder above
      variables:
        - "u_\\d+"
        - "v_\\d+"
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    _default:                   # Option A: also shares the stream decoder
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity
```

In the forward pass:
- `precipitation` runs its own 1-layer TTE.
- `upper_air_wind` and `_default` both read the cached output of the shared 2-layer TTE —
  the shared TTE runs **once**, not once per Option-A group.

---

## 6. Stream config reference

### Top-level keys relevant to decoder groups

| Key | Required | Description |
|---|---|---|
| `embed_target_coords.dim_embed` | Yes | Hidden dimension of all decoders (shared and per-group). Increasing this grows every decoder uniformly. |
| `embed_target_coords.net` | Yes | Embedding network type: `"linear"` or `"mlp"`. |
| `target_readout.num_layers` | Yes* | Number of cross-attention + MLP pairs in the **shared** decoder. Required if any group is Option A (no own `target_readout`). |
| `target_readout.num_heads` | Yes* | Attention heads in the shared decoder. |
| `target_readout.dim_head_proj` | No | Head projection dimension. Defaults to `None` (full-rank). |
| `target_readout.mlp_hidden_factor` | No | MLP hidden width multiplier. Default `2`. |
| `target_readout.softcap` | No | Attention logit soft-cap. Default `0.0` (disabled). |
| `variable_groups` | No | When absent: original single-decoder behaviour. |

*Required when at least one group does not carry its own `target_readout`.

### `variable_groups.<group_name>` keys

| Key | Required | Description |
|---|---|---|
| `variables` | Yes (except `_default`) | List of Python `re.fullmatch` patterns matched against channel names. |
| `target_readout` | No | When present: this group gets its own `TargetPredictionEngineClassic` (Option B). When absent: group shares the stream-level TTE (Option A). |
| `target_readout.num_layers` | Yes (if target_readout present) | Cross-attention layers in this group's decoder. |
| `target_readout.num_heads` | Yes (if target_readout present) | Attention heads. Note: must be compatible with `dim_embed` (divisible). |
| `pred_head` | No | Head config. Falls back to the stream-level `pred_head` if present; error if neither exists. |
| `pred_head.ens_size` | Yes (if pred_head) | Number of parallel ensemble prediction heads. Usually `1`. |
| `pred_head.num_layers` | Yes (if pred_head) | Layers in the head MLP. Usually `1`. |
| `pred_head.final_activation` | No | Activation after the last linear layer. Default `Identity`. See table below. |

### Available `final_activation` values

| Name | Output range | Use-case |
|---|---|---|
| `Identity` | (−∞, +∞) | Default; wind, temperature, geopotential |
| `Softplus` | (0, +∞) | Precipitation, humidity, mixing ratios |
| `Sigmoid` | (0, 1) | Probabilities, cloud fraction |
| `Tanh` | (−1, 1) | Normalised anomalies |
| `GELU` | (−0.17, +∞) | Smooth alternative to ReLU |
| `ReLU` | [0, +∞) | Hard non-negative constraint |

Full registry: `src/weathergen/model/utils.py::ActivationFactory._registry`.

### Variable pattern syntax

Patterns are matched with Python `re.fullmatch` against channel names like `"u_850"`, `"tp"`.

| Pattern | Matches |
|---|---|
| `"tp"` | Exactly `tp` |
| `"u_\\d+"` | `u_50`, `u_100`, `u_850`, `u_1000`, … |
| `"[uv]_\\d+"` | `u_850`, `v_500`, … |
| `"(tp\|cp)"` | `tp` or `cp` |
| `".*_\\d+"` | Any variable with a `_<number>` suffix |

Always use **double backslash** (`\\d+`) in YAML strings since YAML consumes one level of escaping.

---

## 7. Model config setup

The model config controls the encoder, forecasting engine, and training loop.
The only key that connects the model to a decoder-groups stream is `streams_directory`.

### Minimal changes from the baseline `config_mae.yml`

```yaml
# Only this key changes — everything else is identical to config_mae.yml
streams_directory: "./config/streams/era5_1deg_physics_optA/"
```

The rest of the decoder-relevant model config keys:

| Key | Relevant to decoder groups | Notes |
|---|---|---|
| `decoder_type` | Yes | Must be `PerceiverIOCoordConditioning` for `variable_groups` to work. This is the default. `Linear` decoder does not support per-group `target_readout`. |
| `pred_self_attention` | Yes | Whether `TargetPredictionEngineClassic` adds self-attention between cross-attention layers. Applies to all decoders (shared and per-group). |
| `pred_mlp_adaln` | Yes | Adaptive LayerNorm conditioning in the decoder MLP. Applies to all decoders. |
| `with_mixed_precision` / `attention_dtype` | Indirectly | Decoder inputs may arrive in BF16; pred-head outputs are Float32. The implementation casts `grp_pred.to(pred.dtype)` per group — no config action needed. |

### Full working model config (Option A example)

```yaml
# config/config_mae_physics_optA.yml

embed_orientation: "channels"
embed_unembed_mode: "block"
embed_dropout_rate: 0.1

ae_local_dim_embed: 1024
ae_local_num_blocks: 2
ae_local_num_heads: 16
ae_local_dropout_rate: 0.1
ae_local_with_qk_lnorm: True

ae_local_num_queries: 1
ae_local_queries_per_cell: False
ae_adapter_num_heads: 16
ae_adapter_embed: 128
ae_adapter_with_qk_lnorm: True
ae_adapter_with_residual: True
ae_adapter_dropout_rate: 0.1

ae_global_dim_embed: 2048
ae_global_num_blocks: 2
ae_global_num_heads: 32
ae_global_dropout_rate: 0.1
ae_global_with_qk_lnorm: True
ae_global_att_dense_rate: 1.0
ae_global_block_factor: 64
ae_global_mlp_hidden_factor: 2
ae_global_trailing_layer_norm: False

ae_aggregation_num_blocks: 8
ae_aggregation_num_heads: 32
ae_aggregation_dropout_rate: 0.1
ae_aggregation_with_qk_lnorm: True
ae_aggregation_att_dense_rate: 1.0
ae_aggregation_block_factor: 64
ae_aggregation_mlp_hidden_factor: 2

# Must be PerceiverIOCoordConditioning when using variable_groups
decoder_type: PerceiverIOCoordConditioning
pred_adapter_kv: False
pred_self_attention: True
pred_dyadic_dims: False
pred_mlp_adaln: True
num_class_tokens: 1
num_register_tokens: 7

fe_num_blocks: 16
fe_num_heads: 16
fe_dropout_rate: 0.1
fe_with_qk_lnorm: True
fe_layer_norm_after_blocks: [7]
fe_impute_latent_noise_std: 1e-4
forecast_att_dense_rate: 1.0
with_step_conditioning: True

healpix_level: 5
rope_2D: False

with_mixed_precision: True
with_flash_attention: True
compile_model: False
with_fsdp: True
ddp_find_unused_parameters: False
attention_dtype: bf16
mixed_precision_dtype: bf16
mlp_norm_eps: 1e-5
norm_eps: 1e-4

latent_noise_kl_weight: 0.0
latent_noise_gamma: 2.0
latent_noise_saturate_encodings: 5
latent_noise_use_additive_noise: False
latent_noise_deterministic_latents: True

freeze_modules: ""
norm_type: "LayerNorm"
zarr_store: "zip"

# ← Only this line differs from the baseline config_mae.yml
streams_directory: "./config/streams/era5_1deg_physics_optA/"
streams: ???

general:
  istep: 0
  rank: ???
  world_size: ???
  multiprocessing_method: "fork"
  desc: "mae_physics_optA"
  run_id: ???
  run_history: []

train_logging:
  terminal: 10
  metrics: 20
  checkpoint: 250
  log_grad_norms: False

data_loading:
  num_workers: 12
  rng_seed: ???
  repeat_data_in_mini_epoch: False

training_config:
  training_mode: ["masking"]
  num_mini_epochs: 96
  samples_per_mini_epoch: 4096
  shuffle: True

  start_date: 1979-01-01T00:00
  end_date: 2022-12-31T00:00

  time_window_step: 06:00:00
  time_window_len: 06:00:00

  window_offset_prediction: 0

  learning_rate_scheduling:
    lr_start: 1e-6
    lr_max: 5e-5
    lr_final_decay: 2e-6
    lr_final: 0.0
    num_steps_warmup: 256
    num_steps_cooldown: 512
    policy_warmup: "cosine"
    policy_decay: "constant"
    policy_cooldown: "linear"
    parallel_scaling_policy: "sqrt"

  optimizer:
    grad_clip: 1.0
    weight_decay: 0.1
    log_grad_norms: False
    adamw:
      beta1: 0.98125
      beta2: 0.9875
      eps: 2e-08

  losses:
    physical:
      type: LossPhysical
      loss_fcts:
        mse: {}

  model_input:
    mae_random:
      masking_strategy: "random"
      num_samples: 1
      num_steps_input: 1
      masking_strategy_config:
        rate: 0.4
        rate_sampling: False
        diffusion_rn: False

  forecast:
    time_step: 00:00:00
    num_steps: 0
    policy: null

validation_config:
  samples_per_mini_epoch: 256
  shuffle: False
  start_date: 2023-10-01T00:00
  end_date: 2023-12-31T00:00
  validate_with_ema:
    enabled: False
    ema_ramp_up_ratio: 0.09
    ema_halflife_in_thousands: 1e-3
  output:
    num_samples: 0
    normalized_samples: False
    streams: null
  validate_before_training: False

wgtags:
  org: null
  issue: null
  exp: "mae_physics_optA"
  grid: null
```

For Option B, change only `streams_directory` and the tag keys:

```yaml
streams_directory: "./config/streams/era5_1deg_physics_optB/"

general:
  desc: "mae_physics_optB"

wgtags:
  exp: "mae_physics_optB"
```

---

## 8. Ready-made configs

The repository ships four configs covering the two options across two grouping granularities.

### Precipitation-only grouping (proof of concept)

These configs add `tp` and `cp` as targets and give them a `Softplus` head. All other channels
use `Identity`. Good starting point — minimal change from the baseline.

| File | Option | Stream dir |
|---|---|---|
| `config/config_mae_precip_optA.yml` | A | `streams/era5_1deg_precip_optA/` |
| `config/config_mae_precip_optB.yml` | B (precip only owns decoder) | `streams/era5_1deg_precip_optB/` |

**Stream config** (`era5_1deg_precip_optA/era5.yml` excerpt):
```yaml
target_exclude: ['w_', 'slor', 'sdor', 'tcw']   # tp and cp are now targets

variable_groups:
  precipitation:
    variables: ["tp", "cp"]
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus
  _default:
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity
```

### Full physics grouping (recommended)

These configs split ERA5 channels into five physical groups with physics-appropriate activations
and (for Option B) decoder capacities tuned per group.

| File | Option | Stream dir |
|---|---|---|
| `config/config_mae_physics_optA.yml` | A | `streams/era5_1deg_physics_optA/` |
| `config/config_mae_physics_optB.yml` | B (all groups own their decoder) | `streams/era5_1deg_physics_optB/` |

**Stream configs** (`era5_1deg_physics_optA/` vs `era5_1deg_physics_optB/`):

| Group | Option A head | Option B decoder | Option B head |
|---|---|---|---|
| `precipitation` | Softplus | 1 layer, 4 heads | Softplus |
| `upper_air_wind` | Identity | 2 layers, 8 heads | Identity |
| `upper_air_mass` | Identity | 2 layers, 8 heads | Identity |
| `upper_air_moisture` | Softplus | 2 layers, 4 heads | Softplus |
| `_default` | Identity | 1 layer, 4 heads | Identity |

To launch:

```bash
# Option A — shared decoder, physics-aware heads
uv run train --config config/config_mae_physics_optA.yml

# Option B — per-group decoders
uv run train --config config/config_mae_physics_optB.yml
```

---

## 9. Choosing decoder depth and width

### Decoder depth (`num_layers`)

Each layer is one cross-attention block (Q = target tokens, KV = 1-ring latent neighbours)
plus an MLP. Depth increases capacity for learning latent-to-physical mappings at the cost of
compute and memory per prediction step.

| `num_layers` | Suitable for |
|---|---|
| 1 | Locally determined fields: precipitation, boundary-layer surface fields, single-level diagnostics |
| 2 | Fields with moderate spatial scale: upper-air dynamics, moisture |
| 3+ | Only justified if 2 layers demonstrably underfits; rarely needed |

### Decoder width (`num_heads`)

`num_heads` must divide `dim_embed` evenly.
With `embed_target_coords.dim_embed: 256`, valid choices include 1, 2, 4, 8, 16, 32.

| `num_heads` | Effective head dimension (`dim_embed / num_heads`) | Notes |
|---|---|---|
| 4 | 64 | Lightweight; suitable for local fields |
| 8 | 32 | Standard for most ERA5 fields |
| 16 | 16 | Head dimension is small; marginal benefit |

### Parameter count estimate

For `dim_embed=256` and `pred_self_attention=True`, each TTE layer contributes approximately:
- Cross-attention: `4 × dim_embed²` = 262 k parameters
- Self-attention: `4 × dim_embed²` = 262 k parameters
- MLP (hidden_factor=2): `2 × dim_embed × 2 × dim_embed` = 262 k parameters
- Total per layer: ≈ 786 k parameters

A 2-layer per-group TTE therefore adds ≈ 1.6 M parameters per group. A stream with five
Option-B groups adds ≈ 8 M parameters total — modest compared to the encoder (typically > 1 B).

### When to prefer Option A over B

- You only need different activations (Softplus vs. Identity). → **Option A**.
- You want to run an ablation baseline. → **Option A** matches the baseline numerically when
  all groups use Identity (identical computation).
- GPU memory is tight. → **Option A** (fewer TTE instances).
- You suspect group-specific latent routing improves loss for a variable class. → **Option B**.

---

## 10. Checkpoint compatibility

### New runs

Starting fresh with `variable_groups`: no special action needed.

### Loading an old checkpoint into a `variable_groups` model

`pred_heads` and `target_token_engines` keys change:

| State dict key (old, no groups) | State dict key (with groups) |
|---|---|
| `pred_heads.ERA5` | `pred_heads.ERA5/precipitation`, `pred_heads.ERA5/upper_air_wind`, … |
| `target_token_engines.ERA5` | `target_token_engines.ERA5/precipitation` (Option B) or `target_token_engines.ERA5` (Option A shared) |

The `load_model` function in `model_interface.py` uses `strict=False`, so missing keys log
a warning and the corresponding modules are re-initialised from scratch. This means:

- Encoder weights (embeddings, assimilation, global attention) load correctly.
- Decoder weights (TTE, pred_heads) are reinitialised — plan for extra warmup steps.

### Loading a `variable_groups` checkpoint into a plain model

The reverse: grouped keys are present in the checkpoint but the model expects plain keys.
Again `strict=False` handles this gracefully — grouped keys are reported as unused keys.

---

## 11. Troubleshooting

### `omegaconf.errors.ConfigKeyError: Missing key pred_head`

The fallback `si["pred_head"]` was triggered when a group has no `pred_head` and the
stream config also has no top-level `pred_head`.

**Fix:** Either add `pred_head` to every group, or add a stream-level `pred_head` as fallback:

```yaml
ERA5:
  # Stream-level fallback (used when a group omits pred_head)
  pred_head:
    ens_size: 1
    num_layers: 1
    final_activation: Identity

  variable_groups:
    precipitation:
      variables: ["tp", "cp"]
      pred_head:              # overrides stream-level for this group
        final_activation: Softplus
    _default:
      {}                      # inherits stream-level pred_head
```

### `ValueError: N channel(s) unmatched by any variable group but no _default group is defined`

The regex patterns do not cover all channels and `_default` is absent.

**Fix:** Add `_default` to your `variable_groups`. If you genuinely want to cover all channels
without a catch-all, list every pattern exhaustively (error-prone; prefer `_default`).

### `ValueError: decoder_type='Linear' does not support per-group target_readout`

`target_readout` inside a group is only supported with `decoder_type: PerceiverIOCoordConditioning`.

**Fix:** Remove `target_readout` from the group (use Option A instead) or change `decoder_type`
to `PerceiverIOCoordConditioning` in the model config.

### `RuntimeError: Index put requires the source and destination dtypes match`

This was a known bug fixed in the implementation (`grp_pred.to(pred.dtype)` cast). If you
see this error, ensure you are running the latest version of `model.py`.

### `Channels at indices {…} matched by multiple variable groups`

Two groups' patterns overlap — a channel satisfies more than one group's regex.

**Fix:** Make patterns mutually exclusive. Use `re.fullmatch` semantics to verify:
```python
import re
channels = ["u_850", "u_extra"]
pattern = re.compile(r"u_\d+")
print([c for c in channels if pattern.fullmatch(c)])  # ['u_850'] only
```

### Groups appear in the wrong order in `stream_groups`

`_resolve_variable_groups` processes groups in YAML insertion order (Python 3.7+ dicts
preserve order). `_default` is always appended last regardless of where it appears in the YAML.
The order only matters for which group's dtype initialises `pred` first — it has no effect on
correctness.

### Some channels always output zeros

If a group's `ch_indices` is empty (all channels matched by other groups) and `pred` was
already initialised from a different group, the zero-fill is silent. Verify channel coverage
with:

```bash
uv run python -c "
import re
channels = ['u_850', 'v_850', 'tp', 'cp', 't_500', 'sp']  # replace with actual channels
groups = {
    'precipitation': [r'tp', r'cp'],
    'upper_air': [r'u_\\d+', r'v_\\d+', r't_\\d+'],
}
for grp, patterns in groups.items():
    pats = [re.compile(p) for p in patterns]
    matched = [c for c in channels if any(p.fullmatch(c) for p in pats)]
    print(f'{grp}: {matched}')
"
```
