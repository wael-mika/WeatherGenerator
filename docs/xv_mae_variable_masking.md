# XV-MAE: Cross-Variable Masked Autoencoder

## Overview

Standard MAE pretraining masks spatial locations — the encoder sees a subset of grid cells, the decoder reconstructs the rest. XV-MAE adds a second masking axis: **individual atmospheric variables are independently hidden from the encoder**, forcing the model to reconstruct them from other variables.

This teaches the model physical cross-variable relationships that purely spatial MAE cannot: thermal wind balance, hydrostatic balance, moist-adiabatic lapse rates, geostrophic relationships, etc. When the encoder can never see wind from temperature + pressure, it has to internalize those connections.

---

## Why This Was Added

Spatial MAE with random/healpix/swath masking works well — every masked cell has visible neighbours, so there is a dense gradient signal. Cropping fails because the masked complement is a large contiguous region with no nearby visible context, collapsing to the mean. Variable masking creates a fundamentally different task: the spatial coverage can be complete (all cells visible) while the informational bottleneck is cross-variable.

---

## What Was Implemented

### Feature 1 — Variable Channel Dropout (dataset level)

At each training step, for each stream, a `BoolTensor[C]` channel mask is sampled via independent Bernoulli draws. Channels where the mask is `True` are **zeroed out** in the encoder input tokens. The decoder still receives queries for all target cells and all channels — it must reconstruct everything, including the hidden channels.

**Key constraint**: at least one channel is always kept visible (safety guarantee prevents a fully-blank encoder input).

**Where it happens**: `_build_channel_mask()` in `multi_stream_data_sampler.py` → applied in `tokenize_apply_mask_source()` in `tokenizer_utils.py`.

**When it is active**: training only. Validation/test always receive `None` (no masking).

### Feature 2 — Learned Variable Mask Tokens (model level)

When a channel is hidden, Feature 1 leaves its dimensions as `0.0`. Feature 2 replaces those zeros with a **learned per-variable scalar** (`nn.Parameter[C]`) before the channel enters the embedding transformer. This gives the encoder a consistent, learnable signal meaning "this variable is masked here", rather than a potentially misleading zero that could be confused with a true zero measurement.

**Where it lives**: `variable_mask_values` parameter in `StreamEmbedTransformer`, applied in `EmbeddingEngine.forward()`.

**Toggle**: `use_variable_mask_tokens: True` in the model config (top-level, not inside `model_input`). Feature 2 is off by default.

---

## Token Layout (why the indexing works)

Each token assembled by the tokenizer has the structure:

```
token = [stream_id(1) | datetime(5) | coords_local(2) | geoinfos(G) | data(C)]
```

Data channels always occupy the **last C dimensions**. The embedder with `embed_orientation: "channels"` processes each of the C channel slices independently, so zeroing `data[:, k]` for channel `k` is a clean, non-interfering operation. Feature 2 uses `sdata[:, -num_data_channels:]` to locate the data slice without computing explicit offsets.

---

## Configuration

### Minimal setup (uniform dropout rate)

```yaml
model_input:
  "mae_random_xv":
    masking_strategy: "random"
    num_samples: 1
    num_steps_input: 1
    masking_strategy_config:
      rate: 0.15
      rate_sampling: False
      diffusion_rn: False
    variable_masking:
      use_mask_tokens: False      # Feature 2 off
      channel_dropout_rate: 0.3   # each channel independently masked with p=0.3
```

### Per-group rates with regex channel matching

```yaml
variable_masking:
  use_mask_tokens: False
  channel_dropout_rate: 0.2        # fallback for unmatched channels
  variable_groups:
    upper_air_dynamics:
      variables: ["u_\\d+", "v_\\d+", "z_\\d+"]   # wind + geopotential at all levels
      dropout_rate: 0.4
    thermodynamics:
      # Anemoi ERA5 O96: only t_ and q_ are present at pressure levels.
      # r_ (RH) is not produced; w_ is in source_exclude.
      variables: ["t_\\d+", "q_\\d+"]   # temp, specific humidity at all levels
      dropout_rate: 0.3
    surface:
      # Anemoi ERA5 O96 uses ECMWF short names: 10u/10v/2t/2d, not u_10m/v_10m/t_2m/d_2m.
      variables: ["10u", "10v", "2t", "2d", "msl", "sp", "sst"]
      dropout_rate: 0.15           # mask surface less aggressively
```

Channel names are matched with `re.fullmatch(pattern, channel_name)`. The first matching group wins; `channel_dropout_rate` is the fallback.

### Enabling learned mask tokens (Feature 2)

Add at the **top level** of the model config (not inside `model_input`):

```yaml
use_variable_mask_tokens: True     # top-level toggle
```

And inside `model_input`:

```yaml
variable_masking:
  use_mask_tokens: True             # must also be True here
  channel_dropout_rate: 0.3
```

Both flags must be `True`. The model config flag controls whether `StreamEmbedTransformer` allocates the `variable_mask_values` parameter; the `model_input` flag gates its application at forward time.

### Combining with spatial masking

Variable masking stacks on top of spatial masking independently. The existing configs show two ready-to-use combinations:

| Config file | Spatial strategy | Variable masking |
|---|---|---|
| `random_xv.yml` | Random cell dropout (15%) | Channel dropout (30%) |
| `mixed_xv.yml` | Mixed: random + healpix + swath_sparse | Channel dropout (30%) |

---

## Experiment Suggestions

### Baseline comparison

Before investing in variants, establish whether XV-MAE actually helps over vanilla MAE by comparing:

| Experiment | Config | What to measure |
|---|---|---|
| Vanilla random MAE | `random.yml` | Validation loss, downstream finetune RMSE |
| Random + XV-MAE (F1 only) | `random_xv.yml` | Same |
| Random + XV-MAE (F1 + F2) | `random_xv.yml` + `use_variable_mask_tokens: True` | Same |

The gap between F1-only and F1+F2 tells you whether the encoder benefits from a learned "this is masked" signal over a raw zero.

---

### Experiment 1: Asymmetric masking rates by variable group

**Hypothesis**: Upper-air dynamics variables (u, v, z) carry most of the physical relationships. Masking them at higher rates forces stronger cross-variable learning. Surface variables are boundary conditions — masking them too aggressively may destabilise training.

**Config sketch**:

```yaml
variable_groups:
  upper_air_dynamics:
    variables: ["u_\\d+", "v_\\d+", "z_\\d+"]
    dropout_rate: 0.5    # aggressive
  thermodynamics:
    variables: ["t_\\d+", "q_\\d+"]
    dropout_rate: 0.35
  surface:
    # Anemoi ERA5 O96 naming: 10u/10v/2t/2d (not u_10m/v_10m/t_2m/d_2m)
    variables: ["10u", "10v", "2t", "2d", "msl", "sp", "sst"]
    dropout_rate: 0.1    # conservative
```

**What to watch**: Does the loss on upper-air channels converge at a lower absolute value than with uniform rates? Does finetune RMSE on 500hPa geopotential improve?

---

### Experiment 2: XV-MAE with mixed spatial strategy

Mixed spatial masking already trains on three different geometric views. Combining it with variable dropout creates a doubly-diverse training signal: the encoder must handle partial spatial coverage AND partial variable coverage simultaneously.

Use `mixed_xv.yml` directly. The interesting question is whether this joint diversity is better or worse than each form of diversity alone.

**Ablation grid**:

| Spatial | Variable | Expected outcome |
|---|---|---|
| Random | None | Strong spatial reconstruction |
| Random | XV (30%) | Adds cross-variable signal |
| Mixed | None | Diverse geometric views |
| Mixed | XV (30%) | Both axes of diversity |

---

### Experiment 3: Progressive curriculum masking

**Hypothesis**: Start with a low channel dropout rate so the model first learns spatial reconstruction well, then ramp the variable masking rate up during training to force harder cross-variable tasks.

This is not directly supported by a config knob today, but could be approximated by staging two pretraining runs:

1. Phase 1 (~50% of pretraining steps): `channel_dropout_rate: 0.15`
2. Phase 2 (~50% of pretraining steps): resume from checkpoint, `channel_dropout_rate: 0.4`

---

### Experiment 4: High variable masking rate as a stress test

Set `channel_dropout_rate: 0.6` — on average 60% of variables hidden per step. At this rate the encoder regularly sees only a handful of variables and must fully reconstruct the rest. This is the XV-MAE analogue of BEiT-style high masking (75%+ spatial masking).

**Risk**: the training task may become under-constrained if too many anchor variables are hidden simultaneously. The safety guarantee keeps at least one visible, but that one variable may carry insufficient information to constrain the rest. Monitor the training loss carefully; if it does not decrease past random initialization after ~5k steps, reduce the rate.

---

### Experiment 5: Variable masking during finetuning

Finetuning configs exist for IASI and forecasting (`*_iasi_finetuning.yml`, `*_forecast_finetuning.yml`). Adding a mild `variable_masking` block (`channel_dropout_rate: 0.1`) during finetuning acts as a regulariser — it prevents the model from over-relying on any single variable and may improve generalisation at the cost of a slightly higher finetuning loss. This is analogous to using dropout at fine-time.

---

## Diagnostics to monitor

- **Per-group loss**: split the validation loss by variable group. Masked channels should initially have higher loss; convergence of the masked channels' loss below unmasked channels' loss is a signal the model has learned to infer them.
- **Gradient norms of `variable_mask_values`**: if Feature 2 is enabled, these should be non-zero and increasing in the first few hundred steps. If they are stuck at zero the parameter is not receiving gradients.
- **Channel mask statistics**: log the fraction of masked channels per step (debug level in `_build_channel_mask`). Verify the empirical rate matches `channel_dropout_rate`.

---

## Implementation files changed

| File | Change |
|---|---|
| `src/weathergen/datasets/multi_stream_data_sampler.py` | `_build_channel_mask()`, `_get_variable_masking_cfg()`, threads channel mask through batch assembly |
| `src/weathergen/datasets/tokenizer_utils.py` | `tokenize_apply_mask_source()` applies `channel_mask` to data tensor |
| `src/weathergen/datasets/tokenizer_masking.py` | `get_source()` accepts and forwards `channel_mask` |
| `src/weathergen/datasets/batch.py` | `ModelBatch.channel_masks` field, CPU→GPU transfer in `to_device()` |
| `src/weathergen/model/embeddings.py` | `variable_mask_values` parameter in `StreamEmbedTransformer` |
| `src/weathergen/model/engines.py` | `EmbeddingEngine.forward()` applies learned mask values before embedding |
| `config/config_mae_masking/random_xv.yml` | Ready-to-use random spatial + XV config |
| `config/config_mae_masking/mixed_xv.yml` | Ready-to-use mixed spatial + XV config |
