# WeatherGenerator Training Reference

A comprehensive reference for all training strategies, masking options, and downstream tasks.
Covers pretraining and fine-tuning for JEPA, MTM, Forecasting, and Downscaling.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Config Building Blocks](#2-core-config-building-blocks)
   - [Training Modes](#21-training-modes)
   - [Masking Strategies](#22-masking-strategies)
   - [Target-Source Correspondences](#23-target-source-correspondences)
   - [Loss Types](#24-loss-types)
   - [Teacher Types](#25-teacher-types)
3. [Pretraining Recipes](#3-pretraining-recipes)
   - [JEPA (Latent SSL only)](#31-jepa---latent-ssl-only)
   - [MTM (Masked Token Modeling in physical space)](#32-mtm---masked-token-modeling-in-physical-space)
   - [Combined JEPA + MTM](#33-combined-jepa--mtm)
   - [Forecasting Pretraining](#34-forecasting-pretraining)
4. [Fine-tuning Recipes](#4-fine-tuning-recipes)
   - [Frozen Encoder — Reconstruction](#41-frozen-encoder--reconstruction)
   - [Frozen Encoder — Forecasting](#42-frozen-encoder--forecasting)
   - [Full Fine-tune — Forecasting](#43-full-fine-tune--forecasting)
5. [Task Deep-Dives](#5-task-deep-dives)
   - [Forecasting](#51-forecasting)
   - [MTM as a Downstream Task](#52-mtm-as-a-downstream-task)
   - [Downscaling](#53-downscaling)
6. [Masking Strategy Reference Card](#6-masking-strategy-reference-card)
7. [Config File Inventory](#7-config-file-inventory)
8. [JEPA-from-Forecast Experiment Grid (Exps 01-07)](#8-jepa-from-forecast-experiment-grid-exps-01-07)
9. [Evaluation Guide](#9-evaluation-guide)
10. [Quick-Start Checklists](#10-quick-start-checklists)

---

## 1. Architecture Overview

The model is an encoder-latent-decoder pipeline. Understanding which blocks are active in each
training scenario is critical for knowing what gets trained.

```
Input tokens (HEALPix cells × channels)
        │
        ▼
┌───────────────────────────────────────────────┐
│  ae_local  (local cross-attention per cell)   │  ← per-cell embedding
│  ae_global (global self-attention)            │  ← global context
│  ae_aggregation (aggregation transformer)     │  ← compress to latent
└───────────────────────────────────────────────┘
        │   ENCODER — produces latent tokens
        ▼
   [LATENT SPACE]  ← JEPA / SSL losses operate here
        │
        ▼
┌──────────────────────────────────────────────┐
│  fe_blocks  (forecast engine transformer)    │  ← temporal dynamics
│  (fe_num_blocks, optional)                   │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Decoder  (PerceiverIOCoordConditioning)      │  ← cross-attention to output coords
└──────────────────────────────────────────────┘
        │   DECODER — produces physical weather fields
        ▼
   Output tokens (physical variables)
        │
        ▼
   LossPhysical (MSE in variable space)  ← MTM / Forecasting losses operate here
```

### What is active per training scenario

| Block           | JEPA SSL only | MTM only | JEPA + MTM | Forecasting |
|----------------|:---:|:---:|:---:|:---:|
| Encoder (ae_*)  | ✓ trains | ✓ trains | ✓ trains | ✓ trains |
| fe_blocks       | ✗ (0 blocks) | optional | optional | ✓ trains |
| Decoder         | ✗ no loss | ✓ trains | ✓ trains | ✓ trains |
| JEPA head       | ✓ trains | ✗ | ✓ trains | ✗ |
| EMA Teacher     | EMA only | ✗ | EMA only | ✗ |

---

## 2. Core Config Building Blocks

### 2.1 Training Modes

Set in `training_config.training_mode`. Can be combined as a list.

```yaml
training_mode: ["student_teacher"]           # JEPA SSL only
training_mode: ["masking"]                   # Physical space loss only (MTM or forecasting)
training_mode: ["masking", "student_teacher"] # Combined
```

| Mode | Loss space | Decoder active | Typical use |
|------|-----------|---------------|-------------|
| `"student_teacher"` | Latent | No | JEPA pretraining |
| `"masking"` | Physical | Yes | MTM pretraining, forecasting, fine-tuning |
| `["masking", "student_teacher"]` | Both | Yes | Joint SSL + reconstruction |

---

### 2.2 Masking Strategies

Used in both `model_input` (student / source) and `target_input` (teacher / target).

#### `random`
Uniform random token selection. No spatial structure.

```yaml
masking_strategy: "random"
masking_strategy_config:
  rate: 0.6            # fraction of tokens KEPT (not masked)
  rate_sampling: False # if True, sample rate ~ N(rate, 1/(2.5*pi)) each batch
  diffusion_rn: False  # add diffusion noise to input (optional)
```

#### `forecast` (alias: `causal`)
No masking — all tokens visible. Used as teacher in JEPA or as input in forecasting.

```yaml
masking_strategy: "forecast"
masking_strategy_config: {}
```

#### `healpix`
Masks at a coarser HEALPix hierarchy level. All child cells of a masked parent are masked together,
preserving hierarchical spatial structure.

```yaml
masking_strategy: "healpix"
masking_strategy_config:
  hl_mask: 0       # parent HEALPix level to mask at (must be < healpix_level)
  rate: 0.2        # fraction of PARENT cells kept
  rate_sampling: False
```

#### `cropping_healpix`
Spatially contiguous crop — the model sees a connected region of the globe.
Three spatial selection methods available. Required for `cone_distance`, `subset`, `disjoint`
geometry-aware correspondences.

```yaml
masking_strategy: "cropping_healpix"
masking_strategy_config:
  hl_mask: 3             # HEALPix level for crop boundary resolution (3 = 768 parent cells)
  rate: 0.5              # fraction of globe covered by the crop
  method: "geodesic_disk"  # see below
  rate_sampling: False

  # --- method options ---
  # "disk"          : layer-by-layer neighbor expansion — compact, fast
  # "random_walk"   : random walk through neighbors — irregular, elongated shapes
  # "geodesic_disk" : angular distance selection — smoothest circles, REQUIRED for cone_distance

  # --- optional anchor (fix crop location) ---
  anchor_latitude: 45.0        # degrees, fixes crop centre latitude
  anchor_longitude: -10.0      # degrees, fixes crop centre longitude
  anchor_jitter_degrees: 5.0   # random jitter around anchor (default 0)

  # --- cone_distance relationship params (model_input only) ---
  center_distance_degrees: 45          # fixed angular distance from teacher centre
  # OR random distance:
  center_distance_degrees_random: true
  center_distance_degrees_min: 0
  center_distance_degrees_max: 90
  center_distance_degrees_step: 15    # samples from {0, 15, 30, ..., 90}
  center_azimuth_degrees: 90          # optional: direction (0=N, 90=E, 180=S, 270=W)
```

**Angular radius formula:** `θ = arccos(1 - 2 * rate)`

| `rate` | Angular radius |
|--------|--------------|
| 0.3 | ≈ 66° |
| 0.4 | ≈ 78° |
| 0.5 | 90° (hemisphere) |
| 0.6 | ≈ 102° |

---

### 2.3 Target-Source Correspondences

Defined in the loss under `target_source_correspondence: {target_idx: {source_idx: "relationship"}}`.
Some also require `relationship:` in `model_input`.

```yaml
# Single target, single source (most JEPA experiments):
target_source_correspondence: {0 : {0 : "independent"}}

# Multiple targets/sources (combined losses):
target_source_correspondence: {0 : {0 : "complement"}}   # physical loss
target_source_correspondence: {1 : {1 : "cone_distance"}} # JEPA loss
```

| Correspondence | Mask relation | Needs geometry | Typical use |
|---------------|--------------|:--------------:|-------------|
| `"independent"` | Student generated independently of teacher | No | Base JEPA from global teacher |
| `"complement"` | Student mask = `~teacher` mask (zero token overlap) | No | BERT-style latent MTM |
| `"identity"` | Student mask = teacher mask (same tokens) | No | Auto-encoding, distillation |
| `"subset"` | Student crop geometrically inside teacher crop | Yes (`cropping_healpix`) | Zoom-out prediction |
| `"disjoint"` | Student and teacher crops are spatially separated | Yes (`cropping_healpix`) | Non-overlapping views |
| `"cone_distance"` | Student cone at specified angular distance from teacher | Yes (`cropping_healpix`) | I-JEPA style two-crop |
| `"contained_cone"` | Student cone contained in teacher (fine-grained) | Yes | Hierarchical, always inside |
| `"separated_cone"` | Student cone separated from teacher (fine-grained) | Yes | Non-overlapping, gap-controlled |

**Setting `relationship:` in `model_input`:** Required for geometry-aware correspondences so the
masker knows how to generate the student crop relative to the teacher crop.

```yaml
model_input:
  "student": {
    masking_strategy: "cropping_healpix",
    ...
    relationship: "cone_distance",  # ← masker positions student relative to teacher
  }
```

---

### 2.4 Loss Types

#### `LossPhysical`
Reconstruction loss in physical weather variable space. Requires the decoder to be active.

```yaml
"physical": {
  type: LossPhysical,
  enabled: True,
  weight: 1.0,
  loss_fcts: {
    "mse": {
      weight: 1.0,
      target_source_correspondence: {0 : {0 : "complement"}},
    },
  },
  target_and_aux_calc: "Physical",
}
```

#### `LossLatentSSLStudentTeacher`
JEPA-style loss: student's JEPA head output vs teacher's latent representations.
Decoder is NOT used. Loss is entirely in latent space.

```yaml
"student-teacher": {
  type: LossLatentSSLStudentTeacher,
  enabled: True,
  weight: 1.0,
  loss_fcts: {
    "JEPA": {
      weight: 4,
      loss_extra_args: {},
      out_dim: 2048,           # output dimension of JEPA head
      head: transformer,       # "transformer" for student, "identity" for teacher
      num_blocks: 6,
      num_heads: 12,
      with_qk_lnorm: True,
      intermediate_dim: 768,
      dropout_rate: 0.1,
      target_source_correspondence: {0 : {0 : "independent"}},
    },
  },
  target_and_aux_calc: { "EMATeacher": { ... } },
}
```

---

### 2.5 Teacher Types

#### `EMATeacher`
Momentum-updated copy of the student. Updated each step via exponential moving average.
The standard choice for JEPA/DINO-style SSL.

```yaml
target_and_aux_calc: { "EMATeacher" :
  { ema_ramp_up_ratio: null,          # null = no ramp-up; 0.09 = ramp over first 9% of training
    ema_halflife_in_thousands: 1e-1,  # EMA half-life in thousands of steps
                                      # smaller → faster teacher update
                                      # 1e-1 = fast (adapts quickly), 1e-3 = slow (more stable)
    teacher_run_id: "aev85iny",       # optional: init teacher from pretrained run
    teacher_mini_epoch: -1,           # -1 = latest checkpoint
    model_param_overrides: {          # override specific config keys in teacher model
      training_config: { losses: { student-teacher: { loss_fcts: { JEPA: { head: identity }}}}}
    },
  }
}
```

**EMA half-life guidance:**

| `ema_halflife_in_thousands` | Behaviour | Typical use |
|---|---|---|
| `1e-1` (100 steps) | Fast teacher update, follows student closely | Init from pretrained model (our experiments) |
| `1e-3` (1000 steps) | Slow, stable target | Training from scratch |

#### `FrozenTeacher`
Teacher weights loaded from a checkpoint and completely frozen throughout training.
No EMA update. Good for probing: does a fixed pretrained teacher provide a good learning signal?

```yaml
target_and_aux_calc: { "FrozenTeacher": {
    teacher_run_id: "aev85iny",
    teacher_mini_epoch: -1,
}}
```

#### `Physical` (pseudo-teacher)
Not an actual teacher — this is the label for the physical reconstruction target.
Used when `target_and_aux_calc: "Physical"` in `LossPhysical` losses.

---

## 3. Pretraining Recipes

### 3.1 JEPA — Latent SSL only

**What trains:** Encoder (ae_local, ae_global, ae_aggregation) + JEPA predictor head.
**What does NOT train:** Decoder, fe_blocks (set to 0).
**Loss space:** Latent only.

**Reference config:** [config/config_jepa_from_forecast.yml](config/config_jepa_from_forecast.yml)

```yaml
training_mode: ["student_teacher"]

fe_num_blocks: 0                  # no forecast blocks needed
window_offset_prediction: 0       # no temporal offset

losses:
  "student-teacher":
    type: LossLatentSSLStudentTeacher
    weight: 1.0
    loss_fcts:
      "JEPA":
        head: transformer          # student has a projection head
        target_source_correspondence: {0 : {0 : "independent"}}
    target_and_aux_calc:
      EMATeacher:
        ema_halflife_in_thousands: 1e-1
        teacher_run_id: "aev85iny"
        model_param_overrides:
          # teacher uses identity head (no projection)
          training_config: { losses: { student-teacher: { loss_fcts: { JEPA: { head: identity }}}}}

model_input:
  "student":
    masking_strategy: "random"
    masking_strategy_config: { rate: 0.6 }

target_input:
  "teacher":
    masking_strategy: "forecast"  # teacher sees all tokens
```

**Variants (our 7 experiments):** See [Section 8](#8-jepa-from-forecast-experiment-grid-exps-01-07).

---

### 3.2 MTM — Masked Token Modeling in physical space

**What trains:** Everything — encoder, decoder, optionally fe_blocks.
**Loss space:** Physical variable space (MSE).
**Key idea:** Mask a fraction of tokens; model must reconstruct masked values in physical space.

**Reference config:** [config/config_exp/config_physical_random.yml](config/config_exp/config_physical_random.yml)

```yaml
training_mode: ["masking"]

fe_num_blocks: 6                  # optional; 0 for pure AE, >0 for latent dynamics
window_offset_prediction: 0       # 0 = reconstruction (same timestep)

losses:
  "physical":
    type: LossPhysical
    weight: 1.0
    loss_fcts:
      "mse":
        weight: 1.0
        target_source_correspondence: {0 : {0 : "complement"}}
                                  # student sees unmasked tokens, predicts masked tokens

model_input:
  "source":
    masking_strategy: "random"
    masking_strategy_config:
      rate: 0.4                   # student keeps 40% of tokens
      diffusion_rn: True          # optional: add noise to remaining tokens

target_input:
  "target":
    masking_strategy: "random"
    masking_strategy_config:
      rate: 0.4                   # same rate as source (complement gives the other 60%)
```

**MTM with spatial masking (harder, more structured):**

```yaml
# Replace both model_input and target_input masking_strategy with:
masking_strategy: "cropping_healpix"
masking_strategy_config:
  hl_mask: 3
  rate: 0.4
  method: "geodesic_disk"
# Keep: target_source_correspondence: {0: {0: "complement"}}
# → student sees the rest of the globe, predicts the cropped region in physical space
```

---

### 3.3 Combined JEPA + MTM

**What trains:** Everything — encoder, JEPA head, decoder.
**Loss space:** Both latent (JEPA) and physical (MSE) simultaneously.
**Key idea:** Jointly learn good representations (JEPA) and reconstruct observations (MTM).
Avoids representation collapse while anchoring latents to physical meaning.

**Reference configs:**
- [config/config_physical_jepa.yml](config/config_physical_jepa.yml)
- [config/config_exp/config_jepa_random.yml](config/config_exp/config_jepa_random.yml)
- [config/config_exp/config_jepa_cropping_healpix.yml](config/config_exp/config_jepa_cropping_healpix.yml)

```yaml
training_mode: ["masking", "student_teacher"]

fe_num_blocks: 6

# Two separate target and source pairs — one per loss
# Index 0: physical pair    Index 1: JEPA pair
losses:
  "physical":
    type: LossPhysical
    weight: 0.5
    loss_fcts:
      "mse":
        weight: 1.0
        target_source_correspondence: {0 : {0 : "complement"}}
    target_and_aux_calc: "Physical"

  "student-teacher":
    type: LossLatentSSLStudentTeacher
    weight: 0.5
    loss_fcts:
      "JEPA":
        weight: 8
        target_source_correspondence: {1 : {1 : "subset"}}  # ← index 1 for JEPA pair
    target_and_aux_calc: { EMATeacher: { ... } }

# Physical pair (index 0)
model_input:
  "source_physical":
    masking_strategy: "random"
    masking_strategy_config: { rate: 0.6, diffusion_rn: True }
    relationship: "complement"

# JEPA student pair (index 1)
  "source_jepa":
    masking_strategy: "random"
    masking_strategy_config: { rate: 0.4 }
    relationship: "subset"

target_input:
  "target_physical":             # index 0 — physical target
    masking_strategy: "random"
    masking_strategy_config: { rate: 0.4 }

  "target_jepa":                 # index 1 — teacher target
    masking_strategy: "healpix"
    masking_strategy_config: { rate: 0.2, hl_mask: 0 }
```

**Spatial (cropping_healpix) variant** — cone_distance for JEPA, disjoint for physical:

```yaml
# Reference: config/config_exp/config_jepa_cropping_healpix.yml
# Physical source sees disjoint region from physical target
# JEPA student is at cone_distance from JEPA teacher
target_source_correspondence (physical): {0: {0: "disjoint"}}
target_source_correspondence (JEPA):     {1: {1: "cone_distance"}}
```

---

### 3.4 Forecasting Pretraining

**What trains:** Everything — encoder, fe_blocks, decoder.
**Loss space:** Physical space, future timestep.
**Key idea:** The model sees the current state with light masking or no masking, and must predict
future states via the forecast engine (fe_blocks).

**Reference config:** [config/default_forecast_config.yml](config/default_forecast_config.yml)

```yaml
training_mode: ["masking"]

fe_num_blocks: 8                  # temporal dynamics in latent space
window_offset_prediction: 1       # target is 1 step ahead
forecast:
  time_step: 06:00:00             # step size
  num_steps: 4                    # autoregressive steps during training
  policy: "fixed"

losses:
  "physical":
    type: LossPhysical
    weight: 1.0
    loss_fcts:
      "mse": { weight: 1.0, target_source_correspondence: {0: {0: "independent"}} }

model_input:
  "input":
    masking_strategy: "forecast"  # all tokens visible at input
    masking_strategy_config: {}

target_input:
  "target":
    masking_strategy: "forecast"  # full target at future timestep
    masking_strategy_config: {}

freeze_modules: ""                # train everything from scratch (or from ckpt)
```

**Forecasting with masking (harder input):** Replace `model_input` with `random`, `rate: 0.8`
to make the encoder predict future states from partially observed current states.

---

## 4. Fine-tuning Recipes

For all fine-tuning, you set `load_chkpt` to point to your pretrained run:

```yaml
load_chkpt: { run_id: "YOUR_SSL_RUN_ID", mini_epoch: -1 }
```

---

### 4.1 Frozen Encoder — Reconstruction

**Purpose:** Evaluate encoder quality. If the frozen encoder's latents can be decoded back
to physical fields, the representations are physically meaningful.
**What trains:** Decoder (`PerceiverIOCoordConditioning`) and `fe_blocks` only.
**What is frozen:** Encoder (matched by `freeze_modules`).

**Reference config:** [config/config_jepa_finetuning.yml](config/config_jepa_finetuning.yml)

```yaml
training_mode: ["masking"]

fe_num_blocks: 6
freeze_modules: ".*encoder.*|.*latent_pre_norm.*|.*latent_heads.*"
                             # ↑ freezes ae_local, ae_global, ae_aggregation

load_chkpt: { run_id: "YOUR_SSL_RUN_ID", mini_epoch: -1 }

losses:
  "physical":
    type: LossPhysical
    weight: 1.0
    loss_fcts:
      "mse": { weight: 1.0, target_source_correspondence: {0: {0: "independent"}} }

model_input:
  "input":
    masking_strategy: "random"
    masking_strategy_config: { rate: 1.0 }   # all tokens — no masking

target_input:
  "target":
    masking_strategy: "random"
    masking_strategy_config: { rate: 1.0 }

forecast:
  time_step: 00:00:00
  num_steps: 0
  policy: null

streams_directory: "./config/streams/era5_synop_finetuning/"
```

**Metric:** Validation MSE on held-out dates. Lower = better representations.
**Use:** Compare all 7 SSL experiments with one standardised fine-tuning run each.

---

### 4.2 Frozen Encoder — Forecasting

**Purpose:** Test whether SSL representations transfer to the forecasting task without retraining
the encoder. The fe_blocks and decoder adapt, but the encoder stays fixed.
**What trains:** `fe_blocks` + Decoder.
**What is frozen:** Encoder.

```yaml
# Modify config_jepa_finetuning.yml:
training_mode: ["masking"]
fe_num_blocks: 6
freeze_modules: ".*encoder.*|.*latent_pre_norm.*|.*latent_heads.*"

load_chkpt: { run_id: "YOUR_SSL_RUN_ID", mini_epoch: -1 }

window_offset_prediction: 1       # shift target window forward in time
forecast:
  time_step: 06:00:00
  num_steps: 4                    # multi-step rollout
  policy: "fixed"

losses:
  "physical":
    type: LossPhysical
    weight: 1.0
    loss_fcts:
      "mse": { weight: 1.0, target_source_correspondence: {0: {0: "independent"}} }

model_input:
  "input":
    masking_strategy: "forecast"  # full input (no masking)
    masking_strategy_config: {}
```

**Metric:** RMSE on Z500, T850, U10, etc. at lead times 24h, 48h, 72h.

---

### 4.3 Full Fine-tune — Forecasting

**Purpose:** Gold-standard downstream evaluation. Everything adapts from the SSL-pretrained
initialisation. Tests whether SSL pretraining provides a better starting point than random init.
**What trains:** Everything.
**What is frozen:** Nothing.

**Reference config:** [config/default_forecast_config.yml](config/default_forecast_config.yml)

```yaml
training_mode: ["masking"]

fe_num_blocks: 8
freeze_modules: ""               # nothing frozen

load_chkpt: { run_id: "YOUR_SSL_RUN_ID", mini_epoch: -1 }

forecast:
  time_step: 06:00:00
  num_steps: 4
  policy: "fixed"

model_input:
  "input":
    masking_strategy: "forecast"
    masking_strategy_config: {}
```

**Metric:** Compare convergence speed (val loss vs training steps) and final RMSE across SSL
experiments. A better SSL encoder should reach lower RMSE with fewer fine-tuning steps.

---

## 5. Task Deep-Dives

### 5.1 Forecasting

The forecasting task uses the **forecast engine** (`fe_blocks`) to evolve latent representations
across time steps, then decodes the evolved latent into future weather fields.

**Forward pass for multi-step forecasting:**

```
Input: state at time t (all tokens, no masking)
        │
        ▼
    Encoder → latent z_t
        │
        ▼  [fe_blocks applied num_steps times, each producing z_{t+1}]
    z_{t+1} → Decoder → predicted fields at t+6h
    z_{t+2} → Decoder → predicted fields at t+12h
    ...
    z_{t+N} → Decoder → predicted fields at t+N*6h
        │
        ▼
    LossPhysical (MSE vs ERA5 at each lead time)
```

**Key parameters:**

```yaml
fe_num_blocks: 8                  # more blocks = more temporal capacity
window_offset_prediction: 1       # how many steps ahead the FIRST target is
forecast:
  time_step: 06:00:00             # temporal resolution of each autoregressive step
  num_steps: 4                    # number of autoregressive steps during training
  policy: "fixed"                 # "fixed" = always same num_steps
```

**Pretraining strategy for forecasting:** Use [config/default_forecast_config.yml](config/default_forecast_config.yml)
from scratch, or initialise encoder from JEPA pretraining for better representations.

**Fine-tuning strategy:** Use frozen encoder (Section 4.2) or full fine-tune (Section 4.3).

---

### 5.2 MTM as a Downstream Task

MTM can also serve as a downstream evaluation (not just pretraining). After SSL pretraining,
freeze the encoder and train only the decoder to reconstruct masked inputs. A good SSL encoder
should allow the decoder to reconstruct accurately even from a small fraction of visible tokens.

```yaml
# Probe: how well can a linear/shallow decoder reconstruct from SSL latents?
training_mode: ["masking"]
freeze_modules: ".*encoder.*|.*latent_pre_norm.*|.*latent_heads.*"

model_input:
  "source":
    masking_strategy: "random"
    masking_strategy_config: { rate: 0.2 }   # only 20% visible — very hard

target_input:
  "target":
    masking_strategy: "random"
    masking_strategy_config: { rate: 0.2 }

losses:
  "physical":
    type: LossPhysical
    loss_fcts:
      "mse":
        target_source_correspondence: {0: {0: "complement"}}
```

**Interpretation:** If a frozen encoder + decoder can reconstruct weather fields from 20% of
visible tokens, the latents encode strong global context. Compare reconstruction MSE at different
visible token fractions (20%, 40%, 60%) across SSL experiments.

---

### 5.3 Downscaling

Downscaling is the task of predicting high-resolution weather fields from low-resolution inputs.
In WeatherGenerator this can be approached in two ways depending on how much you want to modify
the data pipeline.

#### Option A — Spatial crop approach (minimal code change)

Use `cropping_healpix` to train the model on patches: the encoder processes a coarse spatial crop
of the globe, and the decoder must predict the corresponding high-resolution patch at the same
location. The "resolution" difference comes from the coarseness of the crop boundary.

**Pretraining with JEPA + spatial crops:**

```yaml
# Teacher sees large regional crop at high resolution
target_input:
  "teacher_region":
    masking_strategy: "cropping_healpix"
    masking_strategy_config:
      hl_mask: 3
      rate: 0.4          # 40% of globe (~78° radius)
      method: "geodesic_disk"

# Student sees only the inner sub-region (lower effective resolution)
model_input:
  "student_inner":
    masking_strategy: "cropping_healpix"
    masking_strategy_config:
      hl_mask: 3
      rate: 0.2          # 20% of globe (inner region only)
      method: "geodesic_disk"
    relationship: "subset"    # student is always inside teacher

# Loss: student must predict teacher's broader regional context
target_source_correspondence: {0: {0: "subset"}}
```

**Fine-tuning for downscaling (physical space):**

```yaml
training_mode: ["masking"]
freeze_modules: ".*encoder.*|.*latent_pre_norm.*|.*latent_heads.*"

# Source: coarse global field (low-res input)
model_input:
  "coarse_input":
    masking_strategy: "cropping_healpix"
    masking_strategy_config:
      hl_mask: 1          # very coarse boundary → fewer, larger parent cells
      rate: 0.4
      method: "geodesic_disk"

# Target: high-res field at same location (labels come from high-res stream)
target_input:
  "highres_target":
    masking_strategy: "cropping_healpix"
    masking_strategy_config:
      hl_mask: 4          # fine boundary → high-res target
      rate: 0.4
      method: "geodesic_disk"

# Physical MSE loss: decode to high-res variables
losses:
  "physical":
    type: LossPhysical
    loss_fcts:
      "mse": { target_source_correspondence: {0: {0: "independent"}} }

streams_directory: "./config/streams/your_highres_stream/"
```

#### Option B — Multi-resolution healpix levels (requires data pipeline)

This uses two `healpix_level` settings — encoder processes at `healpix_level: 4` (lower res),
decoder outputs at `healpix_level: 5` (higher res). Requires separate data streams at both
resolutions and modifications to model instantiation. Not currently in the default configs.

**Recommended approach:** Start with Option A (spatial crops). Once you confirm the training
pipeline works and the model learns spatial upscaling, consider Option B for true resolution
differences.

---

## 6. Masking Strategy Reference Card

Quick lookup for choosing the right strategy for each task.

| Task | Student/Source | Teacher/Target | Correspondence | Notes |
|------|----------------|----------------|---------------|-------|
| JEPA from global teacher | `random`, rate=0.4-0.6 | `forecast` (all) | `independent` | Base config |
| BERT-style latent MTM | `random`, rate=0.5, complement | `random`, rate=0.5 | `complement` | Zero token overlap |
| Variable difficulty | `random`, rate=0.5, `rate_sampling: True` | `forecast` | `independent` | |
| Spatial crop → global | `cropping_healpix`, geodesic | `forecast` | `independent` | |
| Two-crop fixed distance | `cropping_healpix`, geodesic, `dist=45` | `cropping_healpix`, geodesic | `cone_distance` | I-JEPA on sphere |
| Two-crop random distance | `cropping_healpix`, geodesic, random dist | `cropping_healpix`, geodesic | `cone_distance` | Diverse difficulty |
| Nested crops | `cropping_healpix`, rate=0.3, subset | `cropping_healpix`, rate=0.6 | `subset` | Zoom-out prediction |
| Physical MTM | `random`, rate=0.4 | `random`, rate=0.4 | `complement` | Reconstruct masked tokens |
| Forecasting | `forecast` | `forecast` (future step) | `independent` | via fe_blocks |
| Downscaling (spatial) | `cropping_healpix`, inner | `cropping_healpix`, outer | `subset` | Option A |
| Spatial separated MTM | `cropping_healpix`, disjoint | `cropping_healpix` | `disjoint` | Predict from far away |
| Healpix hierarchical | `healpix`, hl_mask=3 | `healpix`, hl_mask=0 | `complement` | Coarse→fine |

---

## 7. Config File Inventory

### Pretraining configs

| File | Mode | Loss | Teacher | Notes |
|------|------|------|---------|-------|
| [config_jepa_from_forecast.yml](config/config_jepa_from_forecast.yml) | `student_teacher` | Latent JEPA | EMATeacher (fast) | Init from `aev85iny`; base for Exps 01-07 |
| [config_jepa.yml](config/config_jepa.yml) | `masking + student_teacher` | Physical + Latent | EMATeacher | Combined MTM + JEPA |
| [config_physical_jepa.yml](config/config_physical_jepa.yml) | `masking + student_teacher` | Physical + Latent | EMATeacher | Combined, era5_nppatms_synop streams |
| [config_dinov2.yml](config/config_dinov2.yml) | — | — | — | DINOv2-style SSL |

### Experiment grid (masking ablation)

| File | Masking | Correspondence |
|------|---------|---------------|
| [config_exp/config_physical_random.yml](config/config_exp/config_physical_random.yml) | Random | complement |
| [config_exp/config_physical_cropping.yml](config/config_exp/config_physical_cropping.yml) | Cropping geodesic | disjoint |
| [config_exp/config_jepa_random.yml](config/config_exp/config_jepa_random.yml) | Random | subset |
| [config_exp/config_jepa_cropping_healpix.yml](config/config_exp/config_jepa_cropping_healpix.yml) | Cropping geodesic | cone_distance (random 0-90°) |
| [config_exp/config_cone_distance.yml](config/config_exp/config_cone_distance.yml) | Cropping geodesic | cone_distance (fixed 45°) |
| [config_exp/config_latent_random.yml](config/config_exp/config_latent_random.yml) | Random | — |
| [config_exp/config_latent_cropping.yml](config/config_exp/config_latent_cropping.yml) | Cropping | — |

### JEPA-from-forecast experiment grid (Exps 01-07)

| File | Key variation |
|------|--------------|
| [config_exp/config_jepa_fc_01_random_high_mask.yml](config/config_exp/config_jepa_fc_01_random_high_mask.yml) | Random, rate=0.4, independent |
| [config_exp/config_jepa_fc_02_random_complement.yml](config/config_exp/config_jepa_fc_02_random_complement.yml) | Random 50/50 complement |
| [config_exp/config_jepa_fc_03_random_rate_sampling.yml](config/config_exp/config_jepa_fc_03_random_rate_sampling.yml) | Random, variable rate |
| [config_exp/config_jepa_fc_04_cropping_geodesic_independent.yml](config/config_exp/config_jepa_fc_04_cropping_geodesic_independent.yml) | Geodesic crop, independent |
| [config_exp/config_jepa_fc_05_cone_fixed_45.yml](config/config_exp/config_jepa_fc_05_cone_fixed_45.yml) | Two crops, fixed 45° |
| [config_exp/config_jepa_fc_06_cone_random.yml](config/config_exp/config_jepa_fc_06_cone_random.yml) | Two crops, random 0-90° |
| [config_exp/config_jepa_fc_07_cropping_subset.yml](config/config_exp/config_jepa_fc_07_cropping_subset.yml) | Nested crops, subset |

### Fine-tuning and forecasting configs

| File | Mode | Frozen | Task |
|------|------|--------|------|
| [config_jepa_finetuning.yml](config/config_jepa_finetuning.yml) | `masking` | Encoder | Reconstruction (all tokens) |
| [default_forecast_config.yml](config/default_forecast_config.yml) | `masking` | Nothing | Full forecast fine-tune |

### Evaluation config

| File | Notes |
|------|-------|
| [config/evaluate/eval_config.yml](config/evaluate/eval_config.yml) | Inference / evaluation |

---

## 8. JEPA-from-Forecast Experiment Grid (Exps 01-07)

All experiments share identical architecture and training hyperparameters from
[config/config_jepa_from_forecast.yml](config/config_jepa_from_forecast.yml).
Only the masking strategy and target-source correspondence differ.

### Architecture (shared)

```yaml
ae_local_dim_embed: 2048,  ae_local_num_blocks: 0
ae_global_dim_embed: 2048, ae_global_num_blocks: 4
ae_aggregation_num_blocks: 0
fe_num_blocks: 0           # no forecast engine during SSL
load_chkpt: { run_id: "aev85iny", mini_epoch: -1 }
EMATeacher: { ema_halflife_in_thousands: 1e-1, teacher_run_id: "aev85iny" }
JEPA head:  transformer, 6 blocks, 12 heads, out_dim 2048
```

### Experiment summary

| Exp | Config | Student masking | Teacher masking | Correspondence | Research question |
|-----|--------|-----------------|-----------------|---------------|-------------------|
| Base | `config_jepa_from_forecast.yml` | random, rate=0.6 | forecast (all) | independent | Baseline |
| 01 | `config_jepa_fc_01_random_high_mask` | random, rate=0.4 | forecast (all) | independent | Does harder masking help? |
| 02 | `config_jepa_fc_02_random_complement` | random, rate=0.5, complement | random, rate=0.5 | complement | BERT-style latent MTM |
| 03 | `config_jepa_fc_03_random_rate_sampling` | random, rate=0.5, variable | forecast (all) | independent | Variable difficulty robustness |
| 04 | `config_jepa_fc_04_cropping_geodesic_independent` | geodesic disk, rate=0.5 | forecast (all) | independent | Spatial vs random structure |
| 05 | `config_jepa_fc_05_cone_fixed_45` | geodesic disk, rate=0.4, 45° from teacher | geodesic disk, rate=0.6 | cone_distance | Fixed spatial separation |
| 06 | `config_jepa_fc_06_cone_random` | geodesic disk, rate=0.4, 0-90° from teacher | geodesic disk, rate=0.6 | cone_distance | Variable spatial separation |
| 07 | `config_jepa_fc_07_cropping_subset` | geodesic disk, rate=0.3, inside teacher | geodesic disk, rate=0.6 | subset | Zoom-out prediction |

### Hypothesis matrix

| Comparison | Isolates |
|-----------|---------|
| 01 vs Base | Masking rate (difficulty) |
| 02 vs 01 | Complement vs independent teacher |
| 03 vs 01 | Fixed vs variable rate |
| 04 vs 01 | Spatial structure vs random (same rate) |
| 05 vs 04 | Two crops (local-to-local) vs one crop (local-to-global) |
| 06 vs 05 | Fixed vs random spatial separation |
| 07 vs 05/06 | Nested (always overlap) vs offset crops |

---

## 9. Evaluation Guide

### During SSL pretraining (online monitoring)

From collapse monitoring (already in configs 01-07):

| Metric | Good sign | Collapse sign |
|--------|-----------|--------------|
| `effective_rank` | High and stable | Drops to 1 |
| `dimension_variance` | Uniform across dims | One dim dominates |
| `singular_values` | Slow decay | Sharp drop (one large singular value) |
| `ema_beta` | Tracks well | — |
| JEPA val loss | Decreasing | Plateaus immediately |

### Downstream evaluation pipeline

For each SSL experiment `run_id_XX`:

**Step 1 — Reconstruction probe (fast, ~1 mini-epoch warmup)**

```
config_jepa_finetuning.yml with load_chkpt: {run_id: "run_id_XX"}
→ Metric: val MSE (reconstruction from all tokens)
→ Interpretation: can the frozen encoder's latents be decoded?
```

**Step 2 — Masked reconstruction probe (harder)**

```
Same config but model_input.rate: 0.2 (only 20% visible)
→ Metric: val MSE
→ Interpretation: how much global context does the frozen encoder encode?
```

**Step 3 — Frozen-encoder forecasting**

```
config_jepa_finetuning.yml + window_offset_prediction: 1 + forecast.num_steps: 4
→ Metric: RMSE at 24h / 48h / 72h
→ Interpretation: do latents transfer to temporal prediction?
```

**Step 4 — Full fine-tune forecasting (gold standard)**

```
default_forecast_config.yml with load_chkpt: {run_id: "run_id_XX"}
→ Metric: convergence speed + final RMSE
→ Interpretation: does SSL init outperform random init?
```

### What a good result looks like

- **Reconstruction probe:** Lower MSE than a random-init encoder trained the same number of steps.
- **Forecasting fine-tune:** Reaches lower RMSE in fewer steps than a randomly initialised baseline.
- **Collapse monitoring:** Effective rank stays above 50 for a 2048-dim latent.

---

## 10. Quick-Start Checklists

### Starting a new JEPA SSL experiment

- [ ] Choose a masking strategy from [Section 6](#6-masking-strategy-reference-card)
- [ ] Copy [config/config_jepa_from_forecast.yml](config/config_jepa_from_forecast.yml)
- [ ] Set `model_input`, `target_input`, and `target_source_correspondence`
- [ ] If using `cropping_healpix` with geometry: add `relationship:` to `model_input`
- [ ] Set `load_chkpt` (or remove if training from scratch)
- [ ] Set `wgtags.exp` and `wgtags.masking_type` for MLFlow tracking
- [ ] Verify collapse monitoring is enabled

### Starting a fine-tuning run

- [ ] Note the `run_id` from MLFlow for the SSL experiment to evaluate
- [ ] Copy [config/config_jepa_finetuning.yml](config/config_jepa_finetuning.yml)
- [ ] Set `load_chkpt: { run_id: "YOUR_SSL_RUN_ID", mini_epoch: -1 }`
- [ ] Confirm `freeze_modules` pattern matches the encoder modules you want to freeze
- [ ] Set `streams_directory` to the correct data stream
- [ ] Set `validate_before_training: True` to get a baseline before any training

### Common pitfalls

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Immediate collapse (effective rank → 1) | EMA half-life too small or LR too high | Increase `ema_halflife_in_thousands` or decrease `lr_max` |
| JEPA loss does not decrease | `target_source_correspondence` indices mismatch | Check that target/source indices in loss match order in `target_input`/`model_input` |
| NaN in loss | `center_distance_degrees` too large for given `rate` | Reduce distance or increase rate; ensure student + distance ≤ teacher radius |
| Fine-tuning MSE = random | `freeze_modules` regex too broad, freezing decoder too | Check regex only matches encoder blocks |
| `cone_distance` masking fails | `method` not set to `"geodesic_disk"` | Always use `method: "geodesic_disk"` with cone_distance |
| Multi-GPU hang | `with_fsdp: False` and `with_ddp: False` | Set `with_ddp: True` for multi-GPU |
