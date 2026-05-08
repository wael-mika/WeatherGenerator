# Advanced Masking Features

This document covers three masking capabilities added to the WeatherGenerator training pipeline:

1. [Skewed rate distribution](#1-skewed-rate-distribution-beta-sampling)
2. [Channel dropout](#2-channel-dropout)
3. [Per-variable-group masking strategies](#3-per-variable-group-masking-strategies)

All features are backward-compatible — existing configs continue to work without modification.

---

## Background: how masking works

Before diving into the new features, it helps to understand the terminology the code uses.

**Keep rate vs. masking rate.** The code operates on a *keep rate*: the fraction of spatial cells
the encoder is allowed to *see*. A keep rate of `0.2` means the encoder sees 20 % of cells —
equivalently, 80 % are masked. All `rate` parameters in the config are keep rates. The term
"masking rate" (80 % in the example) is not used in any config key.

**Spatial cell mask.** A mask is a boolean tensor of shape `(num_cells,)` over the HEALPix grid.
`True` = cell visible to encoder; `False` = cell is masked.

**Channel mask.** An optional secondary mask of shape `(num_channels,)` or
`(num_visible_tokens, num_channels)`. When present, dropped channels are zeroed out in token data.

**MAE complement.** In Mode A (MAE reconstruction), the target is automatically set to the
complement of the source mask — the decoder must reconstruct exactly the cells the encoder did not
see. The new per-group masking extends this to per-group complements.

---

## 1. Skewed Rate Distribution (Beta Sampling)

### Motivation

The original `rate_sampling: true` sampled the keep rate from a clipped absolute normal
distribution: `|N(rate, σ)|`. This produces a roughly symmetric distribution around `rate`. For
MAE pretraining, a distribution that is strongly skewed toward *high masking* (low keep rate) with
only an occasional low-masking sample is often more useful — the model is kept at a challenging
regime most of the time, but sees easy samples rarely to stabilise training.

The Beta distribution provides natural support on `[0, 1]` and can express arbitrary skewness
without clipping artefacts.

### Configuration keys

All keys live **inside `masking_strategy_config`**, next to the existing `rate` and `rate_sampling`
keys. They are optional — omitting any key uses the stated default.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rate` | float | *required* | Mean of the distribution (keep rate). For 80 % masking set `rate: 0.2`. |
| `rate_sampling` | bool | `false` | Must be `true` to activate any distribution sampling. |
| `rate_distribution` | `"normal"` \| `"beta"` | `"normal"` | Which distribution to sample from. `"normal"` is the original behaviour. |
| `rate_alpha` | float | `2.0` | Beta distribution α parameter. Controls how peaked the distribution is. Larger α → narrower peak around the mode. |
| `rate_beta` | float | auto | Beta distribution β parameter. **Auto-computed** from `rate` and `rate_alpha` so that the mean equals `rate`. Formula: `β = α × (1 − rate) / rate`. Override only if you need a specific α/β pair that doesn't honour the mean constraint. |
| `rate_flip` | bool | `false` | Mirror the distribution. When `true`, samples `1 − Beta(α, β)` instead of `Beta(α, β)`. Use this if the convention ever changes from *keep rate* to *masking rate*: flipping preserves the distribution shape while swapping which end has the long tail. |

### How Beta parameters map to distribution shape

| Goal | Suggested `rate` + `rate_alpha` | Result |
|------|--------------------------------|--------|
| High masking (mean 80 %, long tail toward low masking) | `rate: 0.2`, `rate_alpha: 2.0` | `β = 8`. Mode ≈ 0.125, mean = 0.2, right tail extends toward 1.0. |
| Very aggressive masking (mean 90 %, tight peak) | `rate: 0.1`, `rate_alpha: 3.0` | `β = 27`. Mode ≈ 0.074, very narrow distribution, rarely deviates from high masking. |
| Moderate masking with skew (mean 70 %) | `rate: 0.3`, `rate_alpha: 2.0` | `β ≈ 4.67`. Mild right skew. |
| Essentially fixed rate | `rate_alpha: 50.0` | Any `rate`: narrow Beta collapses to a point mass, effectively fixed. |

Increasing `rate_alpha` (while keeping the mean fixed via auto-β) makes the distribution narrower;
decreasing it toward 1 makes it increasingly J-shaped (heavier tail at 1.0).

### Examples

#### Minimal example — Beta sampling with 80 % masking

```yaml
# config/config_mae.yml
training_config:
  model_input:
    mae_random:
      masking_strategy: random
      num_samples: 1
      num_steps_input: 1
      masking_strategy_config:
        rate: 0.2              # mean keep rate (80 % masking)
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.0        # β auto-computed as 8.0; mean = 0.2
```

#### Using an explicit beta parameter

Override `rate_beta` when you want a specific (α, β) pair and don't care that the mean will differ
from `rate`:

```yaml
masking_strategy_config:
  rate: 0.2              # only used as fallback if rate_sampling is false
  rate_sampling: true
  rate_distribution: beta
  rate_alpha: 1.5
  rate_beta: 10.0        # explicit; auto-computation from rate is bypassed
```

#### Preparing for a future masking-rate convention change

If the codebase ever switches `rate` to mean *masking rate* instead of *keep rate*, set
`rate_flip: true`. The distribution is mirrored: `1 − Beta(α, β)` now has its mode near 0 (low
masking / high keep rate side), matching the intended shape after the convention change.

```yaml
masking_strategy_config:
  rate: 0.8              # interpreted as masking rate in the future convention
  rate_sampling: true
  rate_distribution: beta
  rate_alpha: 2.0
  rate_flip: true        # sample 1 − Beta(2, 0.5) ≈ peak near 0.8 keep-rate
```

#### Combined with healpix strategy

`rate_distribution` works with any strategy that calls `_get_sampling_rate`: `random`,
`healpix`, `cropping_healpix`, and `satellite_swath_sparse`.

```yaml
model_input:
  mae_healpix:
    masking_strategy: healpix
    num_samples: 1
    num_steps_input: 1
    masking_strategy_config:
      rate: 0.2
      hl_mask: 4
      rate_sampling: true
      rate_distribution: beta
      rate_alpha: 2.0
```

### Interaction with `mixed` strategy

When using `masking_strategy: mixed`, each sub-strategy has its own `masking_strategy_config`
under `strategy_configs`. Beta sampling must be specified per sub-strategy:

```yaml
masking_strategy_config:
  strategies: [random, healpix]
  strategy_configs:
    random:
      rate: 0.2
      rate_sampling: true
      rate_distribution: beta
      rate_alpha: 2.0
    healpix:
      rate: 0.25
      hl_mask: 4
      rate_sampling: true
      rate_distribution: beta
      rate_alpha: 3.0
```

---

## 2. Channel Dropout

### Motivation

Standard spatial masking drops *locations* from the encoder's view — all variables at a masked
cell are absent. Channel dropout is orthogonal: it drops entire *variables* (all cells of that
variable) with a very low probability. This forces the model to learn representations that are
robust to individual sensors or fields being completely unavailable — a realistic scenario for
observational data where entire instruments can fail.

Channel dropout applies **only to source (encoder) inputs** and **only during training**. Target
values always contain the full channel set so the reconstruction loss is not affected.

### Configuration

Add `channel_drop_rate` as a top-level key inside the stream config, **not** inside
`masking_strategy_config`:

```yaml
# config/streams/era5_1deg/era5.yml
ERA5:
  type: anemoi
  filenames: [...]
  stream_id: 0
  channel_drop_rate: 0.005   # 0.5 % chance each channel is zeroed per sample
  # ... rest of stream config unchanged
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `channel_drop_rate` | float | `0.0` | Per-channel probability of dropping (zeroing) that channel for the current sample. Applied independently to each channel. |

### Semantics

- Each channel is independently sampled: `keep = Uniform(0,1) >= channel_drop_rate`.
- Dropped channels have their values set to `0.0` across **all spatial cells** before tokenisation.
- The RNG used is the masker's shared RNG (same one used for spatial masks), ensuring that
  channel drop patterns change every epoch alongside the spatial masks.
- Channel dropping is **never applied to targets** — the decoder always sees the true channel
  values in the loss.
- Like `randomly_drop_as_source_rate` (which drops the entire stream), channel dropping is
  silently disabled at val/test stages.

### Choosing a value

The rate should be much smaller than the spatial masking rate because dropping a channel removes
*all* cells of that variable, which is a much larger information loss than masking one cell.

| Scenario | Suggested `channel_drop_rate` |
|----------|-------------------------------|
| Gentle regularisation | `0.001` – `0.005` |
| Moderate (simulate 1–2 failed sensors in ERA5 with ~100 channels) | `0.01` |
| Aggressive (stress-test robustness) | `0.02` – `0.05` |

Do not set this above `~0.05` — at high rates it degrades gradient signal without meaningful
regularisation benefit.

### Example — ERA5 with channel dropout

```yaml
# config/streams/era5_1deg/era5.yml
ERA5:
  type: anemoi
  filenames: [...]
  stream_id: 0
  channel_drop_rate: 0.005
  # ... rest unchanged
```

```yaml
# config/config_mae.yml  — no changes needed here
training_config:
  model_input:
    mae_random:
      masking_strategy: random
      num_samples: 1
      masking_strategy_config:
        rate: 0.2
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.0
```

### Combining with stream-level drop

`channel_drop_rate` and `randomly_drop_as_source_rate` are independent:
- `randomly_drop_as_source_rate` drops the **entire stream** (all channels, all cells) with one
  Bernoulli draw.
- `channel_drop_rate` independently drops **individual channels** of the stream.

Both can coexist. When the stream is dropped by `randomly_drop_as_source_rate`, the source mask is
all-False and channel dropping is skipped (there is nothing to drop into since the encoder ignores
the stream entirely).

---

## 3. Per-Variable-Group Masking Strategies

### Motivation

Standard masking applies a single spatial strategy to all variables in a stream. For
multi-variable datasets like ERA5, different physical fields can have very different observational
characteristics:

- **Precipitation** (`tp`, `cp`): sparse, highly localised, similar to satellite swath coverage.
- **Upper-air dynamics** (`u`, `v`, `z` at pressure levels): broad synoptic patterns, better
  suited to random or HEALPix masking.
- **Surface fields** (`2m_temperature`, `mean_sea_level_pressure`): high spatial correlation,
  coarse masking is appropriate.

Per-group masking lets each variable group have its own independently-sampled spatial mask. The
encoder sees the *union* of all group masks. In MAE mode, the decoder must reconstruct each
group's complement independently.

### Configuration modes

There are two configuration modes. They are mutually exclusive — if both are present on the same
stream, Mode A (stream config) takes priority.

---

#### Mode A — masking config inside `variable_groups` in the stream config

This is the most self-contained option: the masking specification lives in the same place as the
variable definition.

##### Full strategy config per group

```yaml
# config/streams/era5_1deg/era5.yml
ERA5:
  type: anemoi
  filenames: [...]
  stream_id: 0

  variable_groups:
    precipitation:
      variables: ["tp", "cp"]
      masking:
        strategy: satellite_swath       # any supported strategy
        config:
          num_swaths: 5
          swath_width_deg: 20
          orbit_drift_deg: -25

    upper_air:
      variables: ["u_\\d+", "v_\\d+", "z_\\d+", "t_\\d+", "q_\\d+"]
      masking:
        strategy: random
        config:
          rate: 0.15
          rate_sampling: true
          rate_distribution: beta
          rate_alpha: 2.0

    surface:
      variables: ["2m_temperature", "msl", "u10", "v10"]
      masking:
        strategy: healpix
        config:
          rate: 0.25
          hl_mask: 4

    _default:                           # all unmatched channels
      masking:
        strategy: random
        config:
          rate: 0.2
```

##### Rate-only shorthand (Option 2)

When all groups should use `random` masking but with different keep rates, you can use the
`masking_rate` shorthand instead of the full `masking` block. This is equivalent to
`masking: {strategy: random, config: {rate: <value>}}`.

```yaml
variable_groups:
  precipitation:
    variables: ["tp", "cp"]
    masking_rate: 0.1        # 10 % keep rate — aggressively masked

  upper_air:
    variables: ["u_\\d+", "v_\\d+"]
    masking_rate: 0.25

  _default:
    masking_rate: 0.2
```

> **Note:** `masking_rate` is a keep rate (not a masking rate). `masking_rate: 0.1` means 10 % of
> cells are visible (90 % are masked).

---

#### Mode B — `variable_groups` tag on `model_input` entries in the training config

This option routes existing `model_input` strategy entries to specific variable groups. Each
`model_input` entry generates one spatial mask that is applied to its tagged groups.

```yaml
# config/config_mae.yml
training_config:
  model_input:
    mae_precip:
      masking_strategy: satellite_swath
      num_samples: 1
      num_steps_input: 1
      variable_groups: [precipitation]           # NEW — which groups this mask applies to
      masking_strategy_config:
        num_swaths: 5
        swath_width_deg: 20
        orbit_drift_deg: -25

    mae_upper_air:
      masking_strategy: random
      num_samples: 1
      num_steps_input: 1
      variable_groups: [upper_air, _default]     # multiple groups can share one mask
      masking_strategy_config:
        rate: 0.2
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.0
```

> **When Mode B is active**, each `model_input` entry's `masking_strategy` is used *only* to
> generate the spatial mask for its tagged groups. The per-entry masks are then combined by the
> masker so that every variable group has its own spatial mask. If two entries tag the same group,
> the last one in the YAML wins.

> **Untagged entries** (no `variable_groups` key) are treated as applying to the whole stream in
> the original manner. Mixing tagged and untagged entries in the same stream is not recommended.

---

### How per-group masking works at runtime

1. **Mask generation** — The masker calls `_generate_cell_mask()` independently for each group,
   using its configured strategy. Each group gets its own `(num_cells,)` boolean tensor.

2. **Union mask** — The stream-level spatial mask is the bitwise OR of all group masks:
   ```
   stream_mask[cell] = True  if  any group covers cell
   ```
   This is the mask the encoder uses to select which cells to attend to.

3. **2-D channel mask** — Inside the tokeniser, for each visible token (cell that appears in the
   union mask), a boolean flag per channel is computed:
   ```
   channel_mask[token_at_cell_C, channels_of_group_G] = group_G_mask[C]
   ```
   Channels from groups that do **not** cover cell C are zeroed out in the token data. This ensures
   the encoder never sees group G's data at a cell that was masked for group G.

4. **MAE complement (Mode A MAE only)** — For each group G, the target is `~group_G_mask`:
   ```
   target_group_G_mask = ~source_group_G_mask
   ```
   The target union mask becomes the OR of all per-group complements. The decoder must reconstruct
   each group's channels at cells that were masked for that group — the loss is applied correctly
   because the 2-D target channel mask mirrors the source channel mask structure.

---

### Variable group patterns

Variable group membership uses Python `re.fullmatch` on the channel names. The same regex
syntax used in `variable_groups` for prediction heads (see `docs/variable_groups.md`) applies
here. Common patterns:

| Pattern | Matches |
|---------|---------|
| `"tp"` | Exact match for `tp` |
| `"u_\\d+"` | `u_50`, `u_100`, `u_500`, etc. (wind at pressure levels) |
| `"[uv]10"` | `u10` or `v10` (10 m wind components) |
| `"(tp\|cp)"` | Exact match for `tp` or `cp` |
| `".*_\\d+"` | Any variable with a numeric suffix |

Channels not matched by any named group fall through to `_default`. **Every channel must be
covered** — if `_default` is absent and any channel goes unmatched, training will error at the
tokenisation step (the 2-D channel mask computation detects ungrouped channels).

> **Tip:** You do not need to define `variable_groups` at all when using per-group masking. If
> groups are only configured for masking (no `pred_head` or `target_readout` keys), the masker
> will generate per-group masks but the model output heads remain at the stream level (one shared
> head for all channels).

---

### Interaction with other features

#### Channel dropout + per-group masking

Both can be active simultaneously. The channel drop mask is a 1-D per-channel boolean that is
applied *on top of* the 2-D group spatial mask:

```yaml
ERA5:
  channel_drop_rate: 0.005       # Feature 3: drop whole channels with low probability
  variable_groups:
    precipitation:
      variables: ["tp", "cp"]
      masking:
        strategy: satellite_swath
        config: {num_swaths: 5}
    _default:
      masking:
        strategy: random
        config: {rate: 0.2}
```

A channel that is dropped by `channel_drop_rate` is zeroed regardless of whether its group's
spatial mask covers the current cell.

#### Beta rate distribution + per-group masking

`rate_distribution: beta` can be specified in the `config` block of any group's `masking` entry:

```yaml
variable_groups:
  upper_air:
    variables: ["u_\\d+", "v_\\d+"]
    masking:
      strategy: random
      config:
        rate: 0.2
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.0
  _default:
    masking:
      strategy: healpix
      config:
        rate: 0.25
        hl_mask: 4
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 3.0
```

Each group samples its keep rate independently from its own Beta distribution.

#### `masking_override` in stream config

The existing `masking_override` mechanism (which overrides global `model_input` / `target_input`
config for a specific stream) is unaffected by per-group masking. `masking_override` operates on
the training-config-level strategies; per-group masking operates on the stream-config-level
`variable_groups` entries. They do not interact.

---

### Complete end-to-end example

The following shows a full config using all three new features together on a single ERA5 stream.

**Stream config** (`config/streams/era5_1deg/era5.yml`):

```yaml
ERA5:
  type: anemoi
  filenames: ["aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v7.zarr"]
  stream_id: 0
  loss_weight: 1.0
  token_size: 8
  tokenize_spacetime: false

  # Feature 3: drop individual variables with 0.5 % probability per sample
  channel_drop_rate: 0.005

  # Feature 1 (Mode A): per-group spatial masking strategies
  variable_groups:
    precipitation:
      variables: ["tp", "cp"]
      masking:
        strategy: satellite_swath
        config:
          num_swaths: 5
          swath_width_deg: 20
          orbit_drift_deg: -25

    upper_air:
      variables: ["u_\\d+", "v_\\d+", "z_\\d+", "t_\\d+", "q_\\d+"]
      masking:
        strategy: random
        config:
          rate: 0.15
          rate_sampling: true
          rate_distribution: beta   # Feature 2: Beta sampling
          rate_alpha: 2.0

    _default:
      masking:
        strategy: healpix
        config:
          rate: 0.2
          hl_mask: 4
          rate_sampling: true
          rate_distribution: beta   # Feature 2: Beta sampling
          rate_alpha: 3.0
```

**Training config** (`config/config_mae.yml`):

```yaml
healpix_level: 5

training_config:
  training_mode: masking

  losses:
    physical:
      type: LossPhysical
      loss_fcts:
        mse: {}

  model_input:
    mae_source:
      masking_strategy: random        # used only when NO variable_groups are set in stream config
      num_samples: 1
      num_steps_input: 1
      masking_strategy_config:
        rate: 0.2

  forecast:
    time_step: "00:00:00"
    num_steps: 0
    offset: 0
    policy: fixed
```

> When Mode A (`variable_groups.masking` in the stream config) is active, the `masking_strategy`
> in `model_input` is **not used for mask generation** — it is overridden by the per-group masks.
> The `model_input` entry is still needed so the correspondence and loss mapping can be resolved.

---

## Reference: which config key goes where

| Feature | Key | Location |
|---------|-----|----------|
| Beta distribution | `rate_distribution`, `rate_alpha`, `rate_beta`, `rate_flip` | Inside `masking_strategy_config` in `model_input` or inside `variable_groups.<group>.masking.config` |
| Channel dropout | `channel_drop_rate` | Top-level key inside the **stream config** (e.g. `era5.yml`) |
| Per-group masking (Mode A) | `variable_groups.<group>.masking` or `variable_groups.<group>.masking_rate` | Inside the **stream config** |
| Per-group masking (Mode B) | `variable_groups: [group1, group2]` | Inside each `model_input` entry in the **training config** |
| Existing: stream-level drop | `randomly_drop_as_source_rate` | Inside the **training config** under `training_config` |
| Existing: strategy override | `masking_override` | Top-level key inside the **stream config** |

---

## Troubleshooting

**`AssertionError: num_cells inconsistent with configured healpix level`**
Occurs when the masker is called before its RNG is initialised. Ensure `tokenizer.reset_rng(rng)`
is called before the first `build_samples_for_stream` call.

**Channel mask is all-True (channel dropout has no effect)**
Check that `channel_drop_rate` is set in the **stream config**, not in `model_input`. Also
confirm the training stage is `"train"` — channel dropout is disabled at val/test.

**`ValueError: N channel(s) unmatched by any variable group but no _default group is defined`**
When per-group masking is active, every channel must be covered. Add `_default` to your
`variable_groups` config.

**Groups in Mode B do not appear to generate independent masks**
Verify that each `model_input` entry has a distinct `variable_groups` list and that the tagged
group names exactly match those defined in the stream config's `variable_groups` section.

**Per-group masking is ignored (stream uses a single mask)**
Check that at least one group in `variable_groups` has a `masking` or `masking_rate` key (Mode A),
or that `model_input` entries have `variable_groups` tags (Mode B). Without these, the masker
falls back to the single-mask behaviour.
