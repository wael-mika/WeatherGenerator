# Masking and Decoder Ablation Guide

This document specifies a progressive ablation sequence for exploring per-group masking rates
and decoder configurations on ERA5 O96. It draws on the empirical variable analysis in
[era5_o96_variables_report.md](era5_o96_variables_report.md) and uses the combined
`variable_groups` machinery documented in [variable_groups_combined.md](variable_groups_combined.md).

---

## Scientific motivation

All existing physics configs (`config_mae_physics_optA/B`) give different variable groups
physics-appropriate output activations and, optionally, per-group decoders — but they apply
a **single global masking rate** to all groups uniformly.

This is problematic because information density varies enormously across ERA5 variables.

| Group | Spatial smooth | ACF₁ range | Effective info at O96 | Reconstructibility |
|---|---|---|---|---|
| T, Z, MSL | ~1.0 | 0.87–0.99 | Equivalent to ~T42 spherical harmonics | Near-perfect at 90% masked |
| U (upper), V (upper) | 0.95–0.99 | 0.85–0.99 | Jet stream structure | High |
| V (lower), 2T, SP | 0.83–0.97 | 0.10–0.88 | BL + diurnal cycle | Moderate |
| Q (troposphere) | 0.87–0.97 | 0.66–0.92 | Log-normal, frontal gradients | Moderate |
| TP, CP | **0.25** | 0.30–0.50 | Sub-grid at O96 | Very low — unresolvable |

With uniform masking, T and Z dominate the reconstruction loss because they return a
strong gradient signal with minimal effort. Q and precipitation get crowded out — the model
learns to reconstruct the easy fields and pays little attention to the hard ones.

**Key principle**: masking rate should be inversely proportional to reconstructibility.
Smooth, redundant variables can absorb very high masking (85%+). Rough, locally-forced
variables need gentle masking or should be treated as target-only.

All `rate` values below are **keep rates** (fraction of cells visible to the encoder).
`rate: 0.15` = 85% masked.

---

## Ablation ladder overview

```
Exp 0  Uniform masking, single head (existing baseline)
  │
  ▼
Exp 1  Coarse 3-group differentiated rates
  │    → Question: does any rate differentiation help over uniform masking?
  ▼
Exp 2  Fine 5-group differentiated rates + beta sampling
  │    → Question: coarse vs. fine grouping?
  ▼
Exp 3  Exp 2 + satellite swath masking for precipitation
  │    → Question: does masking strategy (not just rate) matter for precipitation?
  ▼
Exp 4  Exp 2 + per-group Option B decoders
       → Question: does group-specific decoder capacity add on top of masking?
```

Run in this order. Each experiment is a single config change on top of the prior one.
The biggest expected payoff is at the Exp 1 → 2 transition.

---

## Experiment 0 — Baseline

The existing `config/config_mae.yml` with a uniform global masking rate applied by `model_input`.
No `variable_groups` masking keys. Single shared decoder and head for all channels.

This is the control. All subsequent experiments should be compared against this.

---

## Experiment 1 — Coarse 3-group, differentiated rates

### Hypothesis

Separating precipitation from dynamics is the single most important split. Even a coarse
3-group partition should produce measurable improvement.

### Groups

| Group | Variables | Keep rate | Masking % | Rationale |
|---|---|---|---|---|
| `synoptic` | T all, Z all, U all, V 50–400 hPa, MSL, 2D | 0.15 | 85% | Smooth, redundant at O96; aggressive masking is appropriate |
| `moisture_surface` | V 700–1000 hPa, Q all, 2T, 10u, 10v, SP | 0.35 | 65% | BL + moisture: harder to reconstruct, more localized |
| `precipitation` | TP, CP | 0.75 | 25% | Nearly unresolvable at O96; ask very little |
| `_default` | All unmatched | 0.25 | 75% | Conservative fallback |

### Stream config (`variable_groups` block)

```yaml
variable_groups:

  synoptic:
    variables:
      - "t_\\d+"
      - "z_\\d+"
      - "u_\\d+"
      - "v_(50|100|150|200|250|300|400)"
      - "msl"
      - "2d"
    masking:
      strategy: random
      config:
        rate: 0.15
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.0
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity

  moisture_surface:
    variables:
      - "v_(700|850|925|1000)"
      - "q_\\d+"
      - "2t"
      - "10u"
      - "10v"
      - "sp"
    masking:
      strategy: random
      config:
        rate: 0.35
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.5
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity

  precipitation:
    variables: ["tp", "cp"]
    masking:
      strategy: random
      config:
        rate: 0.75
        rate_sampling: false
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus

  _default:
    masking:
      strategy: random
      config:
        rate: 0.25
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity
```

### Notes

- `skt` (skin temperature, ACF₁ = 0.10) will fall into `_default` unless explicitly listed in
  `moisture_surface`. It fits there physically; add `"skt"` to that group's variables if it
  appears in your target channels.
- `tcw` is typically in `target_exclude`; if not, it belongs in `synoptic` (smooth integrated
  column quantity, ACF₁ = 0.955).
- Stratospheric Q (`q_50`, `q_100`, `q_150`, `q_200`, `q_250`) is captured by `"q_\\d+"` in
  `moisture_surface`. Those levels are essentially numerical noise (ACF₁ ≈ 0). They do not
  harm Exp 1 because they are few, but Exp 2 handles them explicitly.

---

## Experiment 2 — Fine 5-group, differentiated rates + beta sampling

### Hypothesis

The physical regimes within `synoptic` and `moisture_surface` are meaningfully distinct.
Splitting dynamics from mass fields, and moisture from boundary-layer surface fields, should
improve latent routing and loss balance.

### Groups

| Group | Variables | Keep rate | Masking % | Rationale |
|---|---|---|---|---|
| `synoptic_mass` | T all, Z all, MSL | 0.12 | 88% | Most redundant fields in the dataset; Z500 reconstructible from ~10% of tokens |
| `dynamics` | U all, V 50–400 hPa, 2D | 0.22 | 78% | Jet structure is persistent but V is eddy-driven and harder than U |
| `boundary` | V 700–1000 hPa, 10u, 10v, 2T, SP, SKT | 0.38 | 62% | Orographic + BL + diurnal: locally determined |
| `moisture` | Q 300–1000 hPa | 0.45 | 55% | Log-normal distribution, sharp frontal gradients at upper levels |
| `precipitation` | TP, CP | 0.78 | 22% | Only skim the surface: convective-scale, sub-grid at O96 |
| `_default` | Q 50–250 hPa (stratospheric Q) | 0.90 | 10% | Near-zero variability; treat as context, not a reconstruction target |

### Stream config (`variable_groups` block)

```yaml
variable_groups:

  synoptic_mass:
    variables:
      - "t_\\d+"
      - "z_\\d+"
      - "msl"
    masking:
      strategy: random
      config:
        rate: 0.12
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.0
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity

  dynamics:
    variables:
      - "u_\\d+"
      - "v_(50|100|150|200|250|300|400)"
      - "2d"
    masking:
      strategy: random
      config:
        rate: 0.22
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.0
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity

  boundary:
    variables:
      - "v_(700|850|925|1000)"
      - "10u"
      - "10v"
      - "2t"
      - "sp"
      - "skt"
    masking:
      strategy: random
      config:
        rate: 0.38
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 2.5
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity

  moisture:
    variables:
      - "q_(300|400|500|600|700|850|925|1000)"
    masking:
      strategy: random
      config:
        rate: 0.45
        rate_sampling: true
        rate_distribution: beta
        rate_alpha: 3.0
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus

  precipitation:
    variables: ["tp", "cp"]
    masking:
      strategy: random
      config:
        rate: 0.78
        rate_sampling: false
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus

  _default:
    # Catches stratospheric Q (q_50, q_100, q_150, q_200, q_250) and any remainder.
    # Nearly always visible — the model sees it as context but is rarely asked to reconstruct it.
    masking:
      strategy: random
      config:
        rate: 0.90
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity
```

### Notes on stratospheric Q in `_default`

Stratospheric Q (50–250 hPa) has ACF₁ ≈ 0.000–0.033 and smooth ≈ 0.874–1.000. The signal
is float32 noise below the tropopause — values < 10⁻⁶ kg/kg. Reconstruction loss there is
uninformative and wastes gradient capacity. Setting keep rate = 0.90 means the encoder sees
it nearly always (useful context for diagnosing convective outflow via the upper-tropospheric
moisture field), but it is almost never a reconstruction target.

If `q_250` and `q_300` are hard to assign (they straddle the TTL), keep them in `_default`
at this stage. They can be split in a later iteration.

### Beta parameter rationale

| Group | rate | alpha | Implied beta | Distribution shape |
|---|---|---|---|---|
| `synoptic_mass` | 0.12 | 2.0 | 14.7 | Mode ≈ 0.08; long tail toward keep=1 |
| `dynamics` | 0.22 | 2.0 | 7.1 | Mode ≈ 0.15; moderate skew |
| `boundary` | 0.38 | 2.5 | 4.1 | Mode ≈ 0.32; tighter |
| `moisture` | 0.45 | 3.0 | 3.7 | Mode ≈ 0.40; narrow around a hard regime |
| `precipitation` | 0.78 | — | — | Fixed; no sampling |

The general rule: use `rate_alpha: 2.0` for smooth fields where occasional easy samples
stabilise training, and `rate_alpha: 3.0` for harder fields where you want the model to stay
near the configured difficulty.

---

## Experiment 3 — Exp 2 + satellite swath masking for precipitation

### Hypothesis

Random masking at 22% still creates contiguous unmasked regions that look nothing like
real precipitation observations. A satellite swath pattern (contiguous masked strips)
forces the model to reconstruct TP/CP from distant synoptic context, which is closer to
the physical inference problem and may improve representation quality.

### Change from Exp 2

Only the `precipitation` block changes:

```yaml
  precipitation:
    variables: ["tp", "cp"]
    masking:
      strategy: satellite_swath
      config:
        num_swaths: 5
        swath_width_deg: 20
        orbit_drift_deg: -25
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus
```

### Notes

- The satellite swath strategy creates ~5 orbital tracks of ~20° width each. The drift angle
  (-25°) simulates a sun-synchronous orbit inclination. Adjust `num_swaths` to control the
  effective keep rate (more swaths = more coverage).
- Because the swath strategy ignores `rate`, you do not set a `rate` key here.
- All other groups remain unchanged from Exp 2.
- Compare validation loss on precipitation specifically between Exp 2 and Exp 3 to isolate
  the effect of masking strategy vs. rate.

---

## Experiment 4 — Exp 2 + per-group Option B decoders

### Hypothesis

With differentiated masking, different groups may present different latent routing needs.
The cross-attention decoder conditions target tokens on the latent state — precipitation
(convective-scale) should need shallower attention depth than upper-air dynamics
(planetary-scale coherent structures).

### Groups and decoder configs

| Group | Decoder layers | Decoder heads | Rationale |
|---|---|---|---|
| `synoptic_mass` | 2 | 8 | Rossby wave structure: needs latent routing over large spatial range |
| `dynamics` | 2 | 8 | Jet stream + eddy coherence: same reasoning as synoptic_mass |
| `boundary` | 1 | 4 | BL fields are locally determined; shallow decoder is sufficient |
| `moisture` | 2 | 4 | Moisture transport is intermediate in scale; 4 heads sufficient |
| `precipitation` | 1 | 4 | Convective-scale, locally forced at O96; shallow decoder |
| `_default` | 1 | 4 | Stratospheric Q is context, not a primary prediction target |

### Stream config — full combined (masking + Option B decoders)

```yaml
  # Stream-level target_readout: required for schema completeness even when all groups
  # have their own. Also used as fallback if a group is accidentally missing target_readout.
  target_readout:
    num_layers: 2
    num_heads: 4

  variable_groups:

    synoptic_mass:
      variables:
        - "t_\\d+"
        - "z_\\d+"
        - "msl"
      masking:
        strategy: random
        config:
          rate: 0.12
          rate_sampling: true
          rate_distribution: beta
          rate_alpha: 2.0
      target_readout:
        num_layers: 2
        num_heads: 8
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    dynamics:
      variables:
        - "u_\\d+"
        - "v_(50|100|150|200|250|300|400)"
        - "2d"
      masking:
        strategy: random
        config:
          rate: 0.22
          rate_sampling: true
          rate_distribution: beta
          rate_alpha: 2.0
      target_readout:
        num_layers: 2
        num_heads: 8
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    boundary:
      variables:
        - "v_(700|850|925|1000)"
        - "10u"
        - "10v"
        - "2t"
        - "sp"
        - "skt"
      masking:
        strategy: random
        config:
          rate: 0.38
          rate_sampling: true
          rate_distribution: beta
          rate_alpha: 2.5
      target_readout:
        num_layers: 1
        num_heads: 4
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    moisture:
      variables:
        - "q_(300|400|500|600|700|850|925|1000)"
      masking:
        strategy: random
        config:
          rate: 0.45
          rate_sampling: true
          rate_distribution: beta
          rate_alpha: 3.0
      target_readout:
        num_layers: 2
        num_heads: 4
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus

    precipitation:
      variables: ["tp", "cp"]
      masking:
        strategy: random
        config:
          rate: 0.78
          rate_sampling: false
      target_readout:
        num_layers: 1
        num_heads: 4
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus

    _default:
      masking:
        strategy: random
        config:
          rate: 0.90
      target_readout:
        num_layers: 1
        num_heads: 4
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity
```

### Parameter overhead

Each Option B `target_readout` adds approximately 786k parameters per layer (cross-attention
+ self-attention + MLP at `dim_embed=256`, `pred_self_attention=True`). Six groups with a mix
of 1- and 2-layer decoders add roughly:
- synoptic_mass: 2 layers → ~1.6M params
- dynamics: 2 layers → ~1.6M params
- boundary: 1 layer → ~0.8M params
- moisture: 2 layers → ~1.6M params
- precipitation: 1 layer → ~0.8M params
- _default: 1 layer → ~0.8M params
- **Total: ~7.2M additional parameters**

This is modest relative to the encoder (typically > 1B params), but the extra TTE forward
passes add compute. Plan for ~10–15% longer decode steps versus Exp 2.

---

## Decision tree

```
After Exp 1:
  - Loss improved noticeably → proceed to Exp 2
  - Loss unchanged → check per-group loss contributions; if precipitation is
    dragging everything down, try excluding TP/CP from targets entirely

After Exp 2:
  - Further improvement → proceed to Exp 3 and Exp 4 in parallel (they are independent)
  - No improvement over Exp 1 → the coarse grouping is sufficient; don't add complexity

After Exp 3 (swath masking):
  - Precipitation val loss improves → keep swath for Exp 4
  - No effect → stay with random for precipitation; swath adds no structure benefit at O96

After Exp 4 (Option B):
  - Improvement concentrated in specific groups → those groups benefit from decoder depth;
    others can revert to Option A (shared decoder) to save compute
  - No improvement → Option A is sufficient; masking differentiation is the main lever
```

---

## Common pitfalls

### Channel coverage gaps

If training fails at startup with:
```
ValueError: N channel(s) unmatched by any variable group but no _default group is defined
```
A channel is not matched by any regex. Run this to diagnose:

```python
import re
channels = [...]  # your actual target_channels list
groups = {
    "synoptic_mass": [r"t_\d+", r"z_\d+", r"msl"],
    "dynamics":      [r"u_\d+", r"v_(50|100|150|200|250|300|400)", r"2d"],
    "boundary":      [r"v_(700|850|925|1000)", r"10u", r"10v", r"2t", r"sp", r"skt"],
    "moisture":      [r"q_(300|400|500|600|700|850|925|1000)"],
    "precipitation": [r"tp", r"cp"],
}
matched = set()
for grp, patterns in groups.items():
    pats = [re.compile(p) for p in patterns]
    for c in channels:
        if any(p.fullmatch(c) for p in pats):
            matched.add(c)
unmatched = set(channels) - matched
print("Unmatched:", unmatched)
```

All unmatched channels fall to `_default`, so the error only appears if `_default` is absent.

### Mode A vs. Mode B conflict

If `variable_groups.masking` keys are present in the stream config (Mode A) and the training
config's `model_input` entries also carry `variable_groups: [...]` tags (Mode B), Mode A wins
and a `WARNING` is emitted. Use only one mode per stream. All experiments above use Mode A.

### Checkpoint compatibility

Introducing `variable_groups` changes `pred_heads` and `target_token_engines` key names.
Old checkpoints load with `strict=False` — encoder weights are preserved, decoder weights
reinitialise. Budget extra warmup steps (~256–512) when loading an old checkpoint into a
grouped config.

---

## Reference: masking rate ↔ keep rate conversion

| Masking % | Keep rate (`rate`) |
|---|---|
| 88% | 0.12 |
| 85% | 0.15 |
| 78% | 0.22 |
| 75% | 0.25 |
| 65% | 0.35 |
| 62% | 0.38 |
| 55% | 0.45 |
| 25% | 0.75 |
| 22% | 0.78 |
| 10% | 0.90 |

All `rate` keys in the config are keep rates. Higher `rate` = less masking = easier task.
