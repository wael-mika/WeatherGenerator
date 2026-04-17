# Pretraining Experiments

Five XV-MAE pretraining configs designed to isolate and combine the mechanisms
most likely to improve over vanilla random-masking MAE.

Reference baseline configs live in `config/config_mae_masking/` (e.g. `random.yml`,
`random_xv.yml`, `mixed_xv.yml`).  The experiments here are all ablations or
extensions of those baselines.

---

## Experiment overview

| Config | Spatial strategy | Variable masking | Mask tokens | Core question |
|---|---|---|---|---|
| `exp01_random_xv_physics_groups.yml` | Random (15%) | Physics groups (0.45 / 0.30 / 0.10) | No | Do physics-informed rates beat uniform? |
| `exp02_mixed_xv_mask_tokens.yml` | Mixed (random+healpix+swath) | Uniform 30% | **Yes** | Does Feature 2 improve over Feature 1 alone? |
| `exp03_swath_xv_physics_groups.yml` | Swath sparse (25 passes, 30%) | Physics groups (0.45 / 0.25 / 0.10) | No | Best config for IASI-transfer pretraining? |
| `exp04_high_mask_rate_xv.yml` | Random (30%) | Uniform 50% | No | Do high masking rates force better representations? |
| `exp05_mixed_xv_full_stack.yml` | Mixed (random+healpix+swath) | Physics groups (0.40 / 0.30 / 0.10) | **Yes** | Does the full XV-MAE stack outperform all ablations? |

---

## Experiment rationale

### Exp01 — Physics-informed masking rates

**What it tests**: Whether assigning higher dropout probability to upper-air dynamical
variables (u, v, z) than to thermodynamic (t, q, r) or surface variables produces better
representations of cross-variable physical relationships.

**Motivation**: Thermal wind balance connects wind shear to horizontal temperature
gradients.  If the encoder is never forced to infer wind from temperature, it may learn
a shortcut of copying wind values without learning the underlying physics.  By masking
wind fields with p=0.45 while leaving temperature mostly visible, we create a training
signal that specifically rewards learning this relationship.

**Compare against**: `random_xv.yml` (uniform 30%).  If exp01 beats it, physics-informed
rates are worth pursuing.  If not, uniform masking already stresses all channels equally.

---

### Exp02 — Learned mask tokens (Feature 2)

**What it tests**: Whether replacing zeroed-out channels with a learned per-variable
scalar improves the encoder's ability to build an informative latent.

**Motivation**: When variable k is hidden, its token dimensions are set to 0.0.  The
encoder then sees a mix of true near-zero values and masked zeros — an ambiguous signal.
A learned scalar (initialised at 0, updated by gradients) gives the encoder a consistent
"variable k is not available here" signal that it can route around rather than interpret
as data.  This is the equivalent of proper [MASK] tokens in NLP-style BERT pretraining.

**Compare against**: `mixed_xv.yml` (identical except `use_variable_mask_tokens: False`).
Check: are `variable_mask_values` gradients non-zero after 200 steps?

---

### Exp03 — Swath + physics groups (IASI-transfer focused)

**What it tests**: Whether pretraining on the most observationally realistic geometry
(polar-orbit swath) combined with physics-informed variable masking provides the best
starting point for IASI finetuning.

**Motivation**: IASI finetuning asks the model to use radiance observations (which
constrain temperature and humidity profiles) to update the full atmospheric state
including wind.  Exp03 creates exactly this pretraining distribution: the encoder sees
a narrow swath of T/q observations and must reconstruct the global wind+geopotential
state.  The spatial geometry (satellite tracks) and the variable geometry (T visible,
wind masked) both match the downstream task.

**Compare against**: `satellite_swath_sparse.yml` baseline (no variable masking).
The relevant downstream benchmark is IASI finetuning RMSE on wind variables.

---

### Exp04 — High masking rate stress test

**What it tests**: Whether substantially harder pretraining (30% of cells visible,
50% of variables hidden) forces the model to build more useful representations than
easier masking ratios.

**Motivation**: At the standard 15% spatial keep rate, many masked cells have visible
neighbours within 1–2 grid lengths and can be reconstructed by local interpolation.
Driving both masking rates up simultaneously removes both spatial and variable
shortcut paths, potentially forcing the encoder to build more global and physically
meaningful representations.

**Risk**: Monitor training loss for the first 500 steps.  If it fails to decrease
below the initial value by step 300, reduce `channel_dropout_rate` to 0.35 or
`rate` to 0.20 as fallback.

---

### Exp05 — Full XV-MAE stack

**What it tests**: Whether all three components (mixed geometry + physics rates +
mask tokens) are complementary or whether some combination is redundant.

**Motivation**: This is the most capable configuration and the natural endpoint if
exp01, exp02, and exp03 each individually improve over their baselines.  It should
not be the first experiment run — start with exp01 and exp02 to determine which
components matter before running the full stack.

**Run order recommendation**:
```
random.yml         → establish baseline
random_xv.yml      → add uniform variable masking (+XV gain?)
exp01              → physics rates (+group rate gain?)
exp02              → mask tokens (+Feature 2 gain?)
exp03              → swath-focused (for IASI transfer only)
exp04              → high rate (if compute allows)
exp05              → full stack (last, not first)
```

---

## Variable group regex reference

All physics-group configs use `re.fullmatch(pattern, channel_name)`.  Adjust these
patterns to match your actual ERA5 channel names (check `stream_info.source_channels`):

| Group | Pattern examples | Covers |
|---|---|---|
| `upper_air_dynamics` | `u_\\d+`, `v_\\d+`, `z_\\d+` | Wind components, geopotential at pressure levels |
| `thermodynamics` | `t_\\d+`, `q_\\d+` | Temp, specific humidity at pressure levels |
| `surface` | `10u`, `10v`, `2t`, `2d`, `msl`, `sp`, `sst` | Surface/near-surface variables |

`\\d+` matches any numeric pressure level suffix (e.g. `t_500`, `u_850`).

**Anemoi ERA5 O96 naming notes**:
- Surface variables use ECMWF short names (`10u`, `10v`, `2t`, `2d`), **not** `u_10m`/`v_10m`/`t_2m`/`d_2m`.
- `r_\\d+` (relative humidity) and `w_\\d+` (vertical velocity) are not present in the
  aifs-ea-an-oper O96 dataset: `w_` is in `source_exclude`, `r_` is not produced for this analysis.
- `tcw`, `tp` are in `source_exclude` and will never match.
- `sst` may or may not be present depending on dataset version; harmless if absent.

---

## Shared config parameters

All experiments use identical model architecture, optimiser, and training schedule as
the baseline configs in `config/config_mae_masking/`.  The only differences are in
`model_input.variable_masking`, `use_variable_mask_tokens`, and spatial masking rates.
This makes ablation comparisons clean.
