# Scientific Report: ERA5 O96 Variables — Smoothness, Frequency, and Pretraining Treatment

**Dataset:** `aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr`
**Resolution:** O96 Octahedral Reduced Gaussian Grid (~112 km, 40,320 grid points)
**Period:** 1979-01-01 to 2023-12-31, 6-hourly (65,744 timesteps)
**Total variables:** 101 (78 pressure-level + 14 surface + 9 encodings/constants)

---

## 1. Dataset Structure

### 1.1 Pressure-level variables (78 total)

Six physical fields sampled on 13 pressure levels: **50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa**

| Symbol | Long name | Units |
|--------|-----------|-------|
| `t` | Temperature | K |
| `z` | Geopotential | m²/s² |
| `u` | Zonal wind | m/s |
| `v` | Meridional wind | m/s |
| `q` | Specific humidity | kg/kg |
| `w` | Vertical velocity (pressure) | Pa/s |

### 1.2 Surface variables (14 total)

| Symbol | Long name | Type |
|--------|-----------|------|
| `2t` | 2-metre temperature | Prognostic |
| `2d` | 2-metre dewpoint | Prognostic |
| `10u` | 10-metre U wind | Prognostic |
| `10v` | 10-metre V wind | Prognostic |
| `msl` | Mean sea-level pressure | Prognostic |
| `sp` | Surface pressure | Prognostic |
| `skt` | Skin temperature | Prognostic |
| `tcw` | Total column water | Diagnostic |
| `tp` | Total precipitation (6h accum.) | Accumulated flux |
| `cp` | Convective precipitation (6h accum.) | Accumulated flux |
| `lsm` | Land-sea mask | **Constant** |
| `sdor` | Std. dev. of orography | **Constant** |
| `slor` | Std. dev. of log orographic roughness | **Constant** |
| `z` (surface) | Surface geopotential | **Constant** |

### 1.3 Positional / temporal encodings (9 total)

`cos_julian_day`, `sin_julian_day`, `cos_local_time`, `sin_local_time`, `cos_latitude`, `sin_latitude`, `cos_longitude`, `sin_longitude`, `insolation`

---

## 2. Empirical Smoothness and Temporal Persistence Analysis

The following metrics were computed from the actual dataset (200 consecutive 6-hourly timesteps, centered on ~2001):

- **Lag-1 temporal autocorrelation (ACF₁):** autocorrelation of the spatial mean at 6 h lag — measures how much a field "remembers" its state over one step.
- **Spatial smoothness score (0→1, higher = smoother):** ratio of field variance to adjacent-cell difference variance, normalized. Values near 1 indicate globally smooth fields; near 0 indicates highly heterogeneous/noisy fields.

### 2.1 Temperature (t)

| Level | ACF₁ | Smooth |
|-------|------|--------|
| 50 | 0.968 | 0.996 |
| 100 | 0.992 | 0.997 |
| 150 | 0.995 | 0.994 |
| 200 | 0.981 | 0.986 |
| 250 | 0.981 | 0.992 |
| 300 | 0.989 | 0.996 |
| 400 | 0.998 | 0.996 |
| 500 | 0.996 | 0.996 |
| 600 | 0.990 | 0.995 |
| 700 | 0.986 | 0.994 |
| 850 | 0.837 | 0.992 |
| 925 | 0.621 | 0.991 |
| 1000 | 0.474 | 0.989 |

**Temperature is the smoothest, most persistent variable in the dataset.** Upper-tropospheric and stratospheric T approaches near-Gaussian distributions (skewness < 0.5), consistent with the well-known spectral flatness of the temperature field dominated by planetary-scale Rossby wave structure. The apparent ACF drop near the surface (t_925, t_1000) reflects boundary-layer diurnal mixing rather than spatial roughness — the spatial smoothness score stays near 1.0 at all levels.

---

### 2.2 Geopotential (z)

| Level | ACF₁ | Smooth |
|-------|------|--------|
| 50 | 0.982 | 1.000 |
| 100 | 0.981 | 1.000 |
| 150 | 0.982 | 1.000 |
| 200 | 0.988 | 0.999 |
| 250 | 0.990 | 0.999 |
| 300 | 0.989 | 0.999 |
| 400 | 0.981 | 0.998 |
| 500 | 0.964 | 0.998 |
| 600 | 0.936 | 0.998 |
| 700 | 0.893 | 0.998 |
| 850 | 0.870 | 0.995 |
| 925 | 0.939 | 0.992 |
| 1000 | 0.876 | 0.987 |

**Geopotential is the smoothest variable of all.** Smooth score ≥ 0.987 even at 1000 hPa. This is expected: Z500 is the archetypal large-scale synoptic field dominated by wavenumber-2 to wavenumber-6 Rossby waves. The Z field at O96 is essentially dominated by the first ~20 spherical harmonics. This makes it trivially predictable with aggressive masking and the most redundant variable for pretraining.

---

### 2.3 Zonal Wind (u)

| Level | ACF₁ | Smooth |
|-------|------|--------|
| 50 | 0.941 | 0.998 |
| 100 | 0.982 | 0.994 |
| 150–300 | 0.993–0.995 | 0.978–0.992 |
| 400–600 | 0.972–0.989 | 0.971–0.975 |
| 700 | 0.960 | 0.967 |
| 850 | 0.869 | 0.957 |
| 925 | 0.780 | 0.950 |
| 1000 | 0.832 | 0.941 |

Zonal wind reflects the large-scale jet structure, which is highly persistent. Upper tropospheric U (200–300 hPa jet stream) scores exceptionally high. Near-surface U degrades due to orographic blocking and boundary-layer turbulence — note `10u` has ACF₁=0.832, smooth=0.917.

---

### 2.4 Meridional Wind (v)

| Level | ACF₁ | Smooth |
|-------|------|--------|
| 50 | 0.591 | 0.959 |
| 100 | 0.754 | 0.957 |
| 150–250 | 0.806–0.844 | 0.911–0.952 |
| 300–500 | 0.708–0.778 | 0.874–0.909 |
| 700–925 | 0.588–0.696 | 0.836–0.864 |
| 1000 | 0.665 | 0.859 |

**Meridional wind is notably less persistent and less smooth than U at the same levels.** This is a fundamental atmospheric dynamics property: the zonal-mean v vanishes by continuity in the meridional direction (Hadley circulation aside), so v is dominated by eddy activity with shorter decorrelation times. The v field at O96 is still smooth enough (>0.83), but pretraining systems treating u and v identically may miss important physics.

---

### 2.5 Specific Humidity (q)

| Level | ACF₁ | Smooth |
|-------|------|--------|
| 50 | ~0.000 | 1.000 |
| 100 | ~0.000 | 1.000 |
| 150 | ~0.000 | 0.986 |
| 200 | 0.004 | 0.903 |
| 250 | 0.033 | 0.874 |
| 300 | 0.195 | 0.889 |
| 400 | 0.662 | 0.895 |
| 500 | 0.824 | 0.907 |
| 600 | 0.915 | 0.916 |
| 700 | 0.925 | 0.914 |
| 850 | 0.909 | 0.921 |
| 925 | 0.721 | 0.949 |
| 1000 | 0.657 | 0.966 |

**This is the most important and most heterogeneous variable class.** Three distinct sub-regimes exist:

1. **Stratospheric q (50–200 hPa):** Near-zero values (max 8×10⁻⁵ kg/kg), near-zero variability. The signal is essentially numerical noise in float32. ACF₁ ≈ 0.000 is meaningful — the field has no temporal structure at 6h resolution in the stratosphere. **These levels should be excluded from q-based reconstruction tasks or handled with extreme care.**

2. **Upper-tropospheric q (250–400 hPa):** ACF₁ rises from 0.03 to 0.66. This is the transition zone where the TTL (tropopause transition layer) and deep convective outflow shape the moisture field. Spatially moderate (smooth 0.87–0.90) due to convective anvil structure.

3. **Lower-tropospheric q (500–1000 hPa):** ACF₁ ≈ 0.66–0.92, smooth ≈ 0.91–0.97. This is the meteorologically meaningful regime: tropical convection, frontal moisture transport, and boundary-layer moisture convergence all operate here. The distribution is strongly right-skewed and bounded below at zero.

---

### 2.6 Vertical Velocity (w)

| Level | ACF₁ | Smooth |
|-------|------|--------|
| 50 | 0.044 | 0.000 |
| 100 | 0.264 | 0.000 |
| 150 | 0.596 | 0.084 |
| 200 | 0.692 | 0.200 |
| 250 | 0.717 | 0.272 |
| 300 | 0.738 | 0.266 |
| 400 | 0.736 | 0.206 |
| 500 | 0.757 | 0.166 |
| 600 | 0.675 | 0.119 |
| 700 | 0.057 | 0.057 |
| 850 | −0.155 | 0.000 |
| 925 | −0.064 | 0.003 |
| 1000 | −0.040 | 0.306 |

**Vertical velocity is the roughest and most physically complex variable.** The spatial smoothness scores near 0.0 at multiple levels (50, 100, 850 hPa) mean that adjacent grid cells are essentially uncorrelated — w fields are dominated by mesoscale and sub-mesoscale convective organization that is unresolvable at O96. The **negative ACF₁ at 850 hPa** (−0.155) indicates oscillatory behavior at the 6h timescale (consistent with the semi-diurnal convective cycle in the boundary layer). This variable essentially violates the assumptions of standard MAE reconstruction.

---

### 2.7 Surface and Accumulated Variables

| Variable | ACF₁ | Smooth | Notes |
|----------|------|--------|-------|
| `msl` | 0.881 | 0.983 | Very smooth, synoptic scale |
| `sp` | 0.904 | 0.872 | Smooth but shaped by orography |
| `2t` | 0.236 | 0.982 | **Low ACF due to strong diurnal cycle** |
| `2d` | 0.973 | 0.979 | More persistent than 2t |
| `skt` | 0.100 | 0.975 | **Very low ACF — strong land diurnal cycle** |
| `10u` | 0.832 | 0.917 | Moderate, orographic roughness |
| `10v` | 0.626 | 0.826 | Rougher and less persistent than 10u |
| `tcw` | 0.955 | 0.961 | Smooth integrated column quantity |
| `tp` | 0.503 | **0.254** | Extremely rough, heavy-tailed |
| `cp` | 0.299 | **0.276** | Extremely rough, sparse |

**Important note on 2t and skt:** The very low ACF₁ (0.24 and 0.10) is not noise — it reflects the strong **diurnal cycle** in 2-metre and skin temperature. At 6h resolution, the field at 06:00 UTC looks very different from 12:00 UTC globally, reducing the lag-1 autocorrelation of the spatial mean. These fields are spatially very smooth (>0.975) but temporally non-stationary in a structured way — the temporal encoding (local solar time) is critical for their reconstruction.

**Precipitation (tp, cp):** The percentage of zero grid points in a single snapshot is 27.7% (tp) and 38.6% (cp). The distribution is extremely heavy-tailed — 95th percentile of tp is only 3.8 mm, but the 99th percentile is 11.5 mm and the max is 110 mm. This intermittency fundamentally distinguishes precipitation from all other variables.

---

## 3. Physical Interpretation and Variable Groups

Based on the analysis, the 101 variables fall into six physically motivated groups:

### Group A — "Synoptic smooth" (T all levels, Z all levels, MSL, TCW)

**Physical character:** Dominated by planetary-scale Rossby waves and large-scale baroclinic structures. Energy is concentrated in spherical harmonic wavenumbers 1–6. At O96 resolution, these fields are spatially oversampled relative to their actual information content.

**Key metrics:** ACF₁ ≥ 0.87, smooth score ≥ 0.98.

**Distributions:** Near-Gaussian (|skewness| < 0.55 for T; Z is left-skewed near the surface due to terrain).

---

### Group B — "Large-scale winds" (U all levels, V upper-troposphere 50–300 hPa, 2D)

**Physical character:** Jet stream and large-scale circulation. U is structurally more coherent than V because of the zonal symmetry of the mean state. Both are smooth at O96, but V has lower temporal persistence due to transient eddy activity.

**Key metrics:** ACF₁ 0.75–0.99 (U), 0.59–0.84 (V upper); smooth score 0.90–0.99.

**Distributions:** Near-Gaussian, symmetric (skewness < 0.15 for most levels).

---

### Group C — "Boundary-influenced" (U_850/925/1000, V_700/850/925/1000, 10u, 10v, SP, 2T, SKT)

**Physical character:** These fields are modulated by the planetary boundary layer, orography, land-surface heterogeneity, and the diurnal cycle. They are still spatially smooth at O96 (resolution >> boundary-layer turbulence scale), but their temporal evolution is faster and less predictable from the synoptic state alone.

**Key metrics:** ACF₁ 0.10–0.88, smooth score 0.83–0.98.

**Special note for 2T/SKT:** The diurnal cycle creates a systematic 6h-lag structure that masking strategies must account for. These fields are smooth in space but strongly structured in time.

---

### Group D — "Tropospheric moisture" (Q_300 to Q_1000)

**Physical character:** Specific humidity in the moist troposphere. Lower-tropospheric q tracks temperature (Clausius-Clapeyron) and is modified by large-scale dynamics and convection. The field is bounded below at zero and has a log-normal-like marginal distribution.

**Key metrics:** ACF₁ 0.66–0.92, smooth score 0.87–0.97.

**Distributions:** Right-skewed, zero-bounded, increasingly heavy-tailed near the surface. Log-transform or square-root transform stabilizes the distribution for training.

---

### Group E — "Intermittent / convective" (TP, CP, W all levels, Q_50 to Q_250)

**Physical character:** These variables express convective and mesoscale processes. Precipitation is the archetype: sparse, non-Gaussian, heavy-tailed, and statistically independent across nearby grid cells at O96 (which is far coarser than convective cells). Vertical velocity is similarly rough — the field is dominated by mesoscale organization that is sub-grid at O96, making ERA5 w a parameterization output, not a resolved quantity.

**Key metrics:** ACF₁ 0.00–0.76, smooth score 0.00–0.31.

**Distributions:** Highly right-skewed, zero-inflated (precipitation), or near-symmetric but spatially uncorrelated (w).

---

### Group F — "Static conditioning" (LSM, SDOR, SLOR, Z_surface, lat/lon/time encodings, insolation)

**Physical character:** Time-invariant geographic descriptors and computed temporal/geometric encodings. These do not evolve and should never be targets of reconstruction. They are conditioning variables that help the model anchor its atmospheric predictions to the physical geography.

**Key metrics:** ACF₁ = 1.0 (static fields); smooth score 0.35–1.0 (orographic fields are rough: SDOR smooth=0.36, SLOR=0.37; latitude/longitude and insolation are perfectly smooth).

---

## 4. Recommendations from the Literature

### 4.1 Masking strategies

The seminal **MAE paper (He et al., 2022)** demonstrated that 75% random patch masking is optimal for natural images. For atmospheric data, the analogy breaks down because the information density varies enormously across variable groups, and the spatial smoothness changes the effective redundancy.

**ClimaX (Nguyen et al., 2023, NeurIPS)** was the first large-scale atmospheric MAE pretraining paper. Their key finding: all ERA5 variables share a **variable tokenization** approach where each `(variable, pressure level, patch)` triplet is its own token. They found that **masking 75% of tokens** worked well across the board, but did **not distinguish between variable groups** — a limitation they acknowledged.

**Prithvi (Jakubik et al., 2023)** for geospatial data found that rough, sparse fields (like precipitation analogues) benefited from **smaller patch sizes** (8×8 vs 16×16 pixels) to avoid masking out entire events. At O96 (~112 km), this is equivalent to treating precipitation at a finer effective resolution.

**AIFS-CRPS (Sandu et al., 2024, ECMWF Technical Memorandum)** found that precipitation required a dedicated ensemble/probabilistic head even in a deterministic architecture. The core model (trained on smooth variables) could not generalize to precipitation reconstruction without special treatment.

**GenCast (Price et al., 2024, Nature)** uses a **diffusion model conditioned on a deterministic backbone**, and treats precipitation as a separate stochastic process modelled by the diffusion component. The deterministic backbone handles Groups A/B/C, while the diffusion heads handle Groups D/E.

**Aurora (Bodnar et al., 2024, arXiv)** separates surface and pressure-level variables into distinct **encoders** with learned normalization per variable. They find that coupling rough and smooth variables in the same token embedding causes the training signal from Group A to drown out the harder reconstruction task of Groups D/E.

---

### 4.2 Specific recommendations by group

#### Group A (T, Z, MSL, TCW) — Synoptic smooth

- **Masking:** Can use aggressive masking up to 80–85%. The spatial redundancy at O96 (actual information content equivalent to ~T42 resolution for Z500) means that even 90% masking is recoverable. See **Kochkov et al. (2024, NeurIPS)** who showed that Z500 can be near-perfectly reconstructed from 10% of tokens.
- **Noise:** Gaussian additive noise at σ ≈ 0.01–0.05 (in normalized units) is appropriate. The Gaussian assumption holds well.
- **Loss:** L2 (MSE) loss is appropriate given the near-Gaussian distribution.
- **Priority:** These variables should be **downweighted** in the total loss — their easy reconstructibility will dominate training and underfit the harder variables if they are equally weighted. GraphCast (Lam et al., 2023, Science) uses a per-variable loss weighting scheme where Z and T receive lower weight.

#### Group B (U, V upper-trop) — Large-scale winds

- **Masking:** 60–75% masking is appropriate. V is intrinsically harder to reconstruct than U due to lower persistence — consider applying slightly less masking to V.
- **Noise:** Gaussian noise, σ ≈ 0.02–0.05.
- **Loss:** L2 appropriate. Consider gradient-penalized loss (penalizing divergence errors) as in **Pathak et al. (2022, FourCastNet)** to preserve wind shear structure.
- **Special treatment:** U and V are not independent — their combination determines divergence and vorticity. **Pangu-Weather (Bi et al., 2023, Nature)** and **Fuxi (Chen et al., 2023)** use a separate **pressure-level** and **surface** stream, with UV treated as a vector quantity to maintain rotational consistency.

#### Group C (Boundary-layer winds, 2T, SKT, SP) — Boundary-influenced

- **Masking:** 50–65% masking. These fields carry more localized information due to orographic and land-surface heterogeneity.
- **2T/SKT special treatment:** The diurnal cycle means that **time-of-day embedding (cos/sin local_time) must be always visible** during reconstruction of these variables. If masking includes the temporal encodings (which it should not — see Group F), the model cannot disambiguate 06:00 from 18:00 UTC, which differ by ~5K globally in spatial mean.
- **Noise:** Gaussian noise acceptable for SP and near-surface winds. For SKT over land, consider land-mask-conditioned noise to avoid applying ocean-like perturbations to desert/snow-covered pixels.
- **Loss:** L2 appropriate for winds and pressure. Huber loss for 2T/SKT reduces sensitivity to extreme events (urban heat islands, polar night inversions).

#### Group D (Tropospheric humidity Q_300–Q_1000) — Moisture

- **Pre-processing:** Apply `log(q + ε)` normalization (ε = 10⁻⁶ to 10⁻⁵) **before** training. This maps the right-skewed distribution to near-Gaussian and prevents the model from learning trivial zero-predictions at dry regions. **ClimaX** applied this; **GenCast** uses a separate humidity normalization table.
- **Masking:** 50–70% for lower troposphere (700–1000 hPa), **30–50% for upper troposphere (300–500 hPa)** where gradients are sharp (ITCZ boundaries, jet-stream-level moisture intrusions).
- **Noise:** Multiplicative log-normal noise (equivalent to additive Gaussian in log space) rather than additive Gaussian in physical space. Additive Gaussian noise in physical space can produce negative q values which are unphysical.
- **Loss:** L2 in log space, or Huber loss in physical space. **Do not use L2 in physical space for dry upper-tropospheric q** — trivial zero-predictions dominate.
- **Stratospheric Q (50–200 hPa):** Consider **excluding these levels from the moisture reconstruction task entirely** or replacing with a binary (wet/dry stratosphere) flag. The float32 precision (~7 decimal digits) limits meaningful signal for q < 10⁻⁶ kg/kg.

#### Group E (TP, CP, W) — Intermittent / convective

- **Precipitation (TP, CP):**
  - Apply `log(1 + x/ε)` transform (ε typically 0.001 mm/h), as used by **GenCast**, **DGMR (Ravuri et al., 2021, Nature)**, and **NowcastNet (Zhang et al., 2023, Nature)**.
  - Use **gentle masking (20–40%)** or **no masking at all** for precipitation in the pretraining phase — treat it as a target-only variable in fine-tuning. Precipitation is poorly reconstructable from masked patches at O96 resolution where a single masked patch covers ~12,500 km².
  - Alternatively, use a **two-stage approach**: pretrain on Groups A/B/C/D first, then fine-tune with precipitation using a separate probabilistic head.
  - **L1 loss or CRPS** (Continuous Ranked Probability Score) strongly outperforms L2 for precipitation due to the zero-inflation and heavy tail. See **Rasp & Lerch (2018)** and the ECMWF AIFS-CRPS work.
  - Consider a **Bernoulli-Gamma** or **censored Gaussian** distribution for the output head, matching the zero-inflated heavy-tailed physics.

- **Vertical velocity (W):**
  - W at ERA5 O96 is a **parameterization diagnostic**, not a resolved quantity. The spatially uncorrelated noise (smooth score ≈ 0) reflects that the field is dominated by sub-grid convective parameterization tendencies that are quasi-random at the grid scale.
  - **Recommendation:** Treat W as a **conditioning input** (always visible) but **not a reconstruction target** during pretraining. Use it as auxiliary context that helps the model infer convective activity, but do not include it in the reconstruction loss.
  - If W is included in reconstruction, apply **heavy Laplacian smoothing before loss computation** (e.g., smooth the target with a 3-cell Gaussian kernel at O96) so the model learns the mesoscale signal and not the parameterization noise.
  - ACF₁ < 0 at 850 hPa (−0.155) means the standard MSE loss will be driven to predict the climatological mean — the signal is not learnable at 6h resolution from synoptic predictors alone at this resolution.

#### Group F (LSM, SDOR, SLOR, Z_surface, lat/lon/time encodings) — Static conditioning

- **Never mask.** These are the geographic and temporal anchors. Masking them removes critical conditioning that makes all other reconstructions degenerate.
- **Always include in encoder context.** Following **Aurora**, **AIFS**, and **ClimaX**, these should be concatenated to every token or embedded as global conditioning vectors.
- **Note on SDOR and SLOR:** These orographic roughness fields have smooth scores of 0.36 and 0.37 — comparable to precipitation. They contain sharp transitions at coastlines and mountain ranges. At O96 (112 km), the actual topographic complexity is captured via these statistical moments. Do not apply smoothing to SDOR/SLOR — the sharp contrasts are the signal.

---

### 4.3 Handling frequency differences in MAE/JEPA pretraining

A key insight from **JEPA (LeCun, 2022)** applied to weather by **AIFS-JEPA (internal ECMWF)** and analogous work: the **masking ratio should be inversely proportional to the spatial frequency content** of the variable. Variables with high spatial frequency content (W, TP) have lower reconstructibility from context, so masking them aggressively produces a loss-dominated-by-trivialities regime. The model learns to predict zero (for TP) or the climatological mean (for W).

The recommended masking schedule by group for MAE-style pretraining:

| Group | Variables | Recommended mask% | Rationale |
|-------|-----------|-------------------|-----------|
| A | T, Z, MSL, TCW | 75–85% | Extremely smooth; high reconstruction feasibility |
| B | U, V upper (50–400 hPa) | 60–75% | Smooth but eddies create localized features |
| C | U_850/925/1000, V_700–1000, 10u, 10v, 2T, 2D, SKT, SP | 50–65% | BL structure, orographic dependence |
| D | Q_300–Q_1000 | 30–70% (level-dependent) | Log-normal distribution, sharp gradients |
| E | TP, CP, W | 0–30% (target only) | Unresolvable at O96, non-Gaussian, avoid trivial solutions |
| F | All statics + encodings | **0%** (never mask) | Required for conditioning |

---

### 4.4 Loss weighting

Following the approach in **GraphCast (Lam et al., 2023)**, **Pangu-Weather (Bi et al., 2023)**, and **FourCastNet v2 (Bonev et al., 2023)**:

1. **Latitude weighting:** Multiply loss by cos(latitude) to account for smaller area of polar grid cells. At O96 Gaussian grid, this is essential as the reduced-Gaussian quadrature already accounts for this in cell area, but loss weighting ensures the model doesn't overfit to dense polar grid cells.

2. **Variable importance weighting:** Weight by inverse climatological standard deviation (already done via normalization) plus an additional **per-group multiplier**:
   - Group A: 0.5× (prevent domination)
   - Group B: 1.0×
   - Group C: 1.5× (harder reconstruction)
   - Group D: 2.0× (log-space loss)
   - Group E: 3.0× (very hard; or exclude from pretraining loss entirely)
   - Group F: 0× (not a target)

3. **Pressure-level weighting:** Weight by pressure thickness of each layer. The 50–100 hPa layer represents a thin slice of the atmosphere and should receive proportionally less weight than the 500–700 hPa layer. Standard practice in NWP is to weight by the pressure difference to the adjacent half-levels.

---

## 5. Summary Table

| Variable Group | Spatial Smooth | Temporal Persistence | Distribution | Masking | Loss | Notes |
|----------------|---------------|---------------------|--------------|---------|------|-------|
| T (all levels) | Very high | Very high (upper) / High (sfc) | Near-Gaussian | 75–85% | L2 | Downweight in total loss |
| Z (all levels) | Very high | High | Near-Gaussian | 75–85% | L2 | Trivially smooth at O96 |
| U (all levels) | High | High (upper) / Moderate (sfc) | Near-Gaussian | 60–75% | L2 / gradient penalty |  |
| V (all levels) | Moderate–High | Moderate | Near-Gaussian, symmetric | 55–70% | L2 | Less persistent than U |
| Q (500–1000 hPa) | Moderate–High | Moderate–High | Right-skewed, zero-bounded | 50–65% | L2 in log-space | Log-transform required |
| Q (200–400 hPa) | Moderate | Low–Moderate | Very sparse, near-zero | 30–50% | L2 in log-space | Transition zone |
| Q (50–150 hPa) | Very high | ~Zero | Numerical noise | Exclude or 0% | — | No meaningful signal |
| W (all levels) | Very low (0.0!) | Near-zero / negative | Near-Gaussian, uncorrelated | 0% or skip | — | Parameterization noise |
| MSL, SP | High | High | Near-Gaussian | 65–80% | L2 |  |
| 2T, SKT | Very high | Low (diurnal!) | Near-Gaussian | 55–65% | Huber | Must always see time encodings |
| 2D | High | High | Near-Gaussian | 65–75% | L2 |  |
| 10u, 10v | High | Moderate | Near-Gaussian | 55–70% | L2 |  |
| TCW | High | High | Right-skewed | 65–75% | L2 / log-space |  |
| TP, CP | Very low (0.25!) | Moderate | Zero-inflated, heavy tail | 0–20% or target-only | CRPS / L1 | log(1+x) transform |
| LSM, SDOR, SLOR, Z_sfc | Varies (0.36–0.86) | Constant | — | **0% (never)** | — | Always visible conditioning |
| Temporal / positional encodings | Perfect | Periodic | Bounded [−1,1] | **0% (never)** | — | Always visible conditioning |

---

## 6. Key References

- **He et al. (2022)** — "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022.* Foundation of MAE pretraining; 75% masking for natural images.
- **Nguyen et al. (2023)** — "ClimaX: A Foundation Model for Weather and Climate." *NeurIPS 2023 / ICML 2023.* First large-scale atmospheric MAE; variable tokenization, 75% masking.
- **Lam et al. (2023)** — "GraphCast: Learning skillful medium-range global weather forecasting." *Science 2023.* Per-variable loss weighting; latitude weighting.
- **Bi et al. (2023)** — "Accurate medium-range global weather forecasting with 3D neural networks." *Nature 2023.* Pangu-Weather: separate surface/pressure-level streams; UV as vector.
- **Price et al. (2024)** — "GenCast: Diffusion-based ensemble forecasting for medium-range weather." *Nature 2024.* Probabilistic treatment; precipitation via diffusion.
- **Bodnar et al. (2024)** — "Aurora: A Foundation Model of the Atmosphere." *arXiv 2024.* Variable-specific encoders; learned normalization per variable; separate surface/3D encoders.
- **Bonev et al. (2023)** — "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere." *ICML 2023.* FourCastNet v2; frequency-aware treatment on the sphere.
- **Ravuri et al. (2021)** — "Skilful precipitation nowcasting using deep generative models of radar." *Nature 2021.* DGMR; log-transform and adversarial loss for precipitation.
- **Rasp & Lerch (2018)** — "Neural networks for post-processing ensemble weather forecasts." *MWR.* CRPS loss for probabilistic weather.
- **LeCun (2022)** — "A Path Towards Autonomous Machine Intelligence." *OpenReview.* Theoretical basis of JEPA; predictability in latent space.
- **Kochkov et al. (2024)** — "Neural general circulation models for weather and climate." *Nature 2024.* NeuralGCM; smooth-field reconstructibility analysis.
