# Wavelet-Fourier Composite Loss (WFCL) — Phased Evaluation Guide

This document describes the three-phase experimental approach to evaluating WFCL for precipitation prediction in WeatherGenerator.

**Branch:** `wm/dev/raina_wave`

## Overview

The WFCL implementation is mathematically sound and passes all unit tests. However, its effectiveness for the WeatherGenerator architecture is uncertain due to architectural differences from the paper's U-Net baseline:

- The WeatherGenerator operates on irregular HEALPix grids (requiring interpolation to regular grids for FFT/DTCWT)
- It is multi-variable and multi-task (WFCL was tested on precipitation in isolation)
- The model is trained at higher resolutions with different spectral content
- The training schedule and scale are different from the paper's setup

The phased approach lets you test the core hypothesis (frequency-domain losses improve precipitation) with progressively more aggressive configurations, measuring real impacts on validation metrics.

---

## Phase 1: FACL-Only (Fourier Component)

**Config:** `config_forecasting_raina_phase1_facl.yml`

**What it tests:** Can Fourier-domain amplitude and correlation losses (FACL) alone improve precipitation without wavelets?

**Loss weights:**
- MSE: 0.95 (keep most gradient spatial)
- WFCL: 0.05 (small contribution to avoid destabilizing other variables)
- Within WFCL: FACL only (wavelet_levels: 0)

**Key advantages:**
- Simplest experiment: no grid interpolation artifacts from wavelets
- Lowest computational cost (FFT is much faster than DTCWT)
- Clear signal: if it helps, you know spectral thinking is relevant
- If it hurts, stop before investing in wavelets

**What to monitor:**
- **Precipitation metrics:** LPIPS (lower is better), FSS at quantiles 50–95th, Local Phase Coherence
- **Other variables:** Ensure MSE on other fields doesn't increase significantly
- **Loss dynamics:** WFCL loss should be non-zero and decrease monotonically with training

**Decision criteria:**
- ✅ **Proceed to Phase 2** if: LPIPS or FSS improves by >2% and other metrics don't degrade
- ❌ **Stop and analyze** if: FACL hurts other variables or doesn't improve precipitation
- ⚠️ **Adjust and retry** if: Results are ambiguous; try increasing WFCL weight to 0.08 or 0.10

**Expected runtime:** ~1 week (64 mini-epochs on typical GPU cluster)

---

## Phase 2: Full WFCL (Balanced Weights)

**Config:** `config_forecasting_raina_phase2_wfcl_balanced.yml`

**What it tests:** Does adding the multi-scale wavelet component (WACL) improve on Phase 1 gains?

**Loss weights:**
- MSE: 0.9 (reduced to make room for wavelets)
- WFCL: 0.1 (moderate contribution)
- Within WFCL:
  - FACL (Fourier): beta=1.0
  - WACL (Wavelet): 3 levels, equal weighting [1, 1, 1]
  - P_t schedule: amplitude-dominant early, phase-dominant after 10%

**Key assumptions:**
- Phase 1 showed promise
- Interpolation to regular grid doesn't negate benefits
- Multi-variable learning can tolerate 0.1 WFCL weight without significant conflicts

**What to monitor:**
- **Precipitation metrics:** LPIPS, FSS (especially at high quantiles, where structure matters)
- **Spectral metrics:** Compute power spectrum at each scale; wavelets should improve finer scales
- **Other variables:** Watch MSE on non-precipitation fields
- **Loss components:** Log FAL, FCL, WACL separately if possible; verify P_t schedule is running

**Tuning options if needed:**
- If other variables degrade: reduce WFCL weight to 0.05 or 0.07
- If finer scales improve more: change gamma_levels to [1, 2, 4] (Phase 3)
- If coarser scales are the bottleneck: swap to [4, 2, 1]

**Decision criteria:**
- ✅ **Proceed to Phase 3** if: LPIPS/FSS gains persist and scale with wavelets; synoptic and convective scales both improve
- ⚠️ **Stay in Phase 2** if: Gains are smaller than Phase 1; tune weights or gamma_levels
- ❌ **Revert to Phase 1** if: Adding wavelets hurts validation or increases training instability

**Expected runtime:** ~1 week

---

## Phase 3: WFCL Optimized (Precipitation-Only)

**Config:** `config_forecasting_raina_phase3_wfcl_aggressive.yml`

**Prerequisites:**
- Phase 1 and Phase 2 showed consistent gains
- Stream config has been adapted to predict **IMERG_ANEMOI only** (no other variables)
- You've decided to specialize the model for precipitation prediction

**Loss weights:**
- MSE: 0.7 (significantly reduced; spectral loss is the primary signal)
- WFCL: 0.3 (aggressive, but no multi-variable conflicts)
- Within WFCL:
  - FACL: beta=1.0
  - WACL: 3 levels, inverse-frequency weighting [1, 2, 4]

**Rationale:**
- With only precipitation as output, there's no optimization tension from other variables
- Inverse-frequency weighting emphasizes finer scales (convective) where wavelets provide most benefit
- Higher WFCL weight can yield larger sharpness gains

**What to monitor:**
- **Precipitation metrics:** LPIPS, FSS at all quantiles, LPC, spectral coherence
- **Training stability:** Larger WFCL weight may cause higher gradient variance; watch for divergence
- **Loss components:** Verify P_t is ramping correctly (1.0 → 0.0 over 900k steps for 1M total)

**Tuning options:**
- If model is unstable: reduce WFCL weight to 0.20, reduce learning rate by 20%
- If LPIPS plateaus: try alpha=0.2 (faster transition from amplitude to phase), or gamma_levels=[4, 2, 1] (coarser scales)
- If FSS at high quantiles decreases: increase beta to 1.5 or reduce WFCL weight

**Decision criteria:**
- ✅ **Production ready** if: LPIPS improves by >5%, FSS at all quantiles improves, training is stable
- ⚠️ **Further tuning** if: Partial gains but not across all scales; systematic ablations on gamma_levels
- ❌ **Reconsider** if: Gains don't exceed Phase 2, or high-quantile FSS degrades significantly

**Expected runtime:** ~1 week

---

## Switching Between Phases

To run a specific phase:

```bash
# Phase 1: FACL-only
uv run train --base-config config/raina_config/config_forecasting_raina_phase1_facl.yml

# Phase 2: Balanced WFCL
uv run train --base-config config/raina_config/config_forecasting_raina_phase2_wfcl_balanced.yml

# Phase 3: Aggressive WFCL (precipitation-only)
uv run train --base-config config/raina_config/config_forecasting_raina_phase3_wfcl_aggressive.yml
```

Each phase is self-contained; you can run them independently.

---

## Key Metrics to Track

| Metric | Measures | Direction | Paper result | Notes |
|--------|----------|-----------|--------------|-------|
| **LPIPS** | Perceptual distance (deep features) | ↓ better | 46% reduction | Most relevant for sharpness |
| **FSS at p=50** | Structured skill at median | ↑ better | ~+5% | Overall precipitation structure |
| **FSS at p=95** | Structured skill at 95th percentile | ↑ better | ~+10% | Extreme/convective scales |
| **LPC** | Local phase coherence | ↓ better | 21% reduction | Phase alignment across scales |
| **MSE (precip)** | Spatial mean square error | ↓ better | Similar | May not improve with WFCL |
| **MSE (other)** | MSE on non-precipitation vars | ↓ better | N/A | Critical in Phase 1/2; should not degrade |
| **Spectral power** | Energy at different scales | Model-dependent | N/A | Should see power shift to higher frequencies |

---

## Stream Config Adaptation

The three configs expect the stream config to be set up for precipitation prediction. Suggested minimal stream config:

```yaml
# config/streams/raina/imerg_anemoi_precip_only.yml
IMERG_ANEMOI:
  type: anemoi 
  filenames: ["imerg-nasa-grib-o96-1998-2024-6h-v1.zarr"]
  stream_id: 1

  source: ["tp"]
  target: ["tp"]

  target_channel_weights: [1.0]
  location_weight: cosine_latitude

  loss_weight: 1.0

  masking_rate: 0.6
  masking_rate_none: 0.05
  token_size: 16
  tokenize_spacetime: True

  embed:
    net: transformer
    num_tokens: 1
    num_heads: 8
    dim_embed: 256
    num_blocks: 2
  embed_target_coords:
    net: linear
    dim_embed: 256
  target_readout:
    type: "obs_value"
    num_layers: 2
    num_heads: 4
  pred_head:
    ens_size: 1
    num_layers: 2
```

Then reference it in the training config:
```yaml
streams_directory: "./config/streams/raina/"
streams: ["imerg_anemoi_precip_only"]
```

---

## Common Failure Modes and Diagnostics

**Symptom:** WFCL loss is NaN
- **Cause:** Grid interpolation produced NaN; FFT on incomplete grid; DTCWT dimension mismatch
- **Fix:** Check that grid dimensions (180, 360) are divisible by 2^3=8 ✓; verify coordinates don't have gaps

**Symptom:** LPIPS doesn't improve despite WFCL being non-zero
- **Cause:** Interpolation to regular grid smooths away the sharpness WFCL is trying to preserve
- **Fix:** Test Phase 1 with larger WFCL weight (0.10 or 0.15); if still doesn't help, the architecture may not benefit from this approach

**Symptom:** Other variables' MSE increases significantly in Phase 1/2
- **Cause:** WFCL gradients pull precipitation toward sharp, high-frequency structures that hurt smooth variables
- **Fix:** Reduce WFCL weight to 0.03–0.05; or move to Phase 3 (precipitation-only)

**Symptom:** Training is unstable in Phase 3
- **Cause:** WFCL weight=0.3 is too aggressive; gradient variance is high
- **Fix:** Reduce WFCL to 0.15–0.20; reduce learning rate by 10–20%; check that P_t schedule is advancing correctly

---

## Ablations and Extensions (After Phase 3)

If Phase 3 succeeds:

1. **Optimize alpha**: Try alpha=0.2 or 0.3 to extend the amplitude-phase blend
2. **Optimize gamma_levels**: Systematically test [1,2,4], [4,2,1], [2,2,2], etc. to find best scale weighting
3. **Optimize beta**: Reduce to 0.5 if Fourier component dominates; increase to 1.5 if wavelets are the main signal
4. **Test on other variables**: If precipitation gains are strong, carefully add other variables with much lower WFCL weight

---

## References

- **Paper:** An et al. (2026), "Toward Spatially Sharper Precipitation Prediction via Global-Local Frequency Guidance"
- **Implementation:** branch `wm/dev/raina_wave`
- **Config location:** `config/raina_config/config_forecasting_raina_phase*.yml`
- **Code location:** `src/weathergen/train/loss_modules/loss_module_spectral.py`, `spectral_utils.py`

