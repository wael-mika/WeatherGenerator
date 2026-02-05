# Loss functions used in the extreme precipitation experiments

This document describes the loss terms currently used in the WeatherGenerator
extreme-precipitation experiments, how they are computed, and the meaning of
their configuration parameters.

## How losses are combined (overall objective)

For each stream and forecast step, losses are computed over valid target points
and then combined as:

1) Per-loss computation over points and channels.  
2) Average over sub-time steps (if tokenized into substeps).  
3) Weighted sum over loss functions using `loss_fcts` (or `loss_fcts_val`).  
4) Multiply by `stream.loss_weight` (train only) and by any per-fstep weight
   if `timestep_weight` is configured.  
5) Average over forecast steps, then over streams.

**Location and channel weights**
- `stream.target_channel_weights` (train only) multiplies per-channel loss.
- `stream.location_weight` (train only) produces per-point weights; these are
  multiplied into the loss for all loss terms that accept `weights_points`.
- Extreme sampling (if enabled) multiplies sample weights into the location
  weights (or creates uniform weights if none exist).

**Validation mode**
- In validation, stream loss weights and channel weights are disabled
  (`loss_weight = 1`, `target_channel_weights = None`), but the configured
  `loss_fcts_val` weights are still applied.

## Data space and thresholds

Targets and predictions are in the **transformed data space** defined by the
stream (e.g., IMERG transform). For any loss that takes `thresholds_mm`, the
thresholds are converted into model space using the transform parameters
(`transform_type`, `transform_alpha`, `transform_eps`, `transform_offset`,
`transform_mu`, `transform_sigma`). This ensures thresholds are consistent with
the model's internal representation.

---

## Loss terms

### 1) MSE (name: `mse`)
Implementation: `mse_channel_location_weighted`.

For each channel \(c\), with optional per-point weights \(w_i\):

\[
L_c = \frac{1}{N} \sum_i w_i (y_{i,c} - \hat{y}_{i,c})^2
\]

Then apply optional channel weights \(w_c\) and average over channels:

\[
L = \frac{1}{C} \sum_c w_c L_c
\]

**Parameters**
- No `loss_config` parameters.
- Uses `target_channel_weights` (train) and `location_weight` (train) if set.

---

### 2) Intensity-weighted MSE (name: `mse_intensity_weighted`)
Implementation: `mse_intensity_weighted`.

A standard MSE weighted by a nonlinear function of **target intensity** in
model space. Per-point weights \(w_i\) are computed as:

\[
v_i = \text{mean}_c |y_{i,c}| \times \text{scale}
\]

Then
- `mode = log1p`: \(f(v) = \log(1+v)\)
- `mode = power`: \(f(v) = v^{\text{power}}\)
- `mode = linear`: \(f(v) = v\)

\[
w_i = 1 + \alpha f(v_i)
\]

Finally \(w_i\) is clamped to `[min_weight, max_weight]` if `max_weight` is
set, and multiplied by location weights if provided.

**Parameters** (`loss_config.intensity_weighted_mse`)
- `mode`: `"log1p" | "power" | "linear"`
- `alpha`: strength of intensity weighting
- `power`: exponent used in `mode="power"` or after `log1p` if `power != 1`
- `scale`: scales the target intensity before weighting
- `min_weight`, `max_weight`: clamp range for `w_i`

---

### 3) Soft exceedance loss (name: `soft_exceedance`)
Implementation: `soft_exceedance`.

For each threshold \(t\), the target exceedance is a binary label
\( \mathbb{1}(y \ge t) \). The prediction is converted to a logit:

\[
\text{logit} = \frac{\hat{y} - t}{\text{temperature}}
\]

Loss is binary cross-entropy with logits, averaged over points and channels,
then averaged across thresholds (with optional per-threshold weights).

**Parameters** (`loss_config.soft_exceedance`)
- `thresholds_mm`: thresholds in mm (converted to model space)
- `threshold_weights`: optional weights per threshold
- `temperature`: larger = softer decision boundary
- `transform_*`: optional transform parameters for thresholds (see above)

---

### 4) Upper-quantile pinball loss (name: `quantile_upper`)
Implementation: `quantile_upper`.

For each quantile \(q\), the pinball loss is:

\[
L_q = q \cdot \max(y-\hat{y}, 0) + (1-q) \cdot \max(\hat{y}-y, 0)
\]

Losses are averaged over points, then weighted and averaged over quantiles.

**Parameters** (`loss_config.quantile_upper`)
- `quantiles`: list of upper quantiles (e.g., `[0.9, 0.95, 0.99]`)
- `weights`: per-quantile weights (emphasize more extreme quantiles)

---

### 5) Centroid-shift loss (name: `centroid_shift`)
Implementation: `centroid_shift` (new).

This term penalizes **spatial displacement** between predicted and target
extreme precipitation. For each threshold:

1) Compute soft exceedance weights using a sigmoid:
   \[
   w = \sigma\Big(\frac{y - t}{T}\Big)
   \]
2) Compute weighted centroids of target and prediction on the unit sphere
   using latitude/longitude.
3) Compute geodesic distance \(d\) (km) between centroids and normalize:
   \[
   L = d / \text{scale\_km}
   \]

The loss is only applied if **both** target and prediction have at least
`min_points` hard exceedances to avoid unstable gradients.

**Parameters** (`loss_config.centroid_shift`)
- `thresholds_mm`: thresholds in mm (converted to model space)
- `threshold_weights`: per-threshold weights
- `temperature`: sigmoid temperature for soft exceedance
- `min_points`: minimum hard exceedances in both target and pred
- `scale_km`: distance normalization (e.g., 1000 km)
- `earth_radius_km`: radius used for distance (default 6371)
- `transform_*`: optional transform parameters for thresholds (see above)

---

## Notes on scientific interpretation

1) **Intensity-weighted MSE** and **soft exceedance** improve detection but
   often increase FAR.  
2) **Quantile loss** reduces under-prediction in the tail but can bias means.  
3) **Centroid-shift** directly targets location error and should improve
   peak/centroid metrics at high thresholds; it may slightly worsen RMSE/MAE if
   overweighted.

Recommended practice: keep `centroid_shift` weight small (e.g., 0.02-0.05) and
verify improvements with bootstrapped confidence intervals.
