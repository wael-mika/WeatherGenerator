# Brainstorming Note: Data Augmentation for JEPA Pretraining in ERA5 -> IASI Transfer

## Purpose

This note summarizes a set of candidate data augmentation ideas for improving self-supervised pretraining in our WeatherGenerator JEPA setup, with a particular focus on downstream transfer from ERA5 to IASI.

The goal is not to list generic computer vision augmentations, but to identify augmentations that are plausible for weather and observation data, align with the inductive biases of JEPA, and are worth testing in a controlled ablation campaign.

## Context and Motivation

Our recent masking ablations suggest a clear pattern:

- `cropping` produced the worst downstream results
- `healpix` improved over cropping
- `random` masking produced the best results among the tested strategies

At first sight, this can seem surprising because strong data augmentation is often described as important in self-supervised learning. However, the result is not inconsistent with JEPA.

In many contrastive SSL methods, augmentation is the main mechanism for creating positive pairs, so performance depends strongly on hand-crafted invariances. JEPA is different. JEPA learns to predict the representation of a target region from a context region, and the view construction or masking policy is itself a central part of the learning problem. In I-JEPA, the mask design is more important than applying many image-style augmentations, and the method was specifically motivated as an alternative to augmentation-heavy SSL pipelines.

For our weather setup, this leads to an important hypothesis:

> The next gains are more likely to come from physics-aware alternate views than from stronger image-like augmentation.

In other words, the issue is probably not that "weather does not need augmentation", but that the useful augmentations for weather are different from the ones that work for natural images.

## Current Setup Assumptions

This note is based on the current codebase and configuration structure:

- ERA5 is used as a forcing source stream in the ERA5 -> IASI finetuning setup
- IASI is used as a diagnostic target stream
- spatial masking strategies currently implemented include `random`, `healpix`, `cropping_healpix`, and `forecast`
- the masking system already supports geometry-aware relationships such as subset, disjoint, contained cone, separated cone, and cone distance
- channel masking hooks exist in tokenization, but channel-level masking is not implemented yet

These assumptions matter because some ideas below are easy extensions of the current system, while others would require additional plumbing.

Repository context used for this note:

- `config/streams/era5_iasi_finetuning/era5.yml`
- `config/streams/era5_iasi_finetuning/iasi.yml`
- `config/config_jepa_finetuning.yml`
- `src/weathergen/datasets/masking.py`
- `src/weathergen/datasets/tokenizer_utils.py`

## Interpretation of the Current Ablation Results

The masking results already tell us something important about the representation that transfers best to IASI:

- `random` masking likely works well because it preserves broad spatial coverage and forces the model to infer missing information from distributed context across the globe or hemisphere-scale state
- `healpix` masking is also compatible with this idea because it preserves structured regions while still giving access to wide-context information
- `cropping` likely performs poorly because it reduces the available context too aggressively and biases the model toward local continuity rather than large-scale atmospheric state

For weather and radiance-related transfer, the large-scale context matters:

- synoptic-scale structure influences local thermodynamic profiles
- vertical and horizontal dependencies are long-range
- radiances are not simply local image patches; they are tied to atmospheric state, geometry, and broad-scale coherence

This suggests that augmentations which destroy distributed context are risky, while augmentations that create alternate but physically meaningful views of the same atmospheric state are more promising.

## Design Principles for Weather-Specific SSL Augmentation

Any augmentation we add should satisfy most of the following principles:

- it should preserve physically meaningful relationships rather than break them arbitrarily
- it should encourage robustness to nuisance variation, not erase information that is critical for IASI
- it should preserve enough broad context for JEPA to predict semantically rich targets
- it should ideally align with the downstream transfer problem, not just improve pretraining loss
- it should be testable in isolation with a simple ablation

In practice, this means we should prefer:

- alternate temporal views
- alternate variable views
- alternate vertical-resolution views
- alternate observation-geometry views
- moderate scale changes in what is visible

And we should be cautious with:

- strong local crops
- arbitrary rotations or flips on the sphere
- aggressive per-channel scaling that breaks physical balance
- augmentations that force invariance to season, geography, or diurnal cycle

## Why Generic Vision Augmentations Are Not a Good Default Here

Classic SSL augmentations such as random crop, color jitter, blur, solarization, and horizontal flip are useful in computer vision because the semantic label is often unchanged under these transformations. For weather data, this assumption usually does not hold.

Examples:

- a strong crop can remove the synoptic environment that explains the local state
- a horizontal flip on the globe has no physical meaning
- independent channel jitter can destroy balanced relationships between temperature, humidity, wind, and pressure
- aggressive noise can move a field away from a physically plausible atmospheric state

That is why it is better to think in terms of "alternate physical views" rather than "visual perturbations."

## Proposed Augmentation Ideas

### 1. Mixed Multi-Scale Masking

#### Summary

Instead of committing to a single masking strategy, use a mixture of `random` and `healpix`, and randomize the spatial scale of masking across batches or samples.

#### Why it may help

The current ablations suggest that broad distributed context is valuable, but it is still possible that the model is overfitting to a single notion of what a prediction problem looks like. A mixed masking policy could train the model to solve:

- sparse distributed inference under random masking
- region-level interpolation under healpix masking
- multiple difficulty levels via different keep rates and `hl_mask` values

This is a natural next step because it extends the best-performing strategies rather than replacing them.

#### Why it is relevant for ERA5 -> IASI

IASI retrieval-relevant structure depends on both local thermodynamic state and larger-scale air-mass context. A mixed masking policy may help the encoder represent both.

#### Risks

- if the masking distribution becomes too broad, training may become less stable
- if one masking mode dominates optimization, the mixture may act like a noisy version of the stronger baseline rather than a true improvement

#### Expected implementation difficulty

Low to medium. This is close to what the current masking framework already supports conceptually.

### 2. Temporal-Offset Views

#### Summary

Construct student and teacher views from nearby but non-identical times, for example:

- `t` and `t+6h`
- `t` and `t-6h`
- occasionally `t` and `t+12h`
- later, a curriculum that includes rare `t+24h` pairs

#### Why it may help

Weather has natural temporal coherence. Nearby times often share regime-level information while still differing enough to prevent trivial copying. This is exactly the kind of "natural augmentation" that is more meaningful for geophysical data than synthetic image distortions.

This idea is also supported by remote sensing SSL work such as Seasonal Contrast, where temporal variation is treated as a source of useful alternate views rather than something to be replaced with purely artificial transforms.

#### Why it is relevant for ERA5 -> IASI

If the encoder learns state representations that are stable over short time horizons, it may better capture slowly evolving structure that matters for radiance prediction, while still allowing the downstream model to use exact time information when needed.

#### Risks

- too much temporal offset could encourage unwanted invariance to true atmospheric evolution
- the useful offset likely depends on variable family and latitude band
- if local time and solar effects are important, the method must not erase them unintentionally

#### Recommended starting point

Start with small offsets only:

- mostly `0h` to `6h`
- some `12h`
- no `24h` at first

#### Expected implementation difficulty

Medium. This likely touches sampling logic more than masking logic.

### 3. Variable-Family Dropout

#### Summary

Randomly hide one coherent family of variables in one view while keeping them visible in the other. Candidate families include:

- temperature-related fields
- humidity-related fields
- wind-related fields
- surface variables
- cloud or precipitation proxies

#### Why it may help

This forces the model to infer missing physical information from correlated fields instead of relying on a narrow subset of channels. It should encourage cross-variable reasoning and reduce shortcut learning.

This is more physically plausible than independent per-channel jitter because it removes information in a structured way rather than corrupting values arbitrarily.

#### Why it is relevant for ERA5 -> IASI

IASI is sensitive to atmospheric thermodynamic structure, especially temperature and moisture profiles. Variable-family dropout could encourage the encoder to represent redundant information pathways that are useful when mapping to radiances.

#### Risks

- dropping a family that is too important may make the pretext task unrealistically hard
- if the grouping is poorly chosen, the model may learn artifacts of the grouping rather than better structure
- this should be mild at first

#### Important note for the current codebase

Channel masking hooks exist, but channel-level masking is not currently implemented in tokenization. This means the idea is conceptually strong but not immediately available without code work.

#### Expected implementation difficulty

Medium to high.

### 4. Vertical Slab Masking or Vertical Smoothing

#### Summary

Create alternate views by modifying the vertical representation of the state:

- mask contiguous pressure-level slabs
- coarsen or smooth the vertical profile in one view
- use full vertical resolution in the other view

#### Why it may help

This is one of the most downstream-aligned ideas for ERA5 -> IASI. IASI does not observe a single model level directly. It responds to vertically weighted atmospheric structure. If SSL encourages the model to reason across neighboring levels, that could improve transfer more directly than purely spatial perturbations.

#### Why it is relevant for ERA5 -> IASI

Very relevant. The mapping to radiances depends strongly on vertical temperature and humidity structure, not just horizontal fields.

#### Risks

- too aggressive vertical masking may remove information that is essential rather than nuisance
- if applied symmetrically to both views, the task may become too weak
- if applied without care, it may bias the encoder toward overly smooth vertical representations

#### Recommended variants

- start with vertical smoothing in one view
- then try masking only a limited slab, not many slabs
- compare "mild loss of vertical detail" versus "hard missing vertical band"

#### Expected implementation difficulty

Medium to high, depending on how channel and level structure is represented in the input pipeline.

### 5. Observation-Geometry Augmentation

#### Summary

Use masks that mimic satellite observation geometry more closely than circular crops:

- swath-like stripes
- scan-line masks
- latitudinal sampling differences
- structured sparsification that resembles real coverage patterns

#### Why it may help

This creates alternate views that are physically meaningful for an observation-driven downstream task. A swath-like view is much closer to how IASI samples the atmosphere than a geodesic crop.

#### Why it is relevant for ERA5 -> IASI

Very relevant for transfer. The encoder may benefit from learning representations that remain useful under the kinds of partial visibility encountered in satellite geometry.

#### Risks

- if the synthetic geometry is unrealistic, the augmentation may help less than expected
- if it becomes too observation-specific, it may narrow the pretraining task too much

#### Expected implementation difficulty

Medium. It is an extension of spatial masking, but with a different geometry generator.

### 6. Scale-Space Views

#### Summary

Generate paired views at different effective resolutions or frequency content:

- one raw view
- one low-pass filtered or spatially coarsened view
- optionally one anomaly-style view relative to climatology

#### Why it may help

This could encourage the model to separate large-scale background state from finer weather perturbations. For JEPA, this may be useful if the target should reflect semantic atmospheric structure rather than exact small-scale noise.

#### Why it is relevant for ERA5 -> IASI

Radiance transfer often depends strongly on the broad thermodynamic structure, but small-scale variability can still matter. Multi-scale views may help the encoder capture both.

#### Risks

- coarsening may remove important mesoscale information
- anomaly views can unintentionally suppress the absolute state, which often matters for radiances
- climatology subtraction must be handled carefully to avoid changing the task too much

#### Recommended caution

Try spatial smoothing or coarse-graining before anomaly-style views. The anomaly option is more conceptually interesting but also riskier.

#### Expected implementation difficulty

Medium.

### 7. Weak Physics-Scaled Noise

#### Summary

Add small additive noise to selected variables, scaled by each variable's natural variability or estimated uncertainty.

#### Why it may help

A small amount of realistic noise can improve robustness and reduce over-sensitivity to exact numerical values. The key is that the noise must be weak and physically scaled.

#### Why it is relevant for ERA5 -> IASI

ERA5 is already an estimate of atmospheric state, not perfect truth. Mild noise injection could make the encoder more robust to analysis uncertainty and help downstream transfer.

#### Risks

- easy to overdo
- unstructured noise can damage physical consistency
- weaker than the other proposals in expected payoff

#### Recommended use

Treat this as an optional add-on after a stronger augmentation shows promise, not as the first idea to test.

#### Expected implementation difficulty

Low to medium.

### 8. Diversity as Augmentation Through Sampling

#### Summary

Treat sampling strategy itself as augmentation by balancing the pretraining distribution across:

- seasons
- latitude bands
- circulation regimes
- land-ocean contrasts
- extreme versus ordinary conditions

#### Why it may help

For geophysical data, the natural variability of the dataset is itself a major source of representation learning signal. Better sampling can improve robustness without modifying the sample content.

This is especially attractive if the issue is not a lack of augmentation, but an imbalance in what the model sees during pretraining.

#### Why it is relevant for ERA5 -> IASI

IASI performance can differ by regime, latitude, surface type, and season. Better coverage during SSL may improve transfer more reliably than many synthetic transforms.

#### Risks

- if sampling becomes too artificial, pretraining may stop matching the true data distribution
- some forms of regime balancing are hard to define cleanly

#### Expected implementation difficulty

Low to medium conceptually, medium practically.

### 9. Radiative-Transfer Proxy Views

#### Summary

Construct one view from transformed ERA5 features that loosely mimic the vertical weighting behavior of infrared sounding observations.

Examples could include:

- broad layer averages
- weighted temperature and humidity summaries
- observation-inspired feature bundles

#### Why it may help

This is the most downstream-aware augmentation on the list. It could push pretraining toward latent structure that is already aligned with what IASI "cares about."

#### Why it is relevant for ERA5 -> IASI

Extremely relevant. If done well, it may bridge the gap between full-model state representation and observation-space transfer.

#### Risks

- highest design risk
- easy to bake in a poor or too narrow prior
- may reduce generality if the proxy is too tailored to IASI

#### Recommended use

Not a first-wave experiment. This is a second-wave idea after simpler temporal or variable-based augmentations have been tested.

#### Expected implementation difficulty

High.

## Recommended Priority Order

The proposals above are not equally likely to pay off. Based on the current ablations and downstream task, the most sensible order is:

1. mixed multi-scale masking
2. temporal-offset views
3. variable-family dropout
4. vertical slab masking or vertical smoothing
5. observation-geometry augmentation
6. scale-space views
7. diversity-through-sampling
8. weak physics-scaled noise
9. radiative-transfer proxy views

This ordering reflects a balance of:

- expected payoff
- alignment with JEPA
- alignment with ERA5 -> IASI transfer
- ease of adding into the current system

## Recommended First Experiment Batch

If the team wants a compact first campaign, I would suggest the following six experiments:

1. reproduce the best current `random` baseline
2. reproduce the best current `healpix` baseline
3. mixed multi-scale masking
4. temporal-offset views with small offsets only
5. variable-family dropout
6. vertical smoothing or limited vertical slab masking

If one of experiments 3 to 6 clearly improves downstream finetuning, the second batch should test:

1. the best first-wave augmentation combined with the strongest baseline masking policy
2. observation-geometry augmentation
3. the best first-wave augmentation plus weak physics-scaled noise

## Suggested Hypotheses Per Experiment

To make discussions with colleagues easier, here is a concise hypothesis for each of the first-wave ideas.

### Mixed Multi-Scale Masking

Hypothesis:
Training on several spatial visibility patterns will make the encoder less specialized to a single pretext geometry and improve transfer robustness.

Success signal:
Improved finetuning performance without major pretraining instability.

Failure mode:
Noisy optimization and no clear gain over pure random masking.

### Temporal-Offset Views

Hypothesis:
Short-term temporal consistency will create semantically meaningful alternate views that improve regime-level representation learning.

Success signal:
Better downstream transfer at equal or slightly worse pretraining loss.

Failure mode:
Too much invariance to actual temporal evolution, especially for fast-changing situations.

### Variable-Family Dropout

Hypothesis:
Forcing prediction across physically coupled variable families will improve cross-variable latent structure.

Success signal:
Better downstream transfer, especially if the downstream task depends on temperature-humidity coupling.

Failure mode:
Pretext task becomes too hard and the encoder underfits.

### Vertical Slab Masking or Vertical Smoothing

Hypothesis:
Encouraging robustness to partial or coarsened vertical information will improve latent representations relevant to infrared sounding transfer.

Success signal:
Disproportionate improvement on ERA5 -> IASI compared with unrelated downstream tasks.

Failure mode:
Loss of essential vertical fidelity.

## What We Should Avoid Making Invariant

For this project, some invariances are likely harmful rather than helpful. We should avoid training the model to ignore:

- absolute latitude
- season
- local solar time
- large-scale circulation context
- meaningful vertical thermodynamic structure

This is important because many SSL recipes imported from vision implicitly assume that semantics should be invariant to transformations that are, in our case, physically meaningful.

## Practical Notes for Implementation Planning

Before coding begins, it is useful to separate ideas into three buckets.

### Bucket A: Easy Extensions of Current Spatial Masking

- mixed multi-scale masking
- observation-geometry augmentation
- additional healpix-scale randomization

These are the most natural extensions of the current masking framework.

### Bucket B: Requires Sampling Logic Changes

- temporal-offset views
- diversity-through-sampling

These mostly affect how views are drawn rather than how cells are masked.

### Bucket C: Requires Channel or Vertical Feature Plumbing

- variable-family dropout
- vertical slab masking
- vertical smoothing
- radiative-transfer proxy views

These may require explicit handling of variable groups or pressure-level structure in the input pipeline.

## Overall Recommendation

The safest strategic direction is:

- keep the main JEPA bias toward broad distributed context
- avoid stronger crop-based augmentation
- move toward physically meaningful alternate views
- prioritize augmentations that reflect time, variable coupling, vertical structure, and observation geometry

If I had to summarize the central recommendation in one sentence, it would be this:

> For weather JEPA, we should stop thinking in terms of "stronger image augmentation" and start thinking in terms of "better physically grounded alternate views."

## References

These references are useful mainly for framing the discussion, not as prescriptions to copy directly.

- Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- Seitzer et al., [A-JEPA: Audio Joint Embedding Predictive Architecture](https://arxiv.org/abs/2311.15830)
- Cong et al., [SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery](https://arxiv.org/abs/2207.08051)
- Mañas et al., [Seasonal Contrast: Unsupervised Pre-Training From Uncurated Remote Sensing Data](https://openaccess.thecvf.com/content/ICCV2021/html/Manas_Seasonal_Contrast_Unsupervised_Pre-Training_From_Uncurated_Remote_Sensing_Data_ICCV_2021_paper.html)
- Wang et al., [SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation](https://arxiv.org/abs/2211.07044)
- Nguyen et al., [ClimaX: A foundation model for weather and climate](https://arxiv.org/abs/2301.10343)

## Closing Remarks

The current ablations already provide a valuable signal. They suggest that the pretraining task benefits from distributed context and that naive locality-inducing crops are not well aligned with the transfer target. That is a strong result, and it should guide the next round of experimentation.

The best next step is not to add arbitrary augmentation volume, but to design augmentations that reflect the symmetries, nuisances, and partial observability of atmospheric and satellite data.
