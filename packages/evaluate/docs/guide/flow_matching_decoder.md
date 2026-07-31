# The flow-matching decoder

A conditional flow-matching readout for WeatherGenerator: instead of regressing a value per
target point, it **generates** the field jointly over the target points inside a HEALPix cell.

This document is the handover for anyone continuing the work. It covers why the decoder exists,
how it is built, every non-obvious implementation constraint, what has been measured, and what
the open questions are. Code: [`FlowMatchingPointDecoder`](../src/weathergen/model/engines.py).

---

## 1. The problem it solves

The task that motivated it is ERA5 (~1°) → CERRA (5.5 km, 1069×1069 over Europe) downscaling. A
HEALPix-L5 latent cell is ~150 km across and contains ~750 CERRA points. The decoder must produce
5.5 km structure from a latent that does not resolve it.

That 5.5 km variance decomposes into parts with **different information sources**:

| Component | Scale | Where the information is |
|---|---|---|
| Meso / frontal placement | 50–500 km | The large-scale latent. Deterministic in principle. |
| Convective / turbulent texture | 5–25 km | **Nowhere.** Irreducibly aleatoric at t+6 h. |
| Terrain-forced | 5–50 km, land only | Static high-res orography. |

The second row is the point. That texture cannot be *predicted*; it can only be *generated* as a
plausible realization. Any decoder trained to minimise a pointwise proper score will converge to
the conditional mean there, because the conditional mean is the correct answer to the question
being asked. The output is then blurred by construction, not by under-training.

### Why the earlier attempts did not fix this

Two designs were tried first and both failed for the same structural reason. They are recorded
here so nobody re-derives them:

- **Pinball / quantile heads** (`config_quantile_cerra.yml`). Recovered the tail but left central
  products at blur level. A per-point quantile map is not a field: the τ=0.97 map is "the 97th
  percentile everywhere at once", which no atmosphere ever realises.
- **Noise-conditioned regression head + CRPS.** A noise-conditioned regressor judged by a proper
  score retains a degenerate optimum — ignore the noise, emit the conditional mean. The
  permission to blur is still there.

The common failure is that both factorise `p(y | latent)` over points, so they can only ever get
one-point statistics right. Spatial coherence is a *joint* property.

### Why flow matching does not have that failure mode

Flow matching offers no "ignore the noise" basin. Training asks for the velocity along the linear
path

```
y_t = (1 - t) * y0 + t * y1        y0 ~ N(0, I),  y1 = target
u   = y1 - y0                      (the target velocity, constant along the path)
```

At small `t` the network's input is essentially pure noise, so the network **must** be a function
of it. The stochasticity is the state being transported, not an optional side input. Combined
with self-attention over the points in a cell (below), the model learns a joint distribution.

---

## 2. Design

### 2.1 Per-cell joint generation

`TargetPredictionEngineClassic` already contains `MultiSelfAttentionHeadVarlen` over **target
points grouped by cell** (`tcs_lens`) when `pred_self_attention: True`. The velocity network is
that same stack with two extra inputs, so it models `p(y_cell | latent)` jointly over the ~750
points in a cell. This is exactly what pointwise quantiles structurally cannot do.

**Consequence: `pred_self_attention: False` defeats the decoder.** Points are then sampled
independently and the output is spatially incoherent noise. The model logs a warning; it is not
an assert only because the config is still runnable for debugging.

### 2.2 How `y_t` and `t` enter the network

Both ride existing machinery, so **no change to `attention.py` was needed**:

- `y_t` is embedded per point (`embed_state: Linear(num_channels → dims_embed[0])`) and **added
  to the query token**.
- `t` gets a sinusoidal embedding (`flow_time_embedding`, width `flow_dim_time`, default 32) and
  is **concatenated to the coordinate frame** in the AdaLN conditioning — the `coordinates=`
  path already exists.

Because the aux vector is widened, the inherited stack must be constructed for
`dim_coord_in + dim_time`. That happens in `__init__` before `super().__init__(...)`.

### 2.3 Training vs evaluation

```python
if not self.training:
    return self.sample(...)          # [ens_size, N, C] — the ODE loop
# training: one velocity evaluation
return (y0 + v).unsqueeze(0)         # [1, N, C]
```

The training return value is deliberate: returning `y0 + v_theta` means the **existing plain
`mse` loss is exactly the flow-matching objective**, since

```
|| (y0 + v) - y1 ||^2  =  || v - (y1 - y0) ||^2  =  || v - u ||^2
```

No new loss function, no registration, no change to `LossPhysical`. This is the single most
important trick in the implementation — and the reason the returned tensor is *not* a field.

### 2.4 Sampling

`sample()` integrates `t: 0 → 1` in `flow_num_steps` steps under `torch.no_grad()`, drawing
`pred_head.ens_size` independent trajectories. Each integration is one coherent sample.

Members are **looped, not batched** (`for _ in range(ens_size): for i_step in ...`). Memory is
therefore bounded by one member's decode, but cost is `ens_size × num_steps` full decoder passes.
At ens 8 / 24 steps that is **192 passes per sample** versus 1 for a regression arm — plan
inference accordingly (see §6).

---

## 3. Non-obvious implementation constraints

Every item here was discovered the hard way. Changing any of them silently breaks something.

### 3.1 Targets are not reachable from `batch`

The obvious implementation — read the target from the batch already passed to
`predict_decoders` — **does not work**, and fails silently in a way that looks like a shape bug.

Target *values* and target *coordinates* live on **different sample sets**: `source_select`
carries `target_coords`, `target_select` carries `target_values`. The trainer calls
`self.model(batch=batch.get_source_samples())`, where `target_tokens` is empty.

The fix is an opt-in second argument:

- `Model.requires_targets_in_forward` — `True` only when a `FlowMatchingPointDecoder` is present.
- `Model.forward(..., target_batch=None)` and `predict_decoders(..., target_batch=None)`.
- `Trainer` passes `target_batch=batch if self._model_requires_targets else None`.

Targets are then fetched through the source→target matching:

```python
i_t = target_batch.get_target_idx_for_source(i_b)
target_batch.get_target_sample(i_t).streams_data[stream].target_tokens[step]
```

Every other decoder keeps the usual strict separation and never sees a target. There is an assert
on the decoded-vs-target point count, because `LossPhysical` pairs the two by a plain reshape;
if that ever diverges you would train on mispaired data instead of crashing.

### 3.2 Conditioning dropout must be a blend, not a branch

DDP runs with `_set_static_graph()`, which requires the **same set of participating parameters
every iteration**. An `if random() < p:` branch leaves `null_latent` unused on ~90% of steps and
trips it. So:

```python
mask = (torch.rand(()) < self.cond_dropout).to(latent.dtype)
lat  = (1.0 - mask) * latent + mask * self.null_cond.expand_to(latent)
```

With `mask == 0` this is bit-identical to `latent` (verified).

### 3.3 The null token needs its own module

`load_model_state` re-initialises missing checkpoint keys by calling `to_empty()` +
`reset_parameters()` on **the highest-level module covering them**. A bare
`self.null_latent = Parameter(...)` on the decoder would make that root the entire decoder and
**discard every inherited weight** on warm start. Hence `FlowNullConditioning` as a separate
`nn.Module`. Verified: loading a pre-guidance checkpoint reports `null_cond.null_latent` as the
only missing key and `null_cond` as the only re-init root.

It is always constructed (2048 floats) so checkpoints stay interchangeable whether or not
guidance is in use.

### 3.4 `vel_head` must NOT be zero-initialised

Zero-initialising an output head is the standard trick for a *gated residual* branch. Here it is
fatal: `vel_head` is the **only** path from the block stack to the loss, so with zero weights the
gradient w.r.t. the decoded tokens is exactly zero and the whole stack plus `embed_state` receives
no gradient at all. Caught by a gradient assert during development. Default init.

### 3.5 NaN targets

Masked / spoofed targets are NaN. Substituting `y0` gives them zero velocity instead of poisoning
`y_t`; the loss masks those points anyway, so they contribute no gradient.

### 3.6 `decode_soft_blend_k` must be off

Asserted at build time. Blending averages independent samples inside the transition band, which
re-smooths exactly the fine structure the generator produces.

### 3.7 Never apply `LossStructureFunction` to the training output

In training the decoder returns `y0 + v_theta` — a noise-shifted velocity, **not a field**. An SF
loss would score the structure function of iid noise.

The tempting repair is to return the clean-field estimate instead, which on the linear path is
`ŷ1 = y_t + (1-t)·v`. That backfires: plain `mse` against `y1` then expands to
`(1-t)² · ||u - v||²`, i.e. the flow-matching loss weighted by `(1-t)²`, which is **zero at
t → 1** — precisely the refinement end of the path where fine detail is made. You would be
optimising for sharpness while switching off the part of training that creates it.

An SF term on `ŷ1` is well posed only as an **auxiliary** term, with plain `mse` still on
`y0 + v`. That remains the most promising untried lever (§7).

---

## 4. Configuration

Defaults live in [`config/default_config.yml`](../config/default_config.yml); all stage-2 levers
default to the plain behaviour exactly.

| Key | Default | Meaning |
|---|---|---|
| `decoder_type` | — | set to `FlowMatching` |
| `flow_num_steps` | 24 | ODE steps at inference |
| `flow_dim_time` | 32 | sinusoidal time-embedding width (must be even) |
| `flow_solver` | `euler` | `euler` \| `heun` (2nd order, 2× evals/step) |
| `flow_time_sampling` | `uniform` | `uniform` \| `logit_normal` (Esser et al. 2024, SD3) |
| `flow_time_logit_mean/_std` | 0.0 / 1.0 | logit-normal parameters |
| `flow_cond_dropout` | 0.0 | **training** knob; enables guidance |
| `flow_guidance` | 1.0 | **sampling** knob, `w`; costs 2× evals when ≠ 1 |
| `pred_head.ens_size` | — | samples drawn at evaluation |

Ready-made configs:

- [`config/cerra_sharpness/config_pretrain_flowmatch.yml`](../config/cerra_sharpness/config_pretrain_flowmatch.yml)
- [`config/cerra_sharpness/config_ft_flowmatch_generic.yml`](../config/cerra_sharpness/config_ft_flowmatch_generic.yml)
- [`config/evaluate/eval_config_flowmatch_kt8g6wey.yml`](../config/evaluate/eval_config_flowmatch_kt8g6wey.yml)

---

## 5. How to evaluate it — and how not to

**Validation MSE is a *sample* MSE.** One stochastic draw carries the usual double penalty, so it
sits above a regression arm's conditional-mean MSE *by construction*, even when the field is
better. It is not the criterion.

Use instead:

1. **Per-bin structure-function ratios** `S_pred / S_target`, on both the ensemble mean and the
   members (`log_bin_ratios: True` on a `LossStructureFunction` block). `> 1` = too much variance
   at that scale, `< 1` = too smooth.
2. **Single-member maps.** The ensemble mean is smooth by construction; judging a generative
   decoder on mean maps measures nothing.
3. **Coherence**: member-SF vs mean-SF. A coherent generator has member-SF close to the target
   while mean-SF collapses.

### Two traps that have already cost real time

- **The scalar SF is a squared log ratio, hence sign-blind.** It scores 1.38 and 0.72 identically.
  An early reading of this arm as "over-dispersed" was inferred from `member-SF > mean-SF` and was
  wrong in the direction that matters — that inequality is explained by the *mean* being smoothed
  further, not by members having excess variance. **Always read the per-bin ratios before choosing
  a lever.** This is why `log_bin_ratios` exists.
- **`ratio_*km` never reaches `log.txt`.** `trainer.py::_log_terminal` prints only keys ending in
  `avg`. Read the JSONL at
  `<results>/<RUN>/<RUN>_train_metrics.json`, filtering `stage == "val"`.

---

## 6. What has been measured

Campaign on CERRA, HEALPix L5, 64-epoch pretrains at matched budget. Comparison arms are
regression decoders (multiband Fourier, multiscale context, multistage MLP).

**Qualitative result, and the reason this branch exists:** the flow-matching maps are
*fundamentally different* from every regression decoder's. For the first time the decoder
produces **separate points inside a cell with real per-point variance**, rather than a smooth
interpolant. Not yet correct, but structurally the right kind of output.

**Quantitative, post-cooldown, per-channel CERRA MSE** (flow arm emits a single draw):

| arm | 2t | tp | 10si | SF (tp) |
|---|---|---|---|---|
| multiband Fourier | 0.0162 | **0.4307** | 0.1611 | **3.426** |
| flow matching (96 ep) | **0.0135** | 0.4460 | **0.1357** | 3.821 |

Flow wins 2t and 10si by ~16%; Fourier still leads on tp, which is the campaign target and the
only channel SF scores. Caveat: the flow arm bought part of that by zeroing `10wdir`'s loss
weight, and the SF gap is ~1 sd of that metric's run-to-run noise.

**Per-bin member ratios** (`S_pred/S_target`, tp):

| bin | dg48kzg6 (offline) | kt8g6wey (4 samples) | in-loop (32 samples) |
|---|---|---|---|
| 10–25 km | 1.379 | 2.253 | 3.619 |
| 25–50 km | 0.468 | 0.217 | 0.685 |
| 50–100 km | 0.358 | 0.283 | 0.474 |
| 100–200 km | 0.287 | 0.231 | 0.440 |

**Read these carefully.** The 4-sample column does not replicate — use ≥32 validation samples.
The robust signal across every measurement is a **spectral slope error**: over-dispersed at
10–25 km, under-dispersed at 25–200 km. Fine speckle on a field that is too smooth at meso
scales.

Two things this does *not* license concluding:

- A low 25–200 km ratio is **not** flow-matching-specific. It is campaign-wide: the latent
  upsampling *regression* runs recorded `.14/.15/.09/.30` and `.22/.09/.09/.29` across the same
  bins — worse at 25–100 km than this arm.
- Whether the 25–50 km bin reflects real target variance or is an **estimator artifact** has
  never been settled. `playground/scripts/sf_target_spectrum.py` (target-only, minutes) exists to
  answer it. **Do this before treating any 25–50 km number as a model verdict.**

**Solver/step sweep** (on dg48kzg6): euler-64 is *not* an upgrade over euler-24. Members-SF moved
only 5.58 → 5.19 while the ensemble mean got 27% **worse** and MSE was flat — finer integration
mostly removed ensemble diversity that had been integration noise. Do not "upgrade" the step
count without re-measuring.

---

## 7. Open questions and the next lever

**Classifier-free guidance is probably the wrong tool.** `w` rescales the deviation from the
unconditional mean roughly uniformly in field space — it is **not scale-selective**, so it cannot
tilt a spectrum. Against a slope error it will trade one bin against another in whichever
direction, not fix the shape. The capability is implemented and trained (`flow_cond_dropout`), so
a sweep is cheap, but do not expect it to be the answer. If anything the interesting direction is
`w < 1` (interpolating toward the unconditional field *adds* variance), not the usual `w > 1`.

**The most promising untried lever** is an **auxiliary** structure-function term on the model's
clean-field estimate `ŷ1 = y_t + (1-t)·v`, with plain `mse` still on `y0 + v` for the
flow-matching loss itself. It is scale-selective by construction, costs one forward, and pushes
variance in whichever direction the per-bin ratio is wrong. See §3.7 for why the *naive* version
(replacing the mse target) breaks. This is not implemented.

Other threads, roughly in order of expected value:

1. Settle the 25–50 km bin (`sf_target_spectrum.py`). Cheap, and it gates the interpretation of
   every other number.
2. Widen the decoder's receptive field. Today every target point attends to 9 latent vectors at
   one scale (~450 km 1-ring) and the **keys carry no positional code at all**. Frontal placement
   is set by deformation over 500–1500 km. A multi-scale KV ladder is the `MultiScaleContext`
   decoder on `wm/dev/latent_upsampling`, not ported here.
3. Terrain as auxiliary static input — deliberately *not* load-bearing. Evaluate with a
   terrain-stratified SF (ocean / flat land / mountainous); a terrain-driven "win" that appears
   only in the mountainous stratum is orography-stamping, not skill.

---

## 8. Operational notes

- **`launch-slurm.py --from-run-id` runs the PARENT run's code snapshot, not your working tree**
  (`wgen_dir = copy_root_dir / f"slurm_weathergen_{from_run_id}_dir" / ...`). Configs *are*
  copied from home, so a continuation runs **fresh configs against stale code** and any config key
  whose reader is newer than the parent snapshot is silently ignored. `--wgen-dir` does not help.
  This invalidated a full 9.5 h fine-tune once. Before launching, refresh the parent snapshot and
  grep it for your new symbol.
- **Do not read `LossPhysical.CERRA.mse.avg`** when `target_channel_weights` zeroes a channel: the
  average includes the untrained channel, which drifts freely and makes a good run look like a
  regression. Read per-channel keys.
- Inference cost is `ens_size × flow_num_steps` decoder passes per sample. At `max_num_targets=-1`
  (1.14 M points) with ens 8 / 24 steps that is 192 full-grid passes per sample — use `ens_size`
  1–2 for map inspection.

---

## 9. Branch layout

This branch carries only the flow-matching work, on top of `develop`:

| Commit | Contents |
|---|---|
| `Fix decode 1-ring gather ...` | Upstream bug: `hp_nbours` indices are in `[0, num_cells)` but index a `(batch, cell)`-flattened tensor, so **every batch element read sample 0's latent** for `batch_size_per_gpu > 1`. Independent of flow matching; worth upstreaming. |
| `Add structure-function loss module ...` | Dependency, not a flow feature — the diagnostic the arm is read with. Droppable if unwanted. |
| `Add FlowMatchingPointDecoder ...` | The decoder, wiring, configs, stage-2 levers. |

Sibling decoders (multiband Fourier, multistage MLP, multiscale context) live on
`wm/dev/latent_upsampling` and were deliberately **not** ported.
