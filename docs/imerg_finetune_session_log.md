# IMERG diagnostic-decoder finetunes — working log and hard-won lessons

Record of a long debugging and config-building session on branch `wm/develop-ssl-diffusion-v1`.
Written so the next person (or the next session) does not have to rediscover any of it.

Everything below was verified against code, checkpoints or logs — the "how it was proved" column
matters, because several plausible-sounding explanations turned out to be wrong.

---

## 0. Quick index of traps

| symptom | real cause | §
|---|---|---|
| Precip maps come out blue→red instead of blue→green | `colors:` commented out → `_resolve_cmap` default `coolwarm` | [1](#1-evaluation-plot-colours) |
| `fact`/`tact`/`acc` raise `KeyError: 'mean'` | IMERG climatologies contain only SEEPS params | [2](#2-evaluation-metrics-and-sample-alignment) |
| Comparing two runs' `sample: "0-9"` | sample indices ≠ same init times across runs | [2](#2-evaluation-metrics-and-sample-alignment) |
| Run "completes" instantly, trains nothing, writes a checkpoint | `num_mini_epochs` is an absolute stop point | [4](#4-num_mini_epochs-and-istep) |
| Curriculum `num_steps` list silently constant | list indexed by **absolute** mini_epoch | [4](#4-num_mini_epochs-and-istep) |
| `LossPhysical.ERA5.mse` appears beside IMERG | inherited ERA5 output stream resurrected by the streams-dict union | [6](#6-the-era5-leak-four-failed-fixes) |
| Emptying `target: []` changes nothing | reader reads `*_target_channels`, not `target` | [6](#6-the-era5-leak-four-failed-fixes) |
| `DataLoader worker ... killed by signal: Killed` | **host** RAM; `num_workers` is the lever | [7](#7-memory-host-vs-gpu) |
| `torch.OutOfMemoryError: CUDA out of memory` | **GPU**; `max_num_targets` is the lever | [7](#7-memory-host-vs-gpu) |
| `KeyError: '...tte.5.lnorm.embed_aux.0'` | aliased-module bug in `load_model`, unfixed in snapshots | [9](#9-the-aliased-module-load-bug) |
| `FileNotFoundError: .../<x>_latest.chkpt` | run has only numbered checkpoints → needs `--mini-epoch` | [10](#10-launch-slurm-mechanics) |
| `ConfigKeyError: Missing key type` after re-seeding | stale file in the snapshot; config overlay never deletes | [10](#10-launch-slurm-mechanics) |

---

## 1. Evaluation plot colours

**Symptom.** `map_*_tp_*.png` rendered as a `coolwarm` blue→red field on a dark-blue background
instead of the intended white → light-blue → navy → green precipitation palette.

**Cause.** In `playground/eval_config_imerg_diag_comparison.yml` the `colors:` block under
`IMERG_ANEMOI.tp` had been commented out while `levels:` was left active. With no explicit
colours, `Plotter._parse_map_kwargs` falls back to `"cmap": kw.pop("colormap", "coolwarm")`
(`plotter.py:662`) and `_resolve_cmap` returns `plt.get_cmap("coolwarm")` (`plotter.py:704`).
Identical `levels` in both images is what made the two plots share colourbar ticks while
differing entirely in colour.

**Fix.** Uncomment the eight-colour block. Verified by rendering.

Details worth keeping:

- `BoundaryNorm(levels, cmap.N, extend="both")` needs **`len(levels)+1`** colours: 7 levels → 6
  interior bins + 2 extensions = 8. One more or fewer raises.
- `"none"` as the first colour is RGBA alpha 0.0 — transparent, which shows the white page. That
  is what produces the "dry = white" look, not a white colour.
- `playground/` is **gitignored** (`.gitignore:222`), so that config has no git history and the
  regression left no trace. A palette that matters belongs in a tracked `config/evaluate/` file.
- Output subdirectory changed: with `ensemble: "mean"` maps now land in `maps/preds_ens_mean/`,
  not `maps/preds/` (`plot_orchestration.py:1108-1109`). An empty `preds/` is not a failure.

---

## 2. Evaluation metrics and sample alignment

### Activity metrics are not available for IMERG

`fact`, `tact` and `acc` all need a climatological **mean** (`_calc_act` does
`c.sel(statistic="mean")`). Both IMERG climatologies on this system carry only

```
statistic = ['prob_dry', 'light_heavy_threshold']
```

i.e. the two SEEPS parameters. Requesting `fact` raised
`KeyError: "not all values found in index 'statistic'"` and aborted the whole evaluation after
the plots had already been written.

**Substitute:** `froct` / `troct` (forecast / target rate of change). They need no climatology —
just the mean absolute difference between consecutive steps (`calc_change_rate`) — and answer the
same question: `froct` decaying toward zero while `troct` stays flat is the collapse-to-
climatology signature an activity plot would have shown.

`needs_climatology()`'s `req_clim` list is `["acc", "rps", "rpss", "seeps"]` — note `fact`/`tact`
are **not** in it, so they silently rely on a climatology loaded for another metric.

### Sample indices are not comparable across runs

Two inference runs over the same period emitted their initialisation times **in a different
order**. Verified by reading the `times` array out of both zarr stores:

| init time | yz3h0kyn | z71y2ik8 |
|---|---|---|
| 2023-06-01T12 | 0 | 0 |
| 2023-06-01T18 | 6 | 1 |
| 2023-06-02T00 | 1 | 2 |
| 2023-06-02T06 | 7 | 3 |
| … | … | … |

So `sample: "0-9"` on both would have compared *different forecasts*. Each run needs its own
index list. Note `run_evaluation` **replaces** `default_streams` wholesale when a run defines
`streams` (it does not deep-merge), so each per-run block must be complete.

### Lead-time line plots need an explicit opt-in

In `plot_summary`, every other plot type has a boolean fallback but the lead-time one does not:

```python
do_lead_time = "lead_time" in _sp or "qq_analysis" in _sp        # no eval_opt.get(...) fallback
do_ratio     = "ratio" in _sp or eval_opt.get("ratio_plots", False)
do_heatmap   = "heatmap" in _sp or eval_opt.get("heat_maps", False)
```

`summary_plots: true` only gates whether `plot_summary` is *called*. To actually get score-vs-lead-
time curves you also need `evaluation.score_plots: ["lead_time"]`. **This was found but never
verified end-to-end — treat as probable, not confirmed.**

---

## 3. The new hourly IMERG Late dataset

`nasa-imerg-late-n320-2025-2026-1h-v1.zarr` versus the existing
`nasa-imerg-grib-n320-1998-2024-6h-v1.zarr`:

| | OLD (6h) | NEW (1h) |
|---|---|---|
| zarr format | **2** | **3** |
| compressor | Blosc lz4 clevel 5 + SHUFFLE | **Zstd level 0, no shuffle** |
| variables | 1 (`tp`, metres) | 10 (incl. `precipitation`, **mm/hr**) |
| range | 1998-01-01 → 2024-07-31 | 2025-10-01 → 2026-08-02 |
| shape | (38835, 1, 1, 542080) | (7337, 10, 1, 542080) |
| chunks | (1,1,1,542080) = 2.2 MB | same |
| logical size | 84.2 GB | 159.1 GB |

Checks that passed: `precipitation` and `precipitationQualityIndex` both present; **no gaps** in
the hourly time axis (7337/7337); statistics (`mean`/`stdev`/…) present and finite for all 10
variables; N320 grid (542080 points); `has_nans` true everywhere but the actual NaN fraction is
~0 (a handful of points at the north pole in the first step only).

**Units trap:** this store is IMERG-native **mm/hr**; the ECMWF store is **metres**. Any config
carrying `transform_scale: 1000.0` is calibrated for the latter and would be wrong by 1000× here.

**Channel-name trap:** the variable is `precipitation`, not `tp`. A wrong name yields
`source_channels == []` *silently* — the stream contributes nothing and no error is raised.

### Why it cannot be used with the existing pretrained models

No temporal overlap with the input streams:

| stream | coverage |
|---|---|
| ERA5_in (`aifs-ea-an-oper-…-1979-2024-1h-v3`) | ends **2024-12-31** |
| od-an oper o96 (`…-2016-2025-6h-v1`) | ends **2025-07-31** |
| **IMERG Late** | starts **2025-10-01** |

The only overlapping analysis is `aifs-od-an-oper-0080-mars-n320-2025-2026-6h-v1.zarr`
(2025-11-17 → 2026-04-19), which is **N320 with 542080 gridpoints against the checkpoints' o96
40320** — 13.4× the input tokens, a resolution none of them saw.

A standalone smoke config (IMERG as both source and target) got through data loading, tokenisation
and model construction but died inside flash-attention. Reproduced standalone at shape
`(35514, 10, 8, 64)`; **not** explained by head_dim, softcap, dropout or batch size in isolation.
Investigation was stopped before root cause. Treat as open.

---

## 4. `num_mini_epochs` and `istep`

```python
for mini_epoch in range(mini_epoch_base, self.training_cfg.num_mini_epochs):   # trainer.py:469
    ...
self.save_model(self.training_cfg.num_mini_epochs)                              # trainer.py:489
```

`mini_epoch_base = int(cf.general.istep / len(self.data_loader))` — recovered from the **inherited
istep**, not from the parent's epoch count.

**Consequence.** On a continuation, `num_mini_epochs` is an **absolute stop point**. If it is
below the base, the range is empty: the job trains nothing, falls through to the final
`save_model(num_mini_epochs)` and exits **0** with a checkpoint named for that number.

- Tell-tale: **zero `Mini_epoch N of M` lines** in the log plus a lone
  `<run>_chkpt000<num_mini_epochs>.chkpt`.
- Seen in run `re0jivry`: `num_mini_epochs: 16`, base ≈ 36, wrote `re0jivry_chkpt00016.chkpt`
  after training nothing.

**`general.istep: 0` is the fix**, and the whole raina IMERG family already uses it. With the
reset, `num_mini_epochs` becomes a plain count. Two deliberate side effects, both correct for a
finetune with a fresh decoder:

- optimizer momentum is **not** restored (`_load_optimizer_state` is guarded by `istep != 0`);
- the LR scheduler starts at step 0 with a full warmup.

### `forecast.num_steps` lists share the same trap

`num_steps` may be a list, but `_get_fsm` indexes it by **absolute** mini_epoch:

```python
idx = min(self.mini_epoch, len(self.list_num_forecast_steps) - 1)
```

A 16-entry ramp on a run whose epochs are 32..47 is indexed at 15 every time and silently
collapses to the last value — the curriculum never happens. Pad the list to `num_mini_epochs`
length with the ramp at the absolute indices.

Policies: `fixed`/`sequential` → `fs = fsm` for every batch; `random`/`sequential_random` →
per-batch `randint(list.min(), fsm+1)`. Only `random` ignores the epoch index.

---

## 5. Forecast depth per lineage — decides unfreeze vs not

Read from each parent's saved `model_<run>_*.json`.

| lineage | native `num_steps` | forecast engine |
|---|---|---|
| perrjey5 → c71eo6pu → hmpa42d2 | **2** | trainable until hmpa42d2 **froze** it |
| n0t6ejuo → srdrwfy6 → af90zz71 | **8** | trainable |
| fx276yn3, cw6a4szu | **8** | trainable |
| MTM (od08us1u → … → j6nb50v9 / ra1xax01) | **8** | trainable |
| skkbxfya → oywmz4sz / dy0jlrmw | **8** | trainable |

So an 8-step IMERG finetune of anything except the **c71eo6pu** lineage asks the engine for
nothing it has not already done — no curriculum, no unfreeze. Only hmpa42d2 needed both, because
its whole lineage topped out at 2 steps *and* it froze the engine (61 of 1846 params trainable).

**Freeze-regex asymmetry.** Whether `.*ERA5.*` may stay depends on who owns the decoder:

- fx276yn3 / cw6a4szu / hmpa42d2: the decoder to be **trained** is ERA5-owned in the checkpoint,
  so `.*ERA5.*` must be **dropped** — keeping it resolves to **0 of 2191 trainable** (checked).
- MTM / oywmz4sz / dy0jlrmw: IMERG is a **new** stream with its own modules, so `.*ERA5.*` is
  **kept**, and it usefully freezes the leftover ERA5 decoder.

Always add a forecast-engine term when the engine should stay frozen. Verify with the checkpoint
key list, not by reading the regex.

---

## 6. The ERA5 leak — four failed fixes

Goal: make IMERG the only target. This took four attempts because each layer hid the next.

**Attempt 1 — omit `era5.yml` from the stream dir.** Failed. `streams` is a dict, so
`OmegaConf.merge` **unions** it with the checkpoint's and ERA5 returns complete with its decoder.
Symptom in `zk8jugvb` / `x5uptfho`:

```
LossPhysical.ERA5.mse.avg        : 2.0428E-02
LossPhysical.IMERG_ANEMOI.mse.avg: 9.4637E-01
LossPhysical.loss_avg            : 4.8340E-01     <- (ERA5 + IMERG) / 2
```

Two consequences: ERA5's decoder trains (the regex had dropped `.*ERA5.*`), and
`loss = loss / ctr_streams` with `ctr_streams = 2` **halves IMERG's gradient**.

**Attempt 2 — `streams: {ERA5: {reconstruct: false}}` in the finetune config.** Failed, silently.

```python
def _load_streams_in_config(config):
    if streams_directory is not None:
        config.streams = load_streams(streams_directory)      # ASSIGNMENT, not merge
```

The directory expansion **overwrites** any `streams:` block in the same file, before the merge
with the checkpoint config happens. The override has to live in the streams *directory*.

**Attempt 3 — `era5.yml` in the directory with `reconstruct: false` and `target: []`.** Still
failed. The reader does not read `target`:

```python
if stream_info.get(str(stage) + "_target_channels") is None:   # data_reader_anemoi.py:117
    ...derive from `target`
else:
    self.target_channels = stream_info.get(str(stage) + "_target_channels")
```

`train_target_channels` / `val_target_channels` are stored **resolved** in the checkpoint config
(81 channels) and survive the merge independently of `target`. Runs `e7dltovv` and `efc7zcfy`
still logged all 81.

**Attempt 4 — empty all three.** Works:

```yaml
ERA5:
  reconstruct: false
  target: []
  train_target_channels: []
  val_target_channels: []
```

**Then the same bug again, one stream over.** `ERA5_in` is a *forcing* stream with no decoder, yet
the older samplers collect targets for **every** stream at **every** forecast step with no forcing
check — the `stream_has_targets` guard exists only on newer code. So its 74 channels were read and
discarded: `74 × 542080 × 4 B × 8 steps = 1.28 GB per sample`. Emptied the same way, in
`analysis.yml`.

Cumulative effect on the anemoi target read per sample: **1.58 GB → 1.28 GB → 0.00 GB**.

`loss_weight: 0.0` is *not* an alternative: `ctr_streams` increments regardless of the weight, so
IMERG's gradient stays halved.

`forcing: true` is *not* an alternative either where the checkpoint contains that stream's decoder
modules — the flag skips building them (`model.py:404`) and the load then reports unexpected keys.

---

## 7. Memory — host vs GPU

These look nothing alike in the log, and that is the reliable signal.

| | **GPU** | **host / CPU** |
|---|---|---|
| message | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.04 GiB. GPU 3 has a total capacity of 95.00 GiB…` | `RuntimeError: DataLoader worker (pid …) is killed by signal: Killed.` |
| kind | catchable Python exception | SIGKILL from the OOM killer |
| traceback | model code (`model.py`, `attention.py`, `backward`) | `torch/utils/data/_utils/signal_handling.py` |
| lever | `max_num_targets`, forecast depth, freezing | `num_workers`, `num_workers_validation`, what each sample loads |

**A host OOM usually surfaces as a wall of NCCL errors** — `ncclRemoteError`, collective timeouts,
"remote process exited". Those are the *other* ranks timing out on an all-reduce whose partner
died. Always scroll to the first failure.

Triage:

```bash
grep -c "OutOfMemoryError" logs/<run>/output.*.txt     # >0 -> GPU
grep -c "killed by signal" logs/<run>/output.*.txt     # >0 -> host
sacct -j <jobid> --format=JobID%22,State%12,MaxRSS,AveRSS,NTasks --units=G
```

### `num_workers`

DataLoader subprocesses **per rank**, CPU-side only. Host RAM ≈
`ranks_per_node × workers × prefetch × bytes_per_sample`; with 4 ranks/node, 8 workers is 32
loader processes per node. `trainer.py:339-350` builds **both** a train and a validation loader,
each with its own pool; `num_workers_validation` sets the latter separately.

Silent gotcha:

```python
self.len = ((epoch_len // world_size) // (batch_size * workers)) * (batch_size * workers)
```

`samples_per_mini_epoch` is floored to a multiple of `batch_size × num_workers`. Asking for 16
samples with 6 workers silently yields **12**.

**Measured bracket** (fx276yn3 lineage, 8 forecast steps, 4 ranks on 858 GB nodes):

| workers | MaxRSS/rank | outcome |
|---|---|---|
| 4 | 80.7 G | survived |
| 8 | 125.6 G | killed (3×: `h4rw3fa7`, `b88hm6ku`, `to4xs0gh`) |

≈ 11.2 G per worker on ≈ 36 G fixed. `sacct` samples periodically, so **MaxRSS understates
transient peaks** — 125.6 G "should" fit the 214 G/rank budget and did not.

**The real driver is the IASI stream, not the backbone:**

| | IASI stream | channels | workers that fit |
|---|---|---|---|
| fx276yn3 lineage | `METOP_IASI_PC` | **210** | 6 |
| everything else | `METOP_ABC_IASI` | 17 | 8 |

That 12× difference is why `r2iy7iko` (cw6a4szu) ran at 8 workers while `to4xs0gh` (fx276yn3)
died at 8 with an otherwise identical config.

**What does *not* help host memory:** more nodes (DDP keeps per-GPU batch constant, and each node
runs its own pool), `--mem=` (already `--mem=0`, i.e. the whole node), `expandable_segments`
(already exported by `hpc/jupiter/weathergen_slurm.sh:62`), and `multiprocessing_method: fork`
(reduces *startup* cost, not steady-state RSS — and `spawn` is deliberate because the obs streams'
async zarr stores deadlock with forked workers).

### `max_num_targets` — the GPU lever

Decoded points per sample = `max_num_targets × forecast_steps`, with `-1` meaning all 542080 N320
points. Deepening 2 → 8 steps without re-capping quadruples the decoder work:

| | decoded points/sample |
|---|---|
| h1mrni5n at 2 steps (16 epochs, fine) | 1,084,160 |
| same config at 8 steps | **4,336,640** → GPU OOM in `backward` |

Cap to preserve the budget: `131072 × 8 = 1,048,576` (97%). For dt6oawrf, which also decodes
ERA5, both need capping: `IMERG 131072` **and** `ERA5 27136`.

Training-only — `trainer.inference()` forces `max_num_targets = -1`, so evaluation still scores
full maps, and targets are resampled every step so coverage accumulates.

`--options` **can** reach stream keys (it is applied after the directory expansion), which is
preferable to editing a stream dir shared by several configs:

```
--options streams.IMERG_ANEMOI.max_num_targets=131072 streams.ERA5.max_num_targets=27136
```

### Throughput

`s/sec = (print_freq × batch_size_per_gpu) / dt` — samples/sec **per rank**; with batch 1 it is
steps/sec. Measured on the 8-step fx276yn3 config: 0.025 at 4 workers (≈50 s/step, 6.8 h/epoch)
versus ≈0.11 at 8 (≈9 s/step, 1.2 h/epoch). 512 steps/epoch at 4096 samples over 8 ranks.

For the 8-step cw6a4szu run: steady state ≈0.053 → **2.7 h/epoch, ≈44 h for 16 epochs**. The
Slurm limit is 12 h per job, so that needs **`--chain-jobs 4`**, not 2. `--chain-jobs 2` covers
about 9 epochs.

At that point the GPUs were at **99–100% utilisation with ~96 GB of 97 GB used**, so the run was
compute-bound, not starved. Remaining levers are trade-offs: fewer samples per epoch, more nodes
(this *does* help throughput), or fewer epochs.

---

## 8. `output_clamp` — non-negative precipitation

Requested: guarantee non-negative precipitation output "without hurting performance".

`pred_head.final_activation` exists and accepts `relu`, `softplus`, … but **must not be used
here**. The head emits **normalized** values and normalization is `(x - mean) / stdev`. For IMERG
tp, `mean = 0.676 mm/6h`, so

```
physical 0 mm  ->  normalized -0.2074
```

A `relu`/`softplus` head floors the normalized output at 0, i.e. **≥ 0.676 mm/6h everywhere**.
Sampled over 1.6 M points: **68.8% of IMERG is exactly dry, 88.1% below that floor.** It would
break the dominant part of the field, not be performance-neutral. No registered activation has
the right floor either (`elu` bottoms at −1 ≈ −2.58 mm, `silu` at ≈ −0.278).

**Implemented instead:** a per-stream clamp applied *after* denormalization, predictions only
(`validation_io._apply_output_clamp`, committed as `19e860873`).

```yaml
IMERG_ANEMOI:
  output_clamp:
    min: 0.0        # both bounds optional; absent key is a strict no-op
```

Targets are never clamped — they are the verification truth. The training loss, computed in
normalized space, is untouched.

**Caveat:** this is a *code* feature on this branch, and `--from-run-id` runs the parent
snapshot's code, so on older snapshots the key is silently ignored.

---

## 9. The aliased-module load bug

```
KeyError: 'target_token_engines.IMERG_ANEMOI.tte.5.lnorm.embed_aux.0'
  model_interface.py:321  ->  module_to_init = all_modules[path]
```

When a decoder is created **fresh**, `load_model` derives root modules from the missing checkpoint
keys and looks them up in `named_modules()` — which **deduplicates aliased modules**, so a shared
`lnorm` has no entry under its second path and the bare `[path]` lookup raises.

Fixed on this branch (`all_modules.get(path)` plus a `get_submodule` fallback and an alias skip,
`model_interface.py:319-346`) — but **all parent snapshots checked are unfixed**
(`fx276yn3`, `j6nb50v9`, `dy0jlrmw`, `yqsi7obk`).

Why some runs escaped it: `h1mrni5n` / `dt6oawrf` already *contain* an IMERG decoder, so nothing
is created fresh and the path never runs. Fresh-decoder finetunes of `dy0jlrmw`, `oywmz4sz`,
`j6nb50v9`, `ra1xax01` all hit it.

`qk_norm_type: RMSNorm` was tried and **did not help** — the failing module is the AdaLayerNorm's
`embed_aux`, not a qk norm. It was reverted, and it was the wrong kind of change anyway: every
attention head reads that key from the **global** config
(`qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type)`), so it would alter the frozen
encoder and forecast engine, not just the decoder.

**The fix is to ship newer code** via a seeded snapshot — see next section.

---

## 10. launch-slurm mechanics

`--from-run-id X` copies **code** from `<slurm>/slurm_weathergen_X_dir/WeatherGenerator` — the
code that trained the checkpoint — while refreshing **configs** from the home clone
(`copy_all_configs`, which takes all `config/**/*.y*ml`, tracked *and* untracked, so uncommitted
config edits do ship). There is no flag to override the code source.

Practical consequences:

- **New code fixes never reach a continuation.** Hence `playground/raina/seed_raina_run.py`, which
  builds a parallel `<run>_<suffix>` whose snapshot holds the code you want, with the checkpoint
  **hardlinked** (no extra disk). Default ships this working tree; `--code-from DIR` ships another
  snapshot, needed when this branch cannot build the checkpoint's architecture.
- On Jupiter the script needs `WG_MODELS_DIR=/e/scratch/weatherai/shared_work/models` and
  `WG_SLURM_DIR=/e/scratch/weatherai/slurm` — its defaults are CSCS paths.
- **`--mini-epoch` defaults to `-1` → `<run>_latest.chkpt`.** Runs that only ever wrote numbered
  checkpoints (`fx276yn3`, `cw6a4szu`) need an explicit `--mini-epoch 8`, or every rank dies with
  `FileNotFoundError: .../<run>_latest.chkpt` (run `vsosleus`).
- **Run ids are 8 characters.** `--from-run-id zk8jugv` (7) resolved to a non-existent snapshot
  and raised `ValueError: WeatherGenerator directory … does not exist`.
- **The config overlay never deletes.** A file removed from the home clone survives inside the
  snapshot. A stale partial `era5.yml` (an override with no `type:`) became a full stream
  definition under replace semantics → `ConfigKeyError: Missing key type` (run `wawmsyhn`). After
  editing a seeded snapshot, also fix `tracked_files.json`: the launcher blindly
  `shutil.copy2`s every listed path, so a stale entry crashes it with `FileNotFoundError` — the
  same failure seen on CSCS at the start of this session.
- **Seeding from a live run is a moving target.** `dy0jlrmw` advanced from istep 5870 → 6120 mid-
  session; the hardlink broke (`nlink=1`) and the verifier then reported "not identical" and "no
  shared inode". Neither is corruption: the seed is a coherent frozen copy at the older istep.

### Shell quoting

A missing space before a line continuation silently concatenates arguments:

```bash
test_config.samples_per_mini_epoch=72\      # <- no space
test_config.output.num_samples=72\
```

becomes one argument, so `samples_per_mini_epoch` was set to the **string**
`"72test_config.output.num_samples=72…"` and numpy raised
`ufunc 'less_equal' did not contain a loop with signature matching types (Int64DType, StrDType)`.
The other two options were never set at all.

Also: `uv run inference` exits **0** even on failure, because the post-mortem `pdb` swallows it.
`EXIT_CODE` is meaningless there.

---

## 11. Merge semantics differ per family — check before trusting a recipe

Three parent families, three different behaviours from configs that look identical:

| family | `streams_directory` | `reconstruct: false` | ERA5 removed by | stream dir must hold |
|---|---|---|---|---|
| af90zz71 / fx276yn3 / cw6a4szu | **unions** | supported | `era5.yml` neutraliser | output side only |
| j6nb50v9 / ra1xax01 (MTM) | **replaces** | **absent** | simply omitting it | **all streams** |
| oywmz4sz / dy0jlrmw | **unions** | supported | `era5.yml` neutraliser | output side only |
| any `_fix` snapshot (this branch's code) | **replaces** | supported | omitting it | **all streams** |

The discriminator is whether the snapshot's `config.py` contains
`base_config.streams = None`. Check it, do not assume:

```bash
grep -c "base_config.streams = None" <snapshot>/packages/common/src/weathergen/common/config.py
grep -c "is_stream_reconstructed"    <snapshot>/src/weathergen/utils/utils.py
```

Re-seeding a run with newer code **flips this**, which is why
`config/streams/imerg_diag_dy0jlrmw/` had to be rebuilt from output-only to all-streams form.

Other per-family requirements:

| family | loss key for the MSE block | `--mini-epoch` | `num_workers` |
|---|---|---|---|
| fx276yn3 / cw6a4szu | `forecast` | **required** | 6 / 8 |
| MTM, oywmz4sz, dy0jlrmw | **`physical`** | not needed | 8 |
| dt6oawrf / h1mrni5n | `forecast` | not needed | 4 / 6 |

`validation_io.write_output` asserts **exactly one** `LossPhysical` block. The merge *unions* the
losses dict, so declaring the MSE loss under a new key while the parent already has a live
`LossPhysical` under another trips that assert. Parents whose extra blocks are `type: Disabled`
are safe.

### Predicate artefacts when validating statically

`is_stream_forcing` / `is_stream_reconstructed` treat a stream with empty **resolved** channel
lists as forcing. A fresh IMERG stream has those unset in YAML, so a static check reports it as
non-decoded — alarming but wrong. The sampler writes them back
(`stream_info[str(stage) + "_target_channels"] = ds.target_channels`) and is constructed
**before** the model (`trainer.py:336-337` then `:353`), so it resolves at runtime. Simulate the
write-back when validating offline.

---

## 12. Artefacts produced

### Committed

| commit | contents |
|---|---|
| `19e860873` | `output_clamp` implementation (`validation_io.py` only) |
| `074e1e953`, `6c68246ba`, `5be9b7230` | fx276yn3 configs + stream dir, eval/plot configs (by the user) |

### Configs

| file | parent | steps | epochs |
|---|---|---|---|
| `config_finetune_imerg_diag_mse_hmpa42d2_8step.yml` | hmpa42d2 | 2→8 curriculum, `sequential_random` | 48 abs |
| `config_finetune_imerg_diag_mse_hmpa42d2_8step_fixed.yml` | hmpa42d2 | 8 fixed | 48 abs |
| `config_finetune_imerg_diag_mse_rxssbn0v_8step.yml` | rxssbn0v | 8 | 52 abs |
| `config_finetune_imerg_diag_mse_fx276yn3{,_8step}.yml` | fx276yn3 | 2 / 8 | 16 |
| `config_finetune_imerg_diag_mse_cw6a4szu{,_8step}.yml` | cw6a4szu | 2 / 8 | 16 / 8 |
| `config_finetune_imerg_diag_mse_j6nb50v9{,_8step}.yml` | j6nb50v9 | 2 / 8 | 16 |
| `config_finetune_imerg_diag_mse_ra1xax01{,_8step}.yml` | ra1xax01 | 2 / 8 | 16 |
| `config_finetune_imerg_diag_mse_oywmz4sz{,_8step}.yml` | oywmz4sz (`--mini-epoch 2`) | 2 / 8 | 16 |
| `config_finetune_imerg_diag_mse_dy0jlrmw{,_8step}.yml` | dy0jlrmw / `dy0jlrmw_fix` | 2 / 8 | 16 |
| `config_finetune_imerg_diag_mse_dt6oawrf_8step.yml` | dt6oawrf | 2→8 | 6 |
| `config_finetune_imerg_diag_mse_h1mrni5n_8step.yml` | h1mrni5n | 2→8 | 6 |

### Stream directories

`imerg_diag_fx276yn3` · `imerg_diag_fx276yn3_era5tgt` (ERA5 kept as a target) ·
`imerg_diag_cw6a4szu` · `imerg_diag_mtm` (all streams) · `imerg_diag_oywmz4sz` ·
`imerg_diag_dy0jlrmw` (rebuilt to all-streams) · `imerg_late_test`

### Reports

- `docs/imerg_finetune_parents_comparison.md` — the seven-parent comparison (size, architecture,
  I/O, training history), all 28 parameter figures re-verified against the `.chkpt` tensors.
- this file.

---

## 13. The dt6oawrf / h1mrni5n experiment

Both descend from fx276yn3 via the two contaminated runs, and differ in exactly one thing —
which makes them a clean A/B on "does adding ERA5 to the IMERG decoder help?".

| | decoder owners in checkpoint | loss terms | params |
|---|---|---|---|
| **dt6oawrf** (← zk8jugvb) | **ERA5 (131) + IMERG (131)** | `LossPhysical.ERA5` + `LossPhysical.IMERG_ANEMOI` | 2322 |
| **h1mrni5n** (← c6liueoc) | IMERG only (131) | `LossPhysical.IMERG_ANEMOI` | 2191 |

The only config difference is `streams.ERA5.reconstruct` (absent vs `false`) plus
`output_clamp.min` on h1mrni5n.

**For the write-up:** dt6oawrf's IMERG gradient is **halved** relative to h1mrni5n
(`loss = loss / ctr_streams`, `ctr_streams = 2`). That is inherent to the comparison, but it means
any IMERG-skill difference confounds the auxiliary-task effect with an effective-learning-rate
difference. An `ERA5.loss_weight` sweep would separate them.

`ERA5_in`'s wasted targets were emptied in both arms — it is a forcing stream with no decoder, so
that changes no science.

---

## 14. Upstream merge `ee3b326b8` — which incoming changes alter results

Merged 8 commits from `ecmwf/develop-ssl-diffusion-v1`. Some change **numerics**, not just crash
behaviour. Sorted by whether they affect the IMERG runs.

### Will change results

**#2722 `torch.empty` → `torch.zeros` for the token buffer.** Unwritten rows were previously
**uninitialised GPU memory**, i.e. sporadic NaNs. Any run where a stream's tokens were skipped
during embedding was reading garbage. A genuine correctness fix — and it means older runs that hit
that path are **not reproducible**, by construction, since the values were non-deterministic. The
same commit adds an assert that now fails loudly instead of silently zero-filling.

**#2722 also adds `LayerNorm` on intermediate encoder levels** before the SSL fusion concat
("they arrive on very different and much larger scales"). Changes the forward pass for anything
using `deep_ssl`. Our IMERG configs set `deep_ssl.enabled: False`, so this should not touch them —
worth confirming per parent in case one inherits it enabled.

**#2739 `RMSNorm.reset_parameters`.** Fresh modules are initialised via `reset_parameters()`;
without it an RMSNorm created fresh kept whatever `to_empty()` left behind. This lands squarely on
**the fresh IMERG decoders**. It only matters when `qk_norm_type: RMSNorm` — the fx276yn3 /
cw6a4szu lineage, not the MTM / dy0jlrmw ones.

**#2737 target masks for empty-target streams.** Directly relevant to §6: streams with
`target: []` previously got a **zero** target cell mask, dropping their source tokens from the
latent target entirely. They now get the normal mask. Since ERA5 and ERA5_in were deliberately
emptied, this changes how those streams contribute to the latent target — but **only under
student-teacher / latent-loss training**, which these configs disable. Low risk; sanity-check the
first loss values.

### Will not change results

- **`lr_scaling_parallel_policy` rename** (`"const"` → `"constant"`). All 21 IMERG configs use
  `"sqrt"`, matched correctly before and after. No effect.
- **#2738 / #2729** — checkpoint loading only (`module.` prefix handling; `with_ddp`/`with_fsdp`
  passed explicitly instead of read from `cf`). Affects whether a load succeeds, not what the
  model computes.
- **#2743** — inference-time timestep indexing; changes which steps get written, not the model.
  (It also supersedes the local fix `a47c8fa0e`; see §1 of this section's note in
  `validation_io.py`.)

### What this means practically

Exposure for the IMERG finetunes is **#2739** (fresh decoder init, fx276yn3 lineage only) and the
**`torch.empty` fix** if any stream hits the skip path.

**Do not mix pre- and post-merge runs within one comparison arm.** The dt6oawrf / h1mrni5n pair in
particular must both be launched from the same code state, or the ERA5-auxiliary effect being
measured is confounded with an initialisation change.

### `_fix` snapshots

`dy0jlrmw_fix` was originally seeded at `6c68246ba`, before the merge. All four snapshots are now
on merged code `ee3b326b8`, so the family is internally comparable:

| snapshot | istep | notes |
|---|---|---|
| `dy0jlrmw_fix` | 5870 | code refreshed **in place** — its checkpoint is the only surviving copy of that state (dy0jlrmw has since moved to 8668), so deleting to re-seed would have destroyed it |
| `j6nb50v9_fix` | 10942 | seeded from `_latest` |
| `ra1xax01_fix` | 7644 | seeded from `_latest` |
| `oywmz4sz_fix` | 1536 | epoch 2, as requested. `--source-json` only changes the json — the seeder hardcodes `<run>_latest.chkpt` (line 152) — so the checkpoint link was **repointed to `chkpt00002`** afterwards, or json and weights would have disagreed. No `--mini-epoch` needed now. |

Because the merged code **replaces** streams (§11), `imerg_diag_oywmz4sz` and
`imerg_diag_dy0jlrmw` had to be rebuilt from output-only to all-streams form.
`imerg_diag_mtm` already was.

Refreshing a snapshot in place must use the seeder's own `working_tree_files()` rule — `git
ls-files` **plus** all `config/**/*.yml` including untracked. Using `git ls-files` alone flags 41
files as stale, including the run's own untracked config.

Each snapshot was verified to carry **both** sides after the refresh — ours (`all_modules.get(path)`
alias fix, `_apply_output_clamp`, `stream_has_targets` forcing skip) and upstream's
(`tokens_all = torch.zeros`, `RMSNorm.reset_parameters`, `model_has_prefix_module`,
`fe_diffusion_model` gate, `is_stream_diagnostic` skip). 464 files each, manifests consistent,
0 missing.

### Configs re-pointed at the `_fix` parents

All eight configs for these four parents now use `--from-run-id <parent>_fix`:

```bash
../WeatherGenerator-private/hpc/launch-slurm.py --from-run-id j6nb50v9_fix --config config/raina_config/config_finetune_imerg_diag_mse_j6nb50v9_8step.yml --nodes 2 --chain-jobs 2 --account=e-ext-2025e01-128 --partition=booster
../WeatherGenerator-private/hpc/launch-slurm.py --from-run-id ra1xax01_fix --config config/raina_config/config_finetune_imerg_diag_mse_ra1xax01_8step.yml --nodes 2 --chain-jobs 2 --account=e-ext-2025e01-128 --partition=booster
../WeatherGenerator-private/hpc/launch-slurm.py --from-run-id oywmz4sz_fix --config config/raina_config/config_finetune_imerg_diag_mse_oywmz4sz_8step.yml --nodes 2 --chain-jobs 2 --account=e-ext-2025e01-128 --partition=booster
../WeatherGenerator-private/hpc/launch-slurm.py --from-run-id dy0jlrmw_fix --config config/raina_config/config_finetune_imerg_diag_mse_dy0jlrmw_8step.yml --nodes 2 --chain-jobs 2 --account=e-ext-2025e01-128 --partition=booster
```

The `--mini-epoch 2` was **removed** from the oywmz4sz configs: `oywmz4sz_fix`'s `_latest` *is* the
epoch-2 state now (json istep 1536 and the checkpoint hardlink both point at `chkpt00002`).

### Effect on jobs already submitted

**None.** `launch-slurm` copies code and configs into
`/e/scratch/weatherai/slurm/slurm_weathergen_<runid>_dir/` at *submit* time; from then on the job
executes that frozen snapshot. The merge changed the home clone only, so running and queued jobs
keep the code they were launched with.

Evaluation is the exception worth remembering: `uv run evaluation` runs from the **working tree**,
not a snapshot. The merge did not touch `packages/evaluate/`, so results are unchanged — but an
evaluation run today and one run last week are not guaranteed to be the same code in general.

---

## 15. Short variants — `ra1xax01` and `oywmz4sz` (2-step × 8 ep, 8-step × 4 ep)

Two shorter runs off the same `ra1xax01_fix` parent, added alongside the 16-epoch pair:

| config | steps | mini epochs | warmup / cooldown |
|---|---|---|---|
| `…_ra1xax01.yml` (original) | 2 | 16 | 256 / 512 |
| `…_ra1xax01_8step.yml` (original) | 8 | 16 | 256 / 512 |
| **`…_ra1xax01_2step_8ep.yml`** | 2 | **8** | **128 / 256** |
| **`…_ra1xax01_8step_4ep.yml`** | 8 | **4** | **64 / 128** |

Everything else is byte-identical to the 16-epoch pair: same `imerg_diag_mtm` stream directory,
same freeze regex, same `physical` loss key, `samples_per_mini_epoch: 4096`, `general.istep: 0`,
`num_workers: 8`.

**Why the LR lengths were scaled and not left at 256/512.** `trainer.py:547` computes

```
lr_steps = int((len_ds * num_mini_epochs) / batch_size_per_gpu)
```

so `lr_steps` is *linear* in `num_mini_epochs` at any node count. Scaling warmup and cooldown by
the same factor therefore preserves the schedule's **shape** — 6.25 % warmup / 81.25 % decay /
12.5 % cooldown — at every `--nodes N`, which is what makes the curves comparable with the
16-epoch runs. Left at 256/512, the 4-epoch run would have spent 25 % of its steps ramping and
50 % in linear cooldown, leaving a quarter of the run at useful LR.

`LearningRateScheduler.__init__` has a safety net for this (`lr_scheduler.py:54`): if
`n_steps_decay < 0.2 * lr_steps` it silently rewrites warmup/cooldown to 10 %/5 % and logs a
warning. Note it would **not** have fired on the unscaled 4-epoch case — decay would have landed
at exactly 25 %, above the threshold — so the bad schedule would have run as written.

**Cost.** An 8-step mini epoch costs roughly 4× a 2-step one, so the 4-epoch/8-step run is about
2× the wall time of the 8-epoch/2-step run, not half.

**Seeding: nothing to do.** `ra1xax01_fix` was already complete and was re-verified here —
`ra1xax01_fix_latest.chkpt` + `model_ra1xax01_fix_latest.json` in the model dir,
`slurm_weathergen_ra1xax01_fix_dir/WeatherGenerator` staged, both code fixes present
(`all_modules.get` at `model_interface.py:321`, `_apply_output_clamp` at `validation_io.py:25`
with the call site at `:189`), and the snapshot's `config/streams/imerg_diag_mtm/` matching the
working tree exactly — five files, **no stale `era5.yml`**, which is the trap that killed
`wawmsyhn` (§9). New config files need no re-seed: launch-slurm overlays `config/` from this
clone at submit time.

### `oywmz4sz` — same two shapes, already at the right epoch counts

The two `oywmz4sz` configs were **already** 8 ep (2-step) and 4 ep (8-step); only the LR schedule
was wrong, still carrying the 16-epoch 256/512. Both were rewritten in place to 128/256 and
64/128 respectively, matching the `ra1xax01` short variants exactly, so the four runs are
mutually comparable. Neither had been launched — verified two ways: no `imerg_diag_mse_from_oywmz4sz`
in any of my last 400 log dirs, and no `model_*.json` under `shared_work/models/` names
`oywmz4sz_fix` as a parent except the snapshot's own.

`num_workers` is 8 on the 2-step and 6 on the 8-step, inherited from how the files were written.

### `oywmz4sz_fix` verification — the epoch-2 hardlink is correct

This is the one snapshot whose checkpoint was repointed by hand (`--source-json` rewrites only
the json; the seeder hardcodes `<run>_latest.chkpt` at line 152), so it is worth restating the
evidence rather than the intent:

| check | result |
|---|---|
| `oywmz4sz_fix_latest.chkpt` inode | **309907524** |
| `oywmz4sz_chkpt00002.chkpt` inode | **309907524** ✅ same file |
| `oywmz4sz_chkpt00001` / `00003` inodes | 309907521 / 309907527 — different |
| `oywmz4sz_latest.chkpt` inode | 309675765 — **different**, correctly not used |
| `model_oywmz4sz_fix_latest.json` | istep **1536**, run_history `[[skkbxfya, 0]]` |
| vs `model_oywmz4sz_chkpt00002.json` | byte-identical outside `general`; same istep/history |
| vs `model_oywmz4sz_latest.json` | istep 12752, six extra run_history entries — the wrong state |

So the finetune starts from epoch-2 weights, as intended. **`--mini-epoch` must therefore be
omitted**, not set to 2: the snapshot's `_latest` *is* epoch 2, and `--mini-epoch 2` would look
for `oywmz4sz_fix_chkpt00002.chkpt`, which does not exist.

Snapshot code and streams also check out: `base_config.streams = None` present (replace
semantics), `all_modules.get` and `_apply_output_clamp` present, `fe_diffusion_model` present,
`src/` **byte-identical to `ra1xax01_fix`'s**, and the snapshot's
`config/streams/imerg_diag_oywmz4sz/` diffs clean against the working tree — five files, no stale
`era5.yml`.

### Stale headers corrected in four configs

Both `ra1xax01` configs and both `oywmz4sz` configs claimed their stream directory holds "just
`imerg_anemoi.yml` and `era5.yml`" with the forcing inputs "inherited … through the streams-dict
union". Both halves are wrong — §11 of this log and §6 of the parents comparison had it right all
along; only these four config headers were stale.

Each directory actually holds **all ten streams** across five files — `analysis.yml` (ERA5_in),
`avhrr.yml` (AVHRR, METOP_ABC_IASI), `geos.yml` (METEOSAT_SEVIRI_IR, GOES_ABI_IR,
HIMAWARI_AHI_IR, GOES_ABI_VIS, HIMAWARI_AHI_VIS), `synop.yml` (SurfaceCombined),
`imerg_anemoi.yml` — and there is **no `era5.yml`** in either. The merge is a **replace**, not a
union: both `_fix` snapshots' `config.py` carries the `base_config.streams = None` guard
(line 448), so the checkpoint's stream dict is discarded and only the directory survives. ERA5 is
excluded by plain omission, which is why no neutraliser file is needed.

For `oywmz4sz` the stale text was doubly misleading, since the *original* `oywmz4sz` snapshot
really did union (§11) — the header was accurate before the re-seed and silently stopped being so
when the parent became `oywmz4sz_fix`. Behaviour was correct throughout; only the comments were
wrong. The `oywmz4sz` headers also now state the epoch-2 hardlink and the "no `--mini-epoch`"
rule explicitly, since that is the config's single easiest thing to get wrong.

---

## 16. `loss_avg` = 2 × the stream loss — a duplicated model-input branch, not a second target

Run `gckt1vzm` logged

```
LossPhysical.IMERG_ANEMOI.mse.avg : 6.7640E-01
LossPhysical.loss_avg             : 1.3526E+00
```

— exactly 2.0000× on every logged step. **IMERG is genuinely the only target.** The log has just
two `LossPhysical.*` keys, `IMERG_ANEMOI.mse.avg` and `loss_avg`; every other stream prints
`target channels: []` and only `IMERG_ANEMOI` prints `['tp']`. The factor 2 is the **batch
dimension**, not a second stream.

### Mechanism

`training_config.model_input` is a **dict of input branches**, and batch size is derived from it:

```python
# train/utils.py:139  get_batch_size_from_config
num_samples = 0
for _, source_cfg in config.model_input.items():
    if source_cfg.get("enabled", True):
        num_samples += source_cfg.get("num_samples", 1)
```

The merge **unions** that dict. Each parent family names its single branch differently:

| parent family | branch name in the checkpoint config |
|---|---|
| `default_config.yml` | `forecasting` |
| JEPA (`af90zz71`, `fx276yn3`, `cw6a4szu`, `dt6oawrf`, `h1mrni5n`) | `random_easy` |
| MTM `ra1xax01`, `j6nb50v9` | **`masking`** |
| MTM `oywmz4sz`, `dy0jlrmw` | `forecasting` |

Every finetune config declared `model_input: forecasting: {masking_strategy: forecast}`. That
**overrides** the branch when the parent's key is already `forecasting`, and it **adds a second
branch** when the parent's key is `masking` — batch size 2.

The JEPA configs escaped by accident: they also carry `random_easy: {enabled: False}`, which
zeroes the inherited branch, so `random_easy + forecasting` still sums to 1. Confirmed
empirically — `lf97ve7d` (cw6a4szu, 8-step) logs `loss_avg / IMERG.avg = 1.0000` over 553 steps.

### Why the factor propagates into the gradient

`LossPhysical` accumulates over batch items and never divides by them
(`loss_module_physical.py:329-334`):

```python
loss_timestep = loss_timestep + loss_st_corr
ctr_batch += 1 if ctr_loss_fcts > 0.0 else 0   # counted, never used as a denominator
...
loss_stream = loss_stream + loss_timestep
ctr_timesteps += 1 if ctr_batch > 0 else 0
denom = ctr_timesteps if ctr_timesteps > 0 else 1.0
loss = loss + (stream_loss_weight * loss_stream) / denom
```

Forecast steps and streams *are* averaged; batch items are **summed**. So the doubled branch
doubles the backprop loss and hence the gradients — an effective **2× learning rate** — on top of
running each sample through the model twice (~2× the step cost). The reported per-stream
`IMERG_ANEMOI.mse.avg` is unaffected: `losses_all` is keyed
`[stream][timestep][loss_fct][channel]`, so the second batch item simply overwrites the first.

The two branches were not even doing different work. The inherited `masking` branch carries
`masking_strategy_config: {rate: 0.8, rate_sampling: False}`, but for
`masking_strategy: "forecast"` the mask is unconditionally `np.ones(num_cells)`
(`masking.py:632`) — `rate` is read only by the `"random"` strategy. Two identical forward passes.

### Fix

Renamed the block from `forecasting:` to `masking:` in the six configs whose parent uses that key
— `ra1xax01` ×4 (incl. both new short variants) and `j6nb50v9` ×2 — so it overrides the inherited
branch instead of joining it. Verified by replaying the real merge (parent `_fix` json ⊕ config)
for all ten MTM configs and all seven JEPA configs: **every one now resolves to `batch_size = 1`.**

`oywmz4sz` and `dy0jlrmw` needed no change — their parents happen to name the branch
`forecasting`, so the block always overrode correctly.

**`gckt1vzm` ran under the bug** and is the only affected run: it is 2× LR and 2× cost against
every other member of the family, so it is not comparable and should be relaunched from the fixed
config.

**General lesson:** with a union merge, *the key name is the override*. Two configs that read
identically can differ in batch size, loss scale and cost depending only on what the parent
called its branch. Check `model_input` key names against the parent json, the same way §11 says
to check stream-merge semantics.

### …and it is also what killed the run

`gckt1vzm` then died, and the visible failure was a wall of NCCL text —
`Watchdog caught collective operation timeout: WorkNCCL(SeqNum=38234, OpType=_ALLGATHER_BASE …)
ran for 600087 milliseconds`. That is the **symptom**. The cause is 120 lines earlier, on ranks
6 and 7 only:

```
File src/weathergen/model/engines.py, line 134, in forward
  assert batch.tokens_lens.flatten(0, 2).sum(0).max() <= max_tokens
AssertionError: max number of tokens per cell for positional encoding exceeded.
```

Stack: `trainer.run:607 → validate:875 → _process_validation_chunks:333 → ema.py:118
forward_eval → model.forward:810 → _get_initial_conditions:995 → encoder.py:151 → engines.py:134`.

Ranks 6 and 7 raised it and exited; ranks 0–5 stayed blocked in the FSDP all-gather and the
watchdog took the job down 600 s later. `srun` bears this out — it reports tasks 0–5 aborted and
**not** 6 and 7, which were already gone. Both chained jobs died identically at
`Mini_epoch 0 of 8: validate.` on different nodes (`jpbo-018-[33-34]`, `jpbo-045-[31-32]`), so
this is deterministic, not node flakiness. `sacct` confirms it is not host OOM either: MaxRSS
120.20 G and 118.15 G per task against a 214 G/rank budget, `ExitCode 0:6` (SIGABRT).

**Why the assert fired.** `tokens_lens` is 4-D with streams on dim 2 (`engines.py:90` sums dim 2;
`:213` indexes `[:, :, streams_active]`). `flatten(0, 2).sum(0)` therefore collapses **batch ×
forecast step × stream** into a per-cell total, and compares it to `ae_local_max_tokens_per_cell`,
which this lineage never sets — so the default **64** applies. The duplicated `model_input` branch
put a factor 2 on the batch dimension, halving the effective per-sample budget to 32 tokens/cell.

Independent confirmation of the doubled batch from the log itself: 4096 samples ÷ 8 ranks = 512
samples/rank, and the trainer logged **256** steps per mini epoch (`000 : 00250/00256`) — batch 2.
At batch 1 it would have logged 512.

Training survived 256 steps on the 2016–2022 window and only validation (2023-10-01 → 2023-12-31,
denser obs coverage) crossed the limit. The parent `ra1xax01` validated **7 times** on that exact
window at batch 1, and our stream set adds no source tokens to it (`IMERG_ANEMOI` is diagnostic,
`source: []`), so batch 1 is known-good on this data.

**Do not "fix" this by raising `ae_local_max_tokens_per_cell`.** It sizes `pe_embed`
(`model.py:129-133`) *and* enters the sinusoid frequencies as
`-(math.log(max_tokens_local_per_cell) / dim_embed)` (`model.py:227`). Changing it changes the
positional encoding of **every** token, which the frozen pretrained encoder was never trained
with — on top of a shape mismatch on checkpoint load. The batch fix above is the correct lever.

---

## 17. Validation window widened to 6 months

Applied to **all 16 uncommitted** `config/raina_config/config_finetune_imerg_diag_mse_*.yml`
(the 15 untracked ones plus the modified `fx276yn3_8step`). The 8 already-committed configs were
left alone.

```yaml
validation_config:
  samples_per_mini_epoch: 256      # unchanged — same cost
  start_date: 2023-07-01T00:00     # was 2023-10-01
  end_date:   2023-12-31T00:00     # unchanged
  shuffle: True                    # was False (inherited)
```

**`shuffle: True` is load-bearing, not cosmetic.** `_calc_baseperms`
(`multi_stream_data_sampler.py:224-230`) returns `np.arange(max_input_steps, perms_len)` — a
*sorted contiguous* index range — and only the first `samples_per_mini_epoch` entries are consumed.
So without shuffling, validation is always the **earliest** N slots of the window:

| window | slots (6 h) | 256 used, unshuffled | effect |
|---|---|---|---|
| old, 3 months | 92 d × 4 = 368 | 2023-10-01 → ~12-04 | December never scored |
| new, 6 months, `shuffle: False` | 184 d × 4 = 736 | 2023-07-01 → ~09-03 | **narrower and a different season** |
| new, 6 months, `shuffle: True` | 736 | 256 drawn across Jul–Dec | what was actually wanted |

**It does not make the curve noisy.** `reset()` reseeds `self.rng` from the constant
`self.data_loader_rng_seed` at the top of *every* mini epoch
(`multi_stream_data_sampler.py:306`) before permuting, so the same 256 samples are drawn each
epoch and the validation curve stays epoch-to-epoch comparable. The seed mutation at `:941`
happens only inside forked worker copies (for per-sample masking randomness) and never reaches
this permutation. 736 slots ≫ 256, so no `repeat_data` tiling either.

Data coverage is not a concern: every store in these stream sets spans Jul–Dec 2023 — IMERG
1998-2024, ERA5_in 1979-2024, SEVIRI 2014-2024, GOES 2017-2024, Himawari-9 2022-2024,
METOP-B/C IASI to 2023, SurfaceCombined 1979-2025.

**Cross-run comparability breaks here.** Runs already validated on the old 3-month window —
`lf97ve7d`, `ti7k7l5s`, `vpj2sbpa`, `o8m370eh`, `w1flta02`, `bif4t205` and earlier — score a
different sample set. Their validation numbers are not comparable with anything launched from
these configs; only same-window runs can be put on one axis.

---

## 18. Open items

- **flash-attention failure** on the standalone `imerg_late_test` config (§3) — root cause unknown.
- **`oywmz4sz` vs `dy0jlrmw` initial condition.** Described as OperAn vs ERA5, but their stored
  configs are *identical* on this point: both use `ERA5_in` of type `anemoi_operan` reading
  `aifs-ea-an-oper-…-with-era51.zarr` (the **ERA5** store), and for oywmz4sz that holds at epochs
  0, 2, 9 and latest. As configured the two differ only in parent weights. Candidate OperAn store
  for the 2016–2022 window: `aifs-od-an-oper-0001-mars-n320-2016-2023-6h-v8.zarr`.
- ~~`j6nb50v9`, `ra1xax01`, `oywmz4sz` need `_fix` snapshots~~ — **done**, see §14. All four
  `_fix` snapshots are on merged code `ee3b326b8`; `imerg_diag_oywmz4sz` and `imerg_diag_dy0jlrmw`
  were rebuilt to all-streams form.
- **`score_plots: ["lead_time"]`** requirement (§2) not verified end-to-end.
- **`num_workers` for the `_fix` runs** is unmeasured — the bracket in §7 was measured under a
  load (ERA5/ERA5_in reads) that no longer exists.
- **8-step cw6a4szu needs `--chain-jobs 4`**, not 2 (§7).
- Runs `zk8jugvb`, `x5uptfho`, `c6liueoc` and their descendants trained against a **diluted
  objective**; treat their IMERG scores accordingly.
