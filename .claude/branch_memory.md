# Branch memory — `wm/develop-ssl-diffusion-v1`

## Goal
Port the IMERG diagnostic-decoder finetunes to this branch and turn the resulting inference runs
(`xfaps4wq`, `r15j90ns`, `z71y2ik8`) into a presentation.

## Evaluation traps found the hard way
- **`calc_seeps` returns `1.0 - seeps_error`** (`score.py:1393`), so the stored `seeps` is a
  **skill**: higher is better, and it must *decrease* with lead time. Plotting it as an error
  inverts every conclusion.
- **Sample indices are not initialisation times.** `xfaps4wq` and `r15j90ns` emitted their inits in
  different orders, so the circulated `sample: "0-63"` comparison averaged the two runs over
  different forecast sets; only 37 of those 64 inits are shared. The 40-lead pair
  (`z71y2ik8`/`yz3h0kyn`) *is* aligned. Always map index → `target/times[0]` before comparing runs.
- Per-sample scores are already stored in
  `results/<run>/evaluation/<run>_IMERG_ANEMOI_<region>_<metric>_chkpt00000.json`
  (`scores[0].data`, dims `[sample, forecast_step, channel, ens]`), so subsetting to matched inits
  needs no re-run of `uv run evaluation`.

## Results worth remembering
- **2 steps:** the two backbones are within ~1% on every metric. JEPA is behind at +6 h and ahead at
  +12 h (ETS +0.005, SEEPS +0.005, RMSE −0.09 mm/6h) — and the separation is almost entirely
  northern-hemisphere; the south is a tie.
- **The 2-step comparison is confounded.** `xfaps4wq`'s backbone sees 339 input channels on an N320
  analysis (210 IASI PCs, 2048×2 local encoder, no registers, no cross-stream attention);
  `r15j90ns`'s sees 126 on o96 (18 IASI radiances, 1024×4, 64 registers, XSA). It is a
  pretrained-model comparison, not an SSL-vs-supervised ablation.
- **8 steps:** `z71y2ik8` was trained at +48 h and holds ETS ≥ 0.20 out to **exactly +144 h**, with
  FBI in a stable 1.15–1.45 band the whole way, then collapses to a dry static field between +144 h
  and +180 h (collapse onset, FBI < 0.8, is +156 h). `yz3h0kyn` (Forecast backbone, same 10 inits)
  reaches only ETS 0.093 at +144 h — the natural baseline, currently left out of the figures.
- Model sizes: `hmpa42d2` 840 M (289 encoder), `nescpnb2`/`v3gs9fwn` 1164 M (613 encoder). Forecast
  engine is 537 M in all three. Only the 13.8 M IMERG decoder trains (1–2%).

## Running an evaluation on Jupiter
- **Use `../WeatherGenerator-private/hpc/launch-slurm.py --stage evaluation --run-ids <ids>
  --eval-config <yml> --account=e-ext-2025e01-128`.** A hand-written `sbatch` calling
  `uv run evaluation` dies in ~20 s with an *empty log and no traceback* — it is missing the
  module loads and env that `hpc/jupiter/weathergen_slurm.sh` provides.
- The launcher snapshots the config and runs from `/e/scratch/weatherai/slurm/slurm_weathergen_
  <runid>_dir/`, so `summary_dir` / `runplot_base_dir` must be **absolute** or the figures land in
  the snapshot. Logs: `<snapshot>/logs/<runid>/output.evaluation.<jobid>.txt`.
- `--run-ids` *filters* the config's `run_ids` block; the ids must already be in the config.
- **Pass `--no-register`.** The launcher defaults to `--register`, which sets `--push-metrics`, and
  the MLflow push block instantiates the abstract `WeatherGenReader` directly
  (`run_evaluation.py:400`) → `TypeError: Can't instantiate abstract class ... 'get_ensemble',
  'get_forecast_steps', 'get_samples'`. It runs *before* `plot_summary`, so it kills the run right
  after the score maps and before every cross-run summary plot. Seen on job 1370727.
- Compute nodes can hit a bogus `ModuleNotFoundError` from stale GPFS metadata (seen for
  `mlflow.utils.validation`); `touch` the parent directories under `.venv/.../site-packages`.

## Running the IMERG eval on Santis (2026-08-20)
- **`config/evaluate/eval_config_imerg.yml` was written for Jupiter.** Every `results_base_dir`
  and the `climatology_path` pointed at `/e/scratch/weatherai/...`, which does not exist on
  Santis. Santis equivalents: results `/iopsstor/scratch/cscs/thunter/shared_work/results/<run>`
  (= `path_shared_working_dir` from `WeatherGenerator-private/hpc/santis/config/paths.yml`).
- **The `assets/climatology/*.zarr` stores on Santis are empty shells** — 72 K, only `.zattrs`
  and `.zgroup`, no chunks and no `.zmetadata`. Auto-resolution from `data_path_aux` therefore
  yields nothing usable for ANY stream, not just IMERG. The only real IMERG SEEPS asset is
  `assets/climatology_seeps/nasa-imerg-grib-n320-1998-2024-6h-v1_SEEPS.nc`, monthly
  `p_wet` / `threshold_heavy` on `(month, ipoints)` — the wrong layout for `align_clim_data`,
  which wants a `data` variable of `(time, statistic, grid_points, channels)` matched by
  day-of-year + hour. Converted by `playground/scripts/build_imerg_seeps_climatology.py` into
  `/iopsstor/scratch/cscs/walmikae/assets/climatology/nasa-imerg-grib-n320-1998-2024-6h-v1_climatology.zarr`
  (`prob_dry = 1 - p_wet`, `light_heavy_threshold = threshold_heavy` in mm/6h, broadcast onto a
  6-hourly **2023** non-leap reference year, 1460 steps). Logical size 6 GB, 18 MB on disk —
  the per-timestep chunks are byte-identical within a month and compress away.
- **A diagnostic-only stream writes no fstep 0 group at all**, and `ZarrIO.forecast_offset`
  assumed one existed to infer the offset from (`AssertionError: Zarr group: 3/IMERG_ANEMOI/0
  does not exist`). Fixed in `packages/common/src/weathergen/common/io.py`: an absent fstep 0
  now means offset 1. Same patch sorts the fstep group keys **numerically** — they are strings,
  so the old `sorted(...)[1:]` slice dropped fstep **1** (lexicographic "10" < "2") on any store
  with 10+ steps, and `zio.forecast_steps[1]` returned step 10.
- **These 7 stores hold 40 forecast steps, not 2** (8 samples/rank x 4 ranks = 32 samples,
  fsteps 1-40), despite the config header describing fsteps 1-2. `forecast_step: "all"` is
  therefore a 20x bigger job than the comments imply. ~6-8 min per run on a login node at
  `max_workers: 16`, so a full 8-run pass is around an hour.
- **The canonical JSC climatology is now at
  `/iopsstor/scratch/cscs/walmikae/wg_from_jsc/climatology/nasa-imerg-grib-n320-1998-2024-6h-v1_climatology.zarr`**
  and is what the config uses. Checked against a local rebuild from the Santis `_SEEPS.nc`:
  `light_heavy_threshold` matches **exactly**, `prob_dry` at 99.97% of points inside 60S-60N
  (max diff 0.012; disagreements are polar, outside IMERG coverage). Both stores are
  month-constant. **The JSC axis is a LEAP year (2000, 1464 steps)** while `align_clim_data`
  matches on day-of-year, so for non-leap targets (2023) the lookup shifts a day after 28 Feb
  and the 1st of each month picks up the previous month's climatology. Small, but it means
  the two stores never give bit-identical SEEPS.
- **`f6t0xcz3` is NOT shaped like the other seven.** 6 samples/rank = **24** inits (so
  `sample: "0-31"` is out of range), its inits are 12/18/00Z only — **no 06Z** — and it writes
  an fstep 0 group (source only) where the others write none. Its 24 inits are a strict subset
  of the others' 32, but **the sample indices do not line up** (index 3 = Jun 2 06Z in
  luzbikci, Jun 2 12Z here). It has its own `streams:` block with `sample: "0-23"`.
- **A per-run `streams:` REPLACES `default_streams` wholesale** — `run_evaluation.py:345` is
  `if "streams" not in run: run["streams"] = default_streams`, not a deep merge — so an
  override block must repeat every key, `climatology_path` included.

## 8-run backbone comparison, June 2023, 40 leads (completed 2026-08-20)
Global SEEPS **skill** (higher better), `eval_config_imerg.yml`, JSC climatology:

| run | backbone | +6h | +48h | +120h | +240h |
|---|---|---|---|---|---|
| mjh75lxo | MTM f7ug724z W/FE,CW | 0.658 | 0.533 | 0.357 | 0.133 |
| w7gd4p5q | MTM ra1xax01 | 0.657 | 0.531 | 0.353 | 0.133 |
| luzbikci | Forecast dy0jlrmw | 0.656 | 0.520 | 0.319 | 0.106 |
| hc58drc1 | S/JEPA lnm4ud42 W/O CW | 0.653 | 0.525 | 0.318 | **0.030** |
| uccblskr | S/JEPA whzrt2cc W/CW | 0.653 | 0.524 | 0.311 | **0.021** |
| spbhsahx | MTM dwzio194 W/O FE,CW | 0.647 | 0.531 | 0.353 | 0.137 |
| iq5c34dx | MTM xq65fhca W/O FE,CW | 0.646 | 0.529 | 0.351 | 0.136 |
| f6t0xcz3 | T/JEPA cw6a4szu CW (24 inits) | 0.663 | 0.519 | 0.346 | 0.132 |

- **Short lead does not predict long lead.** All eight are within 0.017 at +6 h, but by +240 h the
  two S/JEPA runs have collapsed to ~0.02-0.03 while every MTM run holds ~0.13 — the same dry
  static-field collapse seen on `z71y2ik8`. Rank order at +6 h is nearly meaningless.
- The four MTM runs cluster tightly from +48 h out (0.529-0.533, 0.351-0.357, 0.133-0.137)
  regardless of FE or CW, so at these leads the MTM/JEPA split dominates those two switches.
- `luzbikci` (supervised Forecast baseline) is mid-pack early and clearly behind every MTM run
  from +48 h on.
- **`f6t0xcz3`'s column is NOT like-for-like** — 24 different inits (see above). Its apparent
  +6 h lead and +48 h deficit could both be init-set artefacts; subset to matched inits first.

## Evaluation-package gotchas (beyond the ones above)
- **`psd` cannot be used on this stream.** `detect_grid_type` knows only `octahedral` and
  `regular`; N320 is *reduced* Gaussian, so it returns None, `sht_psd` raises, `calc_psd` swallows
  it — and the run then dies far away in `metric_list_to_json` with `KeyError: 'global'`. Not
  fixable from config (`grid_type` is not a `calc_psd` parameter). Verified on job 1370458.
- **Give every eval run a private `metrics_dir`.** `load_scores` skips metrics already cached
  there *per region*, but `metric_list_to_json` then loops every recomputed metric over *every*
  region and dies with `KeyError: '<region>'`. The shared `results/<run>/evaluation/` dirs are
  partially populated from older runs, so the region named moves around between attempts
  (`'global'` on job 1370458, `'nhem'` on 1370647). An empty metrics_dir makes the shapes line up.
- `acc`/`fact`/`tact` need a climatological mean and `rps`/`rpss` need q20–q80; the IMERG
  climatology has only `prob_dry` + `light_heavy_threshold`.
- When `score_plots:` (list form) is present, ALL legacy booleans (`summary_plots`, `ratio_plots`,
  `heat_maps`, `score_cards`, `bar_plots`, `plot_score_maps`, `plot_score_init_timeseries`) are
  ignored. `lead_time` also gates the psd and qq plots.

## Stream date coverage (read from the zarr `dates` arrays, not the filenames)
- Filenames round the years and mislead. Actual: GOES-16 IR/VIS **2017-03-09**→2024-12-31,
  Himawari-8 2015-08-01→2022-11-30, Himawari-9 2022-12-01→2024-12-31, SEVIRI 2014-01-01→2024-12-30,
  IASI PC 2011-02-22→**2023-12-30**, AVHRR 2001-01-01→**2024-01-01**, Surface 1978→2025-05-30,
  ERA5 n320 1979→2024-12-31, IMERG 1998-01-01→**2024-07-31**.
- **All-nine-streams window is 2017-03-09 → 2023-12-30.** The family's training window
  (2016-01-01→2022-12-31) therefore runs its first ~14 months (~17% of windows) with no GOES.
  Extending training earlier is not available: 2014 costs GOES+Himawari, 2011 costs SEVIRI too.
- A validation window past 2024-01-01 silently drops IASI PC and AVHRR. Any June-2023→June-2024
  val set mixes 9-stream and 7-stream samples into one number. Use 2023-01-01→2023-12-30.

## Train/val divergence on the IMERG finetunes
- The train-vs-val *level* gap is a period artefact, not overfitting: val is Oct–Dec 2023 vs train
  2016–2022, and the gap is already +6.7% at the first eval on a freshly initialised decoder.
  Only the *growth* of the gap (6.7% → 35%) is diagnostic.
- `validation_config.shuffle: False` + 256 samples ⇒ every eval scores the **same** 256 contiguous
  windows (2023-10-01 to ~2023-12-04). No between-checkpoint sampling noise, so a monotonic val
  rise is real degradation — but it is one season of one year.
- Val minima: 2 steps → chkpt00004 (fx276yn3 backbone, 20 480 samples), chkpt00006 (cw6a4szu,
  28 672). 8 steps → 2 mini-epochs for a fresh decoder (`lf97ve7d`), and **zero** for 2-step-seeded
  ones (`w1flta02`, `o8m370eh` rise from their first eval). Training runs ~3x past the optimum.
- Magnitude is small: +2.7% MSE from the minimum = +1.3% RMSE. The cost is checkpoint selection,
  not a broken run. ~10 200 unique 6h train windows vs 69 344 samples = 6.8 passes.
- **ERA5 co-training is a null result.** `zk8jugvb` carries 81 real `LossPhysical.ERA5.mse.*` keys,
  `c6liueoc` has none, and their IMERG val curves agree to <0.001 at every checkpoint.
- **2-step-then-8-step BEATS direct 8-step**, contrary to the initial read. Matched pair (both
  IMERG-only, both from fx276yn3, both 8 steps, same val set): `vpj2sbpa` (direct) scores
  **0.5714 at 16 000 samples**, vs `w1flta02` (via h1mrni5n) 0.5561 @14 240 and 0.5600 @18 320 —
  interpolating gives direct ~+0.013 MSE (+2.3%) worse. `vpj2sbpa`'s number came from the manual
  validation run `yl1mp4fr` (see below); it is a single point against a 4-point curve, and its
  minimum is unknown.
- **The 2-step stage buys SHORT-lead skill.** Per-step, direct-8-step is +7.4% worse at step 1
  (0.4024 vs 0.3748) but only +1.3% worse at step 8 (0.6579 vs 0.6492). Both things are true: the
  2-step stage helps at leads 1-2, AND it hands the 8-step stage an already-overfit decoder (which
  is why `w1flta02`'s val rises from its very first eval).
- **The 2x2 completed by `ti7k7l5s` (xqlk2nkf -> 8 steps) settles it.** Best val per run:
  cw6a4szu via-2-step `ti7k7l5s` **0.5436** @4080, cw6a4szu direct `lf97ve7d` **0.5442** @8160,
  fx276yn3 via-2-step `w1flta02` 0.5470 @4080, fx276yn3+ERA5 `o8m370eh` 0.5531, baseline
  `mq8y8qza` 0.5641. `vpj2sbpa` (fx276yn3 direct) never validated in training so its MINIMUM was
  never measured — 0.5714 @16k / 0.5827 @22k are late points, not a fair cell.
- **On the same backbone the path is a lead-dependent TRADE, not a win.** ti7k7l5s vs lf97ve7d:
  **+5.2% better at +6h**, +0.5% at +12h, then **0.4-0.6% WORSE at every lead from +18h to +48h**.
  Crossover at +18h; the averages cancel to a 0.1% tie. So "2-step-then-8-step is no better" is
  true on `mse.avg` and false per-lead — the average erases the trade. Report per-lead, not avg.
- The 2-step seed buys **speed, not ceiling**: ti7k7l5s hits 0.5436 after ONE mini-epoch;
  lf97ve7d starts at 0.5523 and needs two to reach 0.5442. Same destination, ~half the 8-step cost.
- **Backbone beats path.** Both cw6a4szu runs beat both fx276yn3 runs; all four beat the baseline
  by 2-3.6%. ti7k7l5s also has the family's best +6h (0.3645 -> 0.3637), though by mini-epoch 3 it
  reverts to 0.3645 — short lead saturates too, it just degrades far more slowly than long lead.

## MTM (ra1xax01) vs JEPA (cw6a4szu): cost, and an inert ERA5 stream
- **The backbone comparison is NOT compute-matched.** MTM encoder 487.6 M vs JEPA 616.1 M (+26%);
  forecast engine identical at 537 M. Measured throughput on the same 2 nodes / 8 ranks:
  `e5k44gk1` (MTM, 8 steps) ~3700 samples/h vs `ti7k7l5s` (JEPA, 8 steps) ~1080 — about 3.4x.
  Always state the cw6a4szu val advantage alongside its ~3x per-sample training cost.
- Architecture drivers, not channel count: JEPA has `use_xsa` ON (extra attention across all 11
  streams), 64 register tokens vs 0, ae_global 5 blocks vs 4, and step conditioning ON. MTM is
  ae_local 2x2048 / ae_global 4x2048, no XSA, no registers, FSDP on.
- **`ti7k7l5s` carries an inert `ERA5` stream (id 1): 81 source channels, `reconstruct: False`,
  `forcing: None` — no decoder, no loss term** (its only LossPhysical key is IMERG_ANEMOI). It is
  read and discarded every sample. This is the streams-dict-union trap from `era5.yml`'s own
  header (same as zk8jugvb / x5uptfho). Source channels: MTM 148 vs JEPA 215 — the whole 67-channel
  gap IS this stream. Dropping it cuts read volume ~38% for free on the next cw6a4szu 8-step run.

## Validation OOM kills the val rows AND the numbered checkpoints
- **Symptom:** a run trains normally but its metrics JSONL has zero `stage == "val"` rows, and
  `models/<run>/` holds only `<run>_latest.chkpt` with no numbered checkpoints.
- **Cause:** validation opens a SECOND DataLoader (`num_workers_validation`) on top of the still
  live training workers — 4 ranks x 10 worker processes per node, each opening all 9 zarr streams
  at 8 forecast steps. The host OOM killer takes a val worker (`killed by signal: Killed` at
  `trainer.py:717`, `batch.to_device`), that rank dies, and the surviving ranks abort on a 600 s
  NCCL ALLREDUCE watchdog timeout. `sacct` MaxRSS (111 G of a 214 G/rank budget) does NOT show it —
  it polls too coarsely to catch the val-loader startup spike.
- **Why it erases itself:** the loop is train -> validate -> save_model, and `_latest.chkpt` is
  rewritten *inside* train() every `train_logging.checkpoint` batches. The restart therefore
  resumes with istep already past the mini epoch it died validating, `mini_epoch_base` resolves one
  higher, and it skips to the next mini epoch's training. `save_model` is never reached, so no
  numbered checkpoint is ever written. Seen on vpj2sbpa jobs 1364281/1364283.
- **Manual validation without retraining:** `validate_before_training()` runs at `trainer.py:592`,
  *before* the loop at `:596`. Launch with
  `--from-run-id <run>` (no `--run-id`, so a new id is minted) and
  `--options validation_config.validate_before_training=True training_config.num_mini_epochs=1
  data_loading.num_workers=2`.
  Use `num_mini_epochs=1`, NOT 0: `range(mini_epoch_base, 1)` is already empty once base >= 1, and
  0 sets `lr_steps = 0`, which the LR scheduler divides by (`lr_scheduler.py:104,130`).
  Keep the same `--nodes` as the original run — the val loop breaks on
  `bidx * batch_size > samples_per_mini_epoch`, so world size changes how many of the 256 samples
  are consumed and the number stops being comparable. Do not pass `--config`/`streams_directory`;
  inheriting the checkpoint's stream set is what keeps it comparable.
- **Verified working** as run `yl1mp4fr` (job 1372777, from `vpj2sbpa` at `--mini-epoch -1`):
  COMPLETED 0:0 in 48 min, one val row at `num_samples 16000`. **MaxRSS 41.75 G/rank vs 111.62 G**
  for the crashing job — the `num_workers` drop cut resident memory 63% and validation then ran
  clean, which confirms host RAM (not GPU) was the cause.
- **Validation is silent for 20-35 min at 8 steps** — `validate()` logs nothing until the whole
  256-sample loop finishes (`_log_terminal` at `trainer.py:928`), and the config dump at `:588` is
  the last line before it. A quiet log there is normal, not a hang. Measured per-validation
  wall-clock: w1flta02 20-22 min, lf97ve7d 26-30 min, o8m370eh 33-35 min.

## launch-slurm.py pipeline mode
- `--pipeline <file.yml>` chains train/inference/evaluation stages; `STAGE.<name>` in `from_run_id`
  resolves to a prior stage's run in the SAME file. **Hard cap: 8 stages per file**
  (`STAGES_MAX_ALLOWED` in launch-slurm.py) — a train+infer pipeline across 7 parents needs one
  file per parent, not one big file.
- `run_id:` on a stage is honored if set (contrary to a stale comment in the script saying "no
  choice of run id for now") — pin it in the yaml to know run ids before submitting, instead of
  scraping the post-run summary log.
- `config/raina_config/pipelines/pipeline_imerg_<parent>.yml` (7 files, one per
  dwzio194/dy0jlrmw/f7ug724z/hmco4ptc/lnm4ud42/whzrt2cc/xq65fhca): each chains
  train_2step_8ep -> infer_2step_8ep -> train_8step_2ep -> infer_8step_2ep, nodes=2 train /
  nodes=1+`--time=04:00:00` infer, `parallelize: false` (strict sequential). Pinned run ids and
  status tracked in `playground/docs/imerg_pipeline_run_ids.md`. Not yet launched as of 2026-08-19.
- Validated all 7 against the actual `_parse_pipeline_config`/`_validate_parsed_pipeline` (local
  python3 is 3.6 and can't even parse the script's `type X = ...` syntax — use
  `uv run --with omegaconf==2.3.0 --with mlflow-skinny==3.10.0 --with certifi==2026.2.25 --with
  dacite <script>` to import it standalone for a dry-run check).

## Artefacts
- `scripts/imerg_presentation_figures.py` — 15 presentation figures, `--figure <id>` per figure,
  `--verify-matched-inits` re-derives the pinned 37-init lists from the stores.
- `docs/imerg_presentation_figures.md` — captions, numbers and slide order.
- `config/evaluate/eval_config_imerg_2step_matched.yml` and `..._8step_rollout.yml` — the
  evaluation-package figure sets (16 metrics × 4 regions), documented in
  `docs/imerg_evaluation_configs.md`.
- `docs/imerg_finetune_session_log.md`, `docs/imerg_finetune_parents_comparison.md` — earlier work.

## 3-month / 40-step inference: DIRECT-from-backbone path (2026-08-20, SUPERSEDED)
- **`chkpt00001` is the common checkpoint for SEVEN runs**, not eight: `dnq4fmyz` `rpqk51fp`
  `uqjiujv6` `r4874frh` `yex8fsv8` `re7ur23g` `fq85jsg6` (hmco4ptc — was missing from the
  circulated list). All 2 mini epochs, 512 batches each (510 for the two S/JEPA runs, whose
  `num_workers: 6` rounds the per-worker slice down), warm 32 / cool 64, `lr_max` 2.8276e-4, and
  `src/weathergen` in all seven `<parent>_fix` snapshots hashes to **`193332f51fc5`** —
  byte-identical code. `ra1xax01_fix` hashes the same; `cw6a4szu` does not (`8fc95db89e3f`, no
  `all_modules.get` fix) and needs a `cw6a4szu_fix` seed before it can join.
- **`ti7k7l5s` and `e5k44gk1` are not fixable by checkpoint choice.** Both seeded from a *2-step*
  run (`xqlk2nkf` / `sslfnnb4`) rather than the backbone, both 4 mini epochs on 256/512 and 64/128
  schedules, and **their numbered checkpoint weights were deleted** — `ti7k7l5s` has only
  `_latest.chkpt`, `e5k44gk1` only `chkpt00003`. Replacements: new
  `config_finetune_imerg_diag_mse_ra1xax01_8step_2ep.yml` (from `ra1xax01_fix`) and, once seeded,
  the cw6a4szu equivalent. `playground/raina/seed_raina_run.py` is NOT in this tree.
- **`dnq4fmyz`'s `num_samples` axis is inflated 4x and means nothing.** `dy0jlrmw_fix` carries
  `world_size_original: 32`; `Trainer.get_batch_size_total` uses that instead of the live world
  size, so the metrics say 16384/32768 where the others say 4096/8192. Same 512 batches, same
  `lr_max`, same 52 metric rows as `rpqk51fp` — it did NOT see 4x the data.
- `yex8fsv8`/`re7ur23g` wrote a trailing `chkpt00002` that is the same state as `chkpt00001`
  (post-loop `save_model`, consecutive lines in the log). Use `chkpt00001`.
- **`--mini-epoch 1` is mandatory for inference.** Default `-1` loads `_latest.chkpt`, which
  `train()` rewrites every 250 batches — i.e. batch 500 of the 512-batch second mini epoch, 1012
  of 1024 steps, mid-cooldown. The five existing 32-sample 8-step inference runs (`vxvg0fn9`
  `mn9er14f` `if8mdd4v` `m6fao72k` `ukxvf1t2`) all used `_latest`. `Trainer.inference` calls
  `validate(0, ...)`, so the store is named `validation_chkpt00000_*` whatever you load.
- **`test_config.end_date` must clear the whole forecast horizon.** `check_samples` subtracts
  `(time_step*(num_steps+offset) + window_len)//step` = 42 windows at 40 steps/6 h. With
  `end_date=2023-08-31T00` only 322 inits exist and the last is **2023-08-20T00** — 11 days
  silently dropped, no warning. Use `2023-09-12T00:00` + `samples_per_mini_epoch=368` for 368
  inits, 2023-06-01T06 → 2023-09-01T00.
- **368 is chosen for divisibility.** Per-rank len is
  `((spme // world) // (batch*num_workers)) * batch*num_workers`, truncated silently; 368 is exact
  for world 8 (nodes 2) and world 4 (nodes 1). The sample-index → init-time map depends on BOTH
  world size and `num_workers` (ranks take contiguous blocks, workers interleave within a rank —
  verified on `luzbikci`: sample 0 = Jun 1 06Z, 1 = Jun 2 06Z, 2 = Jun 1 12Z, 7 = Jun 3 00Z), so
  every run in a comparison must use the same `--nodes` and `num_workers`.
- Cost: ~203 MB of zarr per sample per run ⇒ **~75 GB/run**, ~525 GB for seven; ~1.5–2 h on 2
  nodes. Evaluation scales ~11.5x from the 6–8 min/run at 32 samples.
- Artefacts: `config/raina_config/pipelines/pipeline_imerg_3month_inference.yml` (7 stages,
  `parallelize: true`, at the 8-stage cap), `..._ra1xax01_8step_2ep.yml` (train+infer),
  `playground/docs/imerg_3month_inference_plan.md`. Nothing submitted.
- Jobs **814473** / **814519** (`r1yzoyn6_cleanup`, `dxmo5bxx_cleanup`) are stuck `PENDING`; their
  inference stages were cancelled at 0:00 and never ran.

## The comparison is the VIA-2-STEP path, not the direct one (2026-08-20)
- **Corrected scope:** the 8-step / 2-mini-epoch stage must start from each lineage's **2-step
  IMERG run at `chkpt00007`**, not from `<backbone>_fix`. The direct-path artefacts above are
  superseded (their files now say so in their own headers); keep them only for a
  direct-vs-via-2-step contrast, never pooled into one ranking.
- **`chkpt00007` is real and shared for EIGHT of nine 2-step parents** — `zbqe7ckw` `ue0ifd0t`
  `uyhl1c32` `q7vren93` `oeimvihc` `bom5igqp` `tn6389hh` `sslfnnb4`. All 8 mini epochs x 4096 at
  2 steps, warm 128 / cool 256, and all eight snapshots hash to **`193332f51fc5`**.
- **`tn6389hh` and `sslfnnb4` restarted at istep 4084**, so their `chkpt00007` is at istep **4596**
  not 4096: `mini_epoch_base` resolved to 7 (`len(dataset)//ws_orig * ws_orig` = 512) and they
  re-ran mini epoch 7 from its start. The extra ~512 steps ran at **lr 6.9e-10** — past the end of
  the schedule — so the weights are equivalent. Their `_latest` is the same istep 4596.
- **`xqlk2nkf` (cw6a4szu / T-JEPA) is out on three counts.** (1) `xqlk2nkf_chkpt00007.chkpt` was
  **deleted** — only `_latest` survives, while all 17 `model_xqlk2nkf_chkpt*.json` remain, so the
  checkpoint looks present in `ls` when it is not. (2) It ran **16** mini epochs on 256/512, so its
  mini epoch 7 was never the others' cooled-down end state and `_latest` (istep 8430) is 2x their
  training. (3) Its snapshot hashes `8b3a217820b5` vs `193332f51fc5` — **18 files differ**,
  `trainer.py` 188 lines, `model.py` 174, `multi_stream_data_sampler.py` 27, `lr_scheduler.py` 2.
  `xqlk2nkf`'s tree is `cw6a4szu`'s with only `validation_io.py` changed, so re-running the 2-step
  stage from `cw6a4szu` does NOT fix the code gap — a `cw6a4szu_fix` seed is required first, and
  `playground/raina/seed_raina_run.py` is **not in this tree**.
- **The nine new configs are the 2-step configs with exactly six edits** (`num_mini_epochs`->2,
  warm->32, cool->64, `forecast.num_steps` 2->8 in train AND val, `num_workers`->6, desc/exp).
  Deriving from the 2-step config — not from the direct-path 8-step one — is what guarantees
  `streams_directory`, `freeze_modules`, the loss key and the `model_input` branch are the
  lineage's own. Verified by diff: only those six lines change.
- **`num_workers: 6` on all nine is deliberate** — 4096/world 8 gives 512 batches at 8 workers but
  **510** at 6, so pinning 6 makes every run exactly 4080 samples/mini-epoch.
- **`oywmz4sz` also carries `world_size_original: 32`** (like `dy0jlrmw`), so its `num_samples`
  axis is inflated 4x too. Two lineages, not one.
- Artefacts: `config/raina_config/config_finetune_imerg_diag_mse_<parent>_via2step_8step_2ep.yml`
  (9 files), `pipelines/pipeline_imerg_via2step_8step_group{1,2}.yml` (8 stages each,
  `parallelize: true`, both validated against the launcher's parser) and
  `..._cw6a4szu.yml` (blocked, kept as the record). Plan: `playground/docs/imerg_via2step_8step_plan.md`.
  Nothing submitted.

## launch-slurm pipeline constraints learned the hard way (2026-08-20)
- **Stage `name:` is capped at 15 characters** and must match `[A-Za-z0-9]+` (no underscores or
  dashes). `train2stepcw6a4szu` (18) was rejected by `_validate_stage_name`; use
  `train2scw6a4szu`. Always dry-run a pipeline through the launcher's own
  `_parse_pipeline_config` + `_validate_parsed_pipeline` before submitting — the local python3 is
  3.6 and cannot parse the script, so import it under
  `uv run --with omegaconf --with mlflow-skinny --with dacite`.
- **`uv run` cold-starts slowly enough here to blow a 2-minute timeout.** For quick config parse
  checks use `.venv/bin/python` (omegaconf is already installed there) and `.venv/bin/ruff`
  directly. Reserve `uv run --with ...` for the launch-slurm dry-run, which genuinely needs
  packages the venv lacks (dacite, mlflow-skinny).
- **Do NOT pin `run_id:` on pipeline stages** — the user reports it does not work in practice.
  Omit it; `_validate_stage` does `run_id = p.run_id if p.run_id else get_run_id()` and downstream
  stages resolve through `from_run_id: STAGE.<name>`, so the chain works without any id being
  named. The minted ids appear ONLY in the launcher's stdout, in the
  "Resolved parameters for stage '<name>'" block (`run_id:` / `from_run_id:`) — pipe it through
  `tee`. Afterwards they are recoverable from slurm job names and `models/` mtimes.
  This supersedes the older note above that said pinning is honoured.
- `mini_epoch: 0` on a stage is silently turned into `-1` (`p.mini_epoch if p.mini_epoch else -1`).

## cw6a4szu / T-JEPA cell: parent RESTORED, not regenerated (2026-08-20)
- **`xqlk2nkf_chkpt00007.chkpt` was copied back from Jupiter** (4,711,706,738 B, byte-exact, with
  .optim and json). It is now on Santis at **istep 4096**. No 2-step re-training; the earlier
  regeneration config `config_finetune_imerg_diag_mse_cw6a4szu_2step_8ep.yml` was deleted.
  `xqlk2nkf_latest` is istep 8430 (= chkpt00016), 2x everyone else's training -- never fall back
  to it.
- **The residual mismatch is the LR schedule, not the sample count.** At the shared checkpoint
  the eight 8-mini-epoch parents (128/256) are fully cooled: bom5igqp 8.3e-9, q7vren93 6.9e-9,
  sslfnnb4 6.9e-10. `xqlk2nkf` ran 16 mini epochs on 256/512, so at istep 4096 it is **halfway
  down its cosine at lr 1.34e-4** -- near peak. Same 4096 optimizer steps, but a hot decoder vs
  annealed ones. No xqlk2nkf checkpoint matches on both amount and anneal state.
- Still not code-matched: `xqlk2nkf`'s snapshot is `8b3a217820b5` vs `193332f51fc5` for the other
  eight (18 files, trainer.py 188 lines, model.py 174).
- **Final layout: one file per parent in `config/raina_config/pipelines_final/`** --
  `pipeline_imerg_via2step_<parent>.yml`, nine files, each a 2-stage strict chain (train from the
  2-step parent at `mini_epoch: 7` -> inference at `mini_epoch: 1`), no `parallelize`, no pinned
  run ids, stage names `train8s2ep` / `infer3month`. The grouped
  `pipelines/pipeline_imerg_via2step_8step_group{1,2,3}.yml` files were deleted.
  `config/raina_config/pipelines/` now holds only the older / superseded material.

## GPU OOM on the cw6a4szu via-2-step run (wx9bzx7q, 2026-08-20)
- **Cause: `config/streams/imerg_diag_cw6a4szu/imerg_anemoi.yml` had `max_num_targets: -1`.** An
  uncapped diagnostic decoder decodes all **542,080** N320 points per forecast step with
  gradients -- 5x the 108,416 every other lineage uses -- so at 8 rollout steps it dies in
  `flash_attn` `varlen_bwd` inside the FIRST `backward()`, at 91-94 of 95 GiB. Fixed to 108416,
  which also makes cw6a4szu match the other eight (fairness fix, not just memory).
- **The file's own header already said 108416 and even named `x5k0r6z0` and `yj9x1jhj` as OOMing
  this exact way at "~92.4 of 95 GiB"** -- the comment was written, the value never changed.
  `wx9bzx7q` was the third casualty. `config/streams/imerg_diag_fx276yn3/imerg_anemoi.yml` still
  carries the same comment-vs-value mismatch and was left alone.
- `ti7k7l5s` survived `-1` only because it ran on **Jupiter**, not Santis. Architecture, streams
  and `num_workers` are otherwise identical (checked field by field); wx9bzx7q even reads *less*
  (its inert ERA5 stream resolves 0 source channels vs ti7k7l5s's 81).
- **Inference is unaffected by the cap.** `Trainer.inference` forces `max_num_targets = -1`
  (`trainer.py:397,403`), so the 3-month run still scores full N320 maps. The cap is training-only.
- **Adding nodes does NOT fix this.** `batch_size_per_gpu` is already 1, so per-rank activation
  memory is independent of world size.
- **A training OOM does not stop the pipeline.** `run_continue` catches the exception and prints
  a traceback, so the process exits 0, sacct shows the step **COMPLETED 0:0**, and the `afterok`
  dependency fires. `wx9bzx7q`'s inference `jighv9wm` duly launched and died in 2:13 with
  `FileNotFoundError: .../wx9bzx7q_chkpt00001.chkpt`. **Check for the numbered checkpoint, not the
  job state** -- `models/wx9bzx7q/` holds only `model_wx9bzx7q.json`, no weights.
- `PYTORCH_CUDA_ALLOC_CONF` is not a clean lever here: the launcher builds its own
  `--export=WEATHERGEN_*` for train stages, and a second `--export` in `slurm_args` collides with
  it. If 108416 still does not fit, halve the cap for that run instead.
- Status at the time: 8 of 9 via-2-step trainings completed clean (3:20-3:27 on 2 nodes), their
  inferences running; only cw6a4szu needed a relaunch.


## 3-month IMERG inference: store verification (2026-08-21)
- All nine `infer3month` runs are USABLE. Eight (`m745z8wi`, `tvmzluy6`, `fvenw2nq`, `xt069k3n`,
  `b96m0co5`, `csuff3rz`, `sctxzwwe`, `lf7aj7df`) hold 8 rank zips x 46 = 368 samples;
  `ce4rujvd` holds 364. Every sample of every run: fsteps 1..40 present, arrays
  (542080, 1, 1), all finite, reads back clean.
- **`ce4rujvd` is short, not corrupt.** SLURM 820419 died on an NCCL collective timeout
  (exit 143); ranks 5/6/7 wrote 45/45/44. The zips still close cleanly and **all 364 samples
  including the tail ones on the short ranks read back finite** — deep-checked, not just
  listed. So there is nothing to repair and no reason to re-run for 4 samples (~1%).
- **Global sample index = rank-file concatenation order** (`wegen_reader.py:538-552`), so a
  short rank SHIFTS every later index. ce4rujvd's indices diverge from the other eight from
  **275** onward. Its 364 inits are a strict SUBSET of the others' 368, missing
  2023-08-09 06Z, 08-20 18Z, 08-26 12Z, 09-01 06Z.
- **The fix is index realignment, not re-inference.** Evaluate all nine on the same 364 inits:
  the eight complete runs use indices `0-274, 276-320, 322-365` (an explicit list; the parser
  takes only "all", one "d-d" range, or a list — `io_reader.py:363-371`), ce4rujvd uses
  `"0-363"` in a per-run `streams:` block. Verified end-to-end in the pipeline, not just on
  paper: m745z8wi sample 365 and ce4rujvd sample 363 emit byte-identical valid-time sequences
  (2023-09-01 00Z -> 09-10 18Z, fsteps 001-040).
- **A per-run `streams:` block REPLACES `default_streams` wholesale** (`run_evaluation.py:345`
  is `if "streams" not in run`, not a deep merge), so ce4rujvd's block repeats every key,
  `climatology_path` included. Change one, change both or they silently diverge.
- fstep-1 window starts are contiguous 6-hourly from **2023-06-01 12Z to 2023-09-01 06Z** for
  the eight complete runs, i.e. start_date + 12 h. Each forecast is 40 x 6 h = +240 h.
- Case-study indices — **two index spaces**: Doksuri -24 h (Jul 19 00Z) = **196** and genesis
  (Jul 20 00Z) = **204** in both; Daniel Aug 31 00Z = **357** (8-run) / **355** (ce4rujvd) and
  Sep 1 00Z = **365** / **363**. Inference was initialised only to Sep 1, so Daniel is only
  reachable at long lead — an init at/near genesis (Sep 3-4) does not exist.

## Evaluate-package fixes made for this comparison
- `eval_config_imerg.yml`'s `tp` levels/colors were **8 and 8**, which raises in
  `BoundaryNorm(levels, cmap.N, extend="both")` (`plotter.py:737`): extend="both" needs
  `len(levels)+1` colors. Now 9 levels / 10 colors, extended to 0.05 and 0.1 m (50/100 mm per 6 h).
- `animation_format: "mp4"` works with no system ffmpeg — imageio picks up the bundled
  `imageio-ffmpeg` binary. It silently pads frames to a multiple of 16 px (harmless warning).
- Keep `regions` minimal — each extra region multiplies scoring cost over 364 inits x 40 fsteps.
  If case-study zooms are ever wanted, `global_plotting_options.regions` overrides the PLOTTING
  regions only, at no scoring cost (`plot_orchestration.py:882`); scoring regions stay in
  `default_streams.<stream>.regions`. (Adding `medicane`/`wpac` to `RegionLibrary` was tried and
  reverted — not wanted.)

## Cost of the 3-month IMERG evaluation (measured 2026-08-21, santis login node)
Two timed runs of `uv run evaluation`, one run_id, 3 regions, 4 metrics, plotting OFF,
`max_workers: 16`: 40 samples -> 7:28 wall / 73 GB peak RSS; 80 samples -> 14:20 / 144 GB.
Both scale almost perfectly linearly (memory ratio 1.97 for 2x the samples):
- **memory ~= 2 GB + 1.78 GB per sample**, **time ~= 36 s + 10.3 s per sample** (at 16 workers).
- **364 samples therefore needs ~650 GB and ~63 min per run** (scoring only; add ~10 min for
  the 4 case-study samples' maps/animations). Nine runs sequential ~= 11 h.
- **This does NOT fit on a login node.** santis-ln00x has 856 GB total but only ~610 GB
  available (shared), so a full-period run OOMs at ~342 samples — i.e. it dies about an hour
  in, during the FIRST run. Must go to a compute node (~850 GB/node), where it fits with
  ~200 GB headroom. One run per node: two concurrent full-period runs would need 1.3 TB.
- `get_data` holds every sample x fstep in memory at once; raw data for 364x40 is only ~63 GB,
  so the other ~10x is intermediates/climatology broadcast. Peak is reached during scoring.
- The 9 runs are processed **sequentially** in one process (`run_evaluation.py:376` is a plain
  list comprehension over `_process_stream`), so parallelism across runs only comes from
  submitting separate jobs (`launch-slurm.py --run-ids <one id>` filters the config).
- **DO NOT raise `max_workers` above ~16 for the full-period run.** Measured on 40 samples,
  same run/regions/metrics: 16 workers -> 7.5 min / 73 GB / 730% CPU; 64 workers -> 6.4 min /
  140 GB / 1081% CPU. That is only a **1.16x speedup for 1.92x the memory** — the job is
  I/O- and GIL-bound, not core-starved. Projected to 364 samples, 64 workers needs ~1.3 TB and
  **will not fit on an 850 GB node**, while 16 workers needs ~650 GB and does. `max_workers` is
  a hard cap (`io_orchestration.py:120-128`, `min(SLURM_CPUS, max_workers)`); the memory lives
  almost entirely in the PARENT process (peak tree RSS 139.5 GB across 67 procs vs 140.5 GB max
  single), so more workers = more in-flight chunks buffered in the parent, not per-worker cost.
- Scores are cached per run in `results/<run>/evaluation/<run>_<stream>_<region>_<metric>_
  chkpt*.json` and reused by `load_scores`, BUT `plot_score_maps` / `plot_score_init_timeseries`
  force a full recompute even when everything is cached. Turn both OFF for a cheap cross-run
  summary-only pass: measured at **72 s and 0.46 GB** for one run (vs 7.5 min / 73 GB to compute).
- **The cache check is by sample MEMBERSHIP and it is correct** — a request for samples 40-79
  against a cached 0-79 file is a legitimate hit; a request for 364 against an 80-sample cache
  recomputes. Tell them apart in the log: "present in **metric file**" = cache hit (no data
  loaded), "present in **Zarr file**" = real recompute.

## Submitting the evaluation on Santis (2026-08-21)
- `hpc/santis/weathergen_slurm.sh` hardcodes `#SBATCH --exclusive --mem=450G`, `--time=24:00:00`
  and `-A ch17`. So **no `--account` is needed**, but the two others must be overridden:
  - **`--mem=450G` will OOM-kill a full-period eval** (364 samples needs ~650 GB). Pass `--mem=0`
    (all node memory, ~856 GB). `--exclusive` reserves the node but `--mem` still caps the cgroup.
  - **`--time=24:00:00` parks the job in the queue.** Pass `--time=02:00:00` — measured need is
    ~75 min/run. The launcher injects a default `--time` only for the TRAIN stage
    (`launch-slurm.py:587`); the evaluation builder (line 681) has none, so it inherits the 24 h.
- **Unknown launcher args are forwarded to sbatch** — `parse_known_args()` at line 1663, then
  `cmd.extend(slurm_args)` at line 681 (before the script path), and CLI sbatch options override
  in-script `#SBATCH`. So `--time=...`/`--mem=...` go straight on the launch-slurm command line.
- Working invocation:
  `../WeatherGenerator-private/hpc/launch-slurm.py --stage evaluation --run-ids <id>
   --eval-config config/evaluate/eval_config_imerg.yml --time=02:00:00 --mem=0`
- `--eval-config` is a **launch-slurm** flag; the `evaluate`/`evaluation` entry point itself takes
  `--config` (both console scripts point at `run_evaluation:evaluate`).
- `push_metrics=upload` and `--register` is the default, but the MLflow block is guarded by
  `if mlflow_client:`, so it is skipped when no tracker is configured — `--no-register` was needed
  on Jupiter, not necessarily here. If it does fire it runs AFTER scoring and BEFORE the cross-run
  summary plots, so the cached score JSONs survive and the cheap summary pass recovers it.
- **Subsetting `evaluation.sample` without subsetting `plotting.sample` crashes**: `_process_stream`
  hands the eval-loaded data to `plot_data`, which intersects loaded samples with the plot set
  (`plot_orchestration.py:1093`); an empty intersection hits `plot_samples[-1]` -> IndexError
  (line 1103). Keep plot samples inside the eval set, or turn plot_maps/histograms/animations off.

## mp4 animations: frame-size trap (hit on job 821431, fixed 2026-08-21)
- `animation_format: "mp4"` crashed a full run with
  `ValueError: All images in a movie should have same size`. Frames are written with
  `plt.savefig(..., bbox_inches="tight")`, so their pixel size varies by a few px with content
  (tick-label widths, title length). The GIF/PIL writer tolerates that; the ffmpeg writer does
  not. Short smoke runs pass because their frames happen to be uniform — it only shows up at
  full scale. Fixed by padding every frame to the sequence max in `_pad_frame`
  (`plot_orchestration.py`), anchored top-left, white fill — padding not resizing, to avoid
  resampling artefacts.
- **Plotting runs BEFORE scoring** (`run_evaluation.py:259` calls `plot_data`, scoring follows),
  so ANY plotting exception discards the whole run — job 821431 burned ~1 h and wrote zero score
  JSONs. Consider splitting: pass A scoring-only (plot_maps/histograms/animations off) to bank the
  score JSONs, then pass B for plots, which reads the cached scores and loads only the 4
  case-study samples (~10 GB, minutes) instead of all 364 (~634 GB).
- Real measured cost of one full run (m745z8wi, 364 samples, 3 regions, 4 metrics, 16 workers):
  **MaxRSS 634 GB** — within 2% of the 648 GB projection, and well over the script's 450 GB cap.
  Data load ~15 min, case-study plotting ~30 min (the 40 "across-samples" histogram tasks alone
  are ~20 min — `plot_histograms` over 364 samples is expensive), scoring still to come.
  **2 h is not enough; use `--time=03:00:00`.**

## Cyclone tracking for Daniel / Doksuri (`playground/scripts/track_cyclones.py`, 2026-08-21)
- IMERG_ANEMOI carries **only `tp`** -- no MSLP, no vorticity, no 10 m wind -- so the repo's
  `example_extras/tropical_cyclones/cyclone_finder.py` (needs pressure + wind) CANNOT be used on
  these outputs. The script tracks the **precipitation centroid** instead: intensity-weighted
  spherical centroid of tp above a threshold within a search radius of the previous position.
  It is a proxy for the circulation centre and is offset from it (eyewall/rainband asymmetry),
  so absolute error is NOT comparable to best-track MSLP verification -- but the SAME algorithm
  runs on model and IMERG, so model-vs-obs and model-vs-model comparisons are like-for-like.
- Addresses forecasts by **first-window timestamp, not sample index**, which sidesteps the
  differing index spaces (ce4rujvd vs the rest) entirely. Per-run time index is cached as JSON
  (`--cache-dir`); building one takes a few min, reading `times[0:1]` (chunks are 672, and times
  are uniform within a step). `coords` is identical for every sample and step -- read once.
- **Validated against reality.** Observed (IMERG) Doksuri track reproduces both real landfalls:
  Luzon 07-26 at 18.9N 121.0E (actual ~18.9N 121.3E) and Fujian 07-28 at 24.2-25.1N ~119.0E
  (actual ~24.7N 118.6E). Observed Daniel track runs Greece -> Ionian loop -> Libyan coast,
  ending 09-10 18Z at 32.87N 21.54E (Derna ~32.8N 22.6E).
- First results for `m745z8wi` (Forecast:dy0jlrmw):
  - **Doksuri, init 07-20**: mean track error 237 km, 100% detected. Good to ~07-25, then drifts
    LEFT/west of track, ~500-600 km off by 07-28.
  - **Daniel, init 09-01** (a +96..+234 h forecast of a mesoscale medicane): mean 876 km; the
    model drifts NE into Anatolia while the storm went SW to Libya, and loses it after +210 h.
  - **Systematic intensity deficit in both**: Doksuri model peak ~96 mm/6h vs 188 observed early,
    ~22 vs 103 late; Daniel ~15 vs 71. Consistent with MSE-trained smoothing of extremes.
- Cartopy downloads Natural Earth data on first use -- run once on a login node (with internet)
  to populate the cache before using it anywhere without network.

## 8-run 3-month comparison, 364 inits, 40 leads (2026-08-21) -- `playground/scripts/compare_scores.py`
Alignment verified first: the 8-run index space and ce4rujvd's are POSITIONALLY identical in
init time over the scored 364 samples, so per-sample comparison is valid.
- **`m745z8wi` has NO scores** -- it died on the mp4 bug and was not in the relaunch loop.
  Eight runs compared, not nine.
- **SEEPS (skill, higher better), global.** `ce4rujvd` (T/JEPA:cw6a4szu) leads at EVERY lead:
  0.653/0.533/0.355/0.214/0.126 at +6/+48/+120/+180/+240 h. Then f7ug724z ~= ra1xax01,
  then xq65fhca ~= dwzio194, then oywmz4sz. Paired bootstrap over inits: ce4rujvd's lead over
  f7ug724z at +240 h is NOT significant (CI spans 0); over everything else it is.
- **RMSE at long lead is MISLEADING here -- do not publish it naively.** At +240 h the two
  S/JEPA runs (`sctxzwwe`/lnm4ud42, `lf7aj7df`/whzrt2cc) rank BEST on RMSE (3.39 mm/6h) while
  ranking WORST on SEEPS by a mile (0.029/0.024 vs 0.126). Cause is a **progressive dry
  collapse**: bias falls to -0.30/-0.24 mm/6h at +240 h (others -0.01..-0.07), and SEEPS
  diverges from ce4rujvd steadily from ~+168 h (0.20 vs 0.24) through +204 h (0.11 vs 0.17)
  to +240 h. A near-dry precipitation forecast wins RMSE because it avoids the double penalty
  on misplaced heavy rain. **Always read RMSE together with bias and SEEPS.**
- Up to +120 h, ce4rujvd is best on RMSE too (2.04/2.72/3.16 at +6/+48/+120 h), so the ranking
  is consistent across metrics before the collapse.
- **Alignment across all 8 runs is now proven, not assumed**: run `compare_scores.py` with
  `--verify-alignment` and it reads each run's timestamps from the zarr, builds the global
  index -> time map (`RunStore.global_time_map()` in `track_cyclones.py` -- one shared
  implementation of the rank-concatenation rule) and refuses to compare if any run differs.
  First call indexes each run (~2-3 min each); cached in `.cache/weathergen_eval/`, after
  which the whole comparison takes ~2 s. Verified 2026-08-21: "8 runs share the same 364
  ordered initialisations." Without the flag only sample COUNTS are checked and it warns.
- Figures/CSVs: `plots/score_comparison/<metric>_<region>_vs_lead.{png,csv}` (global/nhem/shem).
  The CSV carries a `run_id` row under the label header -- labels contain commas and must be
  quoted, which is why the writer uses `csv.writer` and not a manual join.

## TRAP: ce4rujvd's lead_time axis is shifted +6 h in eval-package plots (2026-08-22)
The eval package derives `lead_time = valid_time - init_times`
(`io/data/dataarray_postprocessing.py:add_lead_time_coord`), and `init_times` is inferred from
the store layout. **ce4rujvd is the only run whose store has an fstep-`0` source group**, so it
gets `forecast_offset=0` and its init is taken as the source window START; the other seven have
no fstep 0, get offset 1, and their init is the window END. Same forecast, inits 21600 s apart.
Stored coords prove it:
  - the seven:  lead_time = 0, 6, ... 228, 234   init_times[0] = 1685620800e9
  - ce4rujvd:   lead_time = 6, 12, ... 234, 240  init_times[0] = 1685599200e9
Consequence: **in `compare_*_<region>_..._tp.png` from the eval package, ce4rujvd is plotted 6 h
to the right, so at any x it is credited with a score from one forecast step LESS far out --
which flatters it.** One step is worth ~0.01 SEEPS at long lead, several times its real margin
over the next-best run (+0.0018 at fstep 40, CI spans zero). Do not publish that figure as-is.
**Comparing at equal `forecast_step` is the fair thing to do** -- verified directly: sample k has
an identical fstep-1 target timestamp in both index spaces. `playground/scripts/compare_scores.py` plots by
forecast_step and is therefore unaffected; its ranking stands.
Fix options if the eval-package figure is wanted: strip/ignore the fstep-0 group for ce4rujvd, or
override its init_times so both conventions agree.

## Comparing ce4rujvd / csuff3rz / m745z8wi (2026-08-27)
- **The runs use different init-time conventions.** `ce4rujvd` labels fstep 1 as **+6h**
  with 06Z-based inits; `csuff3rz` labels fstep 1 as **+0h** with 12Z-based inits. Verified
  directly: at equal `forecast_step` **all 364 valid times coincide exactly** (init+lead is
  identical), so comparing at equal forecast_step is right. Pairing on `lead_time` or on
  `init_times` would silently compare *different forecasts* — `init_times` overlap by 360 of
  364 purely because both are 6-hourly grids offset by one step. The convention follows from
  whether the store wrote an fstep-0 group (see the `io.py` offset patch above).
- **`m745z8wi` has no `evaluation/` output at all** (0 files); ce4rujvd and csuff3rz have 12
  JSONs each. Any 3-way comparison needs the eval package run for m745z8wi first.
- Result over 364 shared inits, global, paired moving-block bootstrap (block=4, 95%):
  ce4rujvd beats csuff3rz on **SEEPS at 39 of 40 leads** (~0.004 skill, only +240h unresolved)
  and on RMSE at 40/40 — but **csuff3rz wins on MAE at short leads**. The metrics disagree in
  sign, which is itself the finding: MAE is dominated by the many near-zero points, SEEPS is
  categorical. Driver: `playground/scripts/compare_runs_paired_bootstrap.py`, output in
  `docs/imerg_paired_comparison_results.txt`.
- **`/capstor` was badly degraded this session.** The venv's site-packages are *symlinks* into
  `/capstor/store/cscs/userlab/ch17/uv_cache_shared/`, so when that store stalls: `uv run`
  fails at cache init, scipy raises `cannot import name 'KDTree' ... (unknown location)`, and
  `.venv/bin/*` processes sit at RSS 0 with no CPU forever. `stat -f` on the mount still
  returns instantly, so the mount looks healthy — check a deep path instead. System `python3`
  (numpy 1.17.3 only, no matplotlib/xarray) keeps working and is the escape hatch.

## Export path was broken; sample indices need valid-time matching (2026-08-27)
- **`export_core.py:335` never worked.** Its comment claims `get_model_results` accepts lists
  of epochs and ranks, but that function takes scalars and formats them `:05d`/`:04d`, so every
  call raised `TypeError: unsupported format string passed to list.__format__`. Patched to
  resolve one rank at a time and expand `rank="all"` by globbing beside rank 0 (so the store
  extension is discovered, not assumed). Uncommitted in the working tree.
- **Reading times from a store:** `zio.get_data(sample, stream, fstep)` returns an `OutputItem`
  whose times live at `item.target.times` — there is no `item.times`. An **fstep-0 group is
  source-only** (no target, no prediction), so key on the first *target-bearing* step; that is
  what makes the key identical between a run that wrote an fstep-0 group and one that did not.
- **`playground/scripts/build_matched_sample_lists.py`** resolves each run's own sample indices for the
  same forecasts, keyed on that valid time, and asserts every run's selection matches. For
  ce4rujvd/m745z8wi/csuff3rz: 364 forecasts common to all three. The first 48 chronological
  forecasts are indices **0-46 and 48**, not 0-47 — indices follow rank-file concatenation
  order, so a naive `--samples (0,48,1)` exports a different set from each run.
- Working venvs (both `UV_LINK_MODE=copy`, private cache, no /capstor dependency):
  `/iopsstor/scratch/cscs/walmikae/venvs/wg` (weathergen, torch 2.9.1+cpu, built with
  `--extra cpu` since export needs no GPU) and `.../venvs/raina` (science stack).
- **The default `--n-processes 8` dies on a Santis login node** with
  `RuntimeError: can't start new thread` — the user slice's cgroup `pids.max` is **1000** and a VS
  Code session already sits at ~720, so 8 workers x their BLAS/zarr threads blow the cap. Memory is
  not the constraint (183 GB limit, ~490 GB free). **`--n-processes 2` runs fine**; anything bigger
  belongs in a SLURM job. Verified end to end on b96m0co5: ~65 s per batch of 4 samples,
  ~12.5 min/rank, **~1.7 h and ~97 GB for a full 8-rank / 368-sample / 40-fstep `tp` export**
  (265 MB per sample file, 542080 cells x 40 steps, all finite).
- **The exported `forecast_reference_time` is 6 h EARLIER than the eval package's `init_times`**
  for the seven no-fstep-0 runs. `_reference_from_first_target` subtracts `step * fstep_hours`
  from the first target, giving the *window-start* convention (b96m0co5 -> 2023-06-01T06, which is
  also the filename stamp), while `add_lead_time_coord` gives offset-1 runs the window END
  (2023-06-01T12) — the same +6 h split documented in the ce4rujvd trap above. `valid_time` is read
  straight from the store and is unambiguous, so anything matching on valid time is unaffected;
  only `forecast_reference_time` / `forecast_period` / the filename carry the convention. Note
  `--init-time-reference` is a **no-op** for diagnostic-only runs (the fallback sets
  `source_start == source_end`).
- **`export` resolves stores from the PLATFORM's private config**, not from any CLI flag:
  `get_model_results` builds `<path_shared_working_dir>/results/<run>/validation_chkpt<ep:05d>_
  rank<r:04d>.{zip,zarr}`. Santis gives `/iopsstor/scratch/cscs/thunter/shared_work/`, Jupiter
  `/e/scratch/weatherai/shared_work` — different centres, no shared mount. Running the export on
  Jupiter for a Santis-only run therefore fails with `FileNotFoundError: ... does not exist or is
  not a directory`, which reads like a naming bug but is just the wrong machine. (A genuine
  permissions problem would raise `PermissionError` instead — `Path.exists()` only swallows
  ENOENT/ENOTDIR/EBADF/ELOOP.) The nine 3-month runs exist ONLY on Santis; each is 73 GB
  (8 rank zips x ~9.7 GB), while `results/<run>/evaluation/` is only 32 MB.

## Sibling repo: raina_evaluation
`/users/walmikae/weathergen/raina_evaluation` (JSC GitLab, Sebastian Buschow) holds the RAINA
spatial-structure verification for IMERG `tp`: nearest-neighbour distances, two-point correlation,
and `sadpy` wavelet scale/anisotropy/phase. It consumes this repo's `uv run export --stream
IMERG_ANEMOI --channel tp` netCDFs. Full agent guide lives at `raina_evaluation/CLAUDE.md` — read
that before touching it. Two things to know up front: all its paths are JSC (`/p`, `/e`) and none
resolve on santis, and the precip variable is `tp_imerg_0` in older/O96 exports but `tp` in
newer/N320 ones.

**Audited 2026-08-27 (`raina_evaluation/docs/EVALUATION_REVIEW.md`); fixes applied in that
repo's working tree, uncommitted on `main`.** Consequences for anything already published
from it:
- Every `distance_verif_*.nc` predating the audit is wrong — the two-point correlation
  normalised the *target* by the *prediction's* point count, so the pred/target ratio
  reduced to raw pair counts and carried a `(Np/Nt)^2` factor. Verified on synthetic sets
  with identical clustering: ratio 3.69 instead of 1.0. Delete the files to recompute; the
  script skips runs whose output exists.
- The `era5`/`era5_n320` surrogate runs were built by overwriting the prediction with a
  *global* ERA5 field taken at `forecast_period=0`'s valid times. So the ERA5 benchmark was
  masked-vs-unmasked against a 60S-60N target (inflating prediction->target distance and
  making "frequency bias" ~1.5 on geometry alone) and valid only at the first lead.
- ERA5 was matched to forecast times **positionally**, guarded only by a length check —
  the same sample-order trap recorded above for `xfaps4wq`/`r15j90ns`.
- Wavelet anisotropy/direction ran on raw lat/lon boxes, so it largely measured the
  `1/cos(lat)` map distortion (1.41x at 45N) rather than the weather.

Work lives on branch `review/scientific-audit-and-plots` in that repo (2 commits, not
pushed; `main` untouched). Second commit adds a house plot style, bootstrap bands on every
displacement figure, and three new diagnostics — displacement plane (d1 vs d2 = misses vs
false alarms), a paired-difference forest plot, and a QQ/exceedance intensity figure. Also
`raina_utils.paths`: every JSC root is now behind an env var (`RAINA_DISTANCE_DIR`,
`RAINA_PLOT_DIR`, `RAINA_IMERG_ROOTS`, …), so the repo finally runs on Santis. Figures were
verified by rendering against synthetic netCDFs, not just eyeballed in code.

## Two new backbones: nhv6tkln_fix and aets3ku5_fix (2026-08-28)
- **`nhv6tkln_fix` is cw6a4szu's ENCODER with a different forecast engine.** Loaded both
  checkpoints and compared tensor by tensor: same 1419 parameter names, **1195 bit-identical, 224
  different, and every differing tensor is `module.forecast_engine.*`** (224 is exactly the FE
  parameter count). Its own freeze regex has no forecast_engine term, i.e. it was an FE-only
  finetune off that encoder. So nhv6tkln vs `ce4rujvd` (T/JEPA:cw6a4szu) is a clean FE ablation.
  Size, run_history and istep are all identical to cw6a4szu_chkpt00008 -- do NOT infer identity
  from those.
- **`aets3ku5_fix` is the 8-STEP continuation of `f2esyc18_fix`** -- its run_history ends
  `['f2esyc18', 0]`, istep 2048, 8 mini epochs x 8192, `forecast.num_steps 8`. **Use it, not
  `f2esyc18_fix`**, which stopped at num_steps 2: pairing that against nhv6tkln_fix would have
  confounded the JEPA objective with pretraining depth. With aets3ku5 the pair is symmetric --
  same architecture, same budget (istep 2048), same depth (8 steps), byte-identical resolved
  streams -- and the only difference is the JEPA objective, **temporal (nhv6tkln, `temporal:True`
  + `independent`) vs spatial (aets3ku5, `subset`)**.
- `aets3ku5_fix`'s resolved streams are byte-identical to nhv6tkln_fix's AND to cw6a4szu's, so
  both new stream dirs are verbatim copies of `imerg_diag_cw6a4szu/`. The ERA5
  `max_num_targets: 542080` trap applies only to the discarded `f2esyc18_fix`; aets3ku5 has 108416.
- Both are **code `193332f51fc5`** -- byte-identical to all eight via-2-step parents, so they are
  code-matched to the existing comparison. Both have only `_latest.chkpt` (no numbered ones).
- **Resolved streams: nhv6tkln_fix == cw6a4szu exactly** (all ten streams, every field), so
  `config/streams/imerg_diag_nhv6tkln/` is a verbatim copy of `imerg_diag_cw6a4szu/`.
  **f2esyc18_fix differs only in the ERA5 diagnostic-output stream**: no `channel_weights`,
  different `target_channel_weights`, and **`max_num_targets: 542080`** rather than 108416.
  Following "inherit ERA5's cap" literally there would give an uncapped decoder -- the wx9bzx7q
  OOM. Both IMERG streams use 108416 deliberately.
- Both carry `physical` and `student-teacher` as type **Disabled**, so **`forecast` is the single
  live LossPhysical** -- declare the MSE under that key. model_input branch is `forecasting`.
- Parameter names confirmed from the real checkpoint: `module.encoder.*`,
  `module.forecast_engine.*`, `module.{embed_target_coords,target_token_engines,pred_heads}.<STREAM>.*`
  -- which is what makes the freeze regex
  `.*encoder.*|.*forecast_engine.*|.*latent_pre_norm.*|.*latent_heads.*|.*q_cells.*` correct.
- **BOTH new checkpoints say `parallel_scaling_policy: "const"`**, while the whole IMERG family
  runs `sqrt`. All four configs set `sqrt` EXPLICITLY -- inheriting `const` would make lr_max 1e-4
  mean a peak of 1e-4 here against 1e-4*sqrt(8) = 2.83e-4 everywhere else, i.e. a 2.8x lower LR
  than the runs these are meant to be compared with. Active arm is lr_max 1e-4; the 3e-4 arm is
  present but COMMENTED OUT in each config's learning_rate_scheduling block.
- Final shape: 2 stream dirs, 4 configs (2 backbones x {2step_6ep, 8step_2ep}), 2 eval configs
  (scores / events) and 2 pipelines `pipelines_final/pipeline_imerg_{nhv6tkln,aets3ku5}.yml`,
  5 stages each: train2s6ep -> train8s2ep (mini_epoch 5) -> infer3month (mini_epoch 1) ->
  evalscores -> evalevents. No --mem override anywhere; animations are mp4, which needs the
  `_pad_frame` fix now restored from the `raina evaluation` stash into the working tree.

## Evaluation inside a pipeline: it does work with minted run ids (2026-08-28)
- The older note that "`--run-ids` ids must already be in the config" is **too strong**.
  `run_evaluation.py:150-152` does `cf.run_ids = {k: existing.get(k, {}) for k in args.run_ids}`,
  so an unknown id yields an EMPTY per-run dict; `streams` then falls back to `default_streams`
  (`:345`) and `results_base_dir` falls back to the private config's run path
  (`wegen_reader.py:54-56`). A `run_ids: {}` key MUST still be present in the yaml, or the
  assignment raises on a struct-mode config. `runplot_base_dir` defaults to `results_base_dir`.
- `eval_config:` is a real `ParsedStage` field, so an evaluation stage in a pipeline can name its
  own yaml; `run_ids: [STAGE.<name>]` resolves the upstream inference run.
- The existing 3-month runs (m745z8wi, ce4rujvd, csuff3rz, fvenw2nq, ...) all used **exactly**
  spme 368 / 2023-06-01→2023-09-12 / 40 steps / world 8 / num_workers 2, so any new run on those
  settings shares their index space and the case-study samples **196, 204 (Doksuri) and 357, 365
  (Daniel)** transfer unchanged.

## aets3ku5/nhv6tkln pipelines: blank `num_mini_epochs` crashed the first launch (2026-08-28)
- **`m7mff1ck` (train2s8ep, job 833570) died in 1:35** with `TypeError: unsupported operand
  type(s) for *: 'int' and 'NoneType'` at `trainer.py:547`
  (`lr_steps = int((len_ds * self.training_cfg.num_mini_epochs) / self.batch_size_per_gpu)`).
  `config_finetune_imerg_diag_mse_aets3ku5_2step_8ep.yml`'s `num_mini_epochs:` was left BLANK
  (parses to `None`). sacct showed `COMPLETED 0:0` despite the crash -- same
  run_continue-swallows-exception pattern as the OOM cases above, so check the log, not the job
  state.
- **Decision: these stage-1 runs are 8 mini epochs, matching the eight original via-2-step
  parents** (which all trained 8 x 4096 to `chkpt00007`), NOT the 6-mini-epoch scheme the file's
  own comments briefly described (leftover from an earlier draft). Set
  `num_mini_epochs: 8` in both `config_finetune_imerg_diag_mse_{aets3ku5,nhv6tkln}_2step_8ep.yml`
  (nhv6tkln's had silently drifted to 8 already, so only aets3ku5's was actually broken) and
  `mini_epoch: 7` on each pipeline's `train8s2ep` stage (`pipeline_imerg_{aets3ku5,nhv6tkln}.yml`)
  so stage 2 continues from the fully-cooled-down `chkpt00007`, not a mid-schedule checkpoint.
  All in-file comments (`desc`, `wgtags.exp`, LR-schedule-length note, header) were updated to
  say 8/4096-optimizer-steps/chkpt00007 for consistency -- do not trust a stray "6 mini epochs"
  comment anywhere else in these two config files if one turns up.
- Logs for a train stage live at `<thunter-shared-work>/logs/<runid>/output.<jobid>.txt` (all
  ranks interleaved, prefixed `N:`) -- NOT under the per-run snapshot dir's `logs/`, which is a
  symlink to `/iopsstor/scratch/cscs/thunter/shared_work/logs/`. `models/<runid>/` staying empty
  (no numbered checkpoint) is the tell that a train stage never got past step 0.

## Tooling
- `/capstor` degraded again on 2026-08-28: the repo `.venv` lost `omegaconf`
  (`ModuleNotFoundError`) because site-packages are symlinks into the capstor uv cache.
  **`/iopsstor/scratch/cscs/walmikae/venvs/wg/bin/python` has omegaconf 2.3.0 and works**; the
  `raina` venv does not, and neither venv has `pip` or `dacite`. For torch on a login node set
  `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` or it dies with
  `libgomp: Thread creation failed: Resource temporarily unavailable`.

## The 9-run 3-month IMERG mp4s already existed — `find` just couldn't see them (2026-09-01)
- **`plots/` and `results/` are both top-level symlinks** into
  `/iopsstor/scratch/cscs/thunter/shared_work/`. Plain `find <path> -iname ...` does NOT descend
  into a symlink encountered *during traversal* (only a symlink given directly as the search
  root is followed), so `find . -iname "*imerg_diag_comparison_3months*"` from the repo root (or
  any ancestor) finds nothing even though the directory is right there — always search under
  the resolved iopsstor path, or pass the symlinked path itself as the find root, e.g.
  `find results/<run>/plots ...` (works) vs `find . -iname plots` then descending (doesn't).
- **A full run of `eval_config_imerg.yml` for all nine 3-month runs already completed on
  2026-08-21**, writing 39 mp4s per run (15 histograms + 12 maps/preds_ens_mean +
  12 maps/targets; `maps/bias` has 0) to
  `plots/imerg_diag_comparison_3months/score_maps/<run>/plots/IMERG_ANEMOI/`, plus the
  cross-run `compare_*.png` summary plots. Every one of `ce4rujvd, fvenw2nq, b96m0co5,
  xt069k3n, csuff3rz, m745z8wi, tvmzluy6, sctxzwwe, lf7aj7df` has the full set, m745z8wi
  included (the "m745z8wi died on the mp4 bug, no scores" note above is now stale — it has
  `maps`/`histograms` output, just no `score_maps` subdir, so it likely ran again after the
  `_pad_frame` fix and never got a `results/<run>/evaluation/` score cache written to match).
- These runs' `results/<run>/plots/` never existed (only `han9w7yt`/`hmw4u8bu` had that shape
  natively). **Symlinked `histograms/` and `maps/` from each run's `score_maps/.../plots/
  IMERG_ANEMOI/` into `results/<run>/plots/IMERG_ANEMOI/`** so all nine now match
  `han9w7yt`/`hmw4u8bu`'s structure (`plots/IMERG_ANEMOI/{histograms,maps/{bias,preds_ens_mean,
  targets}}`) without duplicating data. Deliberately skipped symlinking `score_maps` itself
  (the 8 complete runs have it, han9w7yt/hmw4u8bu don't) to keep the shape an exact match.
- **Do not re-run the eval/plot pass for these nine without checking here first** — it already
  exists; a from-scratch `uv run evaluation --config eval_config_imerg.yml` pass costs ~11 h
  sequential (see "Cost of the 3-month IMERG evaluation" above) and would just regenerate what
  is already on disk.


## 11-run comparison: combining the individual evals needs NO job (2026-09-03)
- The stored score JSONs are **per-sample** (`(364, 40, 1, 1)` = sample/fstep/channel/ens), not
  pre-aggregated, and `wegen_reader.load_scores` re-subsets them with
  `score.sel(sample=...)` (`wegen_reader.py:226-230`). Init alignment is therefore enforced
  **at load time from the config's sample list**, not baked into the cache — so a cross-run
  comparison is a pure JSON read.
- `_process_stream` loads zarr only if `plot_score_maps or plot_score_init_time_series or
  recomputable_metrics` (`run_evaluation.py:237-240`). With every metric cached and those two
  switches off, a full 11-run comparison **measured 1.4 GB peak RSS and ran on the login node**,
  vs ~650 GB / ~3 h for the scoring pass. Never submit a job for a comparison-only pass.
- `config/evaluate/eval_config_imerg_compare_all11.yml` is that pass; it produced 660 PNGs under
  `/iopsstor/scratch/cscs/walmikae/imerg_diag_eval_results/compare_all11/summary/`.
- `do_lead_time` in `plot_orchestration.py:1294` has **no old-style boolean fallback** — it reads
  `score_plots` only. `summary_plots: true` works solely because
  `config_compat._SCORE_PLOT_BOOL_MAP` maps it to `lead_time` before that line runs.
- `ratio_plots` emits **one figure per forecast step** (40 per metric/region), i.e. 600 of those
  660 files. Leave it off for large metric sets.
- `baseline:` must name a run_id in `run_ids`; if it doesn't, `line_plots.py:547-549` silently
  falls back to `run_ids[0]` and every ratio plot changes meaning with no error.

## Threshold LISTS now work for ets/fbi/pss (2026-09-03)
- Previously only a scalar `thresh` worked; the old claim that a list auto-expands was false.
  `parse_metric_params` now expands `thresh: [a, b]` into `<metric>_thr<value>` entries
  (`ets_thr0.001`), each scored/cached/plotted independently. Scalar `thresh` is unchanged, so
  existing configs and cached JSONs stay valid.
- `base_metric_name()` (in `utils/dict_utils.py`, a leaf module — import it from anywhere) maps
  those names back. It is consumed by `Scores.get_score` (function + `score_args_map` lookup),
  `plot_utils.lower_is_better`, `score_cards.get_perf_score` and `clim_utils.needs_climatology`.
  **Any new metric-name lookup must go through it**, or thresholded metrics silently take the
  wrong branch (e.g. defaulting to "higher is better").
- Verified numerically identical to calling `calc_ets/calc_fbi/calc_pss` directly.
- Cache-key note: the old scalar-thresh `ets` file is `..._ets_...json`; `ets_thr0.02` writes
  `..._ets_thr0.02_...json`. Different keys — switching to the list form **recomputes**.

## Metrics available vs. useful for an IFS / GraphCast comparison (2026-09-03)
- Registry (`score.py:198`): deterministic — ets, pss, fbi, mae, l1, l2, mse, rmse, vrmse, bias,
  acc, rps, rpss, froct, troct, fact, tact, grad_amplitude, psnr, seeps, qq_analysis, nse, psd;
  probabilistic — ssr, crps, rank_histogram, spread.
- **All four probabilistic scores are unusable here**: every one of the 11 runs is ens_size 1, so
  `get_score` warns and returns None.
- **`psd` must stay off** — `detect_grid_type` knows only octahedral and regular; this is N320
  reduced Gaussian, and it fails far downstream as `KeyError: 'global'`.
- Worth adding: **fbi** (the essential ETS companion — separates "missed everything" from
  "over-forecast everything"), **pss**, **acc** (the headline score IFS/GraphCast are judged on),
  **fact/tact** (their ratio is the damping-toward-climatology diagnostic), **grad_amplitude**
  (blurring), **qq_analysis** (extreme tail). Configured in
  `config/evaluate/eval_config_imerg_metrics_extended.yml` (expensive, reloads data) with the
  cheap read-back in `eval_config_imerg_compare_extended.yml`.
- FBI's perfect value is 1.0 from **either side**, but the score cards treat it as
  "higher is better, perfect 1.0" — so FBI score cards are only meaningful while fbi < 1.
  Read FBI from the line plots.

## S/JEPA long-lead collapse: why RMSE alone would have misled (2026-09-03)
- On the 364 matched inits, `sctxzwwe` and `lf7aj7df` have the **best global RMSE at +240 h**
  (3.39/3.40 mm/6h) and the **worst SEEPS** (0.029/0.024) — ETS@20mm is exactly 0.000 and bias
  falls to -0.30/-0.24 mm/6h past ~+168 h. They dry out to a near-empty field, which RMSE rewards.
- `ce4rujvd` by contrast holds bias -0.047 and ETS 0.010 at +240 h.
- All 11 SSL variants beat both supervised Forecast baselines on both RMSE and SEEPS.
- Label discrepancy to resolve: the circulated list calls `fvenw2nq` "MTM noFE" but the config
  says "W/O FE, **CW**" — same as `b96m0co5`, and their scores nearly coincide (RMSE 3.135 vs
  3.132), consistent with them being the same configuration.

## Smoke-testing the extended metrics found 3 unusable metrics (2026-09-03)
Config `config/evaluate/eval_config_imerg_smoketest.yml` — 1 run, 4 samples, 3 fsteps, and
crucially `metrics_dir:` redirected to scratch (it is a per-run key read at
`wegen_reader.py:73-74`) so partial scores never touch the real cache. Run it TWICE: the second
pass is what proves the cache key is stable.
- **`acc` / `fact` / `tact` cannot be scored on IMERG.** The only IMERG climatology
  (`nasa-imerg-grib-n320-1998-2024-6h-v1_climatology.zarr`) is SEEPS-specific: `statistic` holds
  exactly `['prob_dry', 'light_heavy_threshold']`. `calc_acc` (`score.py:1097`) and `_calc_act`
  (`score.py:1011`, backing both fact and tact) do `c.sel(statistic="mean")` → `KeyError: 'mean'`.
  Needs a mean/std N320 climatology to be generated; the o96 ERA5 ones are not a substitute.
- **A failing metric aborts the ENTIRE run with zero JSONs written** — the exception escapes the
  joblib pool and `calc_scores_per_stream` before anything is banked. One bad metric = the whole
  ~3 h pass lost, per run. Always smoke-test a new metric list first.
- **`grad_amplitude` is dead code.** `calc_geo_spatial_diff`'s inner `check_for_coords` calls
  `coord_names_expected.index()` with no argument (`score.py:1733`) → unconditional TypeError; it
  also returns None while the caller unpacks two values. It also needs lat/lon *dims*, which
  unstructured `ipoint` data lacks.
- **`qq_analysis` can never cache-hit.** `store_metrics_for_region` writes its per-fstep quantiles
  into the DataArray attrs, so stored attrs are `{'fstep_1/p_quantiles': ...}` while config
  parameters are `{}`; `load_single_score` requires `attrs == parameters` (`wegen_reader.py:290`),
  so it reloads and recomputes every single time. Also ~556 KB at 4 samples x 3 fsteps x 1 region
  → ~2 GB/run, ~22 GB over 11 runs, vs ~5 KB per threshold metric.
- **What survives: 18 metrics** — ets/fbi/pss x [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05] m/6h.
  Final reload pass: 18 checks passed, 0 data reloads, 0 recomputes, 555 MB RSS.
- Sanity values for m745z8wi (4 samples): FBI runs 1.56 (0.1 mm) → 0.28 (50 mm), i.e. it
  over-forecasts drizzle and badly under-forecasts extremes — the classic precip signature, and
  the diagnosis ETS alone cannot give.

## Concurrent `uv run` invocations kill each other (2026-09-03)
Two or more `uv run` processes (evaluation, type-check, launch-slurm) each re-sync the shared venv
at `/capstor/store/cscs/userlab/ch17/uv_cache_shared/` and yank packages from under the others —
they die with **exit 141 (SIGPIPE)** after an "Installed N packages" line, having done nothing.
Uncontended sync is 667 ms; contended it was 2+ min then death. Run `./scripts/actions.sh sync`
once first, and stagger a multi-run launch loop (`sleep 60` between iterations) rather than
starting eleven launchers at once.

## Extended 18-metric pass: COMPLETE (2026-09-03)
- 11 jobs (844437-844481), all COMPLETED in **31-38 min each**, not the 3 h budgeted — once the
  data is loaded the contingency-table metrics are cheap arithmetic. 594 JSONs written
  (54/run = ets/fbi/pss x 6 thresholds x 3 regions).
- Comparison read-back (`eval_config_imerg_compare_extended.yml`): **0 recomputes, 0 data reloads,
  726 checks passed** (22 metrics x 3 regions x 11 runs), 14 min, 4.6 GB RSS, 264 figures in
  `/iopsstor/scratch/cscs/walmikae/imerg_diag_eval_results/extended/summary/`. The `_thr` cache
  keys hold at full scale.
- **TRAP: `WEATHERGEN_EVAL_CONFIG` is passed as the LIVE working-copy path**, not the snapshot
  path (`weathergen_slurm.sh:227` appends it as `--config`). The launcher snapshots the *code*
  (WEATHERGEN_HOME) but the eval config is read from
  `/users/walmikae/.../config/evaluate/<name>.yml` at job START. Editing an eval config while
  jobs are queued changes what they compute, and differently per job depending on start time.

## CORRECTED: S/JEPA collapses to DRIZZLE, not to a dry field (2026-09-03)
Earlier entries inferred "dries out to a near-empty field" from negative bias + ETS->0. FBI shows
that was wrong. At +240 h, global:
- `sctxzwwe` / `lf7aj7df` FBI: **3.49 / 3.81 at 0.1 mm**, 0.68 / 0.87 at 1 mm, **0.031 / 0.029 at
  5 mm**, 0.001 at 20 mm, 0.000 at 50 mm.
- So they cover 3.5-3.8x too much area with light rain while producing essentially NO precip above
  5 mm. The precipitation distribution has collapsed onto drizzle; the negative bias comes from
  losing the heavy rain that carries the mass, not from going dry.
- ETS at 5 mm+ is 0.003-0.004 and 0.000 at 20 mm+: no skill beyond light rain.
- Healthy runs for contrast at +240 h / 5 mm: `m745z8wi` FBI 0.892, `ce4rujvd` 0.784,
  `han9w7yt` 0.678. T/JEPA and ST/JEPA also over-forecast drizzle (FBI 2.3-2.7 at 0.1 mm) but
  degrade gracefully.
- This is exactly the discrimination ETS/SEEPS/bias cannot make on their own and why FBI was the
  metric worth adding.
- **Read FBI from the LINE PLOTS, never the score cards**: ScoreCards treats every non-error
  metric as "higher is better, perfect 1.0", so S/JEPA's 3.81 scores as better than a perfect
  1.0 instead of far worse.

## Preprint multi-target precipitation experiment (2026-09-03)
New tree `config/preprint_config/` (+ `pipelines/`), stream dirs `config/streams/preprint_{3target,operanonly}_<bb>/`,
eval configs `config/evaluate/eval_config_preprint_*`. Four backbones: `cw6a4szu`, `nhv6tkln_fix`,
`f7ug724z_fix`, `aets3ku5_fix`. Two arms: 3-target (IMERG+OPERAN+ERA5 tp) and operan-only.
Stages: 2 steps x 16 mini epochs -> 8 steps x 8 mini epochs -> 3-month inference -> 2 eval passes.
- **ERA5 tp is a 1-HOUR accumulation** (mean 0.120 mm) while IMERG and OPERAN tp are 6-HOUR
  (0.676 / 0.741 mm). The only N320 ERA5 store is 1-hourly, so `ERA5_TP` sets
  `frequency: 06:00:00`, which SUBSAMPLES (anemoi frequency selects, never aggregates). Fine as an
  auxiliary signal; the three heads' scores must never share an axis. OPERAN_TP is the only
  like-for-like third target that exists today.
- **Operan-as-target must be `type: anemoi`, NOT `anemoi_operan`.** The operan reader simulates
  INPUT latency: it rewrites timestamps through `nominal_time_mapping` and keeps only the latest
  record available at the cut-off (data_reader_anemoi_operan.py:118-128). Correct for forcing,
  wrong for a target, which must be the analysis valid AT the verification time.
- **`.*ERA5.*` in f7ug724z's freeze regex also matches `pred_heads.ERA5_TP`** --
  `apply_fct_to_blocks` does `re.fullmatch` against MODULE names (model/utils.py:56-60), so the
  new ERA5 decoder would be frozen at random init. Narrowed to `.*\.ERA5|.*\.ERA5\..*`.
  In YAML this MUST be single-quoted: `"\."` is an invalid double-quoted escape.
- **Pipeline `options:` DO reach `streams.*`, config-file `streams:` blocks do not.** launch-slurm
  turns options into config_command_line.yaml via `OmegaConf.from_dotlist` and appends it as the
  LAST overwrite (launch-slurm.py:312-316), whereas `_load_streams_in_config` ASSIGNS
  `config.streams` per overwrite. That is how the 8-step stage retunes max_num_targets.
- **Decoder memory budget is (streams x max_num_targets x forecast steps).** Proven single-IMERG
  8-step load is 1 x 108416 x 8 = 867328. 3 targets at 8 steps therefore use 36139 each; at
  2 steps 108416 each is already under budget. Operan-only needs no override.
- `ctr_streams` counts only streams that produced predictions (loss_module_physical.py:337,346),
  so forcing streams like SurfaceCombined never enter the loss: 3 and 1 respectively.
  IMERG's gradient in the 3-target arm is 1/3 of the single-target runs -- a real confound.
- Verified by full `load_merge_configs` against the real checkpoints: one LossPhysical block each
  (`forecast` for cw6a4szu/nhv6tkln/aets3ku5, `physical` for f7ug724z), one enabled model_input
  branch (`forecasting` vs `masking`), no new decoder frozen.
- **`chain_jobs: 2` is safe with `general.istep: 0`.** All jobs in a chain share one run_id
  (launch-slurm.py:468-470), so `STAGE.<name>` and pinned `chkptNNNNN` still resolve. Job 2+ passes
  NO config files (`current_extra_paths = ""`, :471) and inherits the config saved with the
  checkpoint -- which is what carries pipeline `options` overrides forward. istep does not reset:
  the trainer increments `cf.general.istep` in the live config (trainer.py:795) and saves that.
  Empirical proof: `xqlk2nkf`'s config file says istep 0, its saved chkpt00016 config says 8430.
  chain_jobs is train-only (max 4); inference/evaluation stages ignore it with a warning.
- **max_num_targets as a fraction of N320 (542080 pts):** 108416 = exactly 20%, 36139 = 6.667%.
  The 3-target arm keeps the TOTAL sampled fraction at 20% (3 x 36139 = 108417), split three ways,
  so total decoder load matches the proven single-IMERG runs while each stream sees a third as
  many points per step.
- **CORRECTED (2026-09-03): 36139 was far too conservative; the value is now 135520 (25% of N320).**
  It had been anchored to the last KNOWN-GOOD run rather than a measured ceiling, and left ~60 GiB
  of the 95 GiB card unused. The decoder wraps ALL 12 of its blocks in `torch.utils.checkpoint`
  (engines.py:967-977), so it recomputes instead of storing per-layer activations and memory is far
  cheaper than the naive estimate.
- **Measured on an idle login-node GH200** (same 95 GiB card as the compute nodes, so no job needed)
  with `TargetPredictionEngineClassic` built from the real merged config. Marginal cost is LINEAR at
  **14.1 KiB per point-step**, point-steps = streams x max_num_targets x forecast steps. At
  3 streams x 8 steps: 36139 -> 11.87 GiB, 108416 -> 35.17, 135520 -> 43.87, 162624 -> 52.65,
  180693 -> 58.42 GiB of decoder.
- **Non-decoder baseline is ~23 GiB**, derived from wx9bzx7q: it died with 86.32 GiB allocated
  asking for 4.04 more, and the same geometry (1 x 542080 x 8) measures 67.05 GiB of decoder.
- **Concentrating targets in one stream is MORE expensive than splitting them**: 16.2 KiB per
  point-step at 1 x 542080 vs 14.1 KiB for the same total split three ways, because the varlen
  attention transient tracks the largest single sequence.
- The probe covers the decoder stack ONLY; the trainer also retains per-point predictions/targets/
  coords across steps, which scale with the same number -- so the measured figures are a LOWER
  bound and 25% deliberately leaves ~28 GiB of headroom. 162624 (30%) is the next rung.
- **`decoder_type: "PerceiverIOCoordConditioning"` selects `TargetPredictionEngineClassic`**, not
  `TargetPredictionEngine` -- the condition at model.py:524-526 is inverted relative to the names.
- Latent bug: `engines.py:1049` does `next(self.cf.streams.values())` on a ValuesView (TypeError).
  Unreached in practice because that class is never the one selected.
- **`cw6a4szu` has NO `_latest.chkpt` -- stage 1 from it MUST pin `mini_epoch: 8`.** Its models dir
  holds only `cw6a4szu_chkpt00008.chkpt`. The launcher defaults to mini_epoch -1, which
  model_interface turns into `<run>_latest.chkpt` (model_interface.py:262-264) and load_run_config
  turns into `model_<run>_latest.json`, then SILENTLY falls back to `model_cw6a4szu.json` -- the
  istep 0 config from before cw6a4szu trained (config.py:244-246). The config fallback is silent;
  only the weights load fails, with FileNotFoundError on cw6a4szu_latest.chkpt. Burned run
  sr6zw61d (job 846806) this way. `nhv6tkln_fix`, `aets3ku5_fix` and `f7ug724z_fix` ship ONLY
  `<run>_latest.chkpt`, so -1 is correct for them and pinning a number would break those instead.
  ALWAYS check which of the two forms a parent actually has before writing a pipeline.
- Verified from that same failed run's log.txt that the preprint stream set and freeze regex work:
  every `target_token_engines.OPERAN_TP.*` and `pred_heads.OPERAN_TP.*` block logged "Did not apply
  function freeze_weights" while `latent_heads` / `latent_pre_norm` were frozen.
- launch-slurm copies **git-tracked files from the WORKING TREE** (so tracked-but-uncommitted edits
  like the `_pad_frame` mp4 fix DO ship) plus **ALL** `config/**/*.y*ml` tracked or not
  (`copy_all_configs`, rglob). Untracked files OUTSIDE config/ are NOT copied. Non-.yml files in a
  stream dir (e.g. README.md) are not copied either -- harmless, load_streams reads only *.yml.
- **Third preprint arm added: ERA5-only (`preprint_era5only_*`)**, symmetric with operan-only --
  4 stream dirs, 8 configs, 4 pipelines, 2 eval configs. Derived by transforming the operan-only
  files so every earlier fix (135520, chain_jobs 2, cw6a4szu mini_epoch 8, narrowed f7ug724z regex)
  carries over automatically.
- **The f7ug724z ERA5 freeze trap is CRITICAL in this arm**: ERA5_TP is the only trainable decoder,
  so the original `.*ERA5.*` would have left the run with ZERO trainable parameters rather than
  merely crippling one head. Verified frozen=NONE for all four backbones.
- **This arm is comparable ACROSS BACKBONES but NOT against the operan-only/IMERG arms in absolute
  score** -- ERA5 tp is a 1-hour accumulation (mean 0.120 mm) vs 6-hour for IMERG/OPERAN
  (0.676/0.741 mm), and `frequency: 06:00:00` subsamples rather than aggregates. Its eval colour
  levels are the 6h family levels divided by six.
- Deriving one eval config from another by text extraction is fragile: the stream name appears in
  BOTH `global_plotting_options` and `default_streams`, so an unanchored search grabs the first
  match and swallows everything between. Anchor on the `default_streams:` split first.
