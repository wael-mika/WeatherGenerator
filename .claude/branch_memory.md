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

## Artefacts
- `scripts/imerg_presentation_figures.py` — 15 presentation figures, `--figure <id>` per figure,
  `--verify-matched-inits` re-derives the pinned 37-init lists from the stores.
- `docs/imerg_presentation_figures.md` — captions, numbers and slide order.
- `config/evaluate/eval_config_imerg_2step_matched.yml` and `..._8step_rollout.yml` — the
  evaluation-package figure sets (16 metrics × 4 regions), documented in
  `docs/imerg_evaluation_configs.md`.
- `docs/imerg_finetune_session_log.md`, `docs/imerg_finetune_parents_comparison.md` — earlier work.
