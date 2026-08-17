# IMERG evaluation configs — the `weathergen.evaluate` figure sets

Two tracked configs that drive the evaluation package directly, for internal use. They are
separate on purpose: the 2-step runs and the 8-step run answer different questions and need
different sample sets, so combining them would force one of the two into the wrong shape.

| config | runs | leads | initialisations |
|---|---|---|---|
| [`config/evaluate/eval_config_imerg_2step_matched.yml`](../config/evaluate/eval_config_imerg_2step_matched.yml) | `xfaps4wq`, `r15j90ns` | 1–2 (+6 h, +12 h) | 37 matched |
| [`config/evaluate/eval_config_imerg_8step_rollout.yml`](../config/evaluate/eval_config_imerg_8step_rollout.yml) | `z71y2ik8` | 1–40 (+6 h … +240 h) | 10 |

These complement, not replace, the hand-built presentation figures in
[`imerg_presentation_figures.md`](imerg_presentation_figures.md): the package figures cover far
more metrics and every metric × region × lead combination, but without the narrative framing.

## Running them

Use the project launcher — a bare `sbatch` will not work. The evaluation needs the platform
environment that `hpc/jupiter/weathergen_slurm.sh` sets up (module loads, venv activation, paths);
without it the job dies within ~20 s with an empty log and no traceback.

```bash
../WeatherGenerator-private/hpc/launch-slurm.py --stage evaluation --no-register \
  --run-ids xfaps4wq r15j90ns \
  --eval-config config/evaluate/eval_config_imerg_2step_matched.yml \
  --account=e-ext-2025e01-128

../WeatherGenerator-private/hpc/launch-slurm.py --stage evaluation --no-register \
  --run-ids z71y2ik8 \
  --eval-config config/evaluate/eval_config_imerg_8step_rollout.yml \
  --account=e-ext-2025e01-128
```

`--run-ids` **filters** the config's `run_ids` block down to the ids listed, so it must name the
runs that are already in the config.

`--no-register` skips the MLflow push. It is no longer *required* — see the package fixes below —
but it is faster and these runs do not need a tracker entry; the scores are all on disk in
`metrics_dir` regardless.

## Package fixes this work required

Three bugs in `packages/evaluate` had to be fixed for these configs to run at all. All three killed
the job outright rather than degrading, and two of them did so *after* a lot of successful work, so
they read as partial successes rather than crashes.

| fix | file | what went wrong |
|---|---|---|
| `qq_analysis` is now terminal in the summary-plot loop, like `psd` | `plotting/plot_orchestration.py` | It is a distribution with a `quantile` dimension and a non-unique channel index, but it was still being fed to the ratio / heat map / score card / bar plotters, which reindex on channel. `heat_map` raised `pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects` and took the run down partway through the summary plots (job `1372138`). The loop already special-cased it for lead-time plots; now it produces its quantile plots and moves on. |
| MLflow block uses `get_reader()` | `run_evaluation.py:400` | It instantiated the **abstract** `WeatherGenReader` directly — `WeatherGenZarrReader` / `WeatherGenJsonReader` are the concrete subclasses — giving `TypeError: Can't instantiate abstract class ... 'get_ensemble', 'get_forecast_steps', 'get_samples'`. Because the block runs *before* `plot_summary`, `--register` killed the job after 29 minutes with every score file and score map written and zero summary plots (job `1370727`). |
| new `score_map_regions` option | `run_evaluation.py`, `utils/config_compat.py` | Score maps are one full-resolution map per (metric, region, forecast step). For the 40-lead rollout that is ~2560 maps and the host OOM-killed the job (`srun: task 0: Killed`, job `1372152`). The new key narrows *only* the maps; everything else keeps scoring over the full region list. Unset means "every scored region", i.e. unchanged behaviour. |

```yaml
evaluation:
  score_map_regions: ["global"]    # 8-step config; omit to map every scored region
```

Logs land in the job snapshot, not the working tree:
`/e/scratch/weatherai/slurm/slurm_weathergen_<runid>_dir/WeatherGenerator/logs/<runid>/output.evaluation.<jobid>.txt`

`summary_dir` and `runplot_base_dir` are **absolute** in both configs, pointing at
`plots/imerg_2step_matched/` and `plots/imerg_8step_rollout/` in the working tree. This is
deliberate: the launcher copies the config into a per-job snapshot and runs from there, so a
relative path would scatter the figures across snapshot directories.

## Output layout

```
plots/imerg_<set>/
  <run_ids joined by _>/       # combined, cross-run summary plots
    line_plots/<metric>/<region>/      score vs lead time, runs overlaid
    bar_plots/<metric>/<region>/
    score_cards/<metric>/<region>/
    ratio_plots/<metric>/<region>/     (2-step config only — needs two runs)
    quantile_plots/<metric>/<region>/  from the qq_analysis metric
  score_maps/<run>/plots/IMERG_ANEMOI/
    maps/preds_ens_mean/ , maps/targets/ , histograms/
  score_init_time_series/              score vs initialisation time
  scores/<run>/                        per-metric JSONs (the private metrics_dir, see below)
```

No `psd_plots/` and no `animations/` — both are disabled on purpose; see the two subsections below.

Note `maps/preds_ens_mean/` rather than `maps/preds/`: with `ensemble: "mean"` the plots land in
the `_ens_mean` subdirectory. An empty `preds/` is not a failure.

## Metrics: what is on, and what is deliberately off

Every metric in `Scores.det_metrics_dict` that is valid for this stream is enabled in both configs,
so the two are directly comparable at +6 h/+12 h.

**On** — 16 metrics: `rmse` (latitude-weighted), `mae`, `mse`, `l1`, `l2`, `bias`, `vrmse`, `nse`,
`psnr`, `ets`, `pss`, `fbi`, `seeps`, `qq_analysis`, `froct`, `troct`.

**Off, and why** — these are not oversights, and re-adding them will break the run:

| metric | why not |
|---|---|
| `acc`, `fact`, `tact` | need a climatological **mean**. Both IMERG climatologies on this system carry only `statistic = [prob_dry, light_heavy_threshold]` — the two SEEPS parameters. Requesting them raises `KeyError: "not all values found in index 'statistic'"` and aborts the run **after** the plots have been written. `froct`/`troct` answer the same activity question without a climatology. |
| `rps`, `rpss` | need `q20`/`q40`/`q60`/`q80` in the climatology's `statistic` dimension. Same store, same gap. |
| `psd` | the SHT path cannot identify this grid — see below. |
| `grad_amplitude` | `calc_spatial_variability` requires a regular lat/lon grid; this output is scattered `ipoint`s. |
| `crps`, `ssr`, `spread` | probabilistic. These runs have `ens_size = 1`, so `Scores.get_score` warns and returns `None`. |
| `rank_histogram` | probabilistic, and `xskillscore`/`xhistogram` are not installed in this venv. |

### `psd` is a trap worth spelling out

It looks usable — `calc_psd` regrids internally and takes a `psd_regrid_resolution` — but it is
not, and the way it fails is nasty. `detect_grid_type` recognises only `"octahedral"` and
`"regular"`; **N320 is a reduced Gaussian grid**, so detection returns `None` and `sht_psd` raises
`ValueError: Unknown grid_type: None`.

`calc_psd` swallows that ("PSD computation failed, returning NaN") — so the run keeps going and
looks healthy — but it leaves no per-region entry behind, and the job dies much later in
`metric_list_to_json` with a bare `KeyError: 'global'` that points nowhere near the cause.
Observed on job `1370458`.

There is no config-level fix. `grid_type="reduced"` exists inside `psd.py` but needs
`anemoi.transform` **and** is not reachable from the config: `calc_psd` accepts
`psd_method` / `psd_regrid_resolution` / `psd_sht_truncation` / `lat_range`, not `grid_type`.
`psd_method: "fft"` is not an escape either — it requires a regular lat/lon grid by construction.

Consequence: no `psd_plots/` subdirectory is produced. The over-smoothing question is covered
instead by `qq_analysis` (distribution) and, in the presentation set, by the rain-rate PDFs.

### `metrics_dir` must be private, or the run dies with `KeyError: '<region>'`

Both configs set a per-run `metrics_dir` under `plots/<set>/scores/<run>/` instead of writing into
the shared results store's `evaluation/` directory. This is **required**, not hygiene.

`load_scores` skips any metric already cached in `metrics_dir`, **per region**, and only the
missing ones are recomputed. But `metric_list_to_json` then loops every recomputed metric over
*every* region in `regions_to_compute`:

```python
for metric, metric_stream in metrics_dict.items():
    for region in regions:
        for run_id, metric_data in metric_stream[region][stream].items():   # <- KeyError
```

The shared `evaluation/` directories already hold `rmse`/`mae`/`bias`/`ets`/`fbi`/`seeps` for
`global`/`nhem`/`shem` from earlier evaluations, but nothing for `tropics` and nothing for the ten
metrics added here. The recomputed set is therefore ragged — some metrics cover all four regions,
others only `tropics` — and the loop dies on the first mismatch. Observed as `KeyError: 'global'`
(job `1370458`) and then `KeyError: 'nhem'` (job `1370647`); the region named depends on which
files happened to exist, which is why it moves around between runs.

A private, initially-empty `metrics_dir` means nothing is cached, so every metric is recomputed for
every region and the shapes line up. **Delete that directory to force a clean recomputation** if
you change the metric list or the region list — leaving it populated reintroduces the bug.

It also keeps these runs from writing into `/e/scratch/weatherai/shared_work/results/<run>/evaluation/`,
which other people's evaluations read.

### `animations` — still open

Not requested in either config's `data_plots`, and the only item on this page that is worked around
rather than fixed. `_build_single_animation` renders each lead time as its own figure and hands the
frames to `imageio.mimsave`; the figures come out at **different pixel sizes** — frames resizing
from `(1773, 437)` and `(1544, 915)` appeared in the same movie — so ffmpeg raises
`ValueError: All images in a movie should have same size` and takes the evaluation down with it,
*after* the maps and histograms have been written. Observed on job `1370729` (8-step, 40 leads).

Fixing it means pinning a figure size in `plot_orchestration`, which is a larger change than the
three above and was not attempted. Meanwhile the rollout is covered statically by `F10`/`F11` in
`plots/imerg_presentation/`, and older mp4s from a previous evaluation survive in
`plots/imerg_8step_40lead/animations/`.

Three details worth knowing about the metrics that *are* on:

- **`seeps` is a skill, not an error.** `calc_seeps` returns `1.0 - seeps_error`
  ([`score.py:1393`](../packages/evaluate/src/weathergen/evaluate/scores/score.py#L1393)), so
  higher is better and the curve should **fall** with lead time. Read the 8-step plot accordingly.
- **`tp` is in metres per 6 h window**, so the `ets`/`pss`/`fbi` threshold `0.001` is 1 mm/6 h.
  A metric name can appear only once per config, so a threshold sweep means copying the file,
  changing all three thresholds together, and pointing `summary_dir` elsewhere.
- **`psnr` needs a peak value.** The default `pixel_max: 1.0` would mean 1 m/6 h; both configs set
  `0.05` (50 mm/6 h), a realistic extreme for this field.

## Plot selection

Both configs use the list form:

```yaml
evaluation:
  score_plots: [lead_time, bar, scorecard, ratio, heatmap, score_map, timeseries]
```

**When `score_plots` is present the legacy booleans are ignored entirely** — `summary_plots`,
`ratio_plots`, `heat_maps`, `score_cards`, `bar_plots`, `plot_score_maps` and
`plot_score_init_timeseries` all become dead keys
(`config_compat.parse_score_plots`). Older IMERG configs in this repo still use the boolean form;
do not mix the two in one file.

Supported values: `lead_time`, `ratio`, `heatmap`, `scorecard`, `bar`, `qq`, `rank_histogram`,
`score_map`, `score_animation`, `timeseries`.

Two non-obvious couplings:

- `lead_time` also gates **psd_plots** and the **qq_analysis quantile plots**. Without it neither
  is produced, no matter what is in `metrics`.
- `ratio` needs two runs to divide, so it is enabled in the 2-step config and commented out in the
  8-step one.

Per-stream data plots use the same list form: `data_plots: [maps, target, histograms, animations]`.

## Sample matching — the reason the 2-step config exists

`xfaps4wq` and `r15j90ns` emitted their initialisations in **different orders**, so equal sample
indices are different forecasts:

| init time | `xfaps4wq` | `r15j90ns` |
|---|---|---|
| 2023-06-01T12 | 0 | 0 |
| 2023-06-01T18 | 6 | 10 |
| 2023-06-02T00 | 12 | 20 |
| 2023-06-02T06 | 18 | 30 |

The comparison that was circulated earlier applied `sample: "0-63"` to both and therefore averaged
them over different sets of forecasts. Only **37** of those 64 initialisations are shared; the two
explicit index lists in the config select exactly those, in the same order, so element *k* of one
list and element *k* of the other are the same forecast.

Re-derive and check the lists against the stores with:

```bash
uv run --offline python scripts/imerg_presentation_figures.py --verify-matched-inits
```

`run_evaluation` **replaces** `default_streams` wholesale when a run defines `streams` (it does not
deep-merge), which is why each per-run block in the config is complete rather than an override.

The 8-step config needs no such treatment: `z71y2ik8`'s 10 initialisations are already indices 0–9
in order. The commented-out `yz3h0kyn` baseline block at the bottom of that file *does*, and
carries its own list.
