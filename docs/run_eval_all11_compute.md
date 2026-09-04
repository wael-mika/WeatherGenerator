# Running `eval_config_imerg_scores_all11.yml` on a compute node

## Why it fails on a login node

The config's own header measures **~63 min and ~650 GB peak RSS per run** at
`max_workers: 16`. A login node has neither the memory nor (currently) the
per-user thread headroom, so it dies before producing anything.

Santis compute nodes have **870 GB and 288 CPUs**, so one node fits — but the
platform script `WeatherGenerator-private/hpc/santis/weathergen_slurm.sh`
declares `#SBATCH --exclusive --mem=450G`, which is *below* the requirement.
`--mem=0` on the sbatch command line overrides that directive and requests all
of the node's memory.

## The command

    cd /users/walmikae/weathergen/WeatherGenerator

    ../WeatherGenerator-private/hpc/launch-slurm.py \
        --stage evaluation \
        --eval-config config/evaluate/eval_config_imerg_scores_all11.yml \
        --run-ids m745z8wi tvmzluy6 fvenw2nq xt069k3n b96m0co5 csuff3rz \
                  sctxzwwe lf7aj7df ce4rujvd han9w7yt hmw4u8bu \
        --account=ch17 \
        --no-register \
        --mem=0 --time=08:00:00

Notes on each part:

* `--run-ids` **filters** the config's `run_ids:` block, it does not add to it.
  Every id listed must already be in the yaml (all eleven are). Omit the flag
  to evaluate all of them.
* `--no-register` is **required**, not optional. The launcher defaults to
  `--register`, which sets `--push-metrics`, and the MLflow push block
  instantiates the abstract `WeatherGenReader` directly
  (`run_evaluation.py:400`), raising
  `TypeError: Can't instantiate abstract class ... 'get_ensemble', 'get_forecast_steps', 'get_samples'`.
  It runs *before* the summary step, so it kills the job after the scores are
  computed but before anything is written.
* `--mem=0` and `--time=...` are not launcher options: `parse_known_args()`
  forwards anything it does not recognise straight to `sbatch`, and those land
  after the launcher's own flags, so they win over the script's `#SBATCH`
  directives.
* Time: eight of the eleven runs already have rmse/mae/bias/seeps cached in
  `results/<run>/evaluation/` (a cache hit is near-instant), but `ets` is new
  for every run and m745z8wi, han9w7yt and hmw4u8bu have nothing cached. Budget
  roughly three full passes, ~3 h; 8 h leaves margin.

## If it still runs out of memory

`max_workers` is the memory dial. Halve it rather than reducing the sample list,
which would change what is being measured:

    --options evaluation.max_workers=8

Or evaluate in batches, which caps peak usage per job and lets the cached runs
finish instantly:

    for r in m745z8wi han9w7yt hmw4u8bu; do
      ../WeatherGenerator-private/hpc/launch-slurm.py \
          --stage evaluation \
          --eval-config config/evaluate/eval_config_imerg_scores_all11.yml \
          --run-ids $r --account=ch17 --no-register --mem=0 --time=03:00:00
    done

## Where the output goes

Per-run score JSONs in `results/<run>/evaluation/`. All plotting is disabled in
this config, so nothing is written to `summary_dir`.

The launcher snapshots the repo and runs from
`/iopsstor/scratch/cscs/<user>/slurm/slurm_weathergen_<runid>_dir/`, so logs are
under `<snapshot>/logs/<runid>/weathergen.evaluation.<jobid>.{out,err}` — not in
the working copy.
