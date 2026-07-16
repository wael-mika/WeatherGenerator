# CERRA Sharpness Campaign — configs & run registry

Goal: sharp, correctly-scaled high-frequency CERRA fields from the ERA5->CERRA downscale.
Docs: `playground/docs/cerra_sharpness_plan.md` (roadmap), `playground/docs/sf_finetune_explained.md`
(loss math + why the SF fine-tune worked), `.claude/branch_memory.md` (full session history).

## Run registry

| Config (this dir) | Train run | Inference | Result (one line) |
|---|---|---|---|
| `config_latent_upsampling_cerra.yml` | vbm9r3om | p9yamxb2 | K16 MSE pretrain, 64 ep. Blur baseline: SF 3.86 (~14% fine-scale variance), tp max 21.6/41.2 |
| `config_ft_structure_vbm9r3om.yml` | qeyws5sj | lduzq3y8 | +0.05·SF ft, ~16 ep. SF 3.86→~1.0 (~37% variance), visibly sharp; RMSE +4-7% (expected) |
| `config_quantile_cerra.yml` | ixrvotpi | w125yex0 | 16 pinball quantile heads, 48 ep, no upsampler. Best RMSE all channels (quantile mean); tail fully recovered (41.07/41.24); central products blur-level |
| `config_pretrain_mse_noup.yml` | mnb4hkd7 | ymnbchza | Exp A stage 1: K1 (no upsampler). BEST RMSE all channels (tp .544); SF fingerprint identical to K16 arms ⇒ upsampler dead |
| `config_pretrain_mse_K16_deep4.yml` | fc7kn805 | pur2jxnp | Exp B stage 1: K16 4-block (true expansion). No sharpness gain over K1/1-block; RMSE between them |
| `config_ft_structure_generic.yml` | — | — | Stage 2 overlay for ANY 64-ep/ws8 MSE pretrain (Exp A/B); desc via --options |
| `config_ft_quantile_sf_ixrvotpi.yml` | — | — | Step 3: pinball + per-sorted-member SF ft of ixrvotpi |
| `config_quantile_sf_cerra.yml` | — | — | Step 4 flagship (2c): fresh joint pinball + SF-on-median |

## Launch commands (all training on 2 nodes = world_size 8)

```bash
# Exp A / Exp B pretrains (fresh):
../WeatherGenerator-private/hpc/launch-slurm.py --config config/cerra_sharpness/config_pretrain_mse_noup.yml --nodes 2
../WeatherGenerator-private/hpc/launch-slurm.py --config config/cerra_sharpness/config_pretrain_mse_K16_deep4.yml --nodes 2

# Stage-2 SF fine-tune of a finished 64-ep pretrain (fill in run id + desc):
../WeatherGenerator-private/hpc/launch-slurm.py --from-run-id <PRETRAIN_ID> \
    --config config/cerra_sharpness/config_ft_structure_generic.yml \
    --options general.desc=ft_structure_sf005_<ARM> --nodes 2

# Step 3: per-member SF fine-tune of the quantile run:
../WeatherGenerator-private/hpc/launch-slurm.py --from-run-id ixrvotpi \
    --config config/cerra_sharpness/config_ft_quantile_sf_ixrvotpi.yml --nodes 2

# Step 4 (after Step-3 readout):
../WeatherGenerator-private/hpc/launch-slurm.py --config config/cerra_sharpness/config_quantile_sf_cerra.yml --nodes 2
```

## Conventions (hard-won — do not skip)

- **Smoke every config on 1 GPU before slurm** (8 samples, workers 0; for continuations add
  `--options general.istep=0`, real runs must NOT set istep):
  `uv run train --config <cfg> --options training_config.num_mini_epochs=1 training_config.samples_per_mini_epoch=8 validation_config.samples_per_mini_epoch=2 data_loading.num_workers=0 train_logging.checkpoint=999999`
- **Continuations must run at world_size 8** — the mini-epoch resume arithmetic
  (trainer.py:368-380) depends on it; `num_mini_epochs` in FT overlays = pretrain epochs + ft epochs.
- **Inference on a SINGLE rank** for date-aligned sample strides across runs
  (multi-rank inference interleaves samples across rank files; only rank0000 is read).
  `uv run inference --from-run-id <id> --options streams.CERRA.max_num_targets=-1 test_config.samples_per_mini_epoch=16 test_config.output.num_samples=16`
  Run it from the training run's slurm snapshot dir (`uv sync --all-packages --extra gpu --offline`
  there first) — the current branch checkout may lack the run's code.
- **Quantile heads do NOT self-order** — always build products from per-point SORTED members
  (median = sorted 7/8 of 16; extremes = sorted 15). `LossStructureFunction` reduce modes:
  `mean` (comparability metric), `median` (sorted middle), `members` (per-sorted-member training),
  int (raw member — genuine-ensemble runs only).
- **Never judge these runs on RMSE alone** — sharp fields pay a double penalty; read SF metric +
  histograms + maps together. Keep the validation SF block (tp, reduce: mean) identical in every
  config: it is the single number comparable across the whole campaign since vbm9r3om.
- **Ablation verdict (2026-07-16, `eval_config_pretrain_ablation_cerra.yml`)**: on 8 date-matched
  windows, RMSE tp/2t/10si = K1 .544/1.631/1.467 < deep4 .545/1.673/1.491 < 1-blk .560/1.739/1.522;
  SF ratios identical within noise across all three pure-MSE arms. The upsampler contributes
  nothing at any depth — drop `decode_latent_expand` from future configs. The blur is entirely
  the MSE conditional-mean objective. HEALPix tiling persists in K1 ⇒ it comes from the per-cell
  decode partition, not the upsampler.
- **Soft-blend decode (`decode_soft_blend_k`)**: stream-level flag (default off) that decodes
  boundary-zone target points under their k nearest cells and blends predictions with
  continuous distance weights (`softmax(-d/(tau*cell_spacing))`, `decode_soft_blend_tau`,
  default 0.25) — the anti-tiling mechanism. No new parameters; can be enabled at inference on
  any existing checkpoint: `--options streams.CERRA.decode_soft_blend_k=3`. Seam metric:
  `playground/scripts/seam_metric.py` (baseline seam score tp ≈ 5.6-7.4 across all runs).
- Eval configs live in `config/evaluate/` (`eval_config_ft_vs_base_cerra.yml`,
  `eval_config_three_way_cerra.yml`, `eval_config_pretrain_ablation_cerra.yml` — the latter
  shows the per-run `streams:` override pattern for date-matching mixed-stride inferences);
  streams in `config/streams/{latent_upsampling_cerra,
  cerra_mse_noup, quantile_cerra}/`.
