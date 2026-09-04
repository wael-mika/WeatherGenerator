# Branch memory — wm/dev/raina_crps_dec

## Goal
Import the CRPS (ensemble/probabilistic) decoder onto the raina line so it can run with raina
configs/streams. Source: ecmwf/WeatherGenerator PR #2578.

## Base / provenance (2026-07-02)
Branched from `origin/wm/dev/raina_dev` (up-to-date remote, ~develop #2376 + raina work). Cherry-
picked the 3 PR #2578 commits (f086728a / 374d8826 / 99c38f23) on top. The PR was built on a newer
develop (~#2565), which is why the code files conflicted. PR fetched locally as branch `pr-2578`.
Not yet pushed to origin.

## Conflict resolutions
- `model.py`: kept raina's `self.stream_names` (branch has no `self.streams` attribute the newer
  develop introduced). CRPS decoder core (`EnsPredictionHead`, latent-perturbation ensemble block)
  auto-merged cleanly.
- `loss_module_physical.py`: kept raina's `_parse_loss_fcts` helper (per-loss `args` support) and
  taught it to skip `None` params and the `dynamic_loss` key — required because
  `config_forecasting_crps.yml` uses `loss_fcts: {"mse": null, "kernel_crps": {}}` (the `null`
  would crash the helper's `params.get(...)`). This mirrors the guard the PR's inline version had.
- `utils.py`: took the PR's `if v is not None` guard in `get_target_idxs_from_cfg`, reflowed to
  satisfy the 100-char limit.

## Notes
- `EnsPredictionHead` already existed in this branch's `engines.py` (~line 644); the PR did NOT
  touch engines.py, so no import gap. Verified: model + loss modules import cleanly, ruff passes.
- New config: `config/config_forecasting_crps.yml` (kernel_crps loss, mse weight null).

## Merged origin/develop #2565 (2026-07-02)
Chose MERGE over rebase: develop had architecturally refactored the same subsystems the ~20 raina
commits touch, so a 23-commit rebase would re-conflict ~20×. Merge resolved it once (3 conflicts).
Backup branch: `backup/raina_crps_dec-prerebase-20260702-221837`.
- `multi_stream_data_sampler.py`: develop replaced explicit `import DataReaderX` + per-type `case`
  branches with a **registry** (`get_extra_reader(type_name)` fallback in `readers_extra/registry.py`,
  which knows fesom/iconart/grep/iconesm/cams/mesh/anemoi_operan). Resolution: DROPPED our `fesom`
  import+case (develop moved fesom to `packages/readers_extra` in #2454; old
  `datasets/data_reader_fesom.py` path is gone — registry fallback now serves fesom). KEPT explicit
  cases for our readers `imerg/radklim/icon_dream/anemoi_transform` (they live in `datasets/`, NOT
  registered). Also dropped our redundant two-step filename re-selection (develop's single
  `next(...)` lookup supersedes it).
- `loss_module_physical.py`: kept our `_parse_loss_fcts` (args + skips `null`/`dynamic_loss`) — it's
  a superset of develop's inline parse AND is used by our per-stream `loss_fcts` override path.
  Had to REORDER: develop added `self.dynamic_loss_ema = DynamicLossEMA(...)` inside `__init__` after
  the loss_fcts assignment, so `_parse_loss_fcts` must be a method AFTER `__init__`, not spliced
  into it. Kept our NaN-averaging fix (skips NaN incl. torch.Tensor, `count>0` guard) over develop's
  0.0-substitution. Fixed one develop E501 (dynamic_loss line) via `offset_key` local.
- `validation_io.py`: took develop's simplified spoof branch (`targets = ...`; following code derives
  shapes from `targets`, so our `preds`/`preds_shape` block was dead here). NOTE: on the sibling
  latent_upsampling branch the user asked to keep the old preds approach commented out — did NOT do
  that here (different branch, merge chosen for cleanliness); recoverable from backup if needed.
- Also reflowed a pre-existing raina E501 in `data_reader_anemoi.py` (the tp/IMERG channel-recovery
  `if`) that only surfaced under lint-check now.
- Verified: no markers, ruff clean on all merge-changed files, sampler/loss/model/validation_io all
  import at runtime, registry routes `fesom`→DataReaderFesom and `imerg`→None (explicit case). Merge
  commit is NOT pushed.

## IMERG diagnostic-decoder probe configs for pretrained runs ipnpqow8/rck9wgm7 (2026-07-02)
Added `config/raina_config/config_finetune_imerg_diag_crps_{ipnpqow8,rck9wgm7}.yml` — `train_continue`
overrides that add IMERG-via-anemoi as a new diagnostic stream on top of these two pretrained models
(ERA5+operan / ERA5+Obs→OperAn+obs, trained on `clessig/develop/exps_03062026`) and freeze the whole
backbone so only the new IMERG decoder trains (linear-probe-style test of frozen latents' precip
signal). Uses the CRPS recipe (`pred_head.ens_size: 4` + `loss_fcts: {kernel_crps, mse}`), same combo
as `config/streams/raina_era5_icon_cerra/cerra_tp.yml` / `raina_imerg_forecast/imerg_30min.yml`.
- **Could not read the actual saved configs for these two run_ids** — no trace in git history; the
  checkpoints DO exist on the real cluster (`/iopsstor/scratch/cscs/shickman/models/{id}/`, confirmed
  via `stat` → `Permission denied`, not "no such file"), but not readable with this session's perms.
  `path_shared_working_dir` for Alps/Clariden is `/iopsstor/scratch/cscs/shickman/` (found via a
  `slurm_weathergen_<run_id>_dir` snapshot dir under `/iopsstor/scratch/cscs/thunter/slurm/`, which
  has the full launcher-snapshotted repo + `WeatherGenerator-private/hpc/alps-clariden/config/paths.yml`).
- **Key discovery — don't reconstruct the pretrained `streams_directory`.** Normally adding a stream
  means copying the whole pretrained stream set into a new dir, because `load_merge_configs`
  (`packages/common/.../config.py:438-442`) wipes `base_config.streams` entirely if ANY override
  config sets `streams_directory`. Instead: omit `streams_directory` from the override entirely and
  put the new stream inline under a `streams:` key. Verified directly: `OmegaConf.merge` recursively
  unions nested dict keys rather than replacing them, so `streams: {IMERG_ANEMOI: {...}}` in the
  override gets ADDED to whatever streams the checkpoint's saved config actually has, without needing
  to know/copy them. This generalizes to any "add one diagnostic stream to a pretrained model without
  touching the rest" finetune — much safer than the `streams_directory`-copy pattern used elsewhere
  (`config/streams/era5_iasi_finetuning/`) when the exact pretrained stream set is unknown/unverified.
  Caveat: breaks if any OTHER `--config` passed alongside also sets `streams_directory`.
- `freeze_modules: ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*latent.*|.*ERA5.*"` (same pattern
  as `config/raina_config/config_forecasting_finetuning_raina.yml`, per user steer) freezes the
  encoder body + q_cells + latent + the pretrained ERA5 decoder, but does NOT freeze the forecast
  engine (`fe_*`/`forecast_engine`) — flagged to the user as a possible gap if "freeze totally" was
  meant to include rollout dynamics too; not added unprompted since it deviates from the referenced
  file's convention.
- CRPS mechanics (this branch): NOT a distinct `decoder_type` — it's `EnsPredictionHead` (existing,
  `ens_size` independent MLP heads) + `kernel_crps` loss, requiring `decoder_type:
  PerceiverIOCoordConditioning` (NOT `Linear`, which bypasses `EnsPredictionHead` entirely —
  `model.py:909-924`) and per-stream `pred_head.ens_size > 1`. Optional global
  `decoder_ens_latent_perturbation.num_members` adds a second, model-wide stochastic ensemble axis
  (multiplies with `ens_size`) but affects every stream's decode step, not just one — the two new
  configs rely on `ens_size` alone, matching the established per-stream-only pattern.
- Both new files are unpushed/untracked-by-git-status local additions (only 2 new files, no other
  changes this session on this branch).

## Review of the IMERG probe configs (2026-07-03) — corrections to the session above
- **The saved run configs ARE readable** (the "Permission denied" was shickman's dir, which the
  code never uses): `get_path_model()` → `/iopsstor/scratch/cscs/thunter/shared_work/models/
  {rck9wgm7,ipnpqow8}/model_<id>_latest.json`. Verified against the REAL configs: merge works
  end-to-end (all inherited streams + IMERG_ANEMOI survive), `decoder_type` IS
  PerceiverIOCoordConditioning, stream_id 40 does NOT collide (real ids 0,1,10-14,20,30), load is
  strict=False. Both pretrained 2016-01-01→2022-12-31 (val Oct-Dec 2023), fe_num_blocks 16,
  with_fsdp True. ipnpqow8 streams: ERA5_in(0)+ERA5(1) only.
- **Post-session drift causing issues**: (1) ipnpqow8 config flipped to `streams_directory`
  (wipes inherited streams) and a hand-written `era5.yml` added to `config/streams/
  imerg_diag_crps/` to compensate — its `nominal_time_mapping` is WRONG (6→9,18→21; checkpoint
  truth 6→11,18→23) and the ERA5 diagnostic stream is dropped. (2) era5.yml ALSO poisons the
  rck9wgm7 run: `extra_streams_directory` merges it OVER the real inherited ERA5_in. (3)
  imerg_anemoi.yml still in import-test state: ens_size 1, kernel_crps commented out → plain MSE;
  re-enabling crps with ens_size 1 hits the `ens_size > 1` assert.
- **Other gaps**: finetune `start_date: 2000-01-01` but ERA5_in v6 zarr covers only 2016-2023
  (pre-2016 ipnpqow8 samples have NO source at all) → use 2016-01-01. Freeze regex needs
  `.*encoder.*` (base pattern misses `encoder.embed_engine.embeds.<obs>` and
  `encoder.ae_aggregation_engine`); present in rck9wgm7's file, missing in ipnpqow8's.
  `forecast_engine` (16 blocks) unfrozen in both — not a pure probe unless `.*forecast.*` added.
- `extra_streams_directory` is IGNORED by plain `train` (run_train.py:183 reloads from
  `streams_directory`); only `train_continue` honors it. `source: []` diagnostic streams are safe
  (EmbeddingEngine assigns `nn.Identity()`).

## REDESIGN after user clarification (2026-07-03) — supersedes the two sections above
User's actual intent: do NOT inherit the pretraining output stream. Finetune = pretraining INPUT
(forcing) streams reproduced exactly + IMERG as the ONLY diagnostic/output stream, so the model
trains purely on IMERG (`streams_directory` wipe is therefore the RIGHT tool here, not
`extra_streams_directory`). Also: CRPS must follow PR #2578's recipe = model-wide
`decoder_ens_latent_perturbation` (latent Gaussian noise at decode, learnable sigma) +
`kernel_crps` global loss with `"mse": null` (skipped by `_parse_loss_fcts`), pred_head
ens_size stays 1 — NOT per-stream EnsPredictionHead ens_size. MSE-vs-CRPS comparison → 4 configs.
- **Generated stream dirs** `config/streams/imerg_diag_{ipnpqow8,rck9wgm7}/` VERBATIM from the
  readable saved checkpoint configs via scratchpad/gen_stream_dirs.py (drops `name` +
  `{train,val}_{source,target}_channels` derived keys, unwraps `${timedelta:}` frequency).
  ipnpqow8: ERA5_in only; rck9wgm7: ERA5_in + IASI + 5 geo + SurfaceCombined. Old hand-written
  `imerg_diag_crps/` dir (wrong nominal_time_mapping 6→9/18→21; truth 6→11/18→23) moved to
  scratchpad backup.
- **4 configs** `config/raina_config/config_finetune_imerg_diag_{mse,crps}_{ipnpqow8,rck9wgm7}.yml`:
  64×4096 samples (user choice), lr_max 1e-4, dates 2016-01-01→2022-12-31 (ERA5_in v6 zarr
  coverage), forecast 6h×4 steps, num_members 4 (user choice; PR default was 2). TWO
  freeze_modules lines per config — active = fe frozen (pure probe, incl. `.*forecast.*` +
  `.*encoder.*`), commented = fe trainable (reference convention); user toggles by comment.
- **All 4 verified against the real checkpoints** (scratchpad/smoke_all4.py): exact stream sets,
  IMERG sole output, losses parse to [mse]/[kernel_crps], perturbation knobs, no id collisions.
  Learnable CRPS sigma (`latent_perturbation_log_sigma`) is a bare top-level nn.Parameter →
  never caught by freeze_modules (named_modules only; q_cells is the sole special case) → trains
  under both freeze options, as wanted.
- The `extra_streams_directory` code change (config.py + default_config.yml) is now UNUSED by
  these configs — generic feature, user may keep or revert.

## Launcher ships CHECKPOINT-ERA code for train_continue — why CRPS jobs fail (2026-07-03)
- Job i2q9vihz (crps_ipnpqow8) crashed in `LossPhysical.__init__`: `params.get("weight")` on None
  — the snapshot's loss module is the OLD June-era version with no `"mse": null` guard and no
  `_parse_loss_fcts`. Snapshot model.py has NO `decoder_ens_latent_perturbation` at all.
- Root cause in `WeatherGenerator-private/hpc/launch-slurm.py` (~line 1266): for a stage with
  `--from-run-id X != run_id`, CODE is copied from `copy_root_dir/slurm_weathergen_X_dir/
  WeatherGenerator` (the pretraining run's snapshot, i.e. the code that TRAINED the checkpoint),
  while CONFIGS are refreshed from the home clone (`copy_all_configs(wgen_dir_home, ...)`). So
  new configs + old code — this branch's CRPS code NEVER ships on train_continue. No CLI flag to
  override (checked argparse). copy_root_dir = thunter's shared slurm dir
  (`path_shared_slurm_dir`), overridable via `--dir`.
- GOOD news from the same log: checkpoint loads under the probe design with missing keys =
  ONLY `embed_target_coords/target_token_engines/pred_heads .IMERG_ANEMOI.*` — stream
  reconstruction + wipe design works.
- Verified: this branch's model attribute tree is IDENTICAL to the checkpoint-era snapshot except
  the 3 new CRPS attrs → shipping branch code keeps checkpoint load clean
  (+`latent_perturbation_log_sigma` expected-missing when sigma_learnable).
- Fix options: (a) add a `--code-from-home` flag to launch-slurm.py forcing
  `wgen_dir = wgen_private_dir.parent/"WeatherGenerator"` even with from_run_id (function
  `run_pipeline`, selection at ~1266); (b) `--dir <own slurm root>` + pre-seed
  `slurm_weathergen_<from_run_id>_dir/WeatherGenerator` with this branch's code (no launcher edit,
  but results/artifacts land outside the shared dir). MSE variant would RUN on old code but
  silently without any branch fixes; CRPS variant REQUIRES branch code.

## `<id>_raina` run copies — the fix for checkpoint-era code shipping (2026-07-04)
Per user decision (instead of patching the launcher): created parallel "runs" `ipnpqow8_raina`
and `rck9wgm7_raina` so `train_continue --from-run-id <id>_raina` ships THIS branch's code:
- `models/<id>_raina/` (thunter's shared_work): `model_<id>_raina_latest.json` = original json
  with `general.run_id` renamed; `<id>_raina_latest.chkpt` = HARDLINK to the original 3.3G chkpt
  (same fs — do not delete the original run's chkpt or the link survives but provenance is lost).
- `slurm/slurm_weathergen_<id>_raina_dir/WeatherGenerator` (thunter's slurm dir): this branch's
  `git ls-files` WORKING-TREE content (incl. uncommitted fixes) + all config/**.y*ml (incl.
  untracked) + `tracked_files.json` (required — launcher falls back to it when the dir has no
  .git). No .venv needed (--link-venv is opt-in; actions.sh sync builds one per job).
- Generator: scratchpad/seed_raina_runs.py; verify: scratchpad/verify_raina_runs.py (all 4
  variant×run combos passed: merge, run_id, null-safe loss parse, CRPS code present in seeded
  dir, manifest complete). Config headers updated to `--from-run-id <id>_raina`.
- IMPORTANT: the seeded dirs are a SNAPSHOT of the branch at 2026-07-04 — re-run
  seed_raina_runs.py (it skips existing dirs; delete them first) after further code changes,
  or new fixes won't ship. Same applies if stream/config files change (launcher takes configs
  from home clone at submit, so config-only changes do NOT need re-seeding).

## CRPS implementation review + sigma-load fix (2026-07-04, pre-submission)
Reviewed the PR #2578 CRPS code end-to-end. VERIFIED CORRECT (scratchpad/test_crps_numerics.py):
kernel_crps matches brute-force reference for E=2/4/8, fair & unfair coefficients right for the
once-per-unordered-pair loop, NaN→0 contribution, perfect forecast→0; predict_decoders
member-major cat+reshape/permute == independent per-member decode. Chain also verified: noise
added OUTSIDE checkpoint regions (no resample on recompute), `mse: null` skipped in BOTH
masking.py:252 and _parse_loss_fcts, kernel_crps tagged via global_params["loss"], lp_loss
averages ens members for mse, per-member TTE loop is deliberate (CUDA grid limit — don't batch).
- **REAL BUG FOUND+FIXED (model_interface.load_model)**: `latent_perturbation_log_sigma` is a
  bare param on the model ROOT (which IS fully_shard'ed, line ~147). On train_continue from a
  pre-CRPS checkpoint it lands in mkeys → the module-init loop did `all_modules[path]` →
  KeyError crash under FSDP; AND sigma_init was never applied on ANY continue path
  (reset_parameters not called) → noise std exp(0)=1.0 instead of 0.2. Fix: sharded branch
  synthesizes the sigma entry into the state dict pre-load (distribute_tensor to the meta
  param's mesh/placements, like checkpoint params); plain branch fills post-load; module-init
  loop now warns+skips non-module keys. Tested (scratchpad/test_sigma_init_load.py): init
  applied, sigma-less models untouched, checkpoint-provided sigma NOT overwritten.
- _raina slurm dirs RE-SEEDED with the fix and re-verified (delete dirs + rerun seeder;
  models/ dirs untouched).

## Storage check (2026-07-04): iopsstor 95% FULL is the likely culprit; inodes fine
- `/iopsstor/scratch/cscs`: **2.8/3.0 PB used (95%), 171 TB free** — machine-wide; Lustre read/
  alloc performance collapses near capacity. Inodes healthy (811M/3.4B = 24%).
- Team footprint: grp ch17 on iopsstor = 212 TiB / 30.7M files (no quota limit). 3490 slurm dirs
  but per-job .venvs ARE cleaned post-job (post_train.sh:42 rm -rf; cleanup.py --slurm) — not an
  inode bomb.
- SEPARATE issue to flag to team: `/capstor/store/cscs/userlab/ch17` is at **95.2% of its FILES
  quota (9.5M/10M)** — dataset writes there will start failing soon. Different FS from the venv
  slowness.

## 4th failure s593ij3o — FS degradation confirmed as the story (2026-07-04)
All four 2-node jobs: v02r8pe7 nid[005254-255] stuck=005254; spk3xsft nid[005226,005256]
stuck=005256; tltujaws nid[005226,005228] stuck=005228; s593ij3o same pair stuck=005228 again.
THREE different stuck nodes → not one bad node: iopsstor/Lustre cold reads of the venv are
wedging on most nodes this morning, and it's worsening (warm nid005226 imports slowed
3m46s→5m40s between 07:53 and 08:32). nid005228 never finished imports in 18 min (twice).
- Raising the rendezvous timeout only helps if imports EVENTUALLY finish — unproven (all jobs
  died ≤18 min). The wedged nodes may need FS recovery, not a bigger timeout.
- **Single-node jobs are immune**: all ranks share the node → import at the same speed → reach
  the rendezvous together (why interactive worked). `--nodes 1` is the safe smoke-test path.
- To dodge known-stuck nodes on multi-node: env `SBATCH_EXCLUDE="nid005228,nid005254,nid005256"`
  before launch-slurm.py (sbatch reads SBATCH_* env). Longer-term fix candidates: staged
  timeout raise + per-node venv cache warm (read .venv OUTSIDE the DDP window, e.g. srun
  --ntasks-per-node=1 'tar cf /dev/null .venv' before the python srun in weathergen_slurm.sh).

## Sonnet's uncommitted launch-slurm{,-single}.py changes — checked, UNRELATED to the hangs
Two changes: (1) pure refactor of the config-path relocation loop into `_relocate_config_path`
(logic identical, old code left commented); (2) `--base-config` files are now ALSO relocated
into the staged copy (previously read live at runtime — likely Sonnet's fix after the yyx9vgju
--base-config misuse). Our train_continue jobs pass no --base-config → path (2) never executes.
Neither touches srun/env/venv/timing; tltujaws's healthy node passed "configs loaded", and the
failing node hung in IMPORTS (before any config is read). Exonerated.

## 3rd failure tltujaws CONFIRMS diagnosis via banners; timeout fix staged NOT deployed (2026-07-04)
tltujaws (crps_ipnpqow8, 2 nodes, --time 30): banners show node 1 (nid005226, warm from prior
job) imported in 3m46s and entered DDP init; node 2 (nid005228, cold) printed "python started"
then NOTHING — stuck >10 min INSIDE the heavy import block. Rank 0's store gave up at its
hard-coded 600 s ("4/8 clients joined"). --time 30 didn't help: the binding constraint is the
600 s rendezvous timeout, not wall time. Root cause: pathologically slow cold imports from the
Lustre venv on some nodes (warm login-node import = 48 s; cold ~4 min normal; >10 min when
Lustre is degraded).
- **Fix STAGED in working tree only (user asked NOT to re-seed until confirmed)**:
  trainer_base.init_ddp now passes `timeout=timedelta(seconds=cf.get("ddp_init_timeout_seconds",
  1800))` to init_process_group (also becomes the NCCL collective/watchdog timeout — training
  hangs take 30 min to fail, acceptable tradeoff); key documented in default_config.yml.
  Seeded _raina dirs have banners but NOT this fix. Re-seed after user confirms.
- Confirmation options: resubmit as-is and check when the cold node's "imports done" banner
  fires vs the 600 s store death; or py-spy/proc-stack the hung python via
  `srun --overlap --jobid <id> -w <node>`; or 2-node import-timing probe using an existing
  job venv.

## 2nd 2-node failure spk3xsft — MIRRORED, so NOT a bad node; startup banners added (2026-07-04)
crps_ipnpqow8 2-node test spk3xsft (07:16-07:31, CONCURRENT with v02r8pe7): this time node 1
(nid005226) was fine — rank 0's store reported "4/8 clients joined" — and node 2's 4 ranks were
silent the whole 14 min. Complementary to v02r8pe7 (there node 1 was silent). Two jobs, two
different silent nodes, same time window → systematic slow/hung pre-init startup on one node,
NOT one bad node and NOT the configs (CRPS config never got evaluated). Prime suspect: cold
`import weathergen` from the fresh per-job venv on Lustre — measured 48 s WARM on the login
node; the two jobs ran concurrently and hammered the same FS.
- **Startup banners added to run_train.py** (`_startup_banner`, print+flush; file-level
  `# ruff: noqa: E402, T201`): before heavy imports, after imports, before config load, before
  torch/DDP init in run_continue. Next hang will show exactly which phase the silent node is in.
- _raina dirs re-seeded with the banners. Advise: --time 30 for tests, submit test jobs
  sequentially not simultaneously.

## 2-node test v02r8pe7 TIMEOUT — infrastructure, not config (2026-07-04)
First 2-node test of mse_rck9wgm7 via `launch-slurm.py --from-run-id rck9wgm7_raina --nodes 2
--time 15`: ranks 4-7 (node 2) timed out after 600s connecting to the TCPStore rendezvous at
nid005254:29514; ranks 0-3 (node 1 = master) produced ZERO output for 15 min despite `python -u`
(srun uses -u, so it's a real hang, not buffering) → rank 0 never reached
trainer_base.py:99 (the pre-init_process_group print). Everything before that line is silent by
design (INFO logging not yet configured). Node 2 ran the IDENTICAL seeded code through
config-merge into rendezvous → the _raina setup/code is fine; failure is node-local on
nid005254 (CUDA-init or FS hang; node showed healthy in sinfo afterwards). Notes:
- sbatch-level .out/.err (echoes, NCCL_DEBUG) were LOST — `logs/weathergen-%x.%j.out` is
  relative to the sbatch cwd, which for these jobs has no logs/ dir. The srun-redirect file
  (`logs/<run_id>/output.<jobid>.txt` in the SHARED logs symlink) is all we get.
- 15-min test window is too tight: node 2 needed ~4 min just to import/start, and the TCPStore
  connect timeout alone is 600s. Use --time 30 for smoke tests.
- Action: resubmit as-is (fresh nodes); if it recurs on the same node, --exclude it; if it
  recurs on different nodes, instrument an early per-rank print before init_torch.

## Final pre-submission state (2026-07-04, all verified against real checkpoints)
Experiment matrix as submitted (user's own tweaks after my writes): ipnpqow8 pair = fsteps 2,
crps num_members 2; rck9wgm7 pair = fsteps 4, crps num_members 4 — consistent WITHIN each
run_id pair, so MSE-vs-CRPS is clean per model; the two run_ids are not cross-comparable on
rollout depth (intentional). mse_rck9wgm7 has explicit train_logging terminal 50 (others
inherit 100). All 4: fe frozen (option a active), streams merge to inputs+IMERG only, ntm
correct, no id collisions, kernel_crps parses. Seeded _raina code dirs re-checked: zero drift
vs working tree (src/, packages/, scripts/, lockfiles).

## ERA5 loss appearing in mse_rck9wgm7 run oesyjclc — same launcher root cause (2026-07-04)
- Job oesyjclc (pre-_raina, from-run-id rck9wgm7) logged `LossPhysical.ERA5.mse.avg` despite our
  streams_directory design dropping ERA5. Cause: the checkpoint-era snapshot's
  `load_merge_configs` has NO wipe logic (`base_config.streams = None` on override
  streams_directory was added later) → override streams merged ADDITIVELY onto the checkpoint's
  streams → inherited ERA5 diagnostic stream survived and got a loss term.
- Evidence the rest worked even there: ERA5.mse was bit-identical across steps (2.2977E-02 —
  fully frozen path, freeze regex effective) while IMERG_ANEMOI.mse fell 0.85→0.65 (new decoder
  learning). The ERA5 term only diluted loss_avg.
- No fix needed beyond resubmitting from `rck9wgm7_raina` (branch code HAS the wipe; verified
  merge = 8 forcing + IMERG only).

## Forcing-stream spurious "target data is EMPTY, spoofing" warnings (2026-07-03)
- Symptom: running the imerg_diag_crps finetune (ERA5_in `forcing: True` + IMERG diagnostic output)
  spammed `Stream fstep N: target data is EMPTY, spoofing` for pre-2016 windows.
- Cause: `imerg_diag_crps/era5.yml` is a FORCING stream but still declares a 69-channel `target:`
  list, so `get_target_num_channels() > 0` → the old guard in `_get_data_windows`
  (multi_stream_data_sampler.py:~635) tried to collect its targets. IMERG spans 1998-2024 so the run
  time range includes 2003-2015, but the ERA5_in v6 zarr is only 2016-2023 → target genuinely empty
  pre-2016 → warning fired every step. (ERA5 forcing targets are discarded anyway — masking.py:372
  empties the target mask for forcing streams.)
- Fix: guard now also checks `not is_stream_forcing(stream_ds[0].stream_info)` — forcing streams
  never collect/warn on targets regardless of a redundant `target:` list. `is_stream_forcing`
  imported from `weathergen.utils.utils` (already used by masking.py, no circular import).

## Single-node smoke yliqgead (2026-07-04) — startup fine, but wrong from-run-id + too short
- `--from-run-id rck9wgm7` (NOT _raina) + mse_rck9wgm7 config, 1 node, --time 30. Single-node
  startup confirmed immune to the FS import hang (whole pipeline up in ~13 min: venv+imports+
  model+checkpoint load done 09:58).
- Old-snapshot code again → inherited ERA5 output stream loaded (source [], 80 targets) on top
  of the intended 9 streams; its decoder frozen by `.*ERA5.*` so gradients unaffected, but it
  wastes decode compute and pollutes metrics (same as oesyjclc). Must use `_raina` for clean runs.
- Killed at 30-min TIME LIMIT before the first loss line: terminal logging is every 50 batches
  and only ~17 min remained after model build. For smoke tests: --time 60, or override
  `train_logging.terminal=1` to see batch 1 immediately.

## Jul 3 evening chain jobs aazgf3xw/vx4ruken/udlrszyw — same 600s rendezvous failure (pre-banner)
- All three 2-node --chain-jobs 2 submissions (from _raina, before the startup banners were
  seeded) died with the identical c10d TCPStore 600s timeout. Per-rank output shows one whole
  node silent each time: aazgf3xw stuck=nid005249, vx4ruken stuck=nid005250 (the MASTER node —
  node 2's ranks timed out connecting to it), udlrszyw stuck=nid005256 (same node stuck again
  Jul 4 morning). Stuck-node set now: 005228, 005249, 005250, 005254, 005256 → filesystem-wide,
  not bad nodes. These predate the morning failures, so the iopsstor degradation started at
  latest Jul 3 ~15:00. Chain follow-up jobs left no logs (only part1 output exists per run).

## k3hnf391 (Jul 4 14:03, 1 node, rck9wgm7_raina MSE) — setup CLEAN, just needs more walltime
- _raina path fully validated: banners present, imports 2m08s, stream set exactly the intended 9
  (ERA5_in + 7 obs + IMERG, NO inherited ERA5 output stream, no TargetPredictionEngine_ERA5),
  freeze applied, checkpoint loaded ("Continuing run with id=rck9wgm7_raina").
- Killed at 30-min limit ONE LINE after checkpoint load — dataset opening + model build +
  checkpoint load took ~28 min post-imports (vs 13 min for yliqgead same config → FS slowness is
  variable). Smoke tests need --time 60+ and `--options train_logging.terminal=1`.

## y4g6op28 (Jul 5, 1 node, 60 min, rck9wgm7_raina MSE) — FIRST SUCCESSFUL TRAINING RUN
- Fully clean: loss_avg == IMERG_ANEMOI.mse only; missing-key init touched ONLY
  target_token_engines.IMERG_ANEMOI.* + pred_heads.IMERG_ANEMOI.*; loss 0.88 -> 0.66 by step
  465/1024 of mini-epoch 0; lr plateau 2.0e-4 = lr_max 1e-4 x sqrt(4 GPUs) (parallel_scaling
  sqrt — expected, not a bug).
- Checkpoint saved at step 250 (y4g6op28_latest.chkpt, 3.6 GB) → resumable with
  --from-run-id y4g6op28 (its slurm dir has the branch code, so chaining from it is safe).
- Pace: ~3.1 s/step → mini-epoch ≈ 55 min; full 64 x 1024 ≈ 2.5 days on 1 node + validation.
- FS much healthier Jul 5: python start->training-ready in ~2.5 min (imports 1m43s) vs 28 min
  on Jul 4. Job step itself started 11:31; killed at 11:59 by the launcher's SIGTERM (60-min
  wall incl. queue/venv phase), losing steps 250-465 (only latest saved checkpoint survives).

## atmofs03-rebased finetune setup (2026-07-05) — third pretrained model in the comparison
- New pair config_finetune_imerg_diag_{mse,crps}_atmofs03_rebased.yml + streams dir
  config/streams/imerg_diag_atmofs03_rebased/ (era5.yml VERBATIM from checkpoint json with
  `forcing: true` ADDED — in pretraining ERA5 was input AND output; imerg_anemoi.yml
  byte-identical to the other dirs). tp/cp are in source_exclude → no precip leakage.
- atmofs03-rebased facts: single ERA5 stream (aifs-ea v8 zarr, 1979-2023), pretrain fc
  num_steps 9, fe_num_blocks 16, fe_layer_norm_after_blocks [7], fe_impute_latent_noise_std
  1e-4, ae_local_num_blocks 0/dim 2048, ae_global_num_blocks 4, no CRPS decoder in
  checkpoint (sigma seeded from sigma_init by the model_interface missing-key fix).
  Original json's general.run_id is "atmofs03" (not "atmofs03-rebased").
- COMPARABILITY AUDIT found the checkpoints disagree on keys the finetune configs didn't
  pin: grad_clip 1.0 vs 0.8, time_window_step 6h vs 1h, val samples 512 vs 256, val EMA
  halflife 1e-3 vs 600, num_workers 8 vs 10. USER DECISION (Jul 5): the 4 running configs
  stay UNTOUCHED (reverted my pins); only the atmofs03 pair pins grad_clip 0.8 / val 256 /
  EMA 600 / workers 10 to match. time_window_step deliberately stays at atmofs03's native
  6h (ERA5 v8 is 6-hourly; the others sample with 1h step) — accepted deviation, "comparable
  not identical". Effective-merge check confirms all else identical across the 6.
  window_offset_prediction is dead in branch code; model_input section name is a label only
  (do NOT pin it — OmegaConf would merge additively and create two maskers).
- atmofs03-rebased_raina seeded: model dir (checkpoint HARDLINKED, json run_id renamed) +
  slurm code dir copied bit-identical from slurm_weathergen_rck9wgm7_raina_dir (same code as
  the other experiments, i.e. still WITHOUT the staged ddp_init_timeout fix).

## Branch cleanup + commits (2026-07-05)
- Committed in 5 logical commits (e88d3c54..ad5f508b): ruff formatting; forcing-stream target
  skip; sigma_init-on-missing-checkpoint fix; configurable ddp rendezvous timeout
  (trainer_base + default_config, default 1800s); the 6 finetune configs + 3 stream dirs.
- REVERTED as debug/abandoned (per user): run_train.py [startup] banners (still live in the
  seeded _raina snapshot dirs — expected drift) and the unused extra_streams_directory feature
  (config.py + default_config). .claude/ is GITIGNORED (line 20) despite CLAUDE.md claiming
  branch memory is tracked — branch memory cannot be committed.
- Lint note: repo-wide `actions.sh lint-check` fails with ~88 pre-existing errors in committed
  files (e.g. smoke_test_imerg_icon_dream.py prints, plot_cerra E741/T201); the committed diffs
  are format-clean and add no new violations.
- Caution: `git stash pop` inside a compound Bash call failed silently once (partial apply
  look-alike); changes sat in stash@{0}. Always verify `git stash list` after scripted pops.

## nfwefvyx failure (Jul 5): atmofs03-rebased embedder shape mismatch — FIXED
- 2-node startup itself was FINE (both nodes imported in 1m53s — FS healthy Jul 5).
- Crash: encoder.embed_engine.embeds.ERA5.unembed.* [25,512] (checkpoint) vs [22,512] (model).
  Cause: checkpoint-era code matched channel excludes by SUBSTRING ('w_' excluded all 13
  w_<level> channels); PR #2361 (commit 2af44f70) changed it to EXACT match, so replaying
  `source_exclude: [w_, ...]` today leaks 13 w channels: 84 src + 9 geo = 93 -> 2048//93 = 22
  vs pretraining 71+9=80 -> 2048//80 = 25. Block-mode unembed out-dim =
  (num_tokens*ae_local_dim_embed)//num_channels (embeddings.py).
- Fix: era5.yml now pins EXPLICIT source (71) / target (72) lists from the checkpoint's
  derived train_*_channels instead of the excludes. Amended into the configs commit
  (ad5f508b -> 22ec9fd6, unpushed) + synced to the _raina snapshot. GENERAL RULE: when
  reproducing old checkpoint stream configs, always materialize explicit channel lists —
  never trust the original exclude/include patterns under current matching semantics.
- User's local toggle: mse_ipnpqow8 currently has freeze option (b) active (forecast engine
  trainable) — deliberate experiment state, left uncommitted.

## Inference CLI documented (2026-07-12)
Verified the post-refactor inference invocation (`uv run inference --from-run-id <id> --options
test_config.k=v ...`) end-to-end and wrote `docs/inference_cli.md` (new, uncommitted). Key facts:
inference = `Trainer.inference` → `validate()` on `test_cfg`; effective config chain is
training_config → validation_config → test_config (test_config empty by default, so inference
inherits validation dates/samples); `output.num_samples` must be > 0 for anything to be written
(zarr per rank at `<run>/validation_chkpt00000_rank0000.zarr` under the NEW run_id unless
`--reuse-run-id`).

## Eval config for the 5 IMERG-diag inference runs (2026-07-13)
`config/evaluate/eval_config_imerg_diag_comparison.yml` (new, uncommitted). Runs: t9phvksg
(MSE ipnpqow8 FE-frozen), f9gjxf4r (MSE ipnpqow8 FE-unfrozen), rqwdc8hu (CRPS ipnpqow8, 4 ens),
d07ywvt9 (MSE rck9wgm7 +obs), xjcru87f (BASELINE = inference of xv255s42, atmofs03-rebased).
All stores: 64 samples, IMERG_ANEMOI channel 'tp' (metres/6h), fsteps 1-2, single rank0000 zip.
- **rqwdc8hu shared-dir zip is TRUNCATED** (died writing sample 56; no zip EOCD). Repaired with
  `yes | zip -FF broken --out fixed` + `zip -d fixed "56/*"` → 56 complete samples (0-55),
  verified fully readable (ens=4, finite). Staged at
  `/iopsstor/scratch/cscs/walmikae/imerg_diag_eval_results/rqwdc8hu/` (other 4 runs symlinked
  into the same base dir so extreme_eval --results-dir works too). Original left untouched.
- **BASELINE PERIOD MISMATCH**: xjcru87f inference left test_config empty → inherited
  validation dates Oct 1-Dec 31 2023 (64 windows Oct 1-17, 6h native step); the other four set
  test_config Jun-Aug 2023 / 1h step / shuffle False → 64 windows Jun 1-3. Not sample-matched;
  re-run baseline inference with the test_config overrides (command in the eval config header;
  1h step NOT reproducible — ERA5 v8 zarr is 6-hourly).
- Saved inference configs live at `models/<run_id>/model_<run_id>_chkpt00000.json` (NOT
  _latest.json — inference runs save chkpt00000 only).
- Inference sample count = `test_config.samples_per_mini_epoch` (default from
  validation_config: 256); `output.num_samples` only caps what is WRITTEN. All prior
  inference runs here set test_config.samples_per_mini_epoch=64 — the baseline re-run
  command must include it or it processes 256 samples (4× compute, same 64 written).
- Evaluate-package facts (verified in code): `results_base_dir` must be the RUN's own dir
  (reader globs `validation_chkpt*_rank*.zip` directly in it); metric list parses to
  {name: params} → only ONE thresh per ets/fbi/pss; prob metrics (crps/ssr/spread/
  rank_histogram) are DEAD — `Scores.get_score` has `return None` before dispatch
  (score.py:265); `ensemble: "mean"` is special-cased in io_reader (line ~242) and works;
  sample supports "0-55" range strings; regions: global/tropics/nhem/shem/europe/...;
  psd/grad_amplitude still crash on scattered ipoint output (same as sibling-branch finding).
- Config smoke-tested: all 5 runs resolve channels=['tp'] fsteps=[1,2] 56 samples ens=['mean']
  via WeatherGenZarrReader.check_availability.
- Supplementary per-member config: `eval_config_rqwdc8hu_members.yml` (uncommitted) —
  `ensemble: "all"` keeps the ens dim through scoring (agg_dims="ipoint"; score_orchestration
  iterates a per-ens criteria dim), smoke-tested → ens=[0,1,2,3]. plot_ensemble "members".
- Baseline re-run = hjxwp89v (user swapped it into the comparison config; symlinked into the
  staged dir). Store complete + verified: 64 samples, June 1-17 2023, 6h-aligned windows
  (native 6h step; NOT window-matched to the others' 1h-stepped Jun 1-3). Sample order is
  non-chronological despite shuffle=False in the command (s0=Jun1, s1=Jun2 18h, s55=Jun10).
- Per-run maps/histograms are written to `<results_base_dir>/plots/` (runplot_base_dir
  defaults to results_base_dir). For the 4 symlinked runs that lands in the shared results
  dirs; rqwdc8hu's went to the staged dir (real dir, repaired zip) — looked like "no plots".
  Symlinked back: shared results/rqwdc8hu/plots -> staged rqwdc8hu/plots. Cross-run
  comparison line plots go to evaluation.summary_dir (./plots/imerg_diag_comparison/),
  cwd-relative to where evaluation was launched.
- Training-loss curves: `uv run plot_train -fy config/runs_plot_train_imerg_diag.yml
  --streams IMERG_ANEMOI --channels avg --metrics mse -o ./plots/imerg_diag_losses/
  --legend-outside` (new yml, uncommitted; TRAINING run ids yzjv1l6f/ldwpc9i7/zwx1z9hs/
  xv255s42). plot_train does NOT mkdir the output dir (crashes at first savefig) — mkdir -p
  first. Metric columns: LossPhysical.IMERG_ANEMOI.mse.{avg,tp.<fstep>}; channels 'avg' =
  fstep-mean, 'tp' + --forecast-steps for per-step. Result read: val curves — rck9wgm7+obs
  0.545 best, baseline 0.575, ipnpqow8 frozen 0.72 / unfrozen 0.73 (unfrozen train lower but
  val worse from ~30k samples = overfitting).
- **Plot-colors gotcha**: per-channel map `colors` lists need len(levels)+1 entries —
  plotter._resolve_norm builds BoundaryNorm(levels, cmap.N, extend="both") = (len-1)+2 bins.
  7 levels → 8 colors (the eval_config.yml example's 8th color is not a typo). Crash
  otherwise; fixed in both new eval configs.

## Synthesized cos/sin lat/lon geoinfo for the IMERG zarr (2026-07-14, uncommitted)
The IMERG zarr `nasa-imerg-grib-n320-1998-2024-6h-v1.zarr` (at `/capstor/store/cscs/userlab/
ch17/data/`) has ONLY variable `tp` — no `cos/sin_latitude`, `cos/sin_longitude`. The
`imerg_diag_rck9wgm7_abs_coords/imerg_anemoi.yml` config lists those four as `geoinfo_channels`,
which would have CRASHED at reader init: `data_reader_anemoi.py:151` did `ds.variables.index(ch)`
→ ValueError (those are anemoi build-time computed forcings; installed anemoi.datasets 0.0.1 has
NO open-time forcings-join, so no `anemoi_config` workaround). The AIFS/ERA5 zarrs have them baked
in; IMERG doesn't.
- **FIX (Option B, reader-side synthesis)**: `data_reader_anemoi.py` — module dict
  `_COMPUTED_GEOINFO_FUNCS` (cos/sin of deg2rad(lat|lon)). Geoinfo select loop now tags each
  channel real (store col) vs computed (`_geoinfo_computed[i]`=name, `geoinfo_idx[i]`=-1); unknown
  names raise. Stats per-channel in order (real=ds.statistics; computed=mean/std of synthesized
  grid values, stored in `_computed_geoinfo_grid`). `_get` builds geoinfos column-by-column;
  computed cols are `np.tile(grid_col, n_steps)` — row order t*G+g matches `coords` exactly.
- **Verified end-to-end** against the real zarr: 4 geoinfo cols, MAX ABS DIFF 0.0 vs coords-derived
  cos/sin, normalize→mean0/std1, multi-step (2-slice) tiling row-aligned. Datasets unit tests pass,
  ruff clean. NOTE: absolute position ALREADY reaches the decoder via `coords` + `embed_target_
  coords` — these geoinfo channels are a redundant/additional encoding (mirrors the pretrain input
  streams). Generic: works for any anemoi stream requesting these 4 names.

## z8ngoama failure (2026-07-16): geoinfo fix never shipped — snapshot patched IN PLACE
Run z8ngoama (abs_coords MSE finetune, 2 nodes) crashed at reader init with `ValueError:
'cos_latitude' is not in list` — the KNOWN launcher gotcha: `--from-run-id rck9wgm7_raina`
ships CODE from `slurm_weathergen_rck9wgm7_raina_dir/WeatherGenerator` (seeded 2026-07-04,
pre-dates the geoinfo-synthesis commit 444b3b02). seed_raina_runs.py is gone (was in an old
session scratchpad). Did NOT full-re-seed because the snapshot holds two snapshot-only,
deliberately-kept diffs: `run_train.py` startup banner + `config.py` extra_streams_directory
(a full mirror of the current tree would delete both). Instead patched MINIMALLY: copied only
`data_reader_anemoi.py` into the snapshot (backup alongside:
`data_reader_anemoi.py.bak-pre-geoinfo-20260716`); snapshot's data_reader_base.py is identical
to current, and the patched reader was verified from the snapshot tree against the real zarr
(4 geoinfo cols, max abs diff 0.0). Snapshot also LACKS post-Jul-4 commits 9ab59eef (ddp
rendezvous timeout) etc. — left as-is (2-node startup succeeded in z8ngoama). The other two
seeded dirs (ipnpqow8_raina, atmofs03-rebased_raina) still have the OLD reader — patch them
the same way before launching any abs_coords-style config from them.
- Colleague's claim that cos/sin lat/lon "are already in the zarr forcings" is WRONG for the
  IMERG zarr: data array shape [38835, 1, 1, 542080], variables ['tp'], constant_fields [] —
  they were looking at the ERA5/AIFS zarrs. The reader-side synthesis IS needed.

## JEPA-based IMERG finetunes set up: n0t6ejuo_raina / srdrwfy6_raina (2026-07-17)
New comparison arms from shickman's JEPA lineage: n0t6ejuo (pure temporal-JEPA pretrain,
student_teacher + deep_ssl, has .ema_teacher files) and srdrwfy6 (its 2-step 6h ERA5 forecast
finetune, run_history [[n0t6ejuo,0],...]). KEY FACTS:
- **Our branch CANNOT build these models** (no use_xsa/with_step_conditioning/swiglu/
  embed_orientation/deep_ssl in our code) → unlike rck9wgm7_raina, BOTH new _raina slurm dirs
  ship a copy of `slurm_weathergen_srdrwfy6_dir/WeatherGenerator` (that code loaded the
  n0t6ejuo chkpt when srdrwfy6 was trained; post-PR#2361 exact channel matching; NO ddp
  timeout fix — 600s default).
- **n0t6ejuo chkpt has NO forecast_engine / decoders** (only encoder + deep_ssl_* +
  latent_heads = JEPA predictor; 1368 params). Its finetune config therefore has freeze
  option (b) ACTIVE (fe fresh-init, must train; 6 blocks inherited, `fe_num_blocks: 16`
  commented to replicate srdrwfy6's fresh-fe recipe). srdrwfy6 chkpt: encoder /
  forecast_engine(16 blk trained) / target_token_engines.ERA5 / pred_heads.ERA5 /
  embed_target_coords.ERA5 (dropped as unused since ERA5 output stream removed).
- **JEPA→physical conversion recipe** (from srdrwfy6's own finetune + the team's
  config_decoder_surfacecombined_n0t6ejuo.yml template in the snapshot): losses
  `student-teacher: {enabled: False, type: Disabled}` + physical block named "forecast";
  `model_input.random_easy.enabled: False` + forecasting masker;
  `target_input.random_easy_target.enabled: False`; `deep_ssl.enabled: False`;
  teacher_time_offset 0. trainer.py:133 `filter_config_by_enabled([losses, model_input,
  target_input])` drops disabled entries pre-instantiation; EMATeacher calculator then never
  built and missing .ema_teacher is a logged no-op. `.optim` never loaded on continuation.
- **Freeze regex differs from our other finetunes** (different module tree):
  `.*encoder.*|.*forecast_engine.*|.*latent_pre_norm.*|.*latent_heads.*|.*q_cells.*` (a);
  drop `.*forecast_engine.*` for (b). Stream embeds live INSIDE encoder.embed_engine.
- Streams dir `config/streams/imerg_diag_jepa/`: analysis/avhrr/geos/synop.yml VERBATIM from
  snapshot's jepa_forecast_multi_data_all_years (8 forcing inputs, explicit channel lists —
  no PR#2361 exposure; ERA5_in is type anemoi_operan, 1h zarr sampled 6h via frequency +
  nominal_time_mapping) + our standard imerg_anemoi.yml (stream_ids unique: 0,2,10-14,20,40).
- Configs: `config/raina_config/config_finetune_imerg_diag_mse_{n0t6ejuo,srdrwfy6}.yml`
  (uncommitted). multiprocessing_method pinned "spawn" (obs zarr fork deadlock class).
  Both merged configs VALIDATED through the snapshot's own load_merge_configs +
  filter_config_by_enabled: 9 streams (8 forcing + IMERG diag), losses {forecast: mse only},
  model_input {forecasting}, target_input {}, fsteps 2/offset 1/6h, budget 64x4096,
  dates 2016-2022, opt pins, val 256/EMA 600. launch-slurm overlays ALL config/**.yml from
  the HOME clone at submit (copy_all_configs), so configs need not live in the snapshots.
- Model dirs: `{n0t6ejuo,srdrwfy6}_raina/` = hardlinked <id>_latest.chkpt + json with ONLY
  general.run_id renamed (same recipe as rck9wgm7_raina).

## Dataset replacement crash + resume of hzhk34bz / y71xygpu (2026-07-18)
- Runs hzhk34bz (fe frozen) / y71xygpu (fe trainable) = rck9wgm7 abs_coords IMERG finetunes;
  crashed at part2 (train_continue) because
  `aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6.zarr` was DELETED, replaced by
  `aifs-od-an-oper-0001-mars-o96-2016-2025-6h-v1-for-single-v2.zarr` (drop-in verified: all
  68+9 channels present, 6h axis hours {0,6,12,18} matching nominal_time_mapping keys,
  starts 2016-01-01T00). Both were ~80% done (istep ~26k/32768) — resumed, not restarted.
- Continuation mechanics (CORRECTED after a bad first resume): the chain's part2+ jobs pass
  EMPTY WEATHERGEN_CONFIG_EXTRA — a continue is driven purely by the saved model json
  (istep, run_history, freeze, streams incl. filenames). NEVER pass the finetune --config on
  a resume: its `general.istep: 0` (required for fresh finetune launches) overwrites the
  saved istep and silently restarts the run from mini_epoch 0 (weights still load from
  _latest, but LR schedule/counters reset, and _latest.chkpt + model_<id>[_latest].json get
  clobbered by the restarted run; numbered chkpts are only clobbered as mini-epochs complete).
- First resume attempt (with --config) did exactly that on 2026-07-18 ~21:30; caught after
  ~15 min, scancel'd. Only _latest.chkpt + model_<id>.json + model_<id>_latest.json were
  overwritten; chkpt00049 (istep 25600, correct run_history/freeze) intact for both runs.
- Correct resume recipe after a dead dataset name: edit the zarr filename inside
  `model_<id>_chkpt<NN>.json` of the newest intact numbered checkpoint (backups in session
  scratchpad json_backups/), then
  `launch-slurm.py --run-id <id> --from-run-id <id> --mini-epoch <NN> --nodes 2` with NO
  --config (json is then the sole config source; --mini-epoch NN loads model_<id>_chkptNN.json
  + <id>_chkptNN.chkpt; trainer recomputes mini_epoch_base = istep/len(data_loader)).
- Also fixed the filename in both staged era5_in.yml + 4 home stream ymls (+jepa comment)
  for future fresh launches.
- nominal_time_mapping semantics (anemoi_operan reader): maps hour-of-day of zarr timestamps
  -> availability hour (00 UTC analysis available 05 UTC); hard dict lookup per hour, so the
  zarr's hour set must equal the mapping keys.
- New zarr stats are 2016-2025 (vs 2016-2023) -> slightly different input normalization for
  the last ~20% of training; accepted (old zarr gone, colleague mandated switch).
- poi264qg = the n0t6ejuo JEPA IMERG finetune, running healthily (uses aifs-ea 1h zarr,
  unaffected by the od deletions).


## IMERG SF-sharpness finetunes of zwx1z9hs / yzjv1l6f (2026-07-18)
- New `config/imerg_sharpness/`: SF (LossStructureFunction) stage-2 finetune overlays for the two finished IMERG tp MSE decoders (zwx1z9hs from rck9wgm7_raina, MSE 0.55, istep 33006; yzjv1l6f from ipnpqow8_raina, MSE 0.71, istep 32756). Launch: `launch-slurm.py --config <overlay> --from-run-id <id> --nodes 2` (fresh finetune, new id; overlay has NO istep key so saved istep survives → epochs 64..79 / 63..79 at flat LR 2e-5, policy_decay constant).
- Params empirically calibrated on the IMERG N320 zarr (script + numbers in that dir): min pair distance 28.3 km → bins [60,120,240,480,960] km; increment_power 1 not 2 (tp ~70% zeros, 250σ max → p=2 log-S bootstrap noise 2.5-5× worse); num_pairs 2^21 (default 262144 leaves 60-120 km bin at ~18 pairs); min_pairs_per_bin 32; weight 0.25 ≈ 0.2-0.7× MSE at start — verify from structure.avg in train metrics after ~100 steps, rescale via --options.
- Both overlays also fix ERA5_in to the new o96 zarr (saved parent configs still name the deleted one).
- Launching FROM an old staged dir (from_run_id != run_id) copies the parent snapshot per its `tracked_files.json`; the zwx1z9hs/yzjv1l6f staged dirs listed 31 config ymls deleted since staging → FileNotFoundError. Fixed by pruning missing entries from both staged `tracked_files.json` (backups in session scratchpad). Uncommitted configs are fine at launch: `copy_all_configs` re-copies ALL home-repo config ymls into the new stage dir.
- Staged snapshots only get CONFIGS refreshed from home at launch — new/changed src code must be patched into the parent staged dir AND appended to its tracked_files.json. Did this for loss_module_structure.py + __init__.py registration in the zwx1z9hs and yzjv1l6f staged dirs (their loss_calculator/loss_module_base/physical are identical to HEAD, so the drop-in is safe; do NOT sync the other drifted files — snapshot fidelity).

## n9aqe1cs eval wired up (2026-07-19)
- n9aqe1cs ("ipnpqow8 with SF fix, FE frozen") inference output lives in the SHARED results dir
  (thunter/shared_work/results); eval_config_imerg_diag_comparison.yml's results_base_dir needs the
  usual per-run symlink under /iopsstor/scratch/cscs/walmikae/imerg_diag_eval_results/ — created it
  (same pattern as the other 5 runs). Store verified intact: 64 samples (0-63), zip testzip OK
  (no rqwdc8hu-style truncation). Eval runs clean; other run_ids currently commented out, so
  "compare" plots are single-run. ETS@1mm/6h 0.326/0.261 (global, +6h/+12h), FBI 1.53/1.81.

## SF FINE-TUNE POST-MORTEM + QUANTILE-SF FIX (2026-07-19)
- **ymlx3c4r (SF ft of yzjv1l6f, inf n9aqe1cs + repeats yb5owyij/ys8mhrjt) FAILED by loss
  gaming**: mean-only SF collapsed 0.38→0.09 in ~300 samples; output = global small-amplitude
  drizzle moiré everywhere (incl. dry areas). Pred median 1.9e-4 vs target 0, std UNCHANGED
  (0.00225 vs 0.00219 base; target 0.00349), ETS 0.352/0.298→0.320/0.262, FBI 1.38/1.48→
  1.57/1.74, MSE 0.71→0.75. Cause: per-bin MEAN |Δ| is location-blind; 70%-zero IMERG tp
  dilutes target bin means so uniform noise reproduces them at negligible MSE cost. CERRA
  escaped mainly via p=2 + lower intermittency (its seam amplification = same failure family).
- **Fix implemented (committed-ready, uncommitted)**: `quantile_levels`/`quantile_eps`/
  `match_mean` in loss_module_structure.py — per-bin |Δ| log-quantile matching; target q50
  sits on the zero atom (>50% of pairs in EVERY 60-960 km bin) → drizzle penalized ~25/bin,
  q90/q99 still demand sharp edges. p is irrelevant under quantile matching (scales logs).
  Backward compat: quantile_levels null = bit-for-bit old loss. 6 new CPU tests incl. a
  drizzle-gaming regression test (11 pass).
- **Empirical validation on real outputs** (sf_calibration_imerg.py NEW --pred-zip mode,
  runs the actual torch loss on inference zips; finite-filter needed — raw store has NaNs):
  mean-only start(t9phvksg) 0.46 vs drizzle(n9aqe1cs) 0.09 (gamed); quantile-only 2.62 vs
  7.02; quantile+mean 1.54 vs 3.55 ⇒ loophole closed, drizzle now 2.3× WORSE than start.
  Part A additions: zero-atom mass per bin 0.58-0.50, boot noise of log-q at 2^21 pairs
  ≤0.46 worst bin/level. Full numbers: config/imerg_sharpness/sf_calibration_quantile_20260719.json.
- **New overlay config_ft_structure_q_yzjv1l6f.yml**: levels [0.5,0.9,0.99], quantile_eps
  1e-3, match_mean True, weight 0.15 (0.15×1.54=0.23 ≈ 0.33×MSE 0.71 — anchored on MEASURED
  initial loss, not guessed). Launch: `launch-slurm.py --config <it> --from-run-id yzjv1l6f
  --nodes 2`. Staged dir /iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_yzjv1l6f_dir
  patched with the quantile-capable loss module (backup .bak-pre-quantile-20260719;
  tracked_files.json already lists the file). HEALTH CHECK: structure.avg must fall
  GRADUALLY — collapse to <0.5 within one epoch = gamed again, stop + look at maps.
  zwx1z9hs twin overlay NOT yet created; its staged dir NOT yet patched.
- User constraint: NO ensemble heads for IMERG (quantile_pinball/CRPS routes off the table).
- **LAUNCH GOTCHA (2026-07-19, unrelated to config): pasting the multi-line launch command
  as one line with literal `\` line-continuations feeds stray space-tokens into
  `parse_known_args`** — one gets swallowed as an extra `--config` value (repo root + a
  trailing space), producing `Stage 'train' config file .../WeatherGenerator/  does not
  exist`. Fix: paste the launch command as a single line with no backslashes.
- **REAL BUG FOUND + FIXED (run rj7mhsai, 2026-07-20, crashed ALL 8 ranks at step 0):
  `torch.quantile` REJECTS bf16/fp16** ("input tensor must be either float or double
  dtype"). Training runs in bf16 autocast, so `inc_p`/`inc_t` (built from pred/target) are
  bf16 → the new quantile branch died immediately in real training, even though every CPU
  unit test passed (they only ever used plain float32 tensors — a coverage gap). FIXED in
  `structure_function_loss`: `.float()`-upcast the increments (and build `levels` as
  float32) right before the `torch.quantile` calls only — upcasting is differentiable, so
  the pred-side gradient survives; the mean-term path is untouched (log/pow work fine in
  bf16). Added `test_sf_quantile_loss_works_with_bf16_inputs` (bf16 in, backward, finite
  grad) — 12/12 tests pass now. Re-synced the fixed file into the yzjv1l6f staged snapshot
  (`/iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_yzjv1l6f_dir`). Job rj7mhsai left
  no running SLURM allocation to cancel (already dead) — safe to relaunch the same command.
  LESSON: any new torch op inside a loss module needs an explicit mixed-precision-dtype
  check/test, not just a plain-float32 CPU test — CPU unit tests here don't run autocast.

## rzk7ribx truncated zarr repaired (2026-07-20)
- Eval crashed `zipfile.BadZipFile: File is not a zip file` opening rzk7ribx's
  validation_chkpt00000_rank0000.zip — the inference job hit the SLURM time limit while writing
  sample 63 (log ends mid-write of 63/GOES_ABI_IR source geoinfo; no traceback, error.txt empty),
  so the ZipStore never wrote its end-of-central-directory. Same class as the rqwdc8hu truncation.
- FIX (same as rqwdc8hu): `yes | zip -FF <broken> --out <repaired>`. Replaced the eval symlink with
  a REAL dir at imerg_diag_eval_results/rzk7ribx/ holding the repaired 8.7 GB zip (that pattern is
  why rqwdc8hu is a real dir there, not a symlink). Repaired store: testzip OK, all 64 groups
  present BUT sample 63 is a PARTIAL write — only ERA5_in + GOES_ABI_IR, MISSING IMERG_ANEMOI (the
  scored stream) + 6 others. Complete samples = 0-62 (63 with IMERG_ANEMOI prediction).
- No config sample-range edit needed: both configs that score rzk7ribx (imerg_diag_comparison.yml,
  and rqwdc8hu_members.yml when its run_id is swapped to rzk7ribx) already cap `sample: "0-55"`
  (inherited rqwdc8hu recovery cap), safely inside 0-62. rzk7ribx has 63 usable samples if the cap
  is ever raised — keep ≤62, never 63.
- NOTE: rqwdc8hu_members.yml on disk currently lists run_id `rqwdc8hu` (sample 0-55), not rzk7ribx;
  the user's failing run had rzk7ribx there, so the file was edited back. Re-ran it as-is → scores
  rqwdc8hu + 4 members cleanly (confirms pipeline healthy post-repair).
- v6kaeh49 ALSO truncated (BadZipFile, job died writing sample 56 IMERG pred) — repaired same way;
  0-56 all complete (57 samples). j8zlr0mw clean (0-63). Integrity sweep of all 5 active
  imerg_diag_comparison runs (2026-07-20): hjxwp89v/d07ywvt9/j8zlr0mw complete 0-63,
  rzk7ribx 0-62, v6kaeh49 0-56 — all testzip OK. LESSON: a symlink existing / zarr file existing
  is NOT enough; the shared-work stores from recent inference jobs are frequently truncated (SLURM
  time limit at the tail) — always testzip + check last sample has IMERG_ANEMOI pred before eval.
- Fixed a config typo: imerg_diag_comparison.yml j8zlr0mw results_base_dir had a stray space
  (`.../imerg_diag_eval_results/ j8zlr0mw`) → would resolve to a nonexistent dir.

## poi264qg (JEPA IMERG ft) inference: MUST run from snapshot (2026-07-19)
- `uv run inference --from-run-id poi264qg` from HEAD CRASHES: `ValueError: geoinfo channel
  'noise_time' is neither a variable in the anemoi store nor a computed forcing`. Root cause: the
  saved ERA5_in config has geoinfo_channels [...,'noise_time'] (from Sophie's JEPA self-flow work,
  commit 6ce1551d — noise_time is NOT a store var, it was meant to be appended at runtime by
  masking.py add_geoinfo_noise, which is OFF for this diagnostic ft).
- The two readers handle geoinfo DIFFERENTLY, and it's a SILENT-CORRECTNESS trap, not just a crash:
  - SNAPSHOT reader (trained poi264qg): select_geoinfo_channels intersects config∩store and
    `np.sort`s by STORE INDEX → drops noise_time, feeds 9 channels as [cos_julian_day,
    cos_local_time, insolation, lsm, sdor, sin_julian_day, sin_local_time, slor, z].
  - HEAD reader (commit 7e26241c): takes config list VERBATIM in CONFIG ORDER (z, lsm, slor,...)
    and RAISES on unknown channels. Even after deleting noise_time, HEAD would feed geoinfo in a
    DIFFERENT PERMUTATION than training → silently wrong preds.
- FIX (chosen: run from training snapshot for bit-exactness): venv synced at
  `/iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_poi264qg_dir/WeatherGenerator`
  (`uv sync --all-packages --extra gpu --offline`). Run inference FROM there with
  `WEATHERGEN_PRIVATE_REPO_PATH=/users/walmikae/weathergen/WeatherGenerator-private` set (else
  get_wg_private_path resolves to <snapshot>/../WeatherGenerator-private, a frozen copy). Do NOT
  pass --config or edit streams — the snapshot reader's drop+sort is what matches training.
- GENERAL: any JEPA-lineage run (imerg_diag_jepa streams, noise_time in geoinfo) must be inferred
  from its snapshot, not raina_crps_dec HEAD. The alt (HEAD + geoinfo reordered to store-sorted
  9-list) was rejected as riskier — other reader/model code may also have diverged.
- `uv run` resolves the project from CWD, so a bare `cd <snap> && uv run` still hit HEAD's code if
  the cd didn't take (traceback frames under /users/walmikae/... = HEAD; under /iopsstor/.../slurm_
  ..._dir = snapshot — that path is the tell). ROBUST invocation: `uv run --project <snapshot>
  inference ...` (cwd-independent). Verified the reader module then loads from the snapshot path.

## Inference sample-count gotcha (2026-07-19)
- `test_config.output.num_samples` is a WRITE gate only (trainer.py:583/627: writes the first
  num_samples×batch_size batches) — it does NOT shorten the inference loop, which runs over
  `test_config.samples_per_mini_epoch` capped at the date-range window count (sampler reduces it,
  e.g. 2023-06-01→08-31 hourly windows → 2160→2159). To run exactly N samples add
  `test_config.samples_per_mini_epoch=N` to --options; with shuffle=False those are the FIRST N
  consecutive windows from start_date (N=64 hourly ≈ 2.7 days), so shrink/shift the date range or
  shuffle=True if coverage of the whole period is wanted.

## Ops note (2026-07-21) — saving runs before scratch cleanup
The `./rc` script reads from `/capstor/store/.../ch17/shared_work`, but active runs
actually live in thunter's scratch: `/iopsstor/scratch/cscs/thunter/shared_work/{models,results,logs}`
and `/iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_<id>_dir`. To preserve runs
before a scratch purge, copy FROM that scratch path (not capstor store) into the local repo.
Corrected driver: `scratchpad/save_runs.sh` (models `*latest*` + results + logs + slurm).

## Eval infra gotchas (2026-07-21)
- Inference killed by SLURM time limit leaves a TRUNCATED zarr ZipStore (missing zip
  central directory -> "not a zip file"). Repair with `zip -FF broken.zip --out fixed.zip`
  (rebuilds index from local headers; drops the partial trailing sample). Verify complete
  samples per run before comparing.
- FastEval `regrid: true` uses earthkit-regrid whose SQLite cache lives on NFS home
  (~/.cache/earthkit-regrid). A killed/`Ctrl-Z`-suspended eval keeps that DB open ->
  every new eval dies with `sqlite3 database is locked`. Fix: kill orphaned/stopped
  `evaluate` processes (they hold the lock), then `rm ~/.cache/earthkit-regrid/cache-2.db*`.
  Don't leave eval runs suspended in the background.
- Eval reader path: with NO `results_base_dir` in the run_id block it resolves via
  get_path_run -> path_shared_working_dir/results/<run_id> (santis: /iopsstor/.../thunter/
  shared_work). If you SET results_base_dir it is used verbatim (NO run_id appended) -> must
  point at the dir directly containing validation_*.zip.
