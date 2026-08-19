# Adapting any pretrained run to this branch: the `_raina` procedure

How to take **any** pretrained run — JEPA or not, yours or a colleague's — and make it
continuable on `wm/dev/raina_crps_dec`.

Everything here is derived from the runs that actually failed and the ones that finally worked
(`i2q9vihz`, `oesyjclc`, `yliqgead`, `nfwefvyx`, `z8ngoama`, `xy7szoy7`, `c4hua7ig`, `bqu9b3cw`
→ `k3hnf391`, `y4g6op28`). Every failure below is one somebody already paid for.

---

## 1. The one thing to understand

`launch-slurm.py` (~line 1266): for a stage with `--from-run-id X` where `X != run_id`,

- **CODE** is copied from `<slurm>/slurm_weathergen_X_dir/WeatherGenerator` — the snapshot of the
  code that *trained the checkpoint*;
- **CONFIGS** are refreshed from your home clone (`copy_all_configs`).

So you get **new configs running on old code**. Your branch's work never ships on
`train_continue`, and there is no CLI flag to override it.

The fix is not to patch the launcher. It is to create a **parallel run** `<id>_raina` whose
snapshot dir contains the code you actually want, then launch `--from-run-id <id>_raina`.

Two consequences that follow from this and explain most of the surprises later:

- **Config-only changes never need re-seeding.** Configs come from home at submit time.
- **Any code change after seeding does.** The seeded dir is frozen at the moment you made it.

---

## 2. Decide which code to ship

This is the fork in the road. Get it wrong and the checkpoint either fails to load or loads
into the wrong architecture.

| Situation | Ship | Why |
|---|---|---|
| This branch **can build** the checkpoint's architecture | **this branch's working tree** (default) | You want the CRPS / geoinfo / reader fixes |
| This branch **cannot build** it (JEPA lineage) | **the pretraining snapshot's code** (`--code-from`) | Our code has no `use_xsa`, `with_step_conditioning`, `swiglu`, `embed_orientation`, `deep_ssl` |

### The compatibility audit (do this first, it is cheap)

```bash
python - <<'PY'
import json
i = "gkm6as6m"
c = json.load(open(f"/iopsstor/scratch/cscs/thunter/shared_work/models/{i}/model_{i}_latest.json"))
for k in ["ae_local_num_queries", "decoder_type", "healpix_level", "num_class_tokens",
          "num_register_tokens", "norm_type", "rope_2D", "embed_unembed_mode", "with_fsdp"]:
    print(f"{k:26s} {c.get(k)}")
print("streams:", list(c["streams"]))
print("forecast:", c["training_config"].get("forecast"))
print("istep:", c["general"].get("istep"), "| run_history:", c["general"].get("run_history"))
PY
```

Then diff the snapshot's code against yours — **the architecture builders are what matter**:

```bash
d=/iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_<id>_dir/WeatherGenerator
diff -rq "$d/src" src | grep -v __pycache__ | grep -v _test.py
```

If `engines.py`, `embeddings.py`, `encoder.py`, `attention.py`, `blocks.py`, `layers.py`,
`norms.py` are **identical**, this branch builds the same parameter tree → ship branch code.
If they differ structurally, or the config carries keys your code never reads that change
shapes → ship snapshot code.

**Hard blockers to check explicitly:**

- `ae_local_num_queries` **must be 1**. HEAD's rewritten `predict_decoders` asserts it.
- `decoder_type` must be `PerceiverIOCoordConditioning` if you want CRPS — `Linear` bypasses
  `EnsPredictionHead` entirely (`model.py:909-924`).
- A key present in the config but read **nowhere** in either tree is inert, not a blocker.
  `embed_orientation: channels` is exactly this case — it looks alarming and means nothing.

---

## 3. The procedure

### Step 1 — seed

```bash
# normal case: ship this branch
scripts/raina/seed_raina_run.py gkm6as6m

# JEPA lineage: ship the snapshot that trained it
scripts/raina/seed_raina_run.py n0t6ejuo \
  --code-from /iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_srdrwfy6_dir/WeatherGenerator
```

Creates:

```
<models>/<id>_raina/
    model_<id>_raina_latest.json   # original json, general.run_id renamed — the ONLY change
    <id>_raina_latest.chkpt        # HARDLINK to the original checkpoint
<slurm>/slurm_weathergen_<id>_raina_dir/WeatherGenerator/
    code + every config/**.y*ml + tracked_files.json
```

The checkpoint is a **hardlink**, not a copy — no extra TB, but deleting the original run's
`_latest.chkpt` leaves the link alive with the provenance lost.

`tracked_files.json` is **required**: the launcher falls back to it when the dir has no `.git`.

The seeder refuses to overwrite an existing dir. To re-seed, delete first — but read §5 before
you do.

### Step 2 — verify

```bash
scripts/raina/verify_raina_run.py gkm6as6m
scripts/raina/verify_raina_run.py n0t6ejuo --snapshot-code   # skips drift/fix checks
```

Checks the json differs in exactly one leaf, the checkpoint shares an inode, the manifest is
complete, and **every branch fix is actually present in the seeded code**. That last check is
the point — it is what catches "I seeded before I fixed the bug".

### Step 3 — generate the stream dir

Never hand-write these. Generate them from the checkpoint's own config:

```bash
scripts/raina/gen_stream_dir.py gkm6as6m imerg_diag_gkm6as6m
```

Keeps every forcing input verbatim, drops the pretraining output stream, adds
`imerg_anemoi.yml` as the sole diagnostic output.

**Why generate rather than copy an existing dir:** PR #2361 changed channel-exclude matching
from substring to exact. Replaying an old `source_exclude: [w_, ...]` today leaks 13 `w_<level>`
channels → 93 instead of 80 → unembed `[22,512]` vs checkpoint `[25,512]`. That is exactly how
`nfwefvyx` died. The generator materializes **explicit** channel lists from the checkpoint, so
matching semantics can never drift again.

Regression-check the generator itself after editing it:

```bash
scripts/raina/gen_stream_dir.py rck9wgm7 imerg_diag_rck9wgm7 --check
```

It should reproduce the committed dir file-for-file (one known diff: the ERA5_in zarr was
renamed by hand after generation).

**Then check every zarr filename it emitted actually exists.** Datasets get deleted and
replaced — that is what killed `hzhk34bz`/`y71xygpu`.

### Step 4 — write the finetune config

Copy the nearest existing arm (`config/raina_config/config_finetune_imerg_diag_mse_rck9wgm7.yml`)
and change:

```yaml
streams_directory: "./config/streams/imerg_diag_<id>/"   # the wipe is what makes the probe work
freeze_modules: ".*encoder.*|.*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*latent.*|.*ERA5.*|.*forecast.*"
decoder_ens_latent_perturbation: null                    # MSE arm; set the block for CRPS
general:
  istep: 0                                               # REQUIRED for a fresh finetune
  desc: "imerg_diag_mse_from_<id>"
```

Points that are easy to get wrong:

- **`streams_directory` wipes inherited streams** (`config.py:438-442` sets
  `base_config.streams = None`). That is *desired* here: the finetune is "pretraining inputs +
  IMERG only". If instead you want to **add** a stream while keeping the rest, omit
  `streams_directory` and put the new stream inline under `streams:` — OmegaConf unions nested
  dicts, so you never need to know the inherited set.
- **`freeze_modules` needs `.*encoder.*`** — the base pattern misses
  `encoder.embed_engine.embeds.<obs>` and `encoder.ae_aggregation_engine`. Add `.*forecast.*`
  for a *pure* probe; drop it to let the forecast engine train.
- **JEPA module trees differ**: use
  `.*encoder.*|.*forecast_engine.*|.*latent_pre_norm.*|.*latent_heads.*|.*q_cells.*`.
- **Match the pretraining forecast block** unless you mean to change it. Check
  `training_config.forecast` in the checkpoint json first — if it already says
  6h/offset 1/2 steps/fixed, you are not overriding anything, you are restating it.
- **`num_workers` is a host-RAM constraint, not a comparability knob.** See §6.

For JEPA→physical conversion you also need:
`losses: {student-teacher: {enabled: False, type: Disabled}}` + a physical block,
`model_input.random_easy.enabled: False` + forecasting masker,
`target_input.random_easy_target.enabled: False`, `deep_ssl.enabled: False`,
`teacher_time_offset: 0`. `filter_config_by_enabled` drops disabled entries before
instantiation, so the EMA teacher is never built and the missing `.ema_teacher` is a no-op.

### Step 5 — validate the merge before burning a job

Merge the overlay onto the real `_raina` json through this branch's own loader and assert the
result. Use `config/raina_config/` arms as the model; the essentials:

```python
from weathergen.common.config import load_merge_configs
cfg = load_merge_configs(None, "gkm6as6m_raina", -1, None, "config/raina_config/<your>.yml")
```

- The signature is **positional**: `(private_home, from_run_id, mini_epoch, base, *overwrites)`.
- **`mini_epoch=-1`** resolves `model_<id>_latest.json`. `None` looks for `model_<id>.json`,
  which seeded dirs do not have.
- `cfg.streams` is a **name-keyed dict**, not a list.
- `forecast.time_step`, `time_window_step` and stream `frequency` resolve through
  `${timedelta:}` to `np.timedelta64` — compare against `np.timedelta64(6, "h")`, not `"06:00:00"`.

Assert at minimum: the pretraining output stream is gone, IMERG is the only non-forcing stream,
stream ids are unique, `loss_fcts` parses to what you expect, `istep == 0`, and ERA5_in's channel
lists are byte-identical to the checkpoint's.

### Step 6 — smoke, then launch

```bash
# interactive smoke, 1 node — immune to the multi-node FS import hang
uv run train_continue --from-run-id gkm6as6m_raina --config config/raina_config/<your>.yml \
  --options training_config.samples_per_mini_epoch=8 train_logging.terminal=1

# real launch
../WeatherGenerator-private/hpc/launch-slurm.py --from-run-id gkm6as6m_raina \
  --config config/raina_config/<your>.yml --nodes 2
```

Smoke tests need `--time 60`, not 30 — model build + checkpoint load alone took 28 min on a bad
FS day, and terminal logging every 50 batches means you see nothing before the wall.

Paste launch commands as **one line with no backslashes**: literal `\` continuations feed stray
space-tokens into `parse_known_args` and produce `Stage 'train' config file ... does not exist`.

---

## 4. What "working" looks like

From `y4g6op28`, the first clean run:

- `loss_avg == <YOUR_STREAM>.mse` only — no inherited stream contributing a term;
- missing keys confined to `target_token_engines.<STREAM>.*` and `pred_heads.<STREAM>.*`
  (the genuinely new decoder), everything else loaded;
- `Continuing run with id=<id>_raina`;
- LR plateau = `lr_max × sqrt(ranks)` — that is `parallel_scaling_policy: sqrt`, not a bug.

---

## 5. Keeping a seeded dir current

The seeded dir is a **snapshot**. New code fixes do not reach it by themselves — this has bitten
three times (`xy7szoy7`, `z8ngoama`, and the obs-reader fix in this session).

**Re-seed** (delete + `seed_raina_run.py`) when the dir has no deliberate local edits.

**Patch in place** when it does. Some seeded dirs intentionally hold snapshot-only diffs (the
`run_train.py` startup banners, `extra_streams_directory`) that a re-seed would wipe:

```bash
d=/iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_<id>_raina_dir/WeatherGenerator
cp src/weathergen/datasets/<file>.py "$d/src/weathergen/datasets/<file>.py.bak-pre-<fix>-$(date +%Y%m%d)"  # backup first
cp src/weathergen/datasets/<file>.py "$d/src/weathergen/datasets/"
# if the file is NEW, append it to that dir's tracked_files.json
```

Re-run `verify_raina_run.py` afterwards — its drift check is precisely for this.

Same applies to **any** old snapshot you continue from, not just `_raina` ones: a continuation
from a pre-change run needs its snapshot patched with the code the overlay depends on.
`xy7szoy7` crashed with `invalid literal for int(): 'members'` for exactly this reason.

If a stage dir's `tracked_files.json` lists files deleted since staging, launch fails with
`FileNotFoundError` — prune the missing entries (the seeder does this automatically for
`--code-from`).

---

## 6. Failure catalogue

| Symptom | Cause | Fix |
|---|---|---|
| `params.get("weight")` on None in `LossPhysical.__init__` | Snapshot loss module has no `"mse": null` guard | Ship branch code (`_parse_loss_fcts`) |
| `KeyError` on `latent_perturbation_log_sigma` under FSDP | Bare root param not in `all_modules` on a pre-CRPS checkpoint | Branch's `model_interface` sigma fix |
| An unexpected `LossPhysical.<OLD_STREAM>.mse` appears | Checkpoint-era `load_merge_configs` has no wipe → override streams merged additively | Launch from `_raina` (branch code wipes) |
| `unembed.* [25,512] vs [22,512]` | `source_exclude` replayed under PR #2361 exact matching | Generate the stream dir (explicit channel lists) |
| `'cos_latitude' is not in list` | Snapshot predates geoinfo synthesis | Patch `data_reader_anemoi.py` into the snapshot |
| `'noise_time' is neither a variable nor a computed forcing` | JEPA config lists a runtime-appended geoinfo | Run from the training snapshot; **do not** just delete the channel — HEAD feeds geoinfo in config order, the snapshot sorted by store index, so you would silently permute inputs |
| `invalid literal for int(): 'members'` | Continuation from a snapshot predating a loss-module change | Patch the file into that snapshot |
| `negative dimensions are not allowed` in `_setup_sample_index` | Missing upstream fix `a72c9d71`; a stream's hourly index ends **exactly** at the requested end date (`METOP_IASI_PC`) | Take the develop fix; see §7 |
| `DataLoader worker exited unexpectedly` / `_queue.Empty` / NCCL watchdog timeout | **Host RAM OOM** — look for `oom_kill` / `Out Of Memory` in the log | Lower `num_workers`; see below |
| Loader hangs forever after `Mini_epoch 0: train` | fork + asyncio zarr deadlock on obs stores | `general.multiprocessing_method: spawn` |
| 600 s TCPStore rendezvous timeout, one node silent | Cold Lustre venv imports | `ddp_init_timeout_seconds`; single-node is immune |

### Host RAM is the one that will surprise you

`trainer.py:260-273` builds a **train and a validation DataLoader, both with
`data_loading.num_workers`**, and the val workers only spawn at `iter(...)` (`trainer.py:581`).
With 4 ranks/node that is `4 × num_workers` processes during training, **doubling** at the
train→validate transition. Each worker buffers every stream.

That is why `c4hua7ig` (h512 obs, `token_size 1536`) died in training at 10 workers with zero
loss lines, while `bqu9b3cw` (o256) trained 510 batches happily and died the instant validation
started. Current values: **4** for the h512 arm, **6** for o256.

`memory_pinning` is *not* implicated on `train_continue`: the base config is the checkpoint json,
not `default_config.yml`, so the key is absent and `trainer.py:218` defaults it to `False`.

---

## 7. When HEAD is the broken one

A pretraining snapshot can be **newer** than this branch for files we both touch. This branch
merged develop #2565 on 2026-07-02; develop fixed the obs reader in `a72c9d71` on 2026-07-03, so
we never got it — and a snapshot staged later did.

So when a snapshot works and HEAD crashes in shared code, before assuming it is your bug:

```bash
diff -u "$d/src/weathergen/datasets/<file>.py" src/weathergen/datasets/<file>.py
git log origin/develop --oneline -S '<the exact broken expression>' -- src/weathergen/datasets/<file>.py
git merge-base --is-ancestor <that_commit> HEAD && echo "in our branch" || echo "NOT in our branch"
```

---

## 8. Fresh finetune vs resume — do not confuse them

|  | Fresh finetune | Resume of an interrupted run |
|---|---|---|
| `--from-run-id` | the **pretrained** run (`<id>_raina`) | the run **itself** |
| `--config` | your overlay | **none — never pass it** |
| `general.istep` | `0` (required) | inherited from the json |

Passing the finetune config on a resume re-applies `istep: 0` and silently restarts from
mini-epoch 0: weights still load, but LR schedule and counters reset and `_latest.chkpt` +
`model_<id>[_latest].json` get clobbered. Numbered checkpoints survive, so recovery is
`--mini-epoch <NN>` off the newest intact one.

Correct resume:

```bash
launch-slurm.py --run-id <id> --from-run-id <id> --mini-epoch <NN> --nodes 2   # no --config
```

---

## 9. Quick reference

```bash
scripts/raina/seed_raina_run.py   <id> [--code-from DIR]
scripts/raina/verify_raina_run.py <id> [--snapshot-code]
scripts/raina/gen_stream_dir.py   <id> <out_dir> [--check]
```

Paths (override with `WG_MODELS_DIR` / `WG_SLURM_DIR`):

```
models  /iopsstor/scratch/cscs/thunter/shared_work/models
slurm   /iopsstor/scratch/cscs/thunter/slurm
```

Existing `_raina` runs: `rck9wgm7`, `ipnpqow8`, `atmofs03-rebased`, `n0t6ejuo`, `srdrwfy6`
(JEPA, snapshot code), `gkm6as6m`, `c71eo6pu`.

`ipnpqow8_raina` and `atmofs03-rebased_raina` still carry the **old** `data_reader_anemoi.py` —
patch them before any abs_coords-style launch.
