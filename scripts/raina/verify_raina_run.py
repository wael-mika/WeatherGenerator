#!/usr/bin/env python3
# ruff: noqa: T201
"""Verify a seeded `<run_id>_raina`: json rename, checkpoint hardlink identity, manifest
completeness, drift vs the working tree, and presence of every branch fix the pretraining
snapshots lack.

  scripts/raina/verify_raina_run.py gkm6as6m c71eo6pu
  scripts/raina/verify_raina_run.py n0t6ejuo --snapshot-code   # JEPA: skip fix/drift checks
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("WG_MODELS_DIR", "/iopsstor/scratch/cscs/thunter/shared_work/models"))
SLURM = Path(os.environ.get("WG_SLURM_DIR", "/iopsstor/scratch/cscs/thunter/slurm"))

# (label, file, needle) -- the deltas between this branch and the pretraining snapshots.
# Add a row whenever a new branch fix becomes load-bearing for a finetune.
FIXES = [
    (
        "CRPS latent perturbation",
        "src/weathergen/model/model.py",
        "decoder_ens_latent_perturbation",
    ),
    (
        "sigma load/init fix",
        "src/weathergen/model/model_interface.py",
        "latent_perturbation_log_sigma",
    ),
    (
        "null-safe loss parse (mse: null)",
        "src/weathergen/train/loss_modules/loss_module_physical.py",
        "_parse_loss_fcts",
    ),
    ("kernel_crps loss", "src/weathergen/train/loss_modules/loss_functions.py", "kernel_crps"),
    (
        "forcing-stream target skip",
        "src/weathergen/datasets/multi_stream_data_sampler.py",
        "is_stream_forcing",
    ),
    (
        "geoinfo cos/sin synthesis",
        "src/weathergen/datasets/data_reader_anemoi.py",
        "_COMPUTED_GEOINFO_FUNCS",
    ),
    (
        "obs reader out-of-range fix (a72c9d71)",
        "src/weathergen/datasets/data_reader_obs.py",
        "diff_in_hours_end - (self.hrly_index.shape[0] - 1)",
    ),
    ("ddp rendezvous timeout", "src/weathergen/train/trainer_base.py", "ddp_init_timeout_seconds"),
    (
        "streams_directory wipe",
        "packages/common/src/weathergen/common/config.py",
        "base_config.streams = None",
    ),
    (
        "structure-function loss",
        "src/weathergen/train/loss_modules/loss_module_structure.py",
        "quantile_levels",
    ),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(run_id: str, snapshot_code: bool) -> bool:
    ok = True
    rid = f"{run_id}_raina"
    md = MODELS / rid
    cd = SLURM / f"slurm_weathergen_{rid}_dir" / "WeatherGenerator"
    print(f"=== {rid}")
    if not md.is_dir() or not cd.is_dir():
        print(f"  [FAIL] missing {md if not md.is_dir() else cd}")
        return False

    # 1. json: exactly one leaf changed
    with open(md / f"model_{rid}_latest.json") as fh:
        j = json.load(fh)
    with open(MODELS / run_id / f"model_{run_id}_latest.json") as fh:
        orig = json.load(fh)
    same_id = j["general"]["run_id"] == rid
    orig["general"]["run_id"] = rid
    rest = json.dumps(j, sort_keys=True) == json.dumps(orig, sort_keys=True)
    print(f"  [{'OK' if same_id else 'FAIL'}] json general.run_id == {rid}")
    print(f"  [{'OK' if rest else 'FAIL'}] json otherwise identical to the original")
    ok &= same_id and rest

    # 2. checkpoint is the SAME inode, not a copy
    c, o = md / f"{rid}_latest.chkpt", MODELS / run_id / f"{run_id}_latest.chkpt"
    same_inode = c.stat().st_ino == o.stat().st_ino
    print(
        f"  [{'OK' if same_inode else 'FAIL'}] chkpt shares inode with original "
        f"({c.stat().st_size / 1e9:.2f} GB, nlink={c.stat().st_nlink})"
    )
    ok &= same_inode

    # 3. manifest complete -- a stale entry is a FileNotFoundError at launch
    with open(cd / "tracked_files.json") as fh:
        manifest = json.load(fh)
    missing = [f for f in manifest if not (cd / f).is_file()]
    print(
        f"  [{'OK' if not missing else 'FAIL'}] all {len(manifest)} manifest entries present"
        f"{'' if not missing else f' -- MISSING {missing[:5]}'}"
    )
    ok &= not missing

    # 4. config resolution: the launcher/train_continue needs model_<id>_latest.json
    has_latest = (md / f"model_{rid}_latest.json").is_file()
    print(
        f"  [{'OK' if has_latest else 'FAIL'}] model_{rid}_latest.json present "
        f"(load_run_config needs mini_epoch=-1 to find it)"
    )
    ok &= has_latest

    if snapshot_code:
        print("  [skip] drift + branch-fix checks (this dir ships SNAPSHOT code by design)")
        return ok

    # 5. zero drift vs working tree
    drift = [f for f in manifest if (REPO / f).is_file() and sha(cd / f) != sha(REPO / f)]
    print(
        f"  [{'OK' if not drift else 'WARN'}] drift vs working tree"
        f"{'' if not drift else f' -- DIFFERS {drift[:6]}'}"
    )

    # 6. every branch fix actually shipped
    for label, rel, needle in FIXES:
        p = cd / rel
        present = p.is_file() and needle in p.read_text()
        print(f"  [{'OK' if present else 'FAIL'}] {label}")
        ok &= present
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument(
        "--snapshot-code",
        action="store_true",
        help="dir intentionally ships another snapshot's code (JEPA lineage)",
    )
    a = ap.parse_args()
    allok = all(check(r, a.snapshot_code) for r in a.run_ids)
    print("\nALL CHECKS PASSED" if allok else "\nSOME CHECKS FAILED")
    sys.exit(0 if allok else 1)
