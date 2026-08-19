#!/usr/bin/env python3
# ruff: noqa: T201
"""Seed a `<run_id>_raina` copy of a pretrained run.

WHY: for a stage with `--from-run-id X` where X != run_id, launch-slurm.py copies CODE from
`<slurm>/slurm_weathergen_X_dir/WeatherGenerator` -- the code that TRAINED the checkpoint --
while refreshing CONFIGS from the home clone. New configs then run on old code, and this
branch's fixes never ship. There is no CLI flag to override it. The workaround is a parallel
"run" whose snapshot dir holds the code we actually want.

Creates:
  <models>/<run_id>_raina/
      model_<run_id>_raina_latest.json   original json, general.run_id renamed (ONLY change)
      <run_id>_raina_latest.chkpt        HARDLINK to the original checkpoint (same filesystem)
  <slurm>/slurm_weathergen_<run_id>_raina_dir/WeatherGenerator/
      code + every config/**.y*ml + tracked_files.json (the launcher falls back to that
      manifest when the dir has no .git)

Which code to ship -- see docs/raina_runs.md:
  default            this branch's working tree (git ls-files, so uncommitted edits included).
                     Correct when this branch can BUILD the checkpoint's architecture.
  --code-from DIR    copy DIR instead (a pretraining snapshot). Required for JEPA-lineage runs
                     whose architecture this branch cannot build (use_xsa, deep_ssl, swiglu...).

Refuses to touch existing dirs -- delete them first to re-seed.

  scripts/raina/seed_raina_run.py gkm6as6m
  scripts/raina/seed_raina_run.py n0t6ejuo \
      --code-from <slurm>/slurm_weathergen_srdrwfy6_dir/WeatherGenerator
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("WG_MODELS_DIR", "/iopsstor/scratch/cscs/thunter/shared_work/models"))
SLURM = Path(os.environ.get("WG_SLURM_DIR", "/iopsstor/scratch/cscs/thunter/slurm"))

SKIP_DIRS = {".venv", "__pycache__", ".git", ".ruff_cache", "logs", "models", "output", "plots"}


def working_tree_files() -> list[str]:
    """Tracked files (working-tree content) + all config ymls, including untracked ones.

    NOTE: uv.lock is gitignored, so it is NOT staged. That matches every working _raina dir --
    `actions.sh sync` resolves dependencies per job. Do not "fix" it.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    files = set(tracked)
    for pat in ("*.yml", "*.yaml"):
        files.update(str(p.relative_to(REPO)) for p in (REPO / "config").rglob(pat))
    return sorted(f for f in files if (REPO / f).is_file())


def copy_working_tree(dst: Path) -> int:
    files = working_tree_files()
    for rel in files:
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, target)
    (dst / "tracked_files.json").write_text(json.dumps(files, indent=1))
    return len(files)


def copy_snapshot(src: Path, dst: Path) -> int:
    """Mirror another snapshot (JEPA case). Its own tracked_files.json is carried over."""
    n = 0
    for path in src.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(src).parts):
            continue
        if path.is_file():
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            n += 1
    manifest = dst / "tracked_files.json"
    if not manifest.exists():
        sys.exit(f"ERROR: {src} has no tracked_files.json and no .git -- launcher would fail.")
    # prune entries that no longer exist: a stale manifest is a FileNotFoundError at launch
    listed = json.loads(manifest.read_text())
    present = [f for f in listed if (dst / f).is_file()]
    if len(present) != len(listed):
        print(f"    pruned {len(listed) - len(present)} missing entries from tracked_files.json")
        manifest.write_text(json.dumps(present, indent=1))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument(
        "--code-from",
        type=Path,
        default=None,
        help="snapshot dir to ship instead of this working tree (JEPA lineage)",
    )
    args = ap.parse_args()

    rid = args.run_id
    src_model = MODELS / rid
    dst_model = MODELS / f"{rid}_raina"
    dst_code = SLURM / f"slurm_weathergen_{rid}_raina_dir" / "WeatherGenerator"

    for d in (dst_model, dst_code.parent):
        if d.exists():
            sys.exit(f"REFUSING: {d} already exists -- delete it first to re-seed.")

    src_json = src_model / f"model_{rid}_latest.json"
    src_ckpt = src_model / f"{rid}_latest.chkpt"
    for p in (src_json, src_ckpt):
        if not p.is_file():
            sys.exit(f"MISSING source file: {p}")

    # ---- model dir: renamed json + hardlinked checkpoint -----------------
    dst_model.mkdir(parents=True)
    with open(src_json) as fh:
        cfg = json.load(fh)
    cfg["general"]["run_id"] = f"{rid}_raina"
    with open(dst_model / f"model_{rid}_raina_latest.json", "w") as fh:
        json.dump(cfg, fh)
    link = dst_model / f"{rid}_raina_latest.chkpt"
    os.link(src_ckpt, link)
    print(f"  model dir : {dst_model}")
    print(
        f"    chkpt hardlink -> {src_ckpt.name} ({link.stat().st_size / 1e9:.2f} GB, "
        f"nlink={link.stat().st_nlink}) -- do NOT delete the original run's checkpoint"
    )

    # ---- code dir --------------------------------------------------------
    dst_code.mkdir(parents=True)
    if args.code_from:
        n = copy_snapshot(args.code_from.resolve(), dst_code)
        print(f"  code dir  : {dst_code}\n    {n} files copied FROM SNAPSHOT {args.code_from}")
    else:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "packages"],
            cwd=REPO,
            capture_output=True,
            text=True,
        ).stdout.strip()
        n = copy_working_tree(dst_code)
        print(
            f"  code dir  : {dst_code}\n    {n} files from {REPO} @ {head}"
            f"{' (WITH uncommitted src/packages edits)' if dirty else ' (src/packages clean)'}"
        )

    print("\nNEXT: scripts/raina/verify_raina_run.py " + rid)


if __name__ == "__main__":
    main()
