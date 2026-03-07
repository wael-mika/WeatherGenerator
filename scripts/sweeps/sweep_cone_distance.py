#!/usr/bin/env python3
"""
JEPA Frozen Teacher - Cone Distance Cropping Sweep

Creates N copies of the pretraining + finetuning configs with randomly sampled
lr_max and mask_rate, then writes a commands.txt with the exact launch commands.

Usage:
    python3 scripts/sweeps/sweep_cone_distance.py 10

This will:
  1. Create config/sweep_runs/ with 10 pretraining + 10 finetuning configs
  2. Write scripts/sweeps/sweep_runs/commands.txt with launch commands
  3. Write scripts/sweeps/sweep_runs/params.csv with run parameters

Configs go under config/ so the launcher's copy_all_configs picks them up.
"""

import math
import random
import re
import string
import sys
from pathlib import Path


# --- Config ---
PRETRAINING_CONFIG = "config/config_jepa_frozen_cropping_cone_distance_2drope.yml"
FINETUNING_CONFIG = "config/config_jepa_finetuning_cropping_cone_distance.yml"
LAUNCHER = "../WeatherGenerator-private/hpc/launch-slurm-multi.py"

LR_RANGE = (1e-6, 5e-5)  # log-uniform
MASK_RANGE = (0.1, 0.8)  # uniform, fraction of cells KEPT


def sample_params():
    run_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    lr = math.exp(random.uniform(math.log(LR_RANGE[0]), math.log(LR_RANGE[1])))
    mask = random.uniform(MASK_RANGE[0], MASK_RANGE[1])
    return run_id, lr, mask


def patch_config(content, lr, mask, is_pretraining):
    """Replace lr_max and student mask rate in config content."""
    # Replace lr_max (appears once per config)
    content = re.sub(
        r"(lr_max:\s*)[\d.eE+-]+",
        f"\\g<1>{lr:.2e}",
        content,
    )

    if is_pretraining:
        # Replace student mask rate: the "rate : 0.4" line under model_input section.
        # Split at model_input/target_input boundaries to only touch the right one.
        mi_marker = "model_input:"
        ti_marker = "target_input:"
        mi_start = content.index(mi_marker)
        ti_start = content.index(ti_marker, mi_start)
        before = content[:mi_start]
        model_section = content[mi_start:ti_start]
        after = content[ti_start:]
        # Replace the first occurrence of "rate : <number>" in model_input
        model_section = re.sub(
            r"(rate\s*:\s*)[\d.]+",
            f"\\g<1>{mask:.2f}",
            model_section,
            count=1,
        )
        content = before + model_section + after

    return content


def main():
    num_experiments = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # Resolve paths from repo root
    repo_root = Path(__file__).resolve().parent
    while repo_root != repo_root.parent:
        if (repo_root / "pyproject.toml").exists():
            break
        repo_root = repo_root.parent
    else:
        print("ERROR: could not find repo root (no pyproject.toml found)")
        sys.exit(1)

    pretrain_src = repo_root / PRETRAINING_CONFIG
    finetune_src = repo_root / FINETUNING_CONFIG

    for label, path in [("pretraining config", pretrain_src), ("finetuning config", finetune_src)]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    # Configs go under config/ so the launcher copies them to the SLURM staging dir.
    # commands.txt and params.csv go under scripts/sweeps/sweep_runs/.
    config_dir = repo_root / "config" / "sweep_runs"
    config_dir.mkdir(exist_ok=True)
    out_dir = repo_root / "scripts" / "sweeps" / "sweep_runs"
    out_dir.mkdir(exist_ok=True)

    pretrain_content = pretrain_src.read_text()
    finetune_content = finetune_src.read_text()

    commands = []
    params_lines = ["run_id,lr_max,mask_rate"]

    print(f"=== Generating {num_experiments} sweep configs ===")
    print(f"  Pretraining : {pretrain_src}")
    print(f"  Finetuning  : {finetune_src}")
    print(f"  Config dir  : {config_dir}")
    print(f"  Output dir  : {out_dir}")
    print()

    for i in range(1, num_experiments + 1):
        run_id, lr, mask = sample_params()

        # Write patched pretraining config
        pretrain_out = config_dir / f"pretrain_cone_distance_{i}.yml"
        patched_pretrain = patch_config(pretrain_content, lr, mask, is_pretraining=True)
        pretrain_out.write_text(patched_pretrain)

        # Write patched finetuning config (same lr, no mask change)
        finetune_out = config_dir / f"finetune_cone_distance_{i}.yml"
        patched_finetune = patch_config(finetune_content, lr, mask=0, is_pretraining=False)
        finetune_out.write_text(patched_finetune)

        # Build command (relative paths, run from repo root)
        pretrain_rel = pretrain_out.relative_to(repo_root)
        finetune_rel = finetune_out.relative_to(repo_root)
        cmd = (
            f"{LAUNCHER} "
            f"--chain-jobs 1 1 "
            f"--config ./{pretrain_rel} ./{finetune_rel} "
            f"--run-id {run_id} "
            f"--nodes 1"
        )
        commands.append(cmd)
        params_lines.append(f"{run_id},{lr:.2e},{mask:.2f}")

        print(f"  [{i}/{num_experiments}] {run_id}  lr={lr:.2e}  mask_rate={mask:.2f}")

    # Write commands.txt
    commands_file = out_dir / "commands.txt"
    commands_file.write_text("\n".join(commands) + "\n")

    # Write params.csv
    params_file = out_dir / "params.csv"
    params_file.write_text("\n".join(params_lines) + "\n")

    print()
    print(f"=== Done ===")
    print(f"  Configs   : {config_dir}/")
    print(f"  Commands  : {commands_file}")
    print(f"  Params    : {params_file}")
    print()
    print("To launch all experiments:")
    print(f"  cd {repo_root}")
    print(f"  while read cmd; do $cmd; done < {commands_file.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
