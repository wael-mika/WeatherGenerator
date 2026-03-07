#!/usr/bin/env python3
"""
JEPA Frozen Teacher - Cropping Masking Hyperparameter Sweep

Creates N copies of the pretraining + finetuning configs with randomly sampled
lr_max and mask_rate, then writes a commands.txt with the exact launch commands.

Usage:
    python3 scripts/sweeps/sweep_jepa_cropping.py <STRATEGY> [NUM_EXPERIMENTS]

    STRATEGY (required):
      contained      - Student disk contained within teacher (subset)
      cone_distance  - Random angular separation 0-90° (variable overlap)
      disjoint       - Fully separated student/teacher regions

    NUM_EXPERIMENTS: default 10

Examples:
    python3 scripts/sweeps/sweep_jepa_cropping.py contained 5
    python3 scripts/sweeps/sweep_jepa_cropping.py cone_distance 10
    python3 scripts/sweeps/sweep_jepa_cropping.py disjoint 8

This will:
  1. Create config/sweep_runs/<strategy>/ with N pretraining + N finetuning configs
  2. Write scripts/sweeps/sweep_runs/commands_<strategy>.txt with launch commands
  3. Write scripts/sweeps/sweep_runs/params_<strategy>.csv with run parameters

Configs go under config/ so the launcher's copy_all_configs picks them up.
"""

import math
import random
import re
import string
import sys
from pathlib import Path

# --- Strategy -> config mapping ---
STRATEGY_MAP = {
    "contained": {
        "pretrain": "config/config_jepa_frozen_cropping_contained_2drope.yml",
        "finetune": "config/config_jepa_finetuning_cropping.yml",
    },
    "cone_distance": {
        "pretrain": "config/config_jepa_frozen_cropping_cone_distance_2drope.yml",
        "finetune": "config/config_jepa_finetuning_cropping_cone_distance.yml",
    },
    "disjoint": {
        "pretrain": "config/config_jepa_frozen_cropping_disjoint_2drope.yml",
        "finetune": "config/config_jepa_finetuning_cropping.yml",
    },
}

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
        # Replace student mask rate under model_input section only.
        # Split at model_input/target_input boundaries to avoid touching teacher rate.
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


def find_repo_root():
    """Walk up from script location to find pyproject.toml."""
    repo_root = Path(__file__).resolve().parent
    while repo_root != repo_root.parent:
        if (repo_root / "pyproject.toml").exists():
            return repo_root
        repo_root = repo_root.parent
    print("ERROR: could not find repo root (no pyproject.toml found)")
    sys.exit(1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STRATEGY_MAP:
        strategies = ", ".join(STRATEGY_MAP.keys())
        print(f"Usage: {sys.argv[0]} <{strategies}> [NUM_EXPERIMENTS]")
        sys.exit(1)

    strategy = sys.argv[1]
    num_experiments = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    repo_root = find_repo_root()
    strategy_cfg = STRATEGY_MAP[strategy]

    pretrain_src = repo_root / strategy_cfg["pretrain"]
    finetune_src = repo_root / strategy_cfg["finetune"]

    for label, path in [("pretraining config", pretrain_src), ("finetuning config", finetune_src)]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    # Configs go under config/ so the launcher copies them to the SLURM staging dir.
    config_dir = repo_root / "config" / "sweep_runs" / strategy
    config_dir.mkdir(parents=True, exist_ok=True)
    out_dir = repo_root / "scripts" / "sweeps" / "sweep_runs"
    out_dir.mkdir(exist_ok=True)

    pretrain_content = pretrain_src.read_text()
    finetune_content = finetune_src.read_text()

    commands = []
    params_lines = ["run_id,lr_max,mask_rate,strategy"]

    print(f"=== Generating {num_experiments} sweep configs ({strategy}) ===")
    print(f"  Pretraining : {pretrain_src}")
    print(f"  Finetuning  : {finetune_src}")
    print(f"  Config dir  : {config_dir}")
    print()

    for i in range(1, num_experiments + 1):
        run_id, lr, mask = sample_params()

        # Write patched pretraining config
        pretrain_out = config_dir / f"pretrain_{strategy}_{i}.yml"
        patched_pretrain = patch_config(pretrain_content, lr, mask, is_pretraining=True)
        pretrain_out.write_text(patched_pretrain)

        # Write patched finetuning config (same lr, no mask change)
        finetune_out = config_dir / f"finetune_{strategy}_{i}.yml"
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
        params_lines.append(f"{run_id},{lr:.2e},{mask:.2f},{strategy}")

        print(f"  [{i}/{num_experiments}] {run_id}  lr={lr:.2e}  mask_rate={mask:.2f}")

    # Write commands.txt
    commands_file = out_dir / f"commands_{strategy}.txt"
    commands_file.write_text("\n".join(commands) + "\n")

    # Write params.csv
    params_file = out_dir / f"params_{strategy}.csv"
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
