#!/usr/bin/env python3

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


DEFAULT_SWEEP_LOGS = (
    "scripts/sweeps/sweep_cropping_disjoint_log.csv",
    "scripts/sweeps/sweep_cropping_contained_log.csv",
    "scripts/sweeps/sweep_cropping_contained_log_0.csv",
    "scripts/sweeps/sweep_cropping_cone_distance_log.csv",
)
DEFAULT_SOURCE_SUFFIX = "-stage1"
DEFAULT_TARGET_SUFFIX = "-stage2-fn-"


class FinetuneJob(NamedTuple):
    base_run_id: str
    source_run_id: str
    target_run_id: str
    strategy: str
    source_files: Tuple[Path, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_launcher(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "launch-slurm.py"


def _platform_env_script(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "platform-env.py"


def _default_finetune_config(repo_root: Path) -> Path:
    return repo_root / "config" / "config_jepa_finetuning.yml"


def _default_output_dir(repo_root: Path) -> Path:
    return repo_root / "config" / "sweep_runs" / "stage2_finetuning"


def _shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(
        description=(
            "Submit JEPA finetuning jobs for all run_ids listed in the sweep CSV logs. "
            "Each source model is treated as <run_id>-stage1 and each new finetuning run "
            "is submitted as <run_id>-stage2-fn using launch-slurm.py --from-run-id."
        )
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[str(repo_root / path) for path in DEFAULT_SWEEP_LOGS],
        help="Sweep CSV log files to read.",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=_default_launcher(repo_root),
        help="Path to WeatherGenerator-private/hpc/launch-slurm.py.",
    )
    parser.add_argument(
        "--finetune-config",
        type=Path,
        default=_default_finetune_config(repo_root),
        help="Finetuning config passed via launch-slurm.py --config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(repo_root),
        help="Directory where the generated manifest is written.",
    )
    parser.add_argument(
        "--source-suffix",
        default=DEFAULT_SOURCE_SUFFIX,
        help="Suffix appended to the CSV run_id to locate the source checkpoint run.",
    )
    parser.add_argument(
        "--target-suffix",
        default=DEFAULT_TARGET_SUFFIX,
        help="Suffix appended to the base run_id for the new finetuning run.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to pass to launch-slurm.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of jobs to prepare/submit.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs. Without this flag, commands are only printed.",
    )
    parser.add_argument(
        "--slurm-copy-root",
        type=Path,
        default=None,
        help=(
            "Optional shared slurm working directory. If set, or if it can be resolved "
            "from platform-env.py, the script creates a stage alias directory such as "
            "slurm_weathergen_<run>-stage1_dir -> slurm_weathergen_<run>_dir when needed."
        ),
    )

    args, launcher_args = parser.parse_known_args()
    return args, launcher_args


def _validate_paths(args: argparse.Namespace, repo_root: Path) -> List[Path]:
    csv_paths = [Path(path).resolve() for path in args.csv]
    missing = [path for path in csv_paths if not path.is_file()]
    if missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing CSV log files:\n{missing_str}")

    launcher_path = args.launcher.resolve()
    if not launcher_path.is_file():
        raise FileNotFoundError(f"Launcher not found: {launcher_path}")

    finetune_config = args.finetune_config.resolve()
    if not finetune_config.is_file():
        raise FileNotFoundError(f"Finetuning config not found: {finetune_config}")

    if repo_root not in launcher_path.parents and launcher_path.parent.name != "hpc":
        print(
            f"Warning: launcher path {launcher_path} is outside the current WeatherGenerator checkout.",
            file=sys.stderr,
        )

    return csv_paths


def _strip_source_suffix(run_id: str, source_suffix: str) -> str:
    return run_id[: -len(source_suffix)] if run_id.endswith(source_suffix) else run_id


def _load_paths_yml_value(config_path: Path, key: str) -> Optional[str]:
    if not config_path.is_file():
        return None

    pattern = re.compile(r"^\s*%s\s*:\s*['\"]?([^'\"]+)['\"]?\s*$" % re.escape(key))
    with config_path.open(encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                return match.group(1).strip()
    return None


def _resolve_slurm_copy_root(repo_root: Path, requested_root: Optional[Path]) -> Optional[Path]:
    if requested_root is not None:
        return requested_root.resolve()

    platform_env = _platform_env_script(repo_root)
    if not platform_env.is_file():
        return None

    try:
        result = subprocess.run(
            ["python3", str(platform_env), "hpc-config"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    config_path = Path(result.stdout.strip())
    slurm_root = _load_paths_yml_value(config_path, "path_shared_slurm_dir")
    if not slurm_root:
        return None
    return Path(slurm_root).resolve()


def _collect_jobs(
    csv_paths: List[Path],
    source_suffix: str,
    target_suffix: str,
    limit: Optional[int],
) -> Tuple[List[FinetuneJob], List[str]]:
    jobs_by_base = {}  # type: Dict[str, FinetuneJob]
    duplicates = []  # type: List[str]

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "run_id" not in (reader.fieldnames or []):
                raise ValueError(f"CSV file {csv_path} does not contain a run_id column.")

            for row_index, row in enumerate(reader, start=2):
                raw_run_id = (row.get("run_id") or "").strip()
                if not raw_run_id:
                    raise ValueError(f"Missing run_id in {csv_path}:{row_index}")

                base_run_id = _strip_source_suffix(raw_run_id, source_suffix)
                source_run_id = f"{base_run_id}{source_suffix}"
                target_run_id = f"{base_run_id}{target_suffix}"
                strategy = (row.get("strategy") or "unknown").strip() or "unknown"

                existing = jobs_by_base.get(base_run_id)
                if existing is None:
                    jobs_by_base[base_run_id] = FinetuneJob(
                        base_run_id=base_run_id,
                        source_run_id=source_run_id,
                        target_run_id=target_run_id,
                        strategy=strategy,
                        source_files=(csv_path,),
                    )
                else:
                    if csv_path not in existing.source_files:
                        jobs_by_base[base_run_id] = FinetuneJob(
                            base_run_id=existing.base_run_id,
                            source_run_id=existing.source_run_id,
                            target_run_id=existing.target_run_id,
                            strategy=existing.strategy,
                            source_files=existing.source_files + (csv_path,),
                        )
                    duplicates.append(f"{base_run_id} ({csv_path.name})")

    jobs = list(jobs_by_base.values())
    if limit is not None:
        jobs = jobs[:limit]
    return jobs, duplicates


def _write_manifest(output_dir: Path, jobs: List[FinetuneJob], finetune_config: Path) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "base_run_id",
                "source_run_id",
                "target_run_id",
                "strategy",
                "source_files",
                "finetune_config",
            ],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "base_run_id": job.base_run_id,
                    "source_run_id": job.source_run_id,
                    "target_run_id": job.target_run_id,
                    "strategy": job.strategy,
                    "source_files": ";".join(path.name for path in job.source_files),
                    "finetune_config": finetune_config,
                }
            )
    return manifest_path


def _ensure_stage_alias(
    job: FinetuneJob,
    slurm_copy_root: Optional[Path],
    submit: bool,
) -> Optional[str]:
    if slurm_copy_root is None:
        return None

    base_copy_dir = slurm_copy_root / ("slurm_weathergen_%s_dir" % job.base_run_id)
    stage_copy_dir = slurm_copy_root / ("slurm_weathergen_%s_dir" % job.source_run_id)

    if stage_copy_dir.exists():
        return None
    if not base_copy_dir.exists():
        return (
            "missing base slurm dir for alias: %s" % base_copy_dir
        )

    alias_msg = "%s -> %s" % (stage_copy_dir, base_copy_dir)
    if submit:
        stage_copy_dir.symlink_to(base_copy_dir)
        return "created stage alias: %s" % alias_msg

    return "would create stage alias: %s" % alias_msg


def _build_command(
    launcher: Path,
    finetune_config: Path,
    from_run_id: str,
    run_id: str,
    nodes: int,
    launcher_args: List[str],
) -> List[str]:
    cmd = [
        str(launcher),
        "--from-run-id",
        from_run_id,
        "--run-id",
        run_id,
        "--config",
        str(finetune_config),
        "--nodes",
        str(nodes),
    ]
    cmd.extend(launcher_args)
    return cmd


def main() -> int:
    args, launcher_args = _parse_args()
    repo_root = _repo_root()
    csv_paths = _validate_paths(args, repo_root)
    jobs, duplicates = _collect_jobs(
        csv_paths=csv_paths,
        source_suffix=args.source_suffix,
        target_suffix=args.target_suffix,
        limit=args.limit,
    )
    slurm_copy_root = _resolve_slurm_copy_root(repo_root, args.slurm_copy_root)

    if not jobs:
        print("No jobs found in the provided CSV logs.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(args.output_dir, jobs, args.finetune_config.resolve())

    print(f"Prepared {len(jobs)} finetuning job(s).")
    print(f"Launcher: {args.launcher.resolve()}")
    print(f"Finetuning config: {args.finetune_config.resolve()}")
    print(f"Output dir: {args.output_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    if slurm_copy_root is not None:
        print(f"Slurm copy root: {slurm_copy_root}")
    if duplicates:
        print(f"Skipped duplicate base run_ids: {len(duplicates)}")

    failures = []  # type: List[Tuple[FinetuneJob, int]]
    for job in jobs:
        alias_note = _ensure_stage_alias(job, slurm_copy_root, args.submit)
        cmd = _build_command(
            launcher=args.launcher.resolve(),
            finetune_config=args.finetune_config.resolve(),
            from_run_id=job.source_run_id,
            run_id=job.target_run_id,
            nodes=args.nodes,
            launcher_args=launcher_args,
        )

        print("")
        print(f"[{job.target_run_id}] source={job.source_run_id} strategy={job.strategy}")
        if alias_note:
            print(alias_note)
        print(_shell_join(cmd))

        if not args.submit:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append((job, result.returncode))

    if failures:
        print("", file=sys.stderr)
        print("Submission failures:", file=sys.stderr)
        for job, returncode in failures:
            print(f"  - {job.target_run_id}: exit code {returncode}", file=sys.stderr)
        return 1

    if args.submit:
        print("")
        print("All submissions completed.")
    else:
        print("")
        print("Dry run only. Re-run with --submit to launch the jobs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
