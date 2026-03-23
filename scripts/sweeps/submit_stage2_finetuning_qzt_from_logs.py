#!/usr/bin/env python3
"""Submit q-only Stage 2 finetuning jobs from qzt Stage 1 sweep CSV logs.

Common usage:

  Dry run using the default qzt q-only CSV logs:
    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py

  Dry run only the q healpix sweep:
    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py \
      --csv scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_healpix.csv

  Try the first 3 runs from one CSV:
    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py \
      --csv scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_random.csv \
      --limit 3

  Try a hand-picked subset from one CSV:
    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py \
      --csv scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_cropping.csv \
      --run-id st8zgih3 i7zspy9z

  Actually submit:
    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py --submit

Notes:
  - By default the script only prepares jobs whose Stage 1 artifacts are present.
  - The default finetuning config is q-only and uses `era5_iasi_finetuning_q_abl`.
  - `--limit` keeps the first N jobs after filtering.
  - `--run-id` filters by base run id or by the full `<run_id>-stage1` name.
  - Extra unknown CLI args are forwarded to `launch-slurm.py`.
"""

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


DEFAULT_SWEEP_LOGS = (
    "scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_cropping.csv",
    "scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_healpix.csv",
    "scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_random.csv",
)
DEFAULT_SOURCE_SUFFIX = "-stage1"
DEFAULT_TARGET_SUFFIX = "-stage2-q-fn-fixed"


class FinetuneJob(NamedTuple):
    base_run_id: str
    source_run_id: str
    target_run_id: str
    strategy: str
    stage1_config: str
    source_files: Tuple[Path, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_launcher(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "launch-slurm.py"


def _platform_env_script(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "platform-env.py"


def _default_finetune_config(repo_root: Path) -> Path:
    return repo_root / "config" / "config_jepa_finetuning_q_abl_stage2.yml"


def _default_output_dir(repo_root: Path) -> Path:
    return repo_root / "config" / "sweep_runs" / "stage2_finetuning_qzt_q"


def _shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _resolve_platform_config(repo_root: Path) -> Optional[Path]:
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

    path = Path(result.stdout.strip())
    return path if path.is_file() else None


def _resolve_default_root(repo_root: Path, key: str) -> Optional[Path]:
    config_path = _resolve_platform_config(repo_root)
    if config_path is None:
        return None

    value = _load_paths_yml_value(config_path, key)
    if not value:
        return None
    return Path(value).resolve()


def _resolve_default_model_root(repo_root: Path) -> Optional[Path]:
    return _resolve_default_root(repo_root, "model_path")


def _resolve_default_output_root(repo_root: Path) -> Optional[Path]:
    shared_root = _resolve_default_root(repo_root, "path_shared_working_dir")
    if shared_root is None:
        return None
    return (shared_root / "output").resolve()


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Submit q-only JEPA finetuning jobs for completed qzt Stage 1 runs listed in the "
            "sweep CSV logs. Each source model is treated as <run_id>-stage1 and each new "
            "finetuning run is submitted with a configurable suffix using launch-slurm.py "
            "--from-run-id."
        ),
        epilog=(
            "Examples:\n"
            "  Dry run the defaults:\n"
            "    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py\n\n"
            "  Restrict to one CSV:\n"
            "    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py \\\n"
            "      --csv scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_healpix.csv\n\n"
            "  Restrict to a few specific runs:\n"
            "    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py \\\n"
            "      --csv scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_random.csv \\\n"
            "      --run-id neo4najl w4qhr4f8 scqfy77t\n\n"
            "  Submit only the first 2 matching runs:\n"
            "    ./.venv/bin/python scripts/sweeps/submit_stage2_finetuning_qzt_from_logs.py \\\n"
            "      --csv scripts/sweeps/sweep_jepa_random_student_rate_stage1_select_qzt_log_q_cropping.csv \\\n"
            "      --limit 2 --submit\n"
        ),
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
        "--model-root",
        type=Path,
        default=_resolve_default_model_root(repo_root),
        help="Shared model directory containing <run_id>/model_<run_id>_latest.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_resolve_default_output_root(repo_root),
        help="Shared output directory containing output_<run_id>-stage1_*.txt files.",
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
        "--run-id",
        nargs="+",
        default=None,
        help=(
            "Optional subset filter. Accepts base run ids like 'fgomoie3' or full source "
            "run ids like 'fgomoie3-stage1'."
        ),
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
    parser.add_argument(
        "--allow-incomplete-stage1",
        action="store_true",
        help="Prepare finetuning jobs even if Stage 1 artifacts/logs cannot be validated.",
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

    if args.model_root is None and not args.allow_incomplete_stage1:
        raise FileNotFoundError(
            "Could not resolve the shared model directory automatically. "
            "Pass --model-root or use --allow-incomplete-stage1."
        )
    if args.model_root is not None and not args.model_root.resolve().is_dir():
        raise FileNotFoundError(f"Model root not found: {args.model_root.resolve()}")
    if args.output_root is not None and not args.output_root.resolve().is_dir():
        raise FileNotFoundError(f"Output root not found: {args.output_root.resolve()}")

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

    config_path = _resolve_platform_config(repo_root)
    if config_path is None:
        return None
    slurm_root = _load_paths_yml_value(config_path, "path_shared_slurm_dir")
    if not slurm_root:
        return None
    return Path(slurm_root).resolve()


def _pick_target_run_id(base_run_id: str, target_suffix: str, model_root: Optional[Path]) -> str:
    candidate = f"{base_run_id}{target_suffix}"
    if model_root is None:
        return candidate

    resolved_model_root = model_root.resolve()
    if not (resolved_model_root / candidate).exists():
        return candidate

    idx = 1
    while (resolved_model_root / f"{candidate}{idx}").exists():
        idx += 1
    return f"{candidate}{idx}"


def _has_stage1_model_artifacts(model_root: Optional[Path], source_run_id: str) -> Tuple[bool, str]:
    if model_root is None:
        return False, "model root unavailable"

    model_dir = model_root.resolve() / source_run_id
    if not model_dir.is_dir():
        return False, f"missing model dir: {model_dir}"

    preferred_files = (
        model_dir / f"{source_run_id}_latest.chkpt",
        model_dir / f"model_{source_run_id}_latest.json",
    )
    if any(path.is_file() for path in preferred_files):
        return True, f"found latest Stage 1 artifacts in {model_dir}"

    chkpt_candidates = sorted(model_dir.glob(f"{source_run_id}_chkpt*.chkpt"))
    json_candidates = sorted(model_dir.glob(f"model_{source_run_id}_chkpt*.json"))
    if chkpt_candidates or json_candidates:
        return True, f"found checkpoint snapshots in {model_dir}"

    return False, f"missing Stage 1 checkpoint artifacts in {model_dir}"


def _stage1_output_status(output_root: Optional[Path], source_run_id: str) -> Tuple[bool, str]:
    if output_root is None:
        return True, "output root unavailable"

    matches = sorted(output_root.resolve().glob(f"output_{source_run_id}_*.txt"))
    if not matches:
        return True, "no Stage 1 output log found"

    latest = matches[-1]
    text = latest.read_text(encoding="utf-8", errors="ignore")
    fatal_markers = (
        "Traceback (most recent call last):",
        "ModuleNotFoundError:",
        "ZeroDivisionError:",
        "RuntimeError:",
        "Exception:",
    )
    for marker in fatal_markers:
        if marker in text:
            return False, f"fatal marker {marker!r} found in {latest.name}"

    return True, f"latest Stage 1 output looks clean: {latest.name}"


def _stage1_is_ready(
    source_run_id: str,
    model_root: Optional[Path],
    output_root: Optional[Path],
) -> Tuple[bool, str]:
    has_artifacts, artifact_note = _has_stage1_model_artifacts(model_root, source_run_id)
    if not has_artifacts:
        return False, artifact_note

    output_ok, output_note = _stage1_output_status(output_root, source_run_id)
    if not output_ok:
        return False, output_note

    return True, f"{artifact_note}; {output_note}"


def _normalise_selected_run_ids(
    run_ids: Optional[List[str]],
    source_suffix: str,
) -> Optional[Set[str]]:
    if not run_ids:
        return None

    selected = set()
    for run_id in run_ids:
        cleaned = run_id.strip()
        if not cleaned:
            continue
        selected.add(_strip_source_suffix(cleaned, source_suffix))
    return selected or None


def _collect_jobs(
    csv_paths: List[Path],
    source_suffix: str,
    target_suffix: str,
    model_root: Optional[Path],
    output_root: Optional[Path],
    selected_run_ids: Optional[Set[str]],
    allow_incomplete_stage1: bool,
    limit: Optional[int],
) -> Tuple[List[FinetuneJob], List[str], List[str]]:
    jobs_by_base = {}  # type: Dict[str, FinetuneJob]
    duplicates = []  # type: List[str]
    skipped = []  # type: List[str]

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
                if selected_run_ids is not None and base_run_id not in selected_run_ids:
                    continue
                target_run_id = _pick_target_run_id(base_run_id, target_suffix, model_root)
                strategy = (
                    (row.get("strategy") or row.get("stage1_config") or csv_path.stem).strip()
                    or "unknown"
                )
                stage1_config = (row.get("stage1_config") or "").strip()

                if not allow_incomplete_stage1:
                    ready, note = _stage1_is_ready(source_run_id, model_root, output_root)
                    if not ready:
                        skipped.append(f"{source_run_id}: {note}")
                        continue

                existing = jobs_by_base.get(base_run_id)
                if existing is None:
                    jobs_by_base[base_run_id] = FinetuneJob(
                        base_run_id=base_run_id,
                        source_run_id=source_run_id,
                        target_run_id=target_run_id,
                        strategy=strategy,
                        stage1_config=stage1_config,
                        source_files=(csv_path,),
                    )
                else:
                    if csv_path not in existing.source_files:
                        jobs_by_base[base_run_id] = FinetuneJob(
                            base_run_id=existing.base_run_id,
                            source_run_id=existing.source_run_id,
                            target_run_id=existing.target_run_id,
                            strategy=existing.strategy,
                            stage1_config=existing.stage1_config,
                            source_files=existing.source_files + (csv_path,),
                        )
                    duplicates.append(f"{base_run_id} ({csv_path.name})")

    jobs = list(jobs_by_base.values())
    if limit is not None:
        jobs = jobs[:limit]
    return jobs, duplicates, skipped


def _write_manifest(
    output_dir: Path,
    jobs: List[FinetuneJob],
    finetune_config: Path,
    skipped: List[str],
) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "base_run_id",
                "source_run_id",
                "target_run_id",
                "strategy",
                "stage1_config",
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
                    "stage1_config": job.stage1_config,
                    "source_files": ";".join(path.name for path in job.source_files),
                    "finetune_config": finetune_config,
                }
            )
    if skipped:
        (output_dir / "skipped.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
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
    model_root = args.model_root.resolve() if args.model_root is not None else None
    output_root = args.output_root.resolve() if args.output_root is not None else None
    selected_run_ids = _normalise_selected_run_ids(args.run_id, args.source_suffix)
    jobs, duplicates, skipped = _collect_jobs(
        csv_paths=csv_paths,
        source_suffix=args.source_suffix,
        target_suffix=args.target_suffix,
        model_root=model_root,
        output_root=output_root,
        selected_run_ids=selected_run_ids,
        allow_incomplete_stage1=args.allow_incomplete_stage1,
        limit=args.limit,
    )
    slurm_copy_root = _resolve_slurm_copy_root(repo_root, args.slurm_copy_root)

    if not jobs:
        print("No jobs found in the provided CSV logs.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(
        args.output_dir,
        jobs,
        args.finetune_config.resolve(),
        skipped,
    )

    print(f"Prepared {len(jobs)} finetuning job(s).")
    print(f"Launcher: {args.launcher.resolve()}")
    print(f"Finetuning config: {args.finetune_config.resolve()}")
    print(f"Output dir: {args.output_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    if model_root is not None:
        print(f"Model root: {model_root}")
    if output_root is not None:
        print(f"Output root: {output_root}")
    if slurm_copy_root is not None:
        print(f"Slurm copy root: {slurm_copy_root}")
    if selected_run_ids is not None:
        print(f"Selected run_ids: {', '.join(sorted(selected_run_ids))}")
    if duplicates:
        print(f"Skipped duplicate base run_ids: {len(duplicates)}")
    if skipped:
        print(f"Skipped incomplete/unready Stage 1 runs: {len(skipped)}")

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
