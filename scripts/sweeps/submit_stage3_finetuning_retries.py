#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


SCHEDULER_ERROR = "ZeroDivisionError: float division by zero"
QABL_MISMATCH_ERROR = "size mismatch for module.encoder.embed_engine.embeds.ERA5.unembed"
DEFAULT_SOURCE_SUFFIX = "-stage1"
DEFAULT_TARGET_SUFFIX = "-stage3-fn"


@dataclass(frozen=True)
class RetryJob:
    base_run_id: str
    source_run_id: str
    failed_run_id: str
    target_run_id: str
    failure_mode: str
    finetune_config: Path
    output_file: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_launcher(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "launch-slurm.py"


def _platform_env_script(repo_root: Path) -> Path:
    return repo_root.parent / "WeatherGenerator-private" / "hpc" / "platform-env.py"


def _default_standard_config(repo_root: Path) -> Path:
    return repo_root / "config" / "config_jepa_finetuning.yml"


def _default_qabl_config(repo_root: Path) -> Path:
    return repo_root / "config" / "config_jepa_finetuning_q_abl_stage3.yml"


def _default_output_dir(repo_root: Path) -> Path:
    return repo_root / "config" / "sweep_runs" / "stage3_finetuning_retries"


def _shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _load_paths_yml_value(config_path: Path, key: str) -> str | None:
    if not config_path.is_file():
        return None

    pattern = re.compile(r"^\s*%s\s*:\s*['\"]?([^'\"]+)['\"]?\s*$" % re.escape(key))
    with config_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = pattern.match(line)
            if match:
                return match.group(1).strip()
    return None


def _resolve_platform_config(repo_root: Path) -> Path | None:
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


def _resolve_default_root(repo_root: Path, key: str) -> Path | None:
    config_path = _resolve_platform_config(repo_root)
    if config_path is None:
        return None

    value = _load_paths_yml_value(config_path, key)
    if not value:
        return None
    return Path(value).resolve()


def _resolve_default_output_root(repo_root: Path) -> Path | None:
    shared_root = _resolve_default_root(repo_root, "path_shared_working_dir")
    if shared_root is None:
        return None
    return (shared_root / "output").resolve()


def _resolve_default_model_root(repo_root: Path) -> Path | None:
    return _resolve_default_root(repo_root, "model_path")


def _resolve_default_slurm_root(repo_root: Path) -> Path | None:
    return _resolve_default_root(repo_root, "path_shared_slurm_dir")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    repo_root = _repo_root()
    today = date.today().isoformat()

    parser = argparse.ArgumentParser(
        description=(
            "Scan Stage 2 output logs, identify failed continuation runs, and prepare Stage 3 "
            "resubmissions. Scheduler failures are retried with the standard finetuning config. "
            "Runs whose Stage 1 checkpoint was trained on era5_1deg_abl are retried with a "
            "q-only finetuning config that keeps the Stage 1 ERA5 channel schema."
        )
    )
    parser.add_argument(
        "--date",
        default=today,
        help="Only inspect Stage 2 output files modified on this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_resolve_default_output_root(repo_root),
        help="Shared output directory containing output_*stage2*.txt files.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=_resolve_default_model_root(repo_root),
        help="Shared model directory containing <run_id>/model_<run_id>_latest.json.",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=_default_launcher(repo_root),
        help="Path to WeatherGenerator-private/hpc/launch-slurm.py.",
    )
    parser.add_argument(
        "--standard-finetune-config",
        type=Path,
        default=_default_standard_config(repo_root),
        help="Finetuning config used for the scheduler-failure retry group.",
    )
    parser.add_argument(
        "--qabl-finetune-config",
        type=Path,
        default=_default_qabl_config(repo_root),
        help="Finetuning config used for era5_1deg_abl Stage 1 checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(repo_root),
        help="Directory where the manifest is written.",
    )
    parser.add_argument(
        "--source-suffix",
        default=DEFAULT_SOURCE_SUFFIX,
        help="Suffix stripped from the source run id to recover the base run id.",
    )
    parser.add_argument(
        "--target-suffix",
        default=DEFAULT_TARGET_SUFFIX,
        help="Suffix appended to the base run id for the new Stage 3 retry run.",
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
        help="Optional cap on the number of retries to prepare.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs. Without this flag, commands are only printed.",
    )
    parser.add_argument(
        "--slurm-copy-root",
        type=Path,
        default=_resolve_default_slurm_root(repo_root),
        help="Optional shared slurm working directory used to create stage aliases when needed.",
    )
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="Include unsupported failures in the manifest summary output.",
    )

    args, launcher_args = parser.parse_known_args()
    return args, launcher_args


def _validate_paths(args: argparse.Namespace) -> None:
    if args.output_root is None:
        raise FileNotFoundError(
            "Could not resolve the shared output directory automatically. "
            "Please pass --output-root."
        )
    if args.model_root is None:
        raise FileNotFoundError(
            "Could not resolve the shared model directory automatically. "
            "Please pass --model-root."
        )

    if not args.output_root.is_dir():
        raise FileNotFoundError(f"Output root not found: {args.output_root}")
    if not args.model_root.is_dir():
        raise FileNotFoundError(f"Model root not found: {args.model_root}")

    launcher_path = args.launcher.resolve()
    if not launcher_path.is_file():
        raise FileNotFoundError(f"Launcher not found: {launcher_path}")

    for config_path in (args.standard_finetune_config, args.qabl_finetune_config):
        resolved = config_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Finetuning config not found: {resolved}")


def _iter_stage2_outputs(output_root: Path, day: date) -> list[Path]:
    start_dt = datetime.combine(day, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()

    files = []
    for path in output_root.glob("output_*stage2*.txt"):
        mtime = path.stat().st_mtime
        if start_ts <= mtime < end_ts:
            files.append(path)

    return sorted(files, key=lambda path: path.stat().st_mtime)


def _classify_failure(text: str) -> str | None:
    if SCHEDULER_ERROR in text:
        return "scheduler_zero_division"
    if QABL_MISMATCH_ERROR in text:
        return "checkpoint_shape_mismatch"
    return None


def _extract_source_run_id(text: str) -> str | None:
    match = re.search(r"Continuing run with id=([^\s]+)", text)
    if match is None:
        return None
    return match.group(1).strip()


def _strip_source_suffix(run_id: str, source_suffix: str) -> str:
    if run_id.endswith(source_suffix):
        return run_id[: -len(source_suffix)]
    return run_id


def _load_run_metadata(model_root: Path, run_id: str) -> dict | None:
    model_dir = model_root / run_id
    preferred = model_dir / f"model_{run_id}_latest.json"
    if preferred.is_file():
        return json.loads(preferred.read_text(encoding="utf-8"))

    candidates = sorted(model_dir.glob(f"model_{run_id}_*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _pick_target_run_id(base_run_id: str, target_suffix: str, model_root: Path) -> str:
    candidate = f"{base_run_id}{target_suffix}"
    if not (model_root / candidate).exists():
        return candidate

    idx = 1
    while (model_root / f"{candidate}{idx}").exists():
        idx += 1
    return f"{candidate}{idx}"


def _choose_finetune_config(
    failure_mode: str,
    source_run_id: str,
    model_root: Path,
    standard_config: Path,
    qabl_config: Path,
) -> tuple[Path | None, str]:
    if failure_mode == "scheduler_zero_division":
        return standard_config, "scheduler retry"

    metadata = _load_run_metadata(model_root, source_run_id)
    if metadata is None:
        return None, "missing source model metadata"

    streams_directory = str(metadata.get("streams_directory", ""))
    if streams_directory.endswith("era5_1deg_abl/"):
        return qabl_config, "q_abl retry"

    return None, f"unsupported mismatch source stream: {streams_directory or 'unknown'}"


def _collect_jobs(
    args: argparse.Namespace,
) -> tuple[list[RetryJob], list[str]]:
    retry_jobs: list[RetryJob] = []
    skipped: list[str] = []
    inspected_day = datetime.strptime(args.date, "%Y-%m-%d").date()
    output_files = _iter_stage2_outputs(args.output_root.resolve(), inspected_day)

    for output_file in output_files:
        text = output_file.read_text(encoding="utf-8", errors="ignore")
        failure_mode = _classify_failure(text)
        if failure_mode is None:
            continue

        source_run_id = _extract_source_run_id(text)
        if source_run_id is None:
            skipped.append(f"{output_file.name}: could not determine source run id")
            continue

        base_run_id = _strip_source_suffix(source_run_id, args.source_suffix)
        target_run_id = _pick_target_run_id(
            base_run_id=base_run_id,
            target_suffix=args.target_suffix,
            model_root=args.model_root.resolve(),
        )
        finetune_config, reason = _choose_finetune_config(
            failure_mode=failure_mode,
            source_run_id=source_run_id,
            model_root=args.model_root.resolve(),
            standard_config=args.standard_finetune_config.resolve(),
            qabl_config=args.qabl_finetune_config.resolve(),
        )

        failed_run_id = output_file.name.removeprefix("output_").removesuffix(".txt")

        if finetune_config is None:
            skipped.append(f"{failed_run_id}: {reason}")
            continue

        retry_jobs.append(
            RetryJob(
                base_run_id=base_run_id,
                source_run_id=source_run_id,
                failed_run_id=failed_run_id,
                target_run_id=target_run_id,
                failure_mode=failure_mode,
                finetune_config=finetune_config,
                output_file=output_file,
            )
        )

    retry_jobs.sort(key=lambda job: (job.failure_mode, job.base_run_id))
    if args.limit is not None:
        retry_jobs = retry_jobs[: args.limit]

    return retry_jobs, skipped


def _write_manifest(output_dir: Path, jobs: list[RetryJob], skipped: list[str]) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "base_run_id",
                "source_run_id",
                "failed_run_id",
                "target_run_id",
                "failure_mode",
                "finetune_config",
                "output_file",
            ],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "base_run_id": job.base_run_id,
                    "source_run_id": job.source_run_id,
                    "failed_run_id": job.failed_run_id,
                    "target_run_id": job.target_run_id,
                    "failure_mode": job.failure_mode,
                    "finetune_config": str(job.finetune_config),
                    "output_file": str(job.output_file),
                }
            )

    if skipped:
        skipped_path = output_dir / "skipped.txt"
        skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")

    return manifest_path


def _ensure_stage_alias(
    source_run_id: str,
    base_run_id: str,
    slurm_copy_root: Path | None,
    submit: bool,
) -> str | None:
    if slurm_copy_root is None:
        return None

    base_copy_dir = slurm_copy_root / f"slurm_weathergen_{base_run_id}_dir"
    stage_copy_dir = slurm_copy_root / f"slurm_weathergen_{source_run_id}_dir"

    if stage_copy_dir.exists():
        return None
    if not base_copy_dir.exists():
        return f"missing base slurm dir for alias: {base_copy_dir}"

    alias_msg = f"{stage_copy_dir} -> {base_copy_dir}"
    if submit:
        stage_copy_dir.symlink_to(base_copy_dir)
        return f"created stage alias: {alias_msg}"

    return f"would create stage alias: {alias_msg}"


def _build_command(
    launcher: Path,
    finetune_config: Path,
    source_run_id: str,
    target_run_id: str,
    nodes: int,
    launcher_args: list[str],
) -> list[str]:
    cmd = [
        str(launcher),
        "--from-run-id",
        source_run_id,
        "--run-id",
        target_run_id,
        "--config",
        str(finetune_config),
        "--nodes",
        str(nodes),
    ]
    cmd.extend(launcher_args)
    return cmd


def main() -> int:
    args, launcher_args = _parse_args()
    _validate_paths(args)

    jobs, skipped = _collect_jobs(args)
    if not jobs:
        print("No supported failed Stage 2 runs found for the requested date.", file=sys.stderr)
        if skipped:
            print("\nSkipped entries:", file=sys.stderr)
            for item in skipped:
                print(f"  - {item}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(args.output_dir.resolve(), jobs, skipped)

    print(f"Prepared {len(jobs)} Stage 3 retry job(s) for {args.date}.")
    print(f"Launcher: {args.launcher.resolve()}")
    print(f"Output root: {args.output_root.resolve()}")
    print(f"Model root: {args.model_root.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    if args.slurm_copy_root is not None:
        print(f"Slurm copy root: {args.slurm_copy_root.resolve()}")
    if skipped:
        print(f"Skipped unsupported failures: {len(skipped)}")

    failures: list[tuple[RetryJob, int]] = []
    for job in jobs:
        alias_note = _ensure_stage_alias(
            source_run_id=job.source_run_id,
            base_run_id=job.base_run_id,
            slurm_copy_root=None if args.slurm_copy_root is None else args.slurm_copy_root.resolve(),
            submit=args.submit,
        )
        cmd = _build_command(
            launcher=args.launcher.resolve(),
            finetune_config=job.finetune_config.resolve(),
            source_run_id=job.source_run_id,
            target_run_id=job.target_run_id,
            nodes=args.nodes,
            launcher_args=launcher_args,
        )

        print("")
        print(
            f"[{job.target_run_id}] source={job.source_run_id} "
            f"failure={job.failure_mode} failed_run={job.failed_run_id}"
        )
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

    if skipped and args.include_unsupported:
        print("")
        print("Skipped unsupported failures:")
        for item in skipped:
            print(f"  - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
