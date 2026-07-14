#!/usr/bin/env python3

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Visualize how IMERG precipitation transforms compress the raw distribution tail.

Example:
    .venv/bin/python scripts/plot_imerg_transform_effects.py \
        --data-dir /path/to/data \
        --stream-config config/streams/raina_trans/imerg_transform.yml
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-weathergen")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/weathergen-cache")

import matplotlib
import numpy as np
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_STREAM_CONFIG = Path("config/streams/raina_trans/imerg_transform.yml")
DEFAULT_OUT_DIR = Path("plots/imerg_transform_effects")
TAIL_PERCENTILES = np.array([50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 99.99, 100.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot how the configured IMERG transforms change the precipitation distribution, "
            "with a focus on the upper tail."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Direct path to the IMERG Anemoi zarr dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base directory containing the dataset filename from the stream config.",
    )
    parser.add_argument(
        "--stream-config",
        type=Path,
        default=DEFAULT_STREAM_CONFIG,
        help=f"IMERG stream config to read defaults from. Default: {DEFAULT_STREAM_CONFIG}",
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        default=None,
        help="Top-level stream key in the stream config. Default: first key in the file.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="tp",
        help="Dataset channel to analyze. Default: tp",
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=96,
        help="Number of evenly spaced timesteps to sample from the dataset. Default: 96",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2_000_000,
        help="Maximum number of precipitation points to retain after loading. Default: 2000000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used if downsampling is needed. Default: 7",
    )
    parser.add_argument(
        "--transform-scale",
        type=float,
        default=1000.0,
        help="Scale applied before the transforms. Default: 1000.0 (m -> mm)",
    )
    parser.add_argument(
        "--arcsinh-alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.15],
        help="Arcsinh alpha values to compare. Default: 0.05 0.15",
    )
    parser.add_argument(
        "--log10-offset",
        type=float,
        default=1.0,
        help="Offset for the log10 transform. Default: 1.0",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for summary tables and figures. Default: {DEFAULT_OUT_DIR}",
    )
    return parser.parse_args()


def load_stream_config(path: Path, stream_name: str | None) -> tuple[str, dict]:
    cfg = OmegaConf.load(path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict) or len(cfg_dict) == 0:
        raise ValueError(f"Expected a non-empty stream config mapping in {path}.")

    selected_name = stream_name or next(iter(cfg_dict.keys()))
    if selected_name not in cfg_dict:
        raise KeyError(f"Stream '{selected_name}' not found in {path}.")

    stream_cfg = dict(cfg_dict[selected_name])
    stream_cfg["name"] = selected_name
    return selected_name, stream_cfg


def resolve_dataset_path(
    dataset_path: Path | None,
    data_dir: Path | None,
    stream_cfg: dict,
) -> Path:
    if dataset_path is not None:
        return dataset_path

    if data_dir is None:
        raise ValueError("Provide either --dataset-path or --data-dir.")

    filenames = stream_cfg.get("filenames", [])
    if len(filenames) == 0:
        raise ValueError("The selected stream config does not define any filenames.")

    resolved = data_dir / filenames[0]
    return resolved


def load_raw_channel_values(
    dataset_path: Path,
    channel: str,
    max_time_steps: int,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import anemoi.datasets as anemoi_datasets

    ds = anemoi_datasets.open_dataset(dataset_path)
    if channel not in ds.name_to_index:
        raise KeyError(f"Channel '{channel}' not found in dataset variables.")

    var_idx = ds.name_to_index[channel]
    num_vars = len(ds.variables)
    num_time_steps = len(ds)
    if num_time_steps == 0:
        raise ValueError(f"Dataset {dataset_path} has no timesteps.")

    chosen_steps = np.unique(
        np.linspace(0, num_time_steps - 1, num=min(max_time_steps, num_time_steps), dtype=int)
    )

    values_all = []
    for step_idx in chosen_steps:
        block = ds[step_idx : step_idx + 1][:, :, 0].astype(np.float32)
        values = extract_channel(block, var_idx, num_vars)
        values = values[np.isfinite(values)]
        if values.size > 0:
            values_all.append(values)

    if len(values_all) == 0:
        raise ValueError("No finite values were loaded from the requested dataset channel.")

    values = np.concatenate(values_all, axis=0)
    if max_samples > 0 and values.size > max_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(values.size, size=max_samples, replace=False)
        values = values[chosen]

    return values.astype(np.float64), chosen_steps


def extract_channel(block: np.ndarray, var_idx: int, num_vars: int) -> np.ndarray:
    if block.ndim != 3:
        raise ValueError(f"Expected a 3D Anemoi block, got shape {block.shape}.")

    if block.shape[1] == num_vars:
        return block[:, var_idx, :].reshape(-1)

    if block.shape[2] == num_vars:
        return block[:, :, var_idx].reshape(-1)

    raise ValueError(
        "Could not infer the variable axis in the Anemoi block. "
        + f"Block shape: {block.shape}, num_vars: {num_vars}"
    )


def to_raw_mm(raw_values: np.ndarray, transform_scale: float) -> np.ndarray:
    raw_mm = raw_values * transform_scale
    return raw_mm[np.isfinite(raw_mm)]


def transform_arcsinh(raw_mm: np.ndarray, alpha: float) -> np.ndarray:
    return np.arcsinh(raw_mm / alpha)


def transform_log10(raw_mm: np.ndarray, offset: float) -> np.ndarray:
    return np.log10(raw_mm + offset)


def compute_summary(values: np.ndarray) -> dict[str, float]:
    positive = values[values > 0.0]
    summary = {
        "count": float(values.size),
        "zeros_fraction": float(np.mean(values == 0.0)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }

    quantiles = np.percentile(values, TAIL_PERCENTILES)
    for q, v in zip(TAIL_PERCENTILES, quantiles, strict=True):
        key = f"q{str(q).replace('.', '_')}"
        summary[key] = float(v)

    if positive.size > 0:
        summary["min_positive"] = float(np.min(positive))
        summary["mean_positive"] = float(np.mean(positive))
    else:
        summary["min_positive"] = float("nan")
        summary["mean_positive"] = float("nan")

    return summary


def compute_all_summaries(
    raw_mm: np.ndarray,
    arcsinh_alphas: list[float],
    log10_offset: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    transformed = {"raw_mm": raw_mm}
    for alpha in arcsinh_alphas:
        transformed[f"arcsinh_alpha_{alpha:g}"] = transform_arcsinh(raw_mm, alpha)
    transformed[f"log10_offset_{log10_offset:g}"] = transform_log10(raw_mm, log10_offset)

    summaries = {name: compute_summary(values) for name, values in transformed.items()}
    return transformed, summaries


def save_summary_csv(out_dir: Path, summaries: dict[str, dict[str, float]]) -> Path:
    fieldnames = ["name"]
    for summary in summaries.values():
        fieldnames.extend(summary.keys())
    fieldnames = list(dict.fromkeys(fieldnames))

    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, summary in summaries.items():
            row = {"name": name}
            row.update(summary)
            writer.writerow(row)
    return path


def save_quantiles_csv(out_dir: Path, transformed: dict[str, np.ndarray]) -> Path:
    percentiles = np.array([0.0, 50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 99.99, 100.0], dtype=float)
    path = out_dir / "quantiles.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", *[f"p{p:g}" for p in percentiles]])
        for name, values in transformed.items():
            writer.writerow([name, *np.percentile(values, percentiles)])
    return path


def plot_mapping(
    raw_mm: np.ndarray, arcsinh_alphas: list[float], log10_offset: float, out_dir: Path
):
    positive = raw_mm[raw_mm > 0.0]
    if positive.size == 0:
        raise ValueError("No positive precipitation values available for plotting.")

    x_min = max(float(np.min(positive)), 1e-6)
    x_max = max(float(np.percentile(raw_mm, 99.99)), x_min * 10.0)
    x = np.geomspace(x_min, x_max, num=800)

    fig, ax = plt.subplots(figsize=(9, 6))
    for alpha in arcsinh_alphas:
        ax.plot(x, transform_arcsinh(x, alpha), label=f"arcsinh alpha={alpha:g}", linewidth=2)
    ax.plot(
        x,
        transform_log10(x, log10_offset),
        label=f"log10 offset={log10_offset:g}",
        linewidth=2,
    )

    for marker in [0.1, 1.0, 10.0, 50.0, 100.0]:
        if marker < x_max:
            ax.axvline(marker, color="0.85", linewidth=0.8, linestyle="--")

    ax.set_xscale("log")
    ax.set_xlabel("Raw precipitation (mm)")
    ax.set_ylabel("Transformed value")
    ax.set_title("IMERG transform mapping curves")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mapping_curves.png", dpi=180)
    plt.close(fig)


def plot_histograms(
    raw_mm: np.ndarray,
    transformed: dict[str, np.ndarray],
    out_dir: Path,
):
    positive = raw_mm[raw_mm > 0.0]
    if positive.size == 0:
        raise ValueError("No positive precipitation values available for histogram plotting.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    raw_bins = np.geomspace(max(float(np.min(positive)), 1e-6), float(np.max(positive)), num=120)
    axes[0].hist(positive, bins=raw_bins, histtype="step", linewidth=1.8, color="#0F4C81")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Raw precipitation (mm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Raw IMERG precipitation tail")
    axes[0].grid(True, which="both", alpha=0.25)

    colors = {
        "raw_mm": "#0F4C81",
        "log10": "#B23A48",
        "arcsinh": "#3A7D44",
    }
    for name, values in transformed.items():
        if name == "raw_mm":
            continue
        label = name.replace("_", " ")
        bins = np.linspace(float(np.min(values)), float(np.max(values)), num=120)
        color_key = "log10" if name.startswith("log10") else "arcsinh"
        axes[1].hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            label=label,
            color=colors[color_key],
            log=True,
        )

    axes[1].set_xlabel("Transformed value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution after transformation")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "histograms.png", dpi=180)
    plt.close(fig)


def plot_tail_quantiles(transformed: dict[str, np.ndarray], out_dir: Path):
    exceedance = np.array([10.0, 5.0, 1.0, 0.5, 0.1, 0.01], dtype=float)
    percentiles = 100.0 - exceedance

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    raw_q = np.percentile(transformed["raw_mm"], percentiles)
    axes[0].plot(exceedance, raw_q, marker="o", linewidth=2, color="#0F4C81")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Exceedance probability (%)")
    axes[0].set_ylabel("Raw precipitation (mm)")
    axes[0].set_title("Raw precipitation upper tail")
    axes[0].grid(True, which="both", alpha=0.25)

    for name, values in transformed.items():
        if name == "raw_mm":
            continue
        q = np.percentile(values, percentiles)
        axes[1].plot(exceedance, q, marker="o", linewidth=2, label=name.replace("_", " "))

    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Exceedance probability (%)")
    axes[1].set_ylabel("Transformed value")
    axes[1].set_title("How the transforms compress the tail")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "tail_quantiles.png", dpi=180)
    plt.close(fig)


def print_summary(
    dataset_path: Path,
    chosen_steps: np.ndarray,
    summaries: dict[str, dict[str, float]],
    out_dir: Path,
):
    print(f"Dataset: {dataset_path}")
    print(f"Sampled timesteps: {len(chosen_steps)}")
    print(f"Output directory: {out_dir}")
    print("")

    headers = [
        "name",
        "count",
        "zeros_fraction",
        "mean",
        "std",
        "q90",
        "q99",
        "q99_9",
        "q99_99",
        "q100",
    ]
    print(",".join(headers))
    for name, summary in summaries.items():
        row = [name]
        for key in headers[1:]:
            value = summary.get(key, float("nan"))
            row.append(f"{value:.8g}")
        print(",".join(row))


def main() -> None:
    args = parse_args()

    stream_name, stream_cfg = load_stream_config(args.stream_config, args.stream_name)
    dataset_path = resolve_dataset_path(args.dataset_path, args.data_dir, stream_cfg)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_values, chosen_steps = load_raw_channel_values(
        dataset_path=dataset_path,
        channel=args.channel,
        max_time_steps=args.max_time_steps,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    raw_mm = to_raw_mm(raw_values, args.transform_scale)

    transformed, summaries = compute_all_summaries(
        raw_mm=raw_mm,
        arcsinh_alphas=list(args.arcsinh_alphas),
        log10_offset=args.log10_offset,
    )

    save_summary_csv(out_dir, summaries)
    save_quantiles_csv(out_dir, transformed)
    plot_mapping(raw_mm, list(args.arcsinh_alphas), args.log10_offset, out_dir)
    plot_histograms(raw_mm, transformed, out_dir)
    plot_tail_quantiles(transformed, out_dir)

    print_summary(dataset_path, chosen_steps, summaries, out_dir)
    print("")
    print(f"Stream config: {args.stream_config} ({stream_name})")
    print("Wrote:")
    print(f"  - {out_dir / 'summary.csv'}")
    print(f"  - {out_dir / 'quantiles.csv'}")
    print(f"  - {out_dir / 'mapping_curves.png'}")
    print(f"  - {out_dir / 'histograms.png'}")
    print(f"  - {out_dir / 'tail_quantiles.png'}")


if __name__ == "__main__":
    main()
