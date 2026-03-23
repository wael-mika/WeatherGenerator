#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Run Anemoi-aligned structural checks and lightweight EDA on a zarr dataset.

This checker is tailored to datasets produced by ``anemoi-datasets``.
It mirrors the public metadata/statistics expectations from
``anemoi.datasets.validate`` and reuses the statistics arrays written by Anemoi,
while reading the zarr store directly. That keeps the checker robust on shared
filesystems where ``anemoi.datasets.open_dataset()`` can be slow.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import get_codec
from numpy.typing import NDArray

try:
    import anemoi.datasets
    from anemoi.utils.dates import frequency_to_timedelta
except ImportError:  # pragma: no cover - the repo expects this dependency
    anemoi = None
    frequency_to_timedelta = None

LOG = logging.getLogger(__name__)

DEFAULT_DATASET = Path(
    "/capstor/store/cscs/userlab/ch17/data/nasa-imerg-grib-n320-1998-2024-6h-v1.zarr"
)
DEFAULT_OUTPUT_ROOT = Path("scripts/reports")


@dataclass
class Issue:
    severity: str
    check: str
    details: str


@dataclass
class SampleRow:
    variable: str
    time_index: int
    date: str
    nan_count: int
    nan_fraction: float
    negative_count: int
    zero_fraction: float
    mean: float
    stdev: float
    minimum: float
    p99: float
    p999: float
    maximum: float


@dataclass
class VariableSummary:
    variable: str
    stored_mean: float
    stored_stdev: float
    stored_minimum: float
    stored_maximum: float
    stored_finite_count: int
    expected_finite_count: int
    missing_points_in_statistics: int
    missing_fraction_in_statistics: float
    stored_has_nans: bool
    sample_rows: int
    sample_rows_with_nans: int
    sample_rows_with_negatives: int
    sample_max_nan_count: int
    sample_median_nan_count: float
    sample_latest_nan_count: int
    sample_earliest_nan_count: int
    sample_union_nan_points: int
    sample_intersection_nan_points: int
    sample_union_nan_lat_min: float | None
    sample_union_nan_lat_max: float | None
    sample_union_nan_lon_min: float | None
    sample_union_nan_lon_max: float | None
    sample_top_nan_latitudes: str


@dataclass
class ReportData:
    dataset_path: str
    dataset_name: str
    anemoi_version: str | None
    sample_time_steps: int
    metadata_summary: dict[str, Any]
    issues: list[Issue] = field(default_factory=list)
    variables: list[VariableSummary] = field(default_factory=list)
    samples: list[SampleRow] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the Anemoi zarr dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where report files will be written. "
            "Defaults to scripts/reports/<dataset-name>/."
        ),
    )
    parser.add_argument(
        "--sample-time-steps",
        type=int,
        default=64,
        help="Number of evenly spaced time steps to sample for the EDA layer.",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Optional list of variables to analyse. Defaults to all dataset variables.",
    )
    return parser.parse_args()


class RawAnemoiZarr:
    """Minimal reader for Anemoi-style zarr stores."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {self.path}")
        if not self.path.is_dir():
            raise ValueError(f"Expected a directory zarr store, got: {self.path}")

    @property
    def root_attrs(self) -> dict[str, Any]:
        return self._read_json(self.path / ".zattrs")

    @property
    def data_meta(self) -> dict[str, Any]:
        return self.array_meta("data")

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    def array_meta(self, name: str) -> dict[str, Any]:
        return self._read_json(self.path / name / ".zarray")

    def load_array(self, name: str) -> NDArray[Any]:
        meta = self.array_meta(name)
        dtype = np.dtype(meta["dtype"])
        shape = tuple(meta["shape"])
        chunks = tuple(meta["chunks"])
        codec = get_codec(meta["compressor"])

        if len(shape) != 1:
            raise ValueError(f"{name} is not 1D. Shape={shape}")

        array = np.empty(shape, dtype=dtype)
        chunk_len = chunks[0]
        n_chunks = (shape[0] + chunk_len - 1) // chunk_len
        for idx in range(n_chunks):
            decoded = codec.decode((self.path / name / str(idx)).read_bytes())
            chunk = np.frombuffer(decoded, dtype=dtype)
            start = idx * chunk_len
            end = min(start + chunk_len, shape[0])
            array[start:end] = chunk[: end - start]
        return array

    def load_small_stat_array(self, name: str) -> NDArray[Any]:
        meta = self.array_meta(name)
        dtype = np.dtype(meta["dtype"])
        codec = get_codec(meta["compressor"])
        decoded = codec.decode((self.path / name / "0").read_bytes())
        return np.frombuffer(decoded, dtype=dtype).reshape(meta["shape"])

    def load_data_slice(
        self,
        time_index: int,
        variable_index: int,
        ensemble_index: int = 0,
    ) -> NDArray[Any]:
        meta = self.data_meta
        shape = tuple(meta["shape"])
        chunks = tuple(meta["chunks"])
        dtype = np.dtype(meta["dtype"])
        codec = get_codec(meta["compressor"])

        if len(shape) != 4:
            raise ValueError(f"Only 4D Anemoi stores are supported. Shape={shape}")
        if chunks[0] != 1 or chunks[1] != 1 or chunks[2] != 1:
            raise ValueError(
                "This checker expects chunking of 1 along time/variable/ensemble dimensions. "
                f"Found chunks={chunks}."
            )

        point_chunks = (shape[3] + chunks[3] - 1) // chunks[3]
        values = np.empty(shape[3], dtype=dtype)

        # Each time/variable/member field is split over the last-axis chunks only.
        for point_chunk in range(point_chunks):
            chunk_name = f"{time_index}.{variable_index}.{ensemble_index}.{point_chunk}"
            decoded = codec.decode((self.path / "data" / chunk_name).read_bytes())
            chunk = np.frombuffer(decoded, dtype=dtype)
            start = point_chunk * chunks[3]
            end = min(start + chunks[3], shape[3])
            values[start:end] = chunk[: end - start]

        return values


def normalise_output_dir(dataset: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return DEFAULT_OUTPUT_ROOT / dataset.stem


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{value:.{digits}f}"


def format_pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def parse_frequency(freq: str) -> np.timedelta64:
    if frequency_to_timedelta is None:
        raise RuntimeError("anemoi.utils.dates.frequency_to_timedelta is not available.")
    td: timedelta = frequency_to_timedelta(freq)
    seconds = int(td.total_seconds())
    return np.timedelta64(seconds, "s")


def expected_steps(start: np.datetime64, end: np.datetime64, step: np.timedelta64) -> int:
    return int(((end - start) // step) + 1)


def top_latitude_counts(mask: NDArray[Any], latitudes: NDArray[Any], limit: int = 8) -> str:
    if not mask.any():
        return ""
    unique_lats, counts = np.unique(latitudes[mask], return_counts=True)
    order = np.argsort(counts)[::-1]
    return "; ".join(f"{unique_lats[i]:.3f}:{counts[i]}" for i in order[:limit])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    dataset: RawAnemoiZarr,
    sample_time_steps: int,
    requested_variables: list[str] | None,
) -> ReportData:
    attrs = dataset.root_attrs
    data_meta = dataset.data_meta

    variables = list(attrs.get("variables", []))
    if requested_variables is not None:
        missing = [name for name in requested_variables if name not in variables]
        if missing:
            raise ValueError(
                f"Requested variables are not present: {missing}. Available: {variables}"
            )
        variables = requested_variables

    variable_to_index = {name: idx for idx, name in enumerate(attrs.get("variables", []))}
    variable_indices = [variable_to_index[name] for name in variables]

    dates = dataset.load_array("dates")
    latitudes = dataset.load_array("latitudes")
    longitudes = dataset.load_array("longitudes")

    frequency = parse_frequency(attrs["frequency"])
    expected_dataset_steps = expected_steps(dates[0], dates[-1], frequency)
    full_field_size = int(np.prod(attrs["field_shape"]))

    report = ReportData(
        dataset_path=str(dataset.path),
        dataset_name=dataset.path.stem,
        anemoi_version=(
            getattr(anemoi.datasets, "__version__", None) if anemoi is not None else None
        ),
        sample_time_steps=sample_time_steps,
        metadata_summary={
            "frequency": attrs.get("frequency"),
            "shape": attrs.get("shape"),
            "field_shape": attrs.get("field_shape"),
            "variables": variables,
            "start_date": str(dates[0]),
            "end_date": str(dates[-1]),
            "statistics_start_date": attrs.get("statistics_start_date"),
            "statistics_end_date": attrs.get("statistics_end_date"),
            "latitudes_unique": int(len(np.unique(latitudes))),
            "longitudes_unique": int(len(np.unique(longitudes))),
            "gridpoint_count": int(len(latitudes)),
            "allow_nans": bool(attrs.get("allow_nans", False)),
            "variables_with_nans": bool(attrs.get("variables_with_nans", False)),
        },
    )

    if tuple(data_meta["shape"]) != tuple(attrs["shape"]):
        report.issues.append(
            Issue(
                severity="error",
                check="shape_attr_vs_data",
                details=(
                    f"Root attrs shape {attrs['shape']} does not match "
                    f"data/.zarray shape {data_meta['shape']}."
                ),
            )
        )

    if len(dates) != int(data_meta["shape"][0]):
        report.issues.append(
            Issue(
                severity="error",
                check="dates_length",
                details=(
                    f"dates length {len(dates)} does not match "
                    f"data time dimension {data_meta['shape'][0]}."
                ),
            )
        )

    if expected_dataset_steps != len(dates):
        report.issues.append(
            Issue(
                severity="error",
                check="date_regular_frequency",
                details=(
                    f"Date coverage implies {expected_dataset_steps} steps "
                    f"at {attrs['frequency']}, "
                    f"but the dataset contains {len(dates)}."
                ),
            )
        )

    diffs = np.diff(dates)
    if not np.all(diffs == diffs[0]):
        report.issues.append(
            Issue(
                severity="error",
                check="date_monotonicity",
                details="Dates are not strictly regular throughout the store.",
            )
        )

    if len(latitudes) != full_field_size or len(longitudes) != full_field_size:
        report.issues.append(
            Issue(
                severity="error",
                check="coordinate_lengths",
                details=(
                    f"Lat/Lon lengths ({len(latitudes)}, {len(longitudes)}) do not match "
                    f"field_shape product {full_field_size}."
                ),
            )
        )

    if not np.all(np.isfinite(latitudes)):
        report.issues.append(
            Issue(
                severity="error",
                check="latitudes_finite",
                details="Latitudes contain NaN/Inf.",
            )
        )
    if not np.all(np.isfinite(longitudes)):
        report.issues.append(
            Issue(
                severity="error", check="longitudes_finite", details="Longitudes contain NaN/Inf."
            )
        )

    if latitudes.min() < -90 or latitudes.max() > 90:
        report.issues.append(
            Issue(
                severity="error",
                check="latitude_range",
                details=(
                    f"Latitude range is outside [-90, 90]: [{latitudes.min()}, {latitudes.max()}]."
                ),
            )
        )
    if longitudes.min() < -180 or longitudes.max() > 360:
        report.issues.append(
            Issue(
                severity="error",
                check="longitude_range",
                details=(
                    "Longitude range is outside [-180, 360]: "
                    f"[{longitudes.min()}, {longitudes.max()}]."
                ),
            )
        )

    stored_mean = dataset.load_small_stat_array("mean")
    stored_stdev = dataset.load_small_stat_array("stdev")
    stored_minimum = dataset.load_small_stat_array("minimum")
    stored_maximum = dataset.load_small_stat_array("maximum")
    stored_count = dataset.load_small_stat_array("count")
    stored_has_nans = dataset.load_small_stat_array("has_nans")

    if bool(attrs.get("allow_nans", False)) is False and bool(np.any(stored_has_nans)):
        report.issues.append(
            Issue(
                severity="warning",
                check="allow_nans_mismatch",
                details=(
                    "Root metadata says allow_nans=false, but the stored has_nans flag is true. "
                    "This is the main inconsistency in the dataset metadata."
                ),
            )
        )

    stats_start = np.datetime64(attrs["statistics_start_date"])
    stats_end = np.datetime64(attrs["statistics_end_date"])
    expected_stats_steps = expected_steps(stats_start, stats_end, frequency)
    expected_stats_count = expected_stats_steps * full_field_size

    if stats_end < dates[-1]:
        report.issues.append(
            Issue(
                severity="warning",
                check="statistics_coverage",
                details=(
                    f"Stored statistics stop at {stats_end}, while the dataset "
                    f"extends to {dates[-1]}. "
                    "The statistics therefore do not cover the full time range."
                ),
            )
        )

    sample_indices = np.unique(np.linspace(0, len(dates) - 1, sample_time_steps, dtype=int))
    samples: list[SampleRow] = []

    for variable_name, variable_index in zip(variables, variable_indices, strict=True):
        union_nan_mask = np.zeros(full_field_size, dtype=bool)
        intersection_nan_mask = np.ones(full_field_size, dtype=bool)
        variable_samples: list[SampleRow] = []

        for time_index in sample_indices:
            values = dataset.load_data_slice(int(time_index), variable_index)
            finite = np.isfinite(values)
            nan_mask = np.isnan(values)
            union_nan_mask |= nan_mask
            intersection_nan_mask &= nan_mask

            row = SampleRow(
                variable=variable_name,
                time_index=int(time_index),
                date=str(dates[time_index]),
                nan_count=int(nan_mask.sum()),
                nan_fraction=float(nan_mask.mean()),
                negative_count=int((values[finite] < 0).sum()),
                zero_fraction=float((values[finite] == 0).mean()) if finite.any() else float("nan"),
                mean=float(np.nanmean(values)),
                stdev=float(np.nanstd(values)),
                minimum=float(np.nanmin(values)),
                p99=float(np.nanquantile(values, 0.99)),
                p999=float(np.nanquantile(values, 0.999)),
                maximum=float(np.nanmax(values)),
            )
            variable_samples.append(row)
            samples.append(row)

        earliest = variable_samples[0]
        latest = variable_samples[-1]
        stored_idx = variable_to_index[variable_name]
        missing_points_in_statistics = int(expected_stats_count - int(stored_count[stored_idx]))
        missing_fraction_in_statistics = missing_points_in_statistics / expected_stats_count

        summary = VariableSummary(
            variable=variable_name,
            stored_mean=float(stored_mean[stored_idx]),
            stored_stdev=float(stored_stdev[stored_idx]),
            stored_minimum=float(stored_minimum[stored_idx]),
            stored_maximum=float(stored_maximum[stored_idx]),
            stored_finite_count=int(stored_count[stored_idx]),
            expected_finite_count=int(expected_stats_count),
            missing_points_in_statistics=missing_points_in_statistics,
            missing_fraction_in_statistics=float(missing_fraction_in_statistics),
            stored_has_nans=bool(stored_has_nans[stored_idx]),
            sample_rows=len(variable_samples),
            sample_rows_with_nans=sum(row.nan_count > 0 for row in variable_samples),
            sample_rows_with_negatives=sum(row.negative_count > 0 for row in variable_samples),
            sample_max_nan_count=max(row.nan_count for row in variable_samples),
            sample_median_nan_count=float(np.median([row.nan_count for row in variable_samples])),
            sample_latest_nan_count=latest.nan_count,
            sample_earliest_nan_count=earliest.nan_count,
            sample_union_nan_points=int(union_nan_mask.sum()),
            sample_intersection_nan_points=int(intersection_nan_mask.sum()),
            sample_union_nan_lat_min=(
                float(latitudes[union_nan_mask].min()) if union_nan_mask.any() else None
            ),
            sample_union_nan_lat_max=(
                float(latitudes[union_nan_mask].max()) if union_nan_mask.any() else None
            ),
            sample_union_nan_lon_min=(
                float(longitudes[union_nan_mask].min()) if union_nan_mask.any() else None
            ),
            sample_union_nan_lon_max=(
                float(longitudes[union_nan_mask].max()) if union_nan_mask.any() else None
            ),
            sample_top_nan_latitudes=top_latitude_counts(union_nan_mask, latitudes),
        )
        report.variables.append(summary)

        if summary.missing_points_in_statistics > 0:
            report.issues.append(
                Issue(
                    severity="warning",
                    check=f"{variable_name}_statistics_missing_points",
                    details=(
                        f"{variable_name} has "
                        f"{summary.missing_points_in_statistics:,} "
                        "missing/non-finite points "
                        f"in the stored statistics interval "
                        f"({format_pct(summary.missing_fraction_in_statistics)} "
                        "of the expected finite count)."
                    ),
                )
            )

        if summary.sample_rows_with_nans > 0:
            report.issues.append(
                Issue(
                    severity="warning",
                    check=f"{variable_name}_sampled_nans",
                    details=(
                        f"{variable_name} had NaNs in "
                        f"{summary.sample_rows_with_nans}/{summary.sample_rows} "
                        "sampled time steps. Sampled NaN counts ranged from "
                        f"{summary.sample_earliest_nan_count} at the "
                        f"start to {summary.sample_latest_nan_count} at the end."
                    ),
                )
            )

        if summary.sample_rows_with_negatives > 0:
            report.issues.append(
                Issue(
                    severity="warning",
                    check=f"{variable_name}_sampled_negatives",
                    details=(
                        f"{variable_name} had negative values in sampled data, "
                        "which is unusual for precipitation."
                    ),
                )
            )

        if variable_name in {"tp", "prate"} and summary.stored_minimum < 0:
            report.issues.append(
                Issue(
                    severity="warning",
                    check=f"{variable_name}_stored_minimum",
                    details=(
                        f"{variable_name} stored minimum is negative ({summary.stored_minimum})."
                    ),
                )
            )

    report.samples = samples
    return report


def write_markdown(report: ReportData, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Dataset Check Report: {report.dataset_name}")
    lines.append("")
    lines.append(f"- Dataset: `{report.dataset_path}`")
    lines.append(f"- Anemoi version: `{report.anemoi_version or 'unavailable'}`")
    lines.append(f"- Sampled time steps: `{report.sample_time_steps}`")
    lines.append("")
    lines.append("## Metadata Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for key, value in report.metadata_summary.items():
        if isinstance(value, list):
            rendered = ", ".join(str(v) for v in value)
        else:
            rendered = str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if report.issues:
        lines.append("| Severity | Check | Details |")
        lines.append("| --- | --- | --- |")
        for issue in report.issues:
            lines.append(f"| {issue.severity} | {issue.check} | {issue.details} |")
    else:
        lines.append("No issues were detected.")
    lines.append("")
    lines.append("## Variable Summary")
    lines.append("")
    lines.append(
        "| Variable | Stored Mean | Stored Stdev | Stored Min | Stored Max | Missing In Stats | "
        "Sampled Rows With NaNs | Max Sample NaNs | NaN Region |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report.variables:
        lines.append(markdown_variable_row(row))
    lines.append("")
    lines.append("## Top Sampled Rows By NaN Count")
    lines.append("")
    lines.append("| Variable | Date | NaN Count | Zero Fraction | Mean | P99 | P999 | Max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(report.samples, key=lambda item: item.nan_count, reverse=True)[:10]:
        lines.append(markdown_sample_row(row))
    lines.append("")
    lines.append("## Top Sampled Rows By Maximum Value")
    lines.append("")
    lines.append("| Variable | Date | NaN Count | Zero Fraction | Mean | P99 | P999 | Max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(report.samples, key=lambda item: item.maximum, reverse=True)[:10]:
        lines.append(markdown_sample_row(row))
    path.write_text("\n".join(lines) + "\n")


def serialise_report(report: ReportData) -> dict[str, Any]:
    payload = asdict(report)
    payload["issues"] = [asdict(issue) for issue in report.issues]
    payload["variables"] = [asdict(item) for item in report.variables]
    payload["samples"] = [asdict(item) for item in report.samples]
    return payload


def markdown_variable_row(row: VariableSummary) -> str:
    nan_region = ""
    if row.sample_union_nan_points > 0:
        nan_region = "".join(
            [
                f"{row.sample_union_nan_points} pts, ",
                "lat [",
                format_float(row.sample_union_nan_lat_min, 3),
                ", ",
                format_float(row.sample_union_nan_lat_max, 3),
                "], lon [",
                format_float(row.sample_union_nan_lon_min, 3),
                ", ",
                format_float(row.sample_union_nan_lon_max, 3),
                "]",
            ]
        )

    return "".join(
        [
            f"| {row.variable} | ",
            f"{format_float(row.stored_mean)} | ",
            f"{format_float(row.stored_stdev)} | ",
            f"{format_float(row.stored_minimum)} | ",
            f"{format_float(row.stored_maximum)} | ",
            f"{row.missing_points_in_statistics:,} ",
            f"({format_pct(row.missing_fraction_in_statistics)}) | ",
            f"{row.sample_rows_with_nans}/{row.sample_rows} | ",
            f"{row.sample_max_nan_count} | ",
            f"{nan_region} |",
        ]
    )


def markdown_sample_row(row: SampleRow) -> str:
    return "".join(
        [
            f"| {row.variable} | ",
            f"{row.date} | ",
            f"{row.nan_count} | ",
            f"{format_float(row.zero_fraction)} | ",
            f"{format_float(row.mean)} | ",
            f"{format_float(row.p99)} | ",
            f"{format_float(row.p999)} | ",
            f"{format_float(row.maximum)} |",
        ]
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset = RawAnemoiZarr(args.dataset)
    report = build_report(
        dataset,
        sample_time_steps=args.sample_time_steps,
        requested_variables=args.variables,
    )

    output_dir = normalise_output_dir(args.dataset, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {
            "metric": key,
            "value": json.dumps(value) if isinstance(value, list | dict) else value,
        }
        for key, value in report.metadata_summary.items()
    ]
    issue_rows = [asdict(issue) for issue in report.issues]
    variable_rows = [asdict(item) for item in report.variables]
    sample_rows = [asdict(item) for item in report.samples]

    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "issues.csv", issue_rows)
    write_csv(output_dir / "variables.csv", variable_rows)
    write_csv(output_dir / "sample_eda.csv", sample_rows)
    (output_dir / "report.json").write_text(json.dumps(serialise_report(report), indent=2))
    write_markdown(report, output_dir / "report.md")

    LOG.info("Report written to %s", output_dir)
    LOG.info("- %s", output_dir / "report.md")
    LOG.info("- %s", output_dir / "report.json")
    LOG.info("- %s", output_dir / "summary.csv")
    LOG.info("- %s", output_dir / "issues.csv")
    LOG.info("- %s", output_dir / "variables.csv")
    LOG.info("- %s", output_dir / "sample_eda.csv")


if __name__ == "__main__":
    main()
