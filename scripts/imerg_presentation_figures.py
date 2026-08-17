#!/usr/bin/env python
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Presentation figures for the IMERG diagnostic-decoder finetunes.

Three inference runs, all decoding IMERG ``tp`` on the N320 grid from a frozen backbone:

===========  ===========================================  =====================
run          backbone lineage                             rollout in store
===========  ===========================================  =====================
xfaps4wq     Forecast: c71eo6pu_ssl -> hmpa42d2            2 steps (+6h, +12h)
r15j90ns     JEPA:     af90zz71_ssl -> nescpnb2            2 steps (+6h, +12h)
z71y2ik8     JEPA:     af90zz71_ssl -> ... -> v3gs9fwn     40 steps (+6h .. +240h)
===========  ===========================================  =====================

Two facts that the figures have to respect, both verified against the code and the stores:

1. ``calc_seeps`` returns ``1.0 - seeps_error`` (``score.py:1393``), so the stored "seeps" is a
   SKILL: higher is better, and it must *decrease* with lead time.
2. The 2-step runs emitted their initialisations in different orders, so equal sample indices are
   NOT the same forecast. Only 37 of the first 64 inits are shared; ``MATCHED_2STEP`` pins them.

Usage
-----
    uv run --offline python scripts/imerg_presentation_figures.py --figure all
    uv run --offline python scripts/imerg_presentation_figures.py --figure F06 F08

Figure-generating work that touches the zip stores or the checkpoints should go through Slurm;
the pure line figures (F02, F03, F06, F07, F08, F12) are light enough to iterate on interactively.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
_logger = logging.getLogger("imerg_figures")

# --------------------------------------------------------------------------------------------
# Paths and run identity
# --------------------------------------------------------------------------------------------

RESULTS = Path("/e/scratch/weatherai/shared_work/results")
MODELS = Path("/e/scratch/weatherai/shared_work/models")
OUTDIR = Path("plots/imerg_presentation")

STREAM = "IMERG_ANEMOI"
CHANNEL = "tp"
CHKPT = "chkpt00000"

# Backbone families. `label` is what goes in a legend, `short` in a tight axis tick.
RUNS = {
    "xfaps4wq": {
        "label": "Forecast backbone (hmpa42d2)",
        "short": "Forecast",
        "family": "forecast",
        "parent": "hmpa42d2",
        "lineage": "c71eo6pu_ssl -> hmpa42d2",
        "istep": 16300,
    },
    "r15j90ns": {
        "label": "JEPA backbone (nescpnb2)",
        "short": "JEPA",
        "family": "jepa",
        "parent": "nescpnb2",
        "lineage": "af90zz71_ssl -> nescpnb2",
        "istep": 7640,
    },
    "z71y2ik8": {
        "label": "JEPA backbone, 8-step rollout (v3gs9fwn)",
        "short": "JEPA 8-step",
        "family": "jepa",
        "parent": "v3gs9fwn",
        "lineage": "af90zz71_ssl -> nescpnb2 -> rxssbn0v -> ... -> v3gs9fwn",
        "istep": 26450,
    },
}

# The 37 initialisations shared between the two 2-step runs, 2023-06-01T12 .. 2023-06-29T18.
# Derived by reading `<sample>/IMERG_ANEMOI/1/target/times[0]` out of both zip stores; pinned here
# so the figures do not depend on re-deriving it. `--verify-matched-inits` re-checks it.
MATCHED_2STEP = {
    "xfaps4wq": [
        0,
        6,
        12,
        18,
        24,
        30,
        36,
        19,
        25,
        31,
        37,
        43,
        49,
        55,
        2,
        38,
        44,
        50,
        56,
        62,
        3,
        9,
        15,
        57,
        63,
        4,
        10,
        16,
        22,
        28,
        34,
        23,
        29,
        35,
        41,
        47,
        53,
    ],
    "r15j90ns": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        2,
        12,
        22,
        32,
        42,
        52,
        62,
        63,
        4,
        14,
        24,
        34,
        44,
        35,
        45,
        55,
        6,
        16,
        7,
        17,
        27,
        37,
        47,
        57,
        9,
        19,
        29,
        39,
        49,
        59,
    ],
}
MATCHED_2STEP_INITS = [
    "2023-06-01T12:00",
    "2023-06-01T18:00",
    "2023-06-02T00:00",
    "2023-06-02T06:00",
    "2023-06-02T12:00",
    "2023-06-02T18:00",
    "2023-06-03T00:00",
    "2023-06-07T12:00",
    "2023-06-07T18:00",
    "2023-06-08T00:00",
    "2023-06-08T06:00",
    "2023-06-08T12:00",
    "2023-06-08T18:00",
    "2023-06-09T00:00",
    "2023-06-12T00:00",
    "2023-06-13T12:00",
    "2023-06-13T18:00",
    "2023-06-14T00:00",
    "2023-06-14T06:00",
    "2023-06-14T12:00",
    "2023-06-17T06:00",
    "2023-06-17T12:00",
    "2023-06-17T18:00",
    "2023-06-19T12:00",
    "2023-06-19T18:00",
    "2023-06-22T12:00",
    "2023-06-22T18:00",
    "2023-06-23T00:00",
    "2023-06-23T06:00",
    "2023-06-23T12:00",
    "2023-06-23T18:00",
    "2023-06-28T12:00",
    "2023-06-28T18:00",
    "2023-06-29T00:00",
    "2023-06-29T06:00",
    "2023-06-29T12:00",
    "2023-06-29T18:00",
]

# z71y2ik8's 10 inits are already index-aligned (0..9 = 2023-06-01T12 .. 2023-06-03T18).
ROLLOUT_SAMPLES = list(range(10))

STEP_HOURS = 6  # forecast step 1 == +6h
TRAINED_HORIZON_H = 48  # 8 steps x 6h -- everything past this is extrapolation
DAY6_H = 144

# --------------------------------------------------------------------------------------------
# Metric metadata
# --------------------------------------------------------------------------------------------

# `higher_better` drives the sign of every difference and the direction words in captions.
# NOTE seeps: the evaluate package stores 1 - SEEPS_error, so it is higher-is-better here.
METRICS = {
    "ets": {"name": "ETS", "long": "Equitable Threat Score (>1 mm/6h)", "higher_better": True},
    "seeps": {
        "name": "SEEPS skill",
        "long": "SEEPS skill (1 - SEEPS error)",
        "higher_better": True,
    },
    "fbi": {"name": "FBI", "long": "Frequency bias (>1 mm/6h)", "higher_better": None},
    "rmse": {"name": "RMSE", "long": "RMSE (mm/6h)", "higher_better": False},
    "mae": {"name": "MAE", "long": "MAE (mm/6h)", "higher_better": False},
    "bias": {"name": "Bias", "long": "Mean bias (mm/6h)", "higher_better": None},
    "froct": {
        "name": "Forecast activity",
        "long": "Forecast rate of change (mm/6h)",
        "higher_better": None,
    },
    "troct": {
        "name": "Observed activity",
        "long": "Observed rate of change (mm/6h)",
        "higher_better": None,
    },
}
# tp is stored in metres per 6h window; these are the metrics that carry those units.
MM_SCALED = {"rmse", "mae", "bias", "froct", "troct"}
MM_PER_M = 1000.0

REGIONS = ["global", "nhem", "shem"]
REGION_LABEL = {"global": "Global", "nhem": "Northern hemisphere", "shem": "Southern hemisphere"}

# --------------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------------

# Run colours. The two JEPA runs share a hue family so the backbone split reads at a glance.
COLOR = {
    "xfaps4wq": "#C2603F",  # Forecast backbone -- warm
    "r15j90ns": "#2F6F8F",  # JEPA backbone -- cool
    "z71y2ik8": "#2F6F8F",
    "truth": "#3C3C3C",
}
INK = "#1A1A1A"
MUTED = "#6E6E6E"
GRID = "#D8D8D8"
ACCENT = "#B03A2E"  # collapse / warning
ACCENT_SOFT = "#F2DEDA"
GOOD_SOFT = "#DCE8EE"

# Precipitation palette, carried over unchanged from
# config/evaluate/eval_config_imerg_8step_40lead.yml so the new maps match the ones already
# circulated. Levels are metres per 6h (0.1 mm .. 20 mm).
PRECIP_LEVELS_M = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
# BoundaryNorm(levels, cmap.N, extend="both") needs len(levels)+1 colours:
# 6 interior bins + 2 extensions. "none" is RGBA alpha 0 -> dry renders as the page, not as white.
PRECIP_COLORS = [
    "none",
    "#BEDAE5",
    "#80B5CC",
    "#5691B3",
    "#4B6E9C",
    "#424B83",
    "#45596D",
    "#5B9E4C",
]


def apply_style() -> None:
    """Global matplotlib defaults. Sized for projection at slide scale."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, name: str) -> None:
    """Write both a raster (slides) and a vector (print) copy."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUTDIR / f"{name}.{ext}"
        fig.savefig(path, format=ext)
    _logger.info("wrote %s.{png,pdf}", OUTDIR / name)
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Score access
# --------------------------------------------------------------------------------------------


def score_path(run: str, region: str, metric: str) -> Path:
    return RESULTS / run / "evaluation" / f"{run}_{STREAM}_{region}_{metric}_{CHKPT}.json"


def load_scores(run: str, region: str, metric: str) -> npt.NDArray:
    """Per-sample scores as ``[sample, forecast_step]``.

    The stored array is ``[sample, forecast_step, channel, ens]``; both runs carry a single
    channel (tp) and a single ensemble member, so the trailing axes collapse.
    """
    d = json.loads(score_path(run, region, metric).read_text())
    a = np.asarray(d["scores"][0]["data"], dtype=float)
    assert a.ndim == 4, f"unexpected score dims {a.shape} in {score_path(run, region, metric)}"
    a = a[:, :, 0, 0]
    if metric in MM_SCALED:
        a = a * MM_PER_M
    return a


def lead_hours(n_steps: int) -> npt.NDArray:
    """Forecast step 1 is +6h, so lead time is ``(index + 1) * 6``."""
    return (np.arange(n_steps) + 1) * STEP_HOURS


def matched(run: str, region: str, metric: str) -> npt.NDArray:
    """2-step scores restricted to the 37 initialisations shared by both runs.

    Equal sample indices are different forecasts in the two stores, so this subsetting is what
    makes the head-to-head a paired comparison rather than two unrelated averages.
    """
    return load_scores(run, region, metric)[MATCHED_2STEP[run]]


def bootstrap_ci(
    diff: npt.NDArray, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean of a paired difference and its percentile bootstrap CI, resampling initialisations."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diff.mean()), float(lo), float(hi)


def annotate_source(fig, text: str) -> None:
    """Footnote naming the sample basis, so no figure travels without its caveat."""
    fig.text(0.005, -0.02, text, fontsize=9, color=MUTED, ha="left", va="top")


# --------------------------------------------------------------------------------------------
# Verification helper
# --------------------------------------------------------------------------------------------


def verify_matched_inits() -> None:
    """Re-derive the matched-init lists from the stores and check them against the pinned ones."""
    import zarr

    found = {}
    for run in ("xfaps4wq", "r15j90ns"):
        store = zarr.storage.ZipStore(
            str(RESULTS / run / "validation_chkpt00000_rank0000.zip"), mode="r"
        )
        g = zarr.open_group(store, mode="r")
        times = {}
        for s in range(64):
            try:
                t = np.asarray(g[f"{s}/{STREAM}/1/target/times"][0:1]).ravel()[0]
            except KeyError:
                continue
            times[str(t)[:16]] = s
        found[run] = times
        store.close()

    common = sorted(set(found["xfaps4wq"]) & set(found["r15j90ns"]))
    ok = common == MATCHED_2STEP_INITS
    _logger.info(
        "shared inits: %d (pinned %d) -- %s",
        len(common),
        len(MATCHED_2STEP_INITS),
        "MATCH" if ok else "MISMATCH",
    )
    for run in ("xfaps4wq", "r15j90ns"):
        idx = [found[run][t] for t in common]
        same = idx == MATCHED_2STEP[run]
        _logger.info("  %s indices %s", run, "MATCH" if same else f"MISMATCH -> {idx}")
        ok = ok and same
    if not ok:
        raise SystemExit("matched-init lists are stale; update MATCHED_2STEP")


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

FIGURES: dict[str, Callable[[], None]] = {}


def figure(name: str):
    """Register a figure builder under its short id."""

    def deco(fn):
        FIGURES[name] = fn
        return fn

    return deco


# --------------------------------------------------------------------------------------------
# Part 2 -- two steps, head to head on the 37 matched initialisations
# --------------------------------------------------------------------------------------------

TWO_STEP = ("xfaps4wq", "r15j90ns")
MATCHED_NOTE = (
    f"{len(MATCHED_2STEP_INITS)} initialisations shared by both runs, "
    f"{MATCHED_2STEP_INITS[0][:10]} to {MATCHED_2STEP_INITS[-1][:10]}, 6-hourly. "
    "Shading / whiskers: 95% percentile bootstrap over initialisations."
)


def _mean_ci(values: npt.NDArray, seed: int = 0) -> tuple[float, float, float]:
    """Mean and bootstrap CI of a single run's per-initialisation scores."""
    return bootstrap_ci(values, seed=seed)


@figure("F02")
def f02_two_step_scorecard() -> None:
    """Head-to-head at 2 steps: absolute skill plus the paired difference with CIs.

    The published comparison put six near-identical two-point lines on one axis, which made
    differences of 0.005 look like separation. Here the absolute values carry their uncertainty
    and the paired difference gets its own panel, so "practically tied" is visible as such.
    """
    import matplotlib.pyplot as plt

    leads = lead_hours(2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # (a)-(c): absolute skill with bootstrap CI, per run.
    for ax, met in zip(axes.flat[:3], ["ets", "seeps", "rmse"], strict=True):
        for run in TWO_STEP:
            a = matched(run, "global", met)
            mu, lo, hi = zip(*[_mean_ci(a[:, i]) for i in range(a.shape[1])], strict=True)
            ax.errorbar(
                leads,
                mu,
                yerr=[np.array(mu) - np.array(lo), np.array(hi) - np.array(mu)],
                color=COLOR[run],
                marker="o",
                capsize=4,
                label=RUNS[run]["short"],
            )
        ax.set_xticks(leads)
        ax.set_xlabel("lead time [h]")
        ax.set_ylabel(METRICS[met]["long"])
        arrow = "higher is better" if METRICS[met]["higher_better"] else "lower is better"
        ax.set_title(f"{METRICS[met]['name']} — {arrow}", loc="left")
        ax.margins(x=0.25)
    axes[0, 0].legend(loc="upper right")

    # (d): paired difference, oriented so that right of zero always means "JEPA better".
    ax = axes[1, 1]
    rows, labels = [], []
    for met in ["ets", "seeps", "rmse", "mae"]:
        for i, lead in enumerate(leads):
            fc = matched("xfaps4wq", "global", met)[:, i]
            jp = matched("r15j90ns", "global", met)[:, i]
            diff = jp - fc
            if not METRICS[met]["higher_better"]:
                diff = -diff  # lower-is-better metric: flip so right == JEPA better
            # Express as a percentage of the Forecast-backbone level: the four metrics differ by
            # three orders of magnitude and would not otherwise share an axis.
            mu, lo, hi = bootstrap_ci(diff / np.abs(fc.mean()) * 100.0)
            rows.append((mu, lo, hi))
            labels.append(f"{METRICS[met]['name']}  +{lead}h")

    y = np.arange(len(rows))[::-1]
    for yi, (mu, lo, hi) in zip(y, rows, strict=True):
        sig = lo > 0 or hi < 0
        c = COLOR["r15j90ns"] if mu > 0 else COLOR["xfaps4wq"]
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.4, alpha=0.85 if sig else 0.35)
        ax.plot([mu], [yi], "o", color=c, ms=7, alpha=0.95 if sig else 0.45)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("paired difference [% of Forecast-backbone level]")
    ax.set_title("JEPA − Forecast, oriented so right = JEPA better", loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.12)

    fig.suptitle(
        "IMERG precipitation at 2 forecast steps: JEPA vs Forecast backbone",
        x=0.008,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    annotate_source(fig, MATCHED_NOTE)
    save(fig, "F02_2step_scorecard")

    # Print the table the caption has to quote, so the numbers can be checked by eye.
    _logger.info(
        "%-6s %-5s %10s %10s %12s %22s", "metric", "lead", "Forecast", "JEPA", "diff", "95% CI"
    )
    for met in ["ets", "seeps", "fbi", "rmse", "mae", "bias"]:
        for i, lead in enumerate(leads):
            fc = matched("xfaps4wq", "global", met)[:, i]
            jp = matched("r15j90ns", "global", met)[:, i]
            mu, lo, hi = bootstrap_ci(jp - fc)
            _logger.info(
                "%-6s %-5s %10.5f %10.5f %+12.5f   [%+.5f, %+.5f]",
                met,
                f"+{lead}h",
                fc.mean(),
                jp.mean(),
                mu,
                lo,
                hi,
            )


@figure("F03")
def f03_two_step_regional() -> None:
    """Does the 2-step ranking hold in both hemispheres, or is it a tropical artefact?"""
    import matplotlib.pyplot as plt

    mets = ["ets", "seeps"]
    leads = lead_hours(2)
    fig, axes = plt.subplots(len(mets), len(REGIONS), figsize=(13, 8), sharex=True)

    for r, met in enumerate(mets):
        # One shared y-range per metric row so the three regions are visually comparable.
        lo_all, hi_all = np.inf, -np.inf
        for c, region in enumerate(REGIONS):
            ax = axes[r, c]
            for run in TWO_STEP:
                a = matched(run, region, met)
                stats = [_mean_ci(a[:, i]) for i in range(a.shape[1])]
                mu = np.array([s[0] for s in stats])
                lo = np.array([s[1] for s in stats])
                hi = np.array([s[2] for s in stats])
                ax.errorbar(
                    leads,
                    mu,
                    yerr=[mu - lo, hi - mu],
                    color=COLOR[run],
                    marker="o",
                    capsize=4,
                    label=RUNS[run]["short"],
                )
                lo_all, hi_all = min(lo_all, lo.min()), max(hi_all, hi.max())
            ax.set_xticks(leads)
            ax.margins(x=0.3)
            if r == 0:
                ax.set_title(REGION_LABEL[region], loc="left")
            if r == len(mets) - 1:
                ax.set_xlabel("lead time [h]")
            if c == 0:
                # Short names here -- the long forms collide between the two rows.
                ax.set_ylabel(f"{METRICS[met]['name']} ↑")
        pad = 0.06 * (hi_all - lo_all)
        for c in range(len(REGIONS)):
            axes[r, c].set_ylim(lo_all - pad, hi_all + pad)
    axes[0, 0].legend(loc="best")

    fig.suptitle(
        "JEPA's +12 h edge comes from the northern hemisphere; the south is a tie "
        "(↑ = higher is better)",
        x=0.008,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    annotate_source(
        fig,
        MATCHED_NOTE + " The anemoi IMERG store carries values at all latitudes; IMERG's native "
        "observing range is 60S-60N, so mid- and high-latitude scores lean on filled data.",
    )
    save(fig, "F03_2step_regional")


# --------------------------------------------------------------------------------------------
# Part 3 -- the 8-step model rolled out to 10 days
# --------------------------------------------------------------------------------------------

ROLLOUT_RUN = "z71y2ik8"
ROLLOUT_NOTE = (
    "z71y2ik8, 10 initialisations 2023-06-01T12 to 2023-06-03T18, 6-hourly. "
    "Shading: 95% percentile bootstrap over initialisations. "
    "Scores are means of per-initialisation values, matching weathergen.evaluate."
)


def rollout_stats(metric: str, region: str = "global") -> tuple[npt.NDArray, ...]:
    """Lead times plus mean and bootstrap CI at every one of the 40 forecast steps."""
    a = load_scores(ROLLOUT_RUN, region, metric)
    lead = lead_hours(a.shape[1])
    stats = [bootstrap_ci(a[:, i], seed=i) for i in range(a.shape[1])]
    mu = np.array([s[0] for s in stats])
    lo = np.array([s[1] for s in stats])
    hi = np.array([s[2] for s in stats])
    return lead, mu, lo, hi


def collapse_onset() -> float:
    """First lead time at which the forecast stops being a usable precipitation field.

    Defined on frequency bias rather than on a skill score: FBI leaving the stable band from
    below (< 0.8) is the moment the field stops raining, which is what the maps show and what
    froct independently confirms. Derived from the data, not hard-coded.
    """
    a = load_scores(ROLLOUT_RUN, "global", "fbi").mean(axis=0)
    lead = lead_hours(len(a))
    below = np.flatnonzero(a < 0.8)
    return float(lead[below[0]]) if below.size else float(lead[-1])


def _decorate_rollout(
    ax, *, collapse: float, day6: bool = True, legend_ax: bool = False, label_y: float = 0.96
) -> None:
    """Shared lead-time furniture: trained horizon, day 6, collapse zone, day ticks.

    ``label_y`` is where the two zone labels sit, in axes fraction — the curves occupy a
    different part of the frame in each figure, so the caller places them.
    """
    xmax = ax.get_xlim()[1]
    ax.axvspan(0, TRAINED_HORIZON_H, color=GOOD_SOFT, zorder=0)
    ax.axvspan(collapse, xmax, color=ACCENT_SOFT, zorder=0)
    ax.axvline(TRAINED_HORIZON_H, color=MUTED, lw=1.1, ls="--", zorder=1)
    if day6:
        ax.axvline(DAY6_H, color=INK, lw=1.3, ls=":", zorder=1)
    ax.set_xlim(0, xmax)
    if legend_ax:
        ax.annotate(
            f"trained rollout\n+{TRAINED_HORIZON_H}h",
            xy=(TRAINED_HORIZON_H / 2, label_y),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=10,
            color=MUTED,
        )
        ax.annotate(
            f"collapse\n(from +{collapse:.0f}h)",
            xy=((collapse + xmax) / 2, label_y),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=10,
            color=ACCENT,
        )


def _day_axis(ax) -> None:
    """Secondary axis in days -- the unit the claim is actually made in."""
    sec = ax.secondary_xaxis("top", functions=(lambda h: h / 24.0, lambda d: d * 24.0))
    sec.set_xlabel("lead time [days]")
    sec.set_xticks(np.arange(0, 11, 1))


@figure("F06")
def f06_rollout_skill() -> None:
    """The headline: graceful decay to day 6 — three times the trained horizon — then collapse."""
    import matplotlib.pyplot as plt

    collapse = collapse_onset()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)

    for ax, met in zip(axes, ["ets", "seeps"], strict=True):
        lead, mu, lo, hi = rollout_stats(met)
        ax.fill_between(lead, lo, hi, color=COLOR[ROLLOUT_RUN], alpha=0.20, lw=0)
        ax.plot(lead, mu, color=COLOR[ROLLOUT_RUN])
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_ylabel(METRICS[met]["long"])
        ax.set_title(f"{METRICS[met]['name']} — higher is better", loc="left")
        _decorate_rollout(ax, collapse=collapse, legend_ax=(ax is axes[0]))

        # Call out the two numbers the slide is built on.
        i48 = int(TRAINED_HORIZON_H / STEP_HOURS) - 1
        i144 = int(DAY6_H / STEP_HOURS) - 1
        ax.annotate(
            f"{mu[i144]:.2f} at day 6",
            xy=(DAY6_H, mu[i144]),
            xytext=(-96, 46),
            textcoords="offset points",
            fontsize=11,
            color=INK,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.1},
        )
        ax.annotate(
            f"{mu[i48]:.2f} at +48h",
            xy=(TRAINED_HORIZON_H, mu[i48]),
            xytext=(26, 30),
            textcoords="offset points",
            fontsize=10,
            color=MUTED,
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.0},
        )

    axes[-1].set_xlabel("lead time [h]")
    _day_axis(axes[0])
    fig.suptitle(
        "Skill decays gracefully to day 6 — 3x beyond the trained 48 h rollout — then collapses",
        x=0.008,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    annotate_source(fig, ROLLOUT_NOTE)
    save(fig, "F06_rollout_skill")

    lead, mu, _, _ = rollout_stats("ets")
    for h in (6, 48, 96, 144, 168, 192):
        _logger.info("ETS at +%3dh = %+.4f", h, mu[int(h / STEP_HOURS) - 1])


@figure("F07")
def f07_rollout_fbi_bias() -> None:
    """Rain frequency and mean bias: the field keeps raining until it abruptly does not."""
    import matplotlib.pyplot as plt

    collapse = collapse_onset()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)

    lead, mu, lo, hi = rollout_stats("fbi")
    ax = axes[0]
    ax.fill_between(lead, lo, hi, color=COLOR[ROLLOUT_RUN], alpha=0.20, lw=0)
    ax.plot(lead, mu, color=COLOR[ROLLOUT_RUN])
    ax.axhline(1.0, color=INK, lw=1.1, ls="-")
    ax.annotate(
        "unbiased rain frequency",
        xy=(lead[-1], 1.0),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color=INK,
    )
    band = mu[: int(DAY6_H / STEP_HOURS)]
    ax.set_ylabel(METRICS["fbi"]["long"])
    ax.set_title(
        f"Frequency bias holds a stable {band.min():.2f}–{band.max():.2f} band through day 6",
        loc="left",
    )
    _decorate_rollout(ax, collapse=collapse, legend_ax=True)

    lead, mu, lo, hi = rollout_stats("bias")
    ax = axes[1]
    ax.fill_between(lead, lo, hi, color=COLOR[ROLLOUT_RUN], alpha=0.20, lw=0)
    ax.plot(lead, mu, color=COLOR[ROLLOUT_RUN])
    ax.axhline(0.0, color=INK, lw=1.1)
    ax.set_ylabel(METRICS["bias"]["long"])
    ax.set_title("Mean bias — the field dries out as it collapses", loc="left")
    ax.set_xlabel("lead time [h]")
    _decorate_rollout(ax, collapse=collapse)

    _day_axis(axes[0])
    fig.suptitle(
        "Rain frequency stays realistic to day 6, then the field empties",
        x=0.008,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    annotate_source(fig, ROLLOUT_NOTE)
    save(fig, "F07_rollout_fbi_bias")


@figure("F08")
def f08_rollout_activity() -> None:
    """Why it is stable, and where it stops: forecast activity against observed activity.

    froct/troct are the mean absolute change between consecutive forecast steps. They need no
    climatology, which matters because both IMERG climatologies on this system carry only the two
    SEEPS parameters and cannot support fact/tact/acc. A froct decaying to zero against a flat
    troct is the collapse-to-static signature an activity plot would have shown.
    """
    import matplotlib.pyplot as plt

    collapse = collapse_onset()
    fig, ax = plt.subplots(figsize=(12, 6))

    lead, mu, lo, hi = rollout_stats("froct")
    ax.fill_between(lead, lo, hi, color=COLOR[ROLLOUT_RUN], alpha=0.20, lw=0)
    ax.plot(lead, mu, color=COLOR[ROLLOUT_RUN], label="forecast activity (froct)")

    lead_t, mu_t, lo_t, hi_t = rollout_stats("troct")
    ax.fill_between(lead_t, lo_t, hi_t, color=COLOR["truth"], alpha=0.15, lw=0)
    ax.plot(lead_t, mu_t, color=COLOR["truth"], ls="--", label="observed activity (troct, IMERG)")

    ax.set_ylabel("mean |change| between consecutive steps [mm/6h]")
    ax.set_xlabel("lead time [h]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")
    # troct runs across the top of the frame here, so the zone labels drop to mid-height.
    _decorate_rollout(ax, collapse=collapse, legend_ax=True, label_y=0.72)
    _day_axis(ax)

    i144 = int(DAY6_H / STEP_HOURS) - 1
    ax.annotate(
        f"{100 * mu[i144] / mu_t[i144]:.0f}% of observed\nvariability at day 6",
        xy=(DAY6_H, mu[i144]),
        xytext=(-104, 52),
        textcoords="offset points",
        fontsize=11,
        color=INK,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.1},
    )

    fig.suptitle(
        "The forecast keeps evolving until day 6, then goes static",
        x=0.008,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    annotate_source(fig, ROLLOUT_NOTE)
    save(fig, "F08_rollout_activity")


@figure("F12")
def f12_skill_horizon() -> None:
    """One number for the slide title: how far the model stays above each skill level."""
    import matplotlib.pyplot as plt

    a = load_scores(ROLLOUT_RUN, "global", "ets")
    lead = lead_hours(a.shape[1])
    mu = a.mean(axis=0)

    thresholds = [0.40, 0.30, 0.20, 0.10]
    horizons = []
    for thr in thresholds:
        below = np.flatnonzero(mu < thr)
        # Last lead time still at or above the threshold.
        horizons.append(float(lead[below[0] - 1]) if below.size and below[0] > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(thresholds))[::-1]
    bars = ax.barh(y, horizons, height=0.55, color=COLOR[ROLLOUT_RUN], zorder=2)
    for yi, h, bar in zip(y, horizons, bars, strict=True):
        ax.annotate(
            f"  +{h:.0f} h  ({h / 24:.1f} days)",
            xy=(bar.get_width(), yi),
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )

    ax.axvline(TRAINED_HORIZON_H, color=MUTED, lw=1.4, ls="--", zorder=3)
    ax.annotate(
        "trained rollout (+48 h)",
        xy=(TRAINED_HORIZON_H, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(6, -4),
        textcoords="offset points",
        fontsize=10,
        color=MUTED,
        va="top",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"ETS ≥ {t:.2f}" for t in thresholds])
    ax.set_xlabel("lead time reached [h]")
    ax.set_xlim(0, max(np.nanmax(horizons) * 1.30, TRAINED_HORIZON_H * 1.3))
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "Usable horizon: ETS stays above 0.20 out to 6 days",
        loc="left",
        fontsize=15,
    )
    fig.tight_layout()
    annotate_source(fig, ROLLOUT_NOTE + f" Collapse onset (FBI < 0.8): +{collapse_onset():.0f}h.")
    save(fig, "F12_skill_horizon")

    for thr, h in zip(thresholds, horizons, strict=True):
        _logger.info("ETS >= %.2f out to +%.0fh (%.1f days)", thr, h, h / 24)


# --------------------------------------------------------------------------------------------
# Field access -- the raw N320 predictions and targets
# --------------------------------------------------------------------------------------------

# Zoom boxes for the case studies. lat/lon in degrees, lon in [-180, 180) to match the stores.
ZOOMS = {
    "maritime": {
        "name": "Maritime Continent\n& W Pacific ITCZ",
        "long": "Maritime Continent & W Pacific ITCZ",
        "lat": (-15.0, 20.0),
        "lon": (90.0, 170.0),
    },
    "monsoon": {
        "name": "South Asian monsoon\n(India, Bay of Bengal)",
        "long": "South Asian monsoon (India, Bay of Bengal)",
        "lat": (5.0, 30.0),
        "lon": (65.0, 95.0),
    },
    "natlantic": {
        "name": "N Atlantic & Europe\n(frontal systems)",
        "long": "N Atlantic & Europe (frontal systems)",
        "lat": (35.0, 60.0),
        "lon": (-60.0, 20.0),
    },
}


def _open_store(run: str):
    import zarr

    return zarr.storage.ZipStore(
        str(RESULTS / run / "validation_chkpt00000_rank0000.zip"), mode="r"
    )


def read_field(group, sample: int, fstep: int, kind: str) -> npt.NDArray:
    """One N320 field in mm/6h.

    ``kind`` is ``"prediction"`` or ``"target"``. Forecast step 0 carries only ``source``;
    predictions start at step 1, so ``fstep`` here is 1-based and equals lead / 6h.
    """
    a = np.asarray(group[f"{sample}/{STREAM}/{fstep}/{kind}/data"][:], dtype=np.float32)
    return a.reshape(a.shape[0], -1)[:, 0] * MM_PER_M


def read_coords(group, sample: int, fstep: int = 1) -> tuple[npt.NDArray, npt.NDArray]:
    """Latitude and longitude in degrees; longitude is already in [-180, 180)."""
    c = np.asarray(group[f"{sample}/{STREAM}/{fstep}/target/coords"][:], dtype=np.float32)
    return c[:, 0], c[:, 1]


def read_init_time(group, sample: int) -> str:
    t = np.asarray(group[f"{sample}/{STREAM}/1/target/times"][0:1]).ravel()[0]
    return str(t)[:16]


def box_mask(lat: npt.NDArray, lon: npt.NDArray, zoom: dict) -> npt.NDArray:
    (la0, la1), (lo0, lo1) = zoom["lat"], zoom["lon"]
    return (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)


def pick_case(run: str, samples: list[int], fsteps: list[int]) -> tuple[int, str]:
    """Choose the initialisation with the strongest observed rain inside the three zoom boxes.

    Selecting on the *observed* field keeps the choice independent of which model is being shown,
    so the case study cannot be accused of being cherry-picked in the model's favour.
    """
    cache = OUTDIR / f".case_{run}_{'_'.join(map(str, fsteps))}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        _logger.info("case for %s from cache: sample %d (%s)", run, d["sample"], d["init"])
        return d["sample"], d["init"]

    store = _open_store(run)
    import zarr

    g = zarr.open_group(store, mode="r")
    lat, lon = read_coords(g, samples[0])
    mask = np.zeros(len(lat), dtype=bool)
    for zoom in ZOOMS.values():
        mask |= box_mask(lat, lon, zoom)

    best, best_score, best_init = samples[0], -np.inf, ""
    for s in samples:
        vals = [read_field(g, s, fs, "target")[mask] for fs in fsteps]
        score = float(np.nanmean([np.nanmean(v) for v in vals]))
        init = read_init_time(g, s)
        _logger.debug("  sample %3d (%s) mean box rain %.4f mm/6h", s, init, score)
        if score > best_score:
            best, best_score, best_init = s, score, init
    store.close()

    _logger.info(
        "case for %s: sample %d (%s), %.4f mm/6h mean observed rain in the zoom boxes",
        run,
        best,
        best_init,
        best_score,
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"sample": best, "init": best_init, "score": best_score}))
    return best, best_init


# --------------------------------------------------------------------------------------------
# Map rendering
# --------------------------------------------------------------------------------------------


def _init_cartopy() -> None:
    """Point cartopy at the shared offline shapefile cache, as weathergen.evaluate does."""
    import cartopy

    from weathergen.common.config import _load_private_conf

    work_dir = Path(_load_private_conf(None)["path_shared_working_dir"]) / "assets/cartopy"
    cartopy.config["data_dir"] = str(work_dir)
    cartopy.config["pre_existing_data_dir"] = str(work_dir)
    os.environ["CARTOPY_DATA_DIR"] = str(work_dir)


def precip_scale():
    """Colormap and norm for precipitation in mm/6h.

    Same levels and colours as the circulated maps, converted from metres to mm. The first
    colour is "none" (alpha 0), so dry points render as the page rather than as a white blob.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    levels = [v * MM_PER_M for v in PRECIP_LEVELS_M]
    cmap = ListedColormap(PRECIP_COLORS[1:-1])
    cmap.set_under(PRECIP_COLORS[0])
    cmap.set_over(PRECIP_COLORS[-1])
    # BoundaryNorm over N interior bins; under/over pick up the two extension colours.
    norm = BoundaryNorm(levels, cmap.N, extend="neither")
    return cmap, norm, levels


def auto_marker_size(ax, n_points: int, *, fill: float = 1.9) -> float:
    """Marker area (points²) that makes ``n_points`` just cover this axis.

    Panels in these figures differ in both size and point count — a N Atlantic crop holds half
    as many N320 points as a Maritime Continent crop over twice the area — so a single hard-coded
    marker size renders some panels as continuous fields and others as visible dot grids.
    """
    pos = ax.get_position()
    fig = ax.get_figure()
    area_pt2 = (pos.width * fig.get_figwidth() * 72.0) * (pos.height * fig.get_figheight() * 72.0)
    return fill * area_pt2 / max(n_points, 1)


def draw_precip(ax, lat, lon, vals, *, zoom: dict | None = None, fill: float = 1.9):
    """Scatter an N320 field onto a cartopy axis.

    The reduced Gaussian grid is dense enough that correctly sized round markers read as a
    continuous field; this is how the existing evaluation maps are produced.
    """
    import cartopy.crs as ccrs

    cmap, norm, _ = precip_scale()
    if zoom is not None:
        ax.set_extent([*zoom["lon"], *zoom["lat"]], crs=ccrs.PlateCarree())
    else:
        ax.set_global()
        # Robinson fills roughly 80% of its bounding box, so the same point count is packed
        # into less area than the rectangle implies.
        fill *= 0.8
    ax.coastlines(linewidth=0.35, color="#4A4A4A")
    return ax.scatter(
        lon,
        lat,
        c=vals,
        s=auto_marker_size(ax, len(vals), fill=fill),
        marker="o",
        linewidths=0,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )


ROBINSON_ASPECT = 2.0  # width / height of a full Robinson globe


def zoom_aspect(zoom: dict) -> float:
    """Width / height of a lat-lon box under PlateCarree, where 1 degree is 1 unit either way."""
    return (zoom["lon"][1] - zoom["lon"][0]) / (zoom["lat"][1] - zoom["lat"][0])


def map_grid(
    nrows: int,
    aspects: list[float],
    projection,
    *,
    panel_h: float = 1.9,
    top: float = 1.15,
    bottom: float = 1.05,
    left: float = 0.95,
    right: float = 0.25,
    wgap: float = 0.14,
    hgap: float = 0.26,
    max_fig_w: float = 18.0,
):
    """A grid of map axes sized to the panels' true aspect ratios.

    Cartopy axes hold a fixed aspect, so a slot of the wrong shape letterboxes the map and leaves
    bands of dead space — which ``tight_layout`` cannot recover because the axes, not the figure,
    are the thing that shrank. Sizing every slot correctly up front avoids the problem entirely;
    all padding is then in inches and converted to figure fractions here.

    ``max_fig_w`` keeps a strip of wide boxes (the N Atlantic crop is 3.2:1) from growing into a
    figure so large that the titles and colourbar are unreadable once it is scaled onto a slide.
    """
    import matplotlib.pyplot as plt

    budget = max_fig_w - left - right - wgap * (len(aspects) - 1)
    panel_h = min(panel_h, budget / sum(aspects))

    widths = [panel_h * a for a in aspects]
    axes_w = sum(widths) + wgap * (len(widths) - 1)
    axes_h = panel_h * nrows + hgap * (nrows - 1)
    fig_w = left + axes_w + right
    fig_h = top + axes_h + bottom

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows,
        len(widths),
        width_ratios=widths,
        left=left / fig_w,
        right=1 - right / fig_w,
        top=1 - top / fig_h,
        bottom=bottom / fig_h,
        wspace=wgap / (axes_w / len(widths)),
        hspace=hgap / panel_h,
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[r, c], projection=projection) for c in range(len(widths))]
            for r in range(nrows)
        ]
    )
    return fig, axes


def precip_colorbar(fig, mappable, *, ax, label: str = "precipitation [mm / 6 h]", **kw):
    _, _, levels = precip_scale()
    cb = fig.colorbar(mappable, ax=ax, extend="both", ticks=levels, **kw)
    cb.set_label(label)
    cb.ax.set_xticklabels([f"{v:g}" for v in levels])
    return cb


# --------------------------------------------------------------------------------------------
# Rain-rate distributions
# --------------------------------------------------------------------------------------------

# Log-spaced bins in mm/6h. Zero and near-zero cannot live on a log axis, so the dry fraction
# (below the first edge) is reported as a number instead of being drawn.
PDF_BINS = np.logspace(np.log10(0.05), np.log10(200.0), 61)
DRY_THRESHOLD = PDF_BINS[0]


def accumulate_pdf(group, samples: list[int], fstep: int, kind: str) -> tuple[npt.NDArray, float]:
    """Histogram counts over many initialisations, plus the dry fraction.

    Counts are accumulated per initialisation rather than concatenating the fields: 37 inits x
    542080 points would be 20 M values per curve and there are six curves.
    """
    counts = np.zeros(len(PDF_BINS) - 1)
    n_dry = n_tot = 0
    for s in samples:
        v = read_field(group, s, fstep, kind)
        v = v[np.isfinite(v)]
        counts += np.histogram(v, bins=PDF_BINS)[0]
        n_dry += int((v < DRY_THRESHOLD).sum())
        n_tot += v.size
    return counts, n_dry / max(n_tot, 1)


def pdf_density(counts: npt.NDArray) -> npt.NDArray:
    """Counts -> density per unit rain rate, so unequal log bin widths do not distort the shape."""
    widths = np.diff(PDF_BINS)
    total = counts.sum()
    return counts / (total * widths) if total else counts


PDF_CENTRES = np.sqrt(PDF_BINS[:-1] * PDF_BINS[1:])


@figure("F04")
def f04_two_step_pdf() -> None:
    """Do the two backbones reproduce the observed rain-rate distribution, and where do they miss?

    This is where the FBI > 1 result becomes visible: an excess of light rain relative to IMERG.
    """
    import matplotlib.pyplot as plt
    import zarr

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    dry = {}

    for i, (fstep, ax) in enumerate(zip([1, 2], axes, strict=True)):
        # Observed first, so it leads the legend; it is the same field in both stores.
        store = _open_store(TWO_STEP[0])
        g = zarr.open_group(store, mode="r")
        tc, tdry = accumulate_pdf(g, MATCHED_2STEP[TWO_STEP[0]], fstep, "target")
        ax.plot(PDF_CENTRES, pdf_density(tc), color=COLOR["truth"], ls="--", label="IMERG observed")
        dry[("truth", fstep)] = tdry
        store.close()

        for run in TWO_STEP:
            store = _open_store(run)
            g = zarr.open_group(store, mode="r")
            counts, dry_frac = accumulate_pdf(g, MATCHED_2STEP[run], fstep, "prediction")
            ax.plot(PDF_CENTRES, pdf_density(counts), color=COLOR[run], label=RUNS[run]["short"])
            dry[(run, fstep)] = dry_frac
            store.close()

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("precipitation [mm / 6 h]")
        ax.set_title(f"+{(i + 1) * STEP_HOURS} h", loc="left")
        ax.annotate(
            "dry fraction (<{:.2g} mm):\n  IMERG {:.1%}\n  {} {:.1%}\n  {} {:.1%}".format(
                DRY_THRESHOLD,
                dry[("truth", fstep)],
                RUNS["xfaps4wq"]["short"],
                dry[("xfaps4wq", fstep)],
                RUNS["r15j90ns"]["short"],
                dry[("r15j90ns", fstep)],
            ),
            xy=(0.03, 0.06),
            xycoords="axes fraction",
            fontsize=10,
            color=MUTED,
            va="bottom",
        )
    axes[0].set_ylabel("density [per mm/6h]")
    axes[0].legend(loc="upper right")

    fig.suptitle(
        "Both backbones track the observed rain-rate distribution and over-produce light rain",
        x=0.008,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    annotate_source(fig, MATCHED_NOTE.split(" Shading")[0] + " All N320 points pooled.")
    save(fig, "F04_2step_pdf")


@figure("F09")
def f09_rollout_pdf() -> None:
    """The distribution is held out to day 6 and then falls apart — the PDF view of the collapse."""
    import matplotlib.pyplot as plt
    import zarr

    leads_h = [24, 48, 96, 144, 168, 192]
    store = _open_store(ROLLOUT_RUN)
    g = zarr.open_group(store, mode="r")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), sharex=True, sharey=True)
    collapse = collapse_onset()

    for lead, ax in zip(leads_h, axes.flat, strict=True):
        fstep = lead // STEP_HOURS
        tc, tdry = accumulate_pdf(g, ROLLOUT_SAMPLES, fstep, "target")
        pc, pdry = accumulate_pdf(g, ROLLOUT_SAMPLES, fstep, "prediction")
        ax.plot(PDF_CENTRES, pdf_density(tc), color=COLOR["truth"], ls="--", label="IMERG observed")
        ax.plot(PDF_CENTRES, pdf_density(pc), color=COLOR[ROLLOUT_RUN], label="forecast")
        ax.set_xscale("log")
        ax.set_yscale("log")
        past = lead >= collapse
        ax.set_title(
            f"+{lead} h  (day {lead / 24:g}){'  — collapsed' if past else ''}",
            loc="left",
            color=ACCENT if past else INK,
        )
        if past:
            ax.set_facecolor(ACCENT_SOFT)
        ax.annotate(
            f"dry: IMERG {tdry:.0%} / forecast {pdry:.0%}",
            xy=(0.03, 0.06),
            xycoords="axes fraction",
            fontsize=10,
            color=MUTED,
        )
    for ax in axes[-1]:
        ax.set_xlabel("precipitation [mm / 6 h]")
    for ax in axes[:, 0]:
        ax.set_ylabel("density [per mm/6h]")
    axes[0, 0].legend(loc="upper right")
    store.close()

    fig.suptitle(
        "The rain-rate distribution survives to day 6, then empties",
        x=0.008,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    annotate_source(fig, ROLLOUT_NOTE.split(" Shading")[0] + " All N320 points pooled.")
    save(fig, "F09_rollout_pdf")


# --------------------------------------------------------------------------------------------
# Map figures
# --------------------------------------------------------------------------------------------


@figure("F05")
def f05_two_step_case_maps() -> None:
    """One initialisation, three regions, both backbones against IMERG at +12 h."""
    import cartopy.crs as ccrs
    import zarr

    _init_cartopy()
    fstep = 2  # +12h -- the lead where the two backbones separate
    sample_fc, init = pick_case("xfaps4wq", MATCHED_2STEP["xfaps4wq"], [fstep])
    # Translate the chosen init time into the other run's own indexing.
    sample_jp = MATCHED_2STEP["r15j90ns"][MATCHED_2STEP["xfaps4wq"].index(sample_fc)]

    fields = {}
    for run, sample in (("xfaps4wq", sample_fc), ("r15j90ns", sample_jp)):
        store = _open_store(run)
        g = zarr.open_group(store, mode="r")
        assert read_init_time(g, sample) == init, "init-time mismatch between the two runs"
        lat, lon = read_coords(g, sample)
        fields[run] = (lat, lon, read_field(g, sample, fstep, "prediction"))
        if "truth" not in fields:
            fields["truth"] = (lat, lon, read_field(g, sample, fstep, "target"))
        store.close()

    rows = ["truth", "r15j90ns", "xfaps4wq"]
    row_label = {
        "truth": "IMERG observed",
        "r15j90ns": RUNS["r15j90ns"]["short"],
        "xfaps4wq": RUNS["xfaps4wq"]["short"],
    }
    aspects = [zoom_aspect(z) for z in ZOOMS.values()]
    fig, axes = map_grid(len(rows), aspects, ccrs.PlateCarree(), panel_h=1.85, top=1.45)
    mappable = None
    for r, key in enumerate(rows):
        lat, lon, vals = fields[key]
        for c, zoom in enumerate(ZOOMS.values()):
            m = box_mask(lat, lon, zoom)
            mappable = draw_precip(axes[r, c], lat[m], lon[m], vals[m], zoom=zoom)
            if r == 0:
                axes[r, c].set_title(zoom["name"], loc="center", fontsize=11)
        axes[r, 0].text(
            -0.025,
            0.5,
            row_label[key],
            transform=axes[r, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
            color=COLOR.get(key, INK),
        )

    fig.suptitle(
        f"+12 h precipitation, initialised {init} — both backbones against IMERG",
        x=0.008,
        y=0.995,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    precip_colorbar(
        fig, mappable, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=60
    )
    annotate_source(
        fig,
        f"Initialisation chosen automatically as the one with the most observed rain inside the "
        f"three boxes, across the {len(MATCHED_2STEP_INITS)} matched initialisations.",
    )
    save(fig, "F05_2step_case_maps")


@figure("F10")
def f10_rollout_map_strip() -> None:
    """Global rollout: IMERG above, forecast below, from day 1 to past the collapse."""
    import cartopy.crs as ccrs
    import zarr

    _init_cartopy()
    leads_h = [24, 72, 120, 144, 192]
    fsteps = [h // STEP_HOURS for h in leads_h]
    sample, init = pick_case(ROLLOUT_RUN, ROLLOUT_SAMPLES, fsteps)
    collapse = collapse_onset()

    store = _open_store(ROLLOUT_RUN)
    g = zarr.open_group(store, mode="r")
    lat, lon = read_coords(g, sample)

    fig, axes = map_grid(
        2,
        [ROBINSON_ASPECT] * len(leads_h),
        ccrs.Robinson(),
        panel_h=2.0,
        left=0.7,
    )
    mappable = None
    for c, (lead, fstep) in enumerate(zip(leads_h, fsteps, strict=True)):
        for r, kind in enumerate(["target", "prediction"]):
            vals = read_field(g, sample, fstep, kind)
            mappable = draw_precip(axes[r, c], lat, lon, vals)
        past = lead >= collapse
        axes[0, c].set_title(
            f"+{lead} h  (day {lead / 24:g})",
            loc="center",
            fontsize=12,
            color=ACCENT if past else INK,
        )
    store.close()

    for r, name in enumerate(["IMERG", "Forecast"]):
        axes[r, 0].text(
            -0.02,
            0.5,
            name,
            transform=axes[r, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
            color=COLOR["truth"] if r == 0 else COLOR[ROLLOUT_RUN],
        )

    fig.suptitle(
        f"Rollout from {init}: coherent rain bands through day 6, empty by day 8",
        x=0.008,
        y=0.995,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    precip_colorbar(
        fig, mappable, ax=axes, orientation="horizontal", fraction=0.05, pad=0.03, aspect=70
    )
    annotate_source(
        fig,
        f"z71y2ik8, initialised {init}. Trained rollout is +48 h; collapse onset (FBI < 0.8) is "
        f"+{collapse:.0f} h. Initialisation chosen automatically on observed rain in the "
        f"zoom boxes.",
    )
    save(fig, "F10_rollout_map_strip")


@figure("F11")
def f11_rollout_zoom_strip() -> None:
    """The day-6 claim at regional scale: does the structure survive, or only the statistics?"""
    import cartopy.crs as ccrs
    import zarr

    _init_cartopy()
    leads_h = [24, 72, 120, 144]
    fsteps = [h // STEP_HOURS for h in leads_h]
    sample, init = pick_case(ROLLOUT_RUN, ROLLOUT_SAMPLES, fsteps)

    store = _open_store(ROLLOUT_RUN)
    g = zarr.open_group(store, mode="r")
    lat, lon = read_coords(g, sample)

    for zkey, zoom in ZOOMS.items():
        m = box_mask(lat, lon, zoom)
        fig, axes = map_grid(
            2,
            [zoom_aspect(zoom)] * len(leads_h),
            ccrs.PlateCarree(),
            panel_h=2.1,
        )
        mappable = None
        for c, (lead, fstep) in enumerate(zip(leads_h, fsteps, strict=True)):
            for r, kind in enumerate(["target", "prediction"]):
                vals = read_field(g, sample, fstep, kind)
                mappable = draw_precip(axes[r, c], lat[m], lon[m], vals[m], zoom=zoom)
            axes[0, c].set_title(f"+{lead} h  (day {lead / 24:g})", loc="center", fontsize=12)
        for r, name in enumerate(["IMERG", "Forecast"]):
            axes[r, 0].text(
                -0.03,
                0.5,
                name,
                transform=axes[r, 0].transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=11,
                fontweight="bold",
                color=COLOR["truth"] if r == 0 else COLOR[ROLLOUT_RUN],
            )
        fig.suptitle(
            f"{zoom['long']} — initialised {init}",
            x=0.008,
            y=0.995,
            ha="left",
            va="top",
            fontsize=15,
            fontweight="bold",
        )
        precip_colorbar(
            fig, mappable, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=60
        )
        annotate_source(fig, f"z71y2ik8, initialised {init}. Trained rollout is +48 h.")
        save(fig, f"F11_rollout_zoom_{zkey}")
    store.close()


# --------------------------------------------------------------------------------------------
# Part 1 -- model size, architecture and training budget
# --------------------------------------------------------------------------------------------

# state_dict key prefixes -> the groups the size table reports. Matches the split used in
# docs/imerg_finetune_parents_comparison.md, so the af90zz71 row there is a usable cross-check.
MODULE_GROUPS = {
    "encoder": ("encoder.",),
    "forecast engine": ("forecast_engine.",),
    "IMERG decoder": ("embed_target_coords.", "target_token_engines.", "pred_heads."),
}


def model_config(run: str) -> dict:
    return json.loads((MODELS / run / f"model_{run}_{CHKPT}.json").read_text())


def count_parameters(parent: str, freeze_regex: str | None) -> dict[str, float]:
    """Parameters per module group, in millions, read from the parent checkpoint.

    ``mmap=True`` keeps the 3-5 GB file off the heap: only tensor metadata is touched, which is
    all ``numel()`` needs. Results are cached because opening the checkpoints dominates runtime.
    """
    import re

    import torch

    cache = OUTDIR / f".params_{parent}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    path = MODELS / parent / f"{parent}_latest.chkpt"
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    # Sharded runs prefix every key with "module."; strip it so the group prefixes still match.
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    out = {g: 0 for g in MODULE_GROUPS}
    out["other"] = 0
    out["total"] = 0
    out["trainable in finetune"] = 0
    pattern = re.compile(freeze_regex) if freeze_regex else None

    for key, tensor in sd.items():
        n = int(np.prod(tensor.shape))
        out["total"] += n
        for group, prefixes in MODULE_GROUPS.items():
            if key.startswith(prefixes):
                out[group] += n
                break
        else:
            out["other"] += n
        if pattern is not None and not pattern.match(key):
            out["trainable in finetune"] += n

    out = {k: v / 1e6 for k, v in out.items()}
    _logger.info(
        "%s: %.1f M total (%s)",
        parent,
        out["total"],
        ", ".join(f"{k} {v:.1f}" for k, v in out.items() if k != "total"),
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def input_channels(cfg: dict) -> tuple[int, list[tuple[str, int]]]:
    """Total forcing-stream input channels, and the per-stream breakdown."""
    parts = [
        (name, len(s.get("train_source_channels") or []))
        for name, s in cfg["streams"].items()
        if s.get("forcing")
    ]
    parts.sort(key=lambda p: -p[1])
    return sum(n for _, n in parts), parts


def model_rows() -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Assemble the model comparison as (column headers, [(row label, cells)])."""
    cols, data = [], {}
    for run, meta in RUNS.items():
        cfg = model_config(run)
        params = count_parameters(meta["parent"], cfg.get("freeze_modules"))
        n_in, parts = input_channels(cfg)
        # Read the trained rollout depth rather than trusting the run table.
        n_steps = cfg["training_config"]["forecast"]["num_steps"]
        n_eval = load_scores(run, "global", "ets").shape[1]
        analysis = cfg["streams"]["ERA5_in"]["filenames"][0]
        grid = "N320" if "n320" in analysis else "o96"
        iasi = next((f"{n.replace('METOP_', '')} ({c})" for n, c in parts if "IASI" in n), "-")
        data[run] = {
            "Inference run": run,
            "Pretrained backbone": meta["parent"],
            "Total parameters": f"{params['total']:.0f} M",
            "  encoder": f"{params['encoder']:.0f} M",
            "  forecast engine": f"{params['forecast engine']:.0f} M",
            "  IMERG decoder": f"{params['IMERG decoder']:.1f} M",
            "Trainable in finetune": f"{params['trainable in finetune']:.1f} M "
            f"({100 * params['trainable in finetune'] / params['total']:.1f}%)",
            "Local encoder (dim x blocks)": f"{cfg['ae_local_dim_embed']} x "
            f"{cfg['ae_local_num_blocks']}",
            "Global encoder (dim x blocks)": f"{cfg['ae_global_dim_embed']} x "
            f"{cfg['ae_global_num_blocks']}",
            "Forecast-engine blocks": str(cfg["fe_num_blocks"]),
            "Register tokens": str(cfg["num_register_tokens"]),
            "Cross-stream attention": "yes" if cfg.get("use_xsa") else "no",
            "Step conditioning": "yes" if cfg.get("with_step_conditioning") else "no",
            "Analysis input grid": grid,
            "Input channels": str(n_in),
            "  IASI stream": iasi,
            "Training steps (istep)": f"{meta['istep']:,}",
            "Decoder finetune rollout": f"{n_steps} steps (+{n_steps * STEP_HOURS} h)",
            "Rollout evaluated": f"{n_eval} steps (+{n_eval * STEP_HOURS} h)",
        }
        cols.append(run)

    labels = list(data[cols[0]].keys())
    rows = [(lab, [data[c][lab] for c in cols]) for lab in labels]
    return cols, rows


@figure("T01")
def t01_model_table() -> None:
    """Model size, architecture and budget — rendered as a slide table and as markdown.

    The two backbones differ in far more than the pretraining objective: input channel count,
    analysis grid, encoder shape, register tokens and cross-stream attention all change together.
    The table exists so that the 2-step result is read as "which pretrained model", not as a
    controlled SSL-vs-supervised ablation.
    """
    import matplotlib.pyplot as plt

    cols, rows = model_rows()
    headers = ["", *[f"{RUNS[c]['short']}  ·  {c}" for c in cols]]

    # The axes are the table: sizing the figure by row count and filling it with bbox=[0,0,1,1]
    # avoids the large dead area that `loc=` plus `scale()` leaves behind.
    row_h = 0.34
    fig, ax = plt.subplots(figsize=(12.5, row_h * (len(rows) + 1) + 1.15))
    ax.axis("off")
    ax.set_position((0.005, 0.055, 0.99, 0.87))
    table = ax.table(
        cellText=[[lab, *cells] for lab, cells in rows],
        colLabels=headers,
        cellLoc="left",
        bbox=[0, 0, 1, 1],
        colWidths=[0.34, 0.22, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#EAEAEA")
        if r == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(COLOR[cols[c - 1]] if c > 0 else INK)
        else:
            label = rows[r - 1][0]
            if not label.startswith("  "):
                cell.set_text_props(fontweight="bold" if c == 0 else "normal")
            if r % 2 == 0:
                cell.set_facecolor("#F7F7F7")

    fig.suptitle(
        "Model size, architecture and training budget",
        x=0.008,
        y=0.99,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    annotate_source(
        fig,
        "Parameter counts read from the parent checkpoints' tensors; architecture and stream "
        "configuration from each run's model_<run>_chkpt00000.json. In every run the backbone is "
        "frozen and only the IMERG decoder trains.",
    )
    save(fig, "T01_model_table")

    md = [
        "| | " + " | ".join(f"**{RUNS[c]['short']}** ({c})" for c in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    md += [f"| {lab} | " + " | ".join(cells) + " |" for lab, cells in rows]
    (OUTDIR / "T01_model_table.md").write_text("\n".join(md) + "\n")
    _logger.info("wrote %s", OUTDIR / "T01_model_table.md")
    for line in md:
        _logger.info(line)


@figure("F01")
def f01_lineage() -> None:
    """How the three runs relate: two pretrained backbones, one frozen-backbone finetune recipe."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    cfgs = {run: model_config(run) for run in RUNS}
    n_in = {run: input_channels(cfgs[run])[0] for run in RUNS}

    def shape_line(run: str) -> str:
        c = cfgs[run]
        return (
            f"{c['ae_local_dim_embed']}x{c['ae_local_num_blocks']} local, "
            f"{c['ae_global_dim_embed']}x{c['ae_global_num_blocks']} global, "
            f"{c['num_register_tokens']} register tokens"
        )

    fig, ax = plt.subplots(figsize=(13, 6.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(1.05, 7.75)
    ax.axis("off")

    def box(x, y, w, h, title, lines, color, *, fill="#FFFFFF", lw=1.6):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06,rounding_size=0.12",
                facecolor=fill,
                edgecolor=color,
                linewidth=lw,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.28,
            title,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=color,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h - 0.72,
            "\n".join(lines),
            ha="center",
            va="top",
            fontsize=10,
            color=INK,
            zorder=3,
        )

    def arrow(x0, y0, x1, y1, color=MUTED):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=16,
                color=color,
                linewidth=1.4,
                zorder=1,
            )
        )

    fc, jp = COLOR["xfaps4wq"], COLOR["r15j90ns"]

    # Row 1: the two pretrained backbones.
    box(
        0.3,
        6.02,
        5.2,
        1.15,
        "Forecast pretraining — c71eo6pu",
        [
            shape_line("xfaps4wq"),
            f"N320 analysis · {n_in['xfaps4wq']} input channels (210 IASI PCs)",
        ],
        fc,
    )
    box(
        6.5,
        6.02,
        5.2,
        1.15,
        "JEPA / SSL pretraining — af90zz71",
        [
            shape_line("r15j90ns"),
            f"o96 analysis · {n_in['r15j90ns']} input channels (18 IASI radiances)",
        ],
        jp,
    )

    # The shared recipe.
    ax.add_patch(
        FancyBboxPatch(
            (0.3, 4.50),
            11.4,
            0.90,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=GOOD_SOFT,
            edgecolor=MUTED,
            linewidth=1.2,
            zorder=2,
        )
    )
    ax.text(
        6.0,
        5.20,
        "IMERG diagnostic-decoder finetune",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    ax.text(
        6.0,
        4.86,
        "backbone frozen (encoder · forecast engine · latent heads) — only a fresh IMERG "
        "decoder trains · MSE on IMERG tp, N320, 6 h windows",
        ha="center",
        va="top",
        fontsize=10,
        color=INK,
        zorder=3,
    )
    for x in (2.9, 9.1):
        arrow(x, 5.96, x, 5.48, fc if x < 6 else jp)

    # Row 3: the finetuned checkpoints.
    box(
        0.3,
        2.90,
        3.5,
        1.18,
        "hmpa42d2",
        ["2-step rollout", f"istep {RUNS['xfaps4wq']['istep']:,}"],
        fc,
    )
    box(
        4.25,
        2.90,
        3.5,
        1.18,
        "nescpnb2",
        ["2-step rollout", f"istep {RUNS['r15j90ns']['istep']:,}"],
        jp,
    )
    box(
        8.2,
        2.90,
        3.5,
        1.18,
        "v3gs9fwn",
        ["8-step rollout (+48 h)", f"istep {RUNS['z71y2ik8']['istep']:,}"],
        jp,
    )
    for x, c in ((2.05, fc), (6.0, jp), (9.95, jp)):
        arrow(x, 4.44, x, 4.16, c)
    # v3gs9fwn continues nescpnb2's lineage rather than branching from the recipe box.
    ax.annotate(
        "",
        xy=(8.14, 3.49),
        xytext=(7.81, 3.49),
        arrowprops={"arrowstyle": "-|>", "color": jp, "lw": 1.4},
    )

    # Row 4: the inference runs being presented.
    box(
        0.3,
        1.35,
        3.5,
        1.18,
        "xfaps4wq",
        ["inference +6 h, +12 h", "126 initialisations"],
        fc,
        fill="#FDF3EF",
    )
    box(
        4.25,
        1.35,
        3.5,
        1.18,
        "r15j90ns",
        ["inference +6 h, +12 h", "120 initialisations"],
        jp,
        fill="#EDF3F7",
    )
    box(
        8.2,
        1.35,
        3.5,
        1.18,
        "z71y2ik8",
        ["inference +6 h … +240 h", "10 initialisations"],
        jp,
        fill="#EDF3F7",
    )
    for x, c in ((2.05, fc), (6.0, jp), (9.95, jp)):
        arrow(x, 2.84, x, 2.61, c)

    ax.text(
        6.0,
        7.70,
        "Two pretrained backbones, one decoder recipe, three inference runs",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    annotate_source(
        fig,
        "The two backbones differ in pretraining objective AND in architecture, input channels and "
        "analysis resolution — the 2-step comparison is between pretrained models, not an isolated "
        "objective ablation.",
    )
    save(fig, "F01_lineage")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--figure",
        nargs="+",
        default=["all"],
        help="figure ids to build (e.g. F02 F06), or 'all'",
    )
    ap.add_argument("--outdir", type=Path, default=None, help="override output directory")
    ap.add_argument(
        "--verify-matched-inits",
        action="store_true",
        help="re-derive the 37 shared initialisations from the stores and exit",
    )
    args = ap.parse_args()

    global OUTDIR
    if args.outdir is not None:
        OUTDIR = args.outdir

    if args.verify_matched_inits:
        verify_matched_inits()
        return

    apply_style()
    wanted = sorted(FIGURES) if args.figure == ["all"] else args.figure
    unknown = [f for f in wanted if f not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; available: {sorted(FIGURES)}")

    for fid in wanted:
        _logger.info("=== %s ===", fid)
        FIGURES[fid]()


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
