# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Diagnostics for an mse_wta (winner-take-all) ensemble finetune of IMERG tp.

Reads an inference/validation store for one WTA run and answers the three questions raised in
playground/docs/imerg_tp_ensemble_loss_plan.md §2.2 -- i.e. whether the 16-head WTA ensemble is
actually diverse or has quietly failed in one of the predicted ways:

  1. WIN-COUNT histogram  -- per head, the fraction of points where that head is the WTA winner
     (argmin over members of the squared error to the target, exactly the mse_wta selection).
     Reported overall AND split by dry (target < wet_threshold) vs wet points, because the
     `argmin` tie on the 67% exact-zero points breaks to head 0 -> head-0 dominance / dead heads.
     Flags:  DOMINANT head (win-share > 3/M),  STARVED/dead head (win-share < 1/(4M)).

  2. PER-HEAD stats + INTENSITY-PARTITION test -- per head: mean, wet-fraction, max, p99, and the
     mean TARGET value at the points it wins ("win-value band"). If heads self-partition by
     intensity (the pinball-redux failure), higher-mean heads win higher-value points, so the
     correlation between head-mean and win-value-band is strongly positive. A high correlation
     means the members are globally-scaled intensity bands, NOT distinct mixed-intensity fields.

  3. WET-ONLY pairwise member correlation -- MxM correlation between member fields restricted to
     wet target points (dry points are ~0 everywhere and inflate correlation toward 1). Mean
     off-diagonal near 1 = members collapsed / redundant; well below 1 = genuine diversity.

Winner identity is unweighted (the mse_wta winner is chosen per point regardless of the
cosine-latitude gradient weight); area-weighting is intentionally not applied here.

Usage:
    uv run python scripts/raina/wta_head_diag.py --run-id v481bcsi
    uv run python scripts/raina/wta_head_diag.py --run-id <id> --fstep 2 --sample-limit 32 \
        --wet-threshold 0.1 --out plots/wta_diag/<id>.png

Notes:
  - Reuses playground/extreme_eval.load_experiment (ens_reduce="member_avg" -> pred [N, M] in mm)
    and its private-config results-dir resolution, so it stages/repairs the store the same way.
  - CPU only; no model or GPU needed (reads the written predictions).
"""

# ruff: noqa: T201
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Reuse the loader + results-path resolution from the extreme_eval tooling.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAYGROUND = _REPO_ROOT / "playground"
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

import extreme_eval as ee  # noqa: E402


def _resolve_results_path(results_dir: str | None) -> Path:
    """Mirror extreme_eval.main's results-dir resolution (CLI > private config > repo/results)."""
    if results_dir:
        return Path(results_dir).expanduser().resolve()
    cf = ee.PRIVATE_CF
    if cf is not None and cf.get("run_path"):
        return Path(str(cf.get("run_path"))).expanduser().resolve()
    if cf is not None and cf.get("path_shared_working_dir"):
        return (Path(str(cf.get("path_shared_working_dir"))).expanduser() / "results").resolve()
    return (_REPO_ROOT / "results").resolve()


def _fmt_bar(frac: float, width: int = 24) -> str:
    n = int(round(frac * width))
    return "#" * n + "-" * (width - n)


def compute_diagnostics(t: NDArray, p: NDArray, wet_threshold: float) -> dict:
    """t: [N] target mm; p: [N, M] member preds mm. Returns a dict of diagnostics."""
    n, m = p.shape
    wet = t >= wet_threshold
    dry = ~wet
    n_wet = int(wet.sum())

    # --- 1. WTA winner per point (argmin squared error, matching mse_wta) -------------------
    se = (p - t[:, None]) ** 2  # [N, M]
    winner = se.argmin(axis=1)  # [N]
    win_share = np.bincount(winner, minlength=m).astype(np.float64) / n
    win_share_dry = np.bincount(winner[dry], minlength=m).astype(np.float64) / max(
        int(dry.sum()), 1
    )
    win_share_wet = np.bincount(winner[wet], minlength=m).astype(np.float64) / max(n_wet, 1)

    # normalized entropy of the win distribution (1.0 = perfectly even, 0 = one head wins all)
    ws = win_share[win_share > 0]
    win_entropy = float(-(ws * np.log(ws)).sum() / np.log(m)) if m > 1 else 0.0

    # --- 2. per-head stats + intensity-partition ---------------------------------------------
    head_mean = p.mean(axis=0)
    head_wetfrac = (p >= wet_threshold).mean(axis=0)
    head_max = p.max(axis=0)
    head_p99 = np.percentile(p, 99, axis=0)
    # mean target value at the points each head wins ("win-value band")
    win_value_band = np.array(
        [float(t[winner == k].mean()) if np.any(winner == k) else np.nan for k in range(m)]
    )
    # correlation between head-mean and its win-value band (high => intensity partition)
    finite = np.isfinite(win_value_band)
    if (
        finite.sum() >= 3
        and np.std(head_mean[finite]) > 0
        and np.nanstd(win_value_band[finite]) > 0
    ):
        partition_corr = float(np.corrcoef(head_mean[finite], win_value_band[finite])[0, 1])
    else:
        partition_corr = np.nan

    # --- 3. wet-only pairwise member correlation ---------------------------------------------
    def _mean_offdiag_corr(mask: NDArray) -> float:
        if int(mask.sum()) < 3:
            return np.nan
        pm = p[mask]  # [n_sel, M]
        # drop members with zero variance on the selection (undefined correlation)
        var_ok = pm.std(axis=0) > 0
        if var_ok.sum() < 2:
            return np.nan
        c = np.corrcoef(pm[:, var_ok].T)  # [m', m']
        off = c[~np.eye(c.shape[0], dtype=bool)]
        return float(np.nanmean(off))

    corr_wet = _mean_offdiag_corr(wet)
    corr_all = _mean_offdiag_corr(np.ones(n, dtype=bool))

    return {
        "n": n,
        "m": m,
        "n_wet": n_wet,
        "wet_frac_target": float(wet.mean()),
        "win_share": win_share,
        "win_share_dry": win_share_dry,
        "win_share_wet": win_share_wet,
        "win_entropy": win_entropy,
        "head_mean": head_mean,
        "head_wetfrac": head_wetfrac,
        "head_max": head_max,
        "head_p99": head_p99,
        "win_value_band": win_value_band,
        "partition_corr": partition_corr,
        "corr_wet": corr_wet,
        "corr_all": corr_all,
        "target_mean": float(t.mean()),
        "target_max": float(t.max()),
    }


def print_report(d: dict, label: str, wet_threshold: float) -> None:
    m = d["m"]
    dom_thr = 3.0 / m
    dead_thr = 0.25 / m

    print("\n" + "=" * 78)
    print(f"  mse_wta head diagnostics : {label}")
    print("=" * 78)
    print(
        f"  points N={d['n']:,}  members M={m}  "
        f"wet(target>={wet_threshold}mm) {d['wet_frac_target'] * 100:.1f}% ({d['n_wet']:,})"
    )
    print(f"  target  mean {d['target_mean']:.3f}  max {d['target_max']:.1f}  (mm/6h)")

    # 1. win-count histogram
    print(
        "\n  [1] WTA win-share per head  (overall | dry | wet)   "
        f"entropy={d['win_entropy']:.3f} (1=even)"
    )
    for k in range(m):
        ws, wd, ww = d["win_share"][k], d["win_share_dry"][k], d["win_share_wet"][k]
        flag = ""
        if ws > dom_thr:
            flag = "  <-- DOMINANT"
        elif ws < dead_thr:
            flag = "  <-- starved/dead"
        print(
            f"    head {k:2d}  {_fmt_bar(ws)} {ws * 100:5.1f}% | "
            f"dry {wd * 100:5.1f}% | wet {ww * 100:5.1f}%{flag}"
        )
    n_dead = int((d["win_share"] < dead_thr).sum())
    n_dom = int((d["win_share"] > dom_thr).sum())
    print(f"    -> {n_dom} dominant, {n_dead} starved/dead (of {m})")

    # 2. per-head stats + intensity partition
    print(
        "\n  [2] per-head stats (sorted by mean)   "
        f"partition_corr(head_mean vs win-value-band) = {d['partition_corr']:.3f}"
    )
    print("      (corr near +1 => heads are globally-scaled INTENSITY BANDS = pinball-redux)")
    order = np.argsort(d["head_mean"])
    print(
        f"    {'head':>4} {'mean':>7} {'wetfrac':>8} {'p99':>7} "
        f"{'max':>7} {'win%':>6} {'winband':>8}"
    )
    for k in order:
        wb = d["win_value_band"][k]
        wb_s = f"{wb:8.3f}" if np.isfinite(wb) else "     nan"
        print(
            f"    {k:>4} {d['head_mean'][k]:7.3f} {d['head_wetfrac'][k] * 100:7.1f}% "
            f"{d['head_p99'][k]:7.2f} {d['head_max'][k]:7.1f} "
            f"{d['win_share'][k] * 100:5.1f}%{wb_s}"
        )

    # 3. wet-only correlation
    print("\n  [3] mean off-diagonal member correlation")
    print(f"      wet points only : {d['corr_wet']:.3f}   (near 1 => members collapsed/redundant)")
    print(f"      all points      : {d['corr_all']:.3f}   (dry points inflate this toward 1)")

    # verdict hints
    print("\n  [verdict hints]")
    if n_dead > 0:
        print(
            f"    - {n_dead} starved head(s): raise eps (0.1-0.2) and/or add a win-share "
            "load-balancing term."
        )
    if d["win_share_dry"].max() > 0.5:
        kdom = int(d["win_share_dry"].argmax())
        print(
            f"    - head {kdom} wins >{d['win_share_dry'][kdom] * 100:.0f}% of DRY points "
            "(argmin tie-break to a dry specialist)."
        )
    if np.isfinite(d["partition_corr"]) and d["partition_corr"] > 0.8:
        print(
            "    - strong intensity partition: members are globally-scaled bands, not distinct "
            "mixed-intensity fields (the pinball failure)."
        )
    if np.isfinite(d["corr_wet"]) and d["corr_wet"] > 0.9:
        print("    - wet-only member correlation >0.9: little real diversity where it matters.")
    print("=" * 78)


def maybe_plot(d: dict, label: str, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[warn] matplotlib unavailable, skipping plot: {exc}")
        return

    m = d["m"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    x = np.arange(m)
    ax.bar(x, d["win_share"] * 100, color="#4477aa")
    ax.axhline(100.0 / m, color="k", ls="--", lw=0.8, label="even (100/M)")
    ax.set_title("WTA win-share per head")
    ax.set_xlabel("head")
    ax.set_ylabel("win-share [%]")
    ax.legend(fontsize=8)

    ax = axes[1]
    order = np.argsort(d["head_mean"])
    ax.plot(range(m), d["head_mean"][order], "o-", label="mean")
    ax.plot(range(m), d["head_p99"][order], "s-", label="p99")
    ax.set_title(f"per-head intensity (partition_corr={d['partition_corr']:.2f})")
    ax.set_xlabel("head (sorted by mean)")
    ax.set_ylabel("tp [mm/6h]")
    ax.legend(fontsize=8)

    ax = axes[2]
    txt = (
        f"wet-only corr : {d['corr_wet']:.3f}\n"
        f"all-point corr: {d['corr_all']:.3f}\n"
        f"win entropy   : {d['win_entropy']:.3f}\n"
        f"target wet    : {d['wet_frac_target'] * 100:.1f}%\n"
        f"target mean   : {d['target_mean']:.3f}\n"
        f"target max    : {d['target_max']:.1f}"
    )
    ax.axis("off")
    ax.text(0.02, 0.95, txt, va="top", family="monospace", fontsize=11)
    ax.set_title("summary")

    fig.suptitle(f"mse_wta head diagnostics : {label}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[info] wrote plot -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-id", required=True, help="run id whose validation/inference store to read"
    )
    ap.add_argument("--label", default=None, help="display label (default: run id)")
    ap.add_argument("--stream", default="IMERG_ANEMOI")
    ap.add_argument("--fstep", default="1")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument(
        "--results-dir", default=None, help="override results dir (else private config)"
    )
    ap.add_argument("--sample-limit", type=int, default=None)
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--sample-ids", default=None, help="comma-separated explicit sample ids")
    ap.add_argument(
        "--wet-threshold", type=float, default=0.1, help="mm/6h; dry/wet split (default 0.1)"
    )
    ap.add_argument("--allow-grid-mismatch", action="store_true")
    ap.add_argument("--out", default=None, help="optional PNG path for the summary figure")
    args = ap.parse_args()

    label = args.label or args.run_id
    results_path = _resolve_results_path(args.results_dir)
    print(f"[info] results dir: {results_path}")

    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(x) for x in args.sample_ids.split(",") if x.strip()]

    t, p, _lat, _lon, _s = ee.load_experiment(
        results_path=results_path,
        run_id=args.run_id,
        epoch=args.epoch,
        stream=args.stream,
        fstep=args.fstep,
        strict_fstep=False,
        allow_grid_mismatch=args.allow_grid_mismatch,
        sample_limit=args.sample_limit,
        sample_offset=args.sample_offset,
        sample_ids=sample_ids,
        ens_reduce="member_avg",
    )

    if p.ndim != 2 or p.shape[1] < 2:
        raise SystemExit(
            f"run {args.run_id} has M={1 if p.ndim == 1 else p.shape[1]} member(s); "
            "this diagnostic needs an ensemble run (ens_size > 1)."
        )

    d = compute_diagnostics(t, p, args.wet_threshold)
    print_report(d, label, args.wet_threshold)
    if args.out:
        maybe_plot(d, label, Path(args.out).expanduser().resolve())


if __name__ == "__main__":
    main()
