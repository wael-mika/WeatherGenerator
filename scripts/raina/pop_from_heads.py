# ruff: noqa: T201
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Probability-of-precipitation (PoP) from the WTA ensemble heads, with a calibration check.

Treats the K prediction heads of an mse_wta / ensemble IMERG run as an empirical sample of the
predictive distribution and derives, per point, the exceedance probability

    PoP(x) = (1/K) * sum_k  1[ pred_k(x) > threshold ]          (hard vote  = 1 - ECDF(threshold))
    PoP_soft(x) = (1/K) * sum_k sigmoid( (pred_k(x) - threshold) / temp )   (smoothed vote)

It then asks the ONLY question that matters for a probability: is it any good?  The heads are
NOT trained with a proper probabilistic score (WTA tiles the output space, it does not match the
CDF), so the raw vote-fraction is expected to be miscalibrated -- some heads carry a GLOBAL wet/dry
bias (see the per-head maps) that compresses PoP toward the middle. So we report:

  - Brier score + Brier skill score vs the climatological base rate (calibration + resolution);
  - ROC AUC (pure discrimination -- can be good even when calibration is off);
  - a RELIABILITY table (observed wet-frequency vs predicted PoP; with K heads the hard PoP takes
    the K+1 discrete levels k/K, which is a clean reliability diagram);
  and, if --out is given, a maps figure (target tp | observed wet mask | PoP) plus a reliability
  diagram.

Discrimination good + reliability off  =>  keep PoP, apply a post-hoc calibration (isotonic/Platt)
fitted on a HELD-OUT set (not done here; the reliability curve here IS that map, but in-sample).
The principled alternative is a hurdle head's BCE rain-probability branch
(playground/docs/imerg_tp_ensemble_loss_plan.md §3.1), which optimizes PoP directly.

Usage:
    uv run python scripts/raina/pop_from_heads.py --run-id i63ostyk --fstep 1 --threshold 0.1 \
        --out plots/wta_diag/i63ostyk_pop.png

Notes:
  - Reuses playground/plot_ens_heads.load_sample_members (one sample) and extreme_eval staging.
  - Points come from the ~quasi-area-uniform reduced-Gaussian n320 grid, so unweighted point
    statistics approximate area-weighted ones; no cos-lat weighting is applied.
  - CPU only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAYGROUND = _REPO_ROOT / "playground"
if str(_PLAYGROUND) not in sys.path:
    sys.path.insert(0, str(_PLAYGROUND))

import extreme_eval as ee  # noqa: E402
import plot_ens_heads as peh  # noqa: E402


def _auc(y: NDArray, score: NDArray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (rank-based; handles ties)."""
    y = y.astype(bool)  # accept float 0/1 or bool labels
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    # average ranks for ties
    i = 0
    r = 1
    while i < len(s_sorted):
        j = i
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[order[i:j]] = (r + (r + (j - i) - 1)) / 2.0
        r += j - i
        i = j
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def quantile_pop(p: NDArray, levels: NDArray, threshold: float) -> NDArray:
    """PoP from a sorted-quantile ensemble via CDF inversion.

    p:[N,K] head values, levels:[K] ascending quantile levels tau_k = P(tp <= q_k).
    Returns PoP = 1 - CDF(threshold), where CDF(threshold) is found by linear interpolation of
    the per-point quantile function q(tau). Points whose whole quantile function is above the
    threshold are floored at 1 - levels[0]; entirely below, at 1 - levels[-1].
    """
    n, k = p.shape
    q = np.sort(p, axis=1)  # enforce monotone quantiles
    tau = levels
    n_above = (q > threshold).sum(axis=1)  # quantiles exceeding threshold
    first_above = k - n_above  # index of first exceeding head (sorted ascending)

    cdf = np.empty(n, dtype=np.float64)
    all_ex = first_above == 0
    none_ex = first_above == k
    mid = ~(all_ex | none_ex)
    cdf[all_ex] = tau[0]
    cdf[none_ex] = tau[-1]
    if mid.any():
        rows = np.where(mid)[0]
        j = first_above[rows]  # in 1..k-1
        q_lo, q_hi = q[rows, j - 1], q[rows, j]
        t_lo, t_hi = tau[j - 1], tau[j]
        denom = q_hi - q_lo
        frac = np.where(denom > 1e-12, (threshold - q_lo) / np.maximum(denom, 1e-12), 0.0)
        frac = np.clip(frac, 0.0, 1.0)
        cdf[rows] = t_lo + frac * (t_hi - t_lo)
    return 1.0 - cdf


def compute_pop(
    t: NDArray, p: NDArray, threshold: float, temp: float, levels: list[float] | None = None
) -> dict:
    """t:[N] target mm, p:[N,K] head preds mm. Returns PoP diagnostics for one threshold.

    levels=None  -> heads are exchangeable samples: PoP = fraction of heads > threshold.
    levels given -> heads are sorted quantiles at those levels (e.g. a pinball run): PoP is the
                    proper CDF inversion  PoP = 1 - tau*, where q(tau*) = threshold (linear interp
                    on the per-point quantile function). This is the right way to read an
                    exceedance probability off a quantile ensemble.
    """
    n, k = p.shape
    y = t > threshold  # observed wet

    if levels is None:
        pop_hard = (p > threshold).mean(axis=1)  # [N] in {0, 1/K, ..., 1}
        pop_soft = (1.0 / (1.0 + np.exp(-(p - threshold) / temp))).mean(axis=1)
        rel_levels = np.arange(k + 1) / k  # discrete reliability bins
        discrete = True
    else:
        pop_hard = quantile_pop(p, np.asarray(levels, dtype=np.float64), threshold)
        pop_soft = pop_hard  # no soft variant for the quantile route
        rel_levels = np.linspace(0.0, 1.0, 11)  # 10 equal-width reliability bins
        discrete = False

    base = float(y.mean())
    brier_climo = base * (1.0 - base)

    def _brier(pop: NDArray) -> float:
        return float(np.mean((pop - y) ** 2))

    brier_hard, brier_soft = _brier(pop_hard), _brier(pop_soft)

    rel = []
    if discrete:
        for lv in rel_levels:
            sel = np.isclose(pop_hard, lv)
            cnt = int(sel.sum())
            obs = float(y[sel].mean()) if cnt > 0 else float("nan")
            rel.append((float(lv), obs, cnt))
    else:
        # equal-width bins; report each bin's mean forecast PoP vs observed frequency
        idx = np.clip(np.digitize(pop_hard, rel_levels[1:-1]), 0, len(rel_levels) - 2)
        for b in range(len(rel_levels) - 1):
            sel = idx == b
            cnt = int(sel.sum())
            fc = float(pop_hard[sel].mean()) if cnt > 0 else float("nan")
            obs = float(y[sel].mean()) if cnt > 0 else float("nan")
            rel.append((fc, obs, cnt))

    return {
        "n": n,
        "k": k,
        "threshold": threshold,
        "base_rate": base,
        "brier_hard": brier_hard,
        "brier_soft": brier_soft,
        "bss_hard": 1.0 - brier_hard / brier_climo if brier_climo > 0 else float("nan"),
        "bss_soft": 1.0 - brier_soft / brier_climo if brier_climo > 0 else float("nan"),
        "auc_hard": _auc(y, pop_hard),
        "auc_soft": _auc(y, pop_soft),
        "reliability": rel,
        "pop_hard": pop_hard,
        "pop_soft": pop_soft,
        "y": y,
    }


def print_report(d: dict, label: str) -> None:
    print("\n" + "=" * 74)
    print(f"  PoP from ensemble heads : {label}")
    print("=" * 74)
    print(
        f"  points N={d['n']:,}  heads K={d['k']}  threshold={d['threshold']} mm  "
        f"base rate (obs wet) {d['base_rate'] * 100:.1f}%"
    )
    print("\n  scores (lower Brier / higher BSS & AUC = better):")
    print(
        f"    hard vote : Brier {d['brier_hard']:.4f}  BSS {d['bss_hard']:+.3f}  "
        f"AUC {d['auc_hard']:.3f}"
    )
    print(
        f"    soft vote : Brier {d['brier_soft']:.4f}  BSS {d['bss_soft']:+.3f}  "
        f"AUC {d['auc_soft']:.3f}"
    )
    print("\n  reliability (hard PoP level -> observed wet freq; want obs ≈ level):")
    print(f"    {'PoP':>6} {'obs':>7} {'count':>10}")
    for lv, obs, cnt in d["reliability"]:
        if cnt == 0:
            continue
        obs_s = f"{obs * 100:6.1f}%" if np.isfinite(obs) else "    nan"
        bar = ""
        if np.isfinite(obs):
            diff = obs - lv
            if diff < -0.08:
                bar = "  over-confident"
            elif diff > 0.08:
                bar = "  under-confident"
        print(f"    {lv * 100:5.0f}% {obs_s} {cnt:10,}{bar}")

    # AUC interpretation
    auc = d["auc_soft"]
    print("\n  [read]")
    if np.isfinite(auc):
        disc = "strong" if auc > 0.85 else ("useful" if auc > 0.7 else "weak")
        print(f"    - discrimination (AUC {auc:.3f}) is {disc}: higher PoP => more likely rain.")
    if d["bss_hard"] < 0:
        print("    - BSS < 0: raw vote-fraction is WORSE than always forecasting the base rate")
        print("      => miscalibrated; needs post-hoc calibration (isotonic/Platt, held-out set).")
    elif d["bss_hard"] < 0.2:
        print("    - BSS positive but small: some skill, likely miscalibrated (see reliability).")
    print("=" * 74)


def maybe_plot(
    d: dict, t: NDArray, lat: NDArray, lon: NDArray, label: str, out: Path, draw_maps: bool = True
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] plotting unavailable: {exc}")
        return

    # --- maps: target tp | observed wet mask | PoP (single sample only) ---
    if draw_maps:
        cmap_p, norm_p = ee._get_cmap_norm(ee.PRECIP_LEVELS, cb_name="ocean_r")
        geo = True
        try:
            fig, axes = plt.subplots(
                1, 3, figsize=(15, 4), subplot_kw={"projection": ccrs.Robinson()}, squeeze=False
            )
        except Exception:
            geo = False
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
        axes = axes.ravel()
        sc0 = peh._draw_panel(axes[0], lat, lon, t, "Target tp (mm)", cmap_p, norm_p, geo)
        fig.colorbar(sc0, ax=axes[0], shrink=0.7)
        sc1 = peh._draw_panel(
            axes[1], lat, lon, d["y"].astype(np.float32), f"Observed wet (>{d['threshold']}mm)",
            "Blues", None, geo
        )
        fig.colorbar(sc1, ax=axes[1], shrink=0.7)
        sc2 = peh._draw_panel(
            axes[2], lat, lon, d["pop_hard"], "PoP", "viridis", None, geo
        )
        fig.colorbar(sc2, ax=axes[2], shrink=0.7)
        fig.suptitle(f"{label} | PoP maps | threshold {d['threshold']} mm", fontsize=11)
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"[info] wrote maps -> {out}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)

    # --- reliability diagram ---
    rel = [(lv, obs, cnt) for lv, obs, cnt in d["reliability"] if cnt > 0 and np.isfinite(obs)]
    if rel:
        lvs = [r[0] for r in rel]
        obss = [r[1] for r in rel]
        cnts = [r[2] for r in rel]
        fig2, (axr, axh) = plt.subplots(
            2, 1, figsize=(5, 6.5), gridspec_kw={"height_ratios": [3, 1]}
        )
        axr.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        axr.plot(lvs, obss, "o-", color="#cc3311", label="head-vote PoP")
        axr.axhline(d["base_rate"], color="gray", ls=":", lw=0.8, label="base rate")
        axr.set_xlabel("forecast PoP")
        axr.set_ylabel("observed wet frequency")
        axr.set_xlim(0, 1)
        axr.set_ylim(0, 1)
        axr.set_title(f"{label} reliability (AUC {d['auc_soft']:.3f}, BSS {d['bss_hard']:+.3f})")
        axr.legend(fontsize=8)
        axh.bar(lvs, cnts, width=0.8 / d["k"], color="#4477aa")
        axh.set_xlim(0, 1)
        axh.set_xlabel("forecast PoP")
        axh.set_ylabel("count")
        axh.set_yscale("log")
        fig2.tight_layout()
        rel_out = out.with_name(out.stem + "_reliability.png")
        fig2.savefig(rel_out, dpi=130, bbox_inches="tight")
        print(f"[info] wrote reliability -> {rel_out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--stream", default="IMERG_ANEMOI")
    ap.add_argument("--fstep", default="1")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--sample-limit", type=int, default=1, help="how many samples to aggregate")
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--sample-ids", default=None, help="comma-separated explicit sample ids")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument(
        "--threshold", default="0.1", help="mm/6h wet threshold(s), comma-separated (def 0.1)"
    )
    ap.add_argument("--temp", type=float, default=0.1, help="soft-vote sigmoid temperature (mm)")
    ap.add_argument(
        "--levels",
        default=None,
        help="comma-separated quantile levels (heads are sorted quantiles, e.g. a pinball run); "
        "switches PoP to proper CDF inversion instead of a head-vote fraction",
    )
    ap.add_argument("--out", default=None, help="optional PNG path for maps + reliability figs")
    args = ap.parse_args()

    label = args.label or args.run_id
    if args.results_dir:
        results_path = Path(args.results_dir).expanduser().resolve()
    elif ee.PRIVATE_CF is not None and ee.PRIVATE_CF.get("run_path"):
        results_path = Path(str(ee.PRIVATE_CF.get("run_path"))).expanduser().resolve()
    elif ee.PRIVATE_CF is not None and ee.PRIVATE_CF.get("path_shared_working_dir"):
        results_path = (
            Path(str(ee.PRIVATE_CF.get("path_shared_working_dir"))).expanduser() / "results"
        ).resolve()
    else:
        results_path = (_REPO_ROOT / "results").resolve()
    print(f"[info] results dir: {results_path}")

    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(x) for x in args.sample_ids.split(",") if x.strip()]

    # load_experiment (member_avg) returns t[N], p[N,K], lat, lon, s across the selected samples,
    # already flattened and finite-filtered and converted to mm.
    t, p, lat, lon, s = ee.load_experiment(
        results_path=results_path,
        run_id=args.run_id,
        epoch=args.epoch,
        stream=args.stream,
        fstep=args.fstep,
        strict_fstep=False,
        allow_grid_mismatch=False,
        sample_limit=args.sample_limit,
        sample_offset=args.sample_offset,
        sample_ids=sample_ids,
        ens_reduce="member_avg",
    )
    if p.ndim != 2 or p.shape[1] < 2:
        raise SystemExit(f"run {args.run_id} is not an ensemble (need >1 head).")
    n_samples = len(np.unique(s))

    levels = None
    if args.levels:
        levels = [float(x) for x in args.levels.split(",") if x.strip()]
        if len(levels) != p.shape[1]:
            raise SystemExit(
                f"--levels has {len(levels)} entries but the run has {p.shape[1]} heads."
            )
    thresholds = [float(x) for x in str(args.threshold).split(",") if x.strip()]
    for thr in thresholds:
        d = compute_pop(t, p, thr, args.temp, levels=levels)
        print_report(d, f"{label} ({n_samples} samples, fstep{args.fstep}, thr {thr}mm)")
        if args.out:
            base = Path(args.out).expanduser().resolve()
            out_thr = base.with_name(f"{base.stem}_thr{thr}{base.suffix}")
            # maps only meaningful for a single sample (points overlap across samples otherwise)
            maybe_plot(d, t, lat, lon, label, out_thr, draw_maps=(n_samples == 1))


if __name__ == "__main__":
    main()
