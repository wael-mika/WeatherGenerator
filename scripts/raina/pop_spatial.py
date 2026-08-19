# ruff: noqa: T201
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spatial probabilistic analysis of a quantile (pinball) IMERG run.

pop_from_heads.py answers "is the exceedance probability calibrated GLOBALLY?". This script asks
the spatial follow-up: does the probabilistic output hold up OVER SPACE -- is it calibrated in
every region, and does the predictive uncertainty vary sensibly (large in the ITCZ / storm tracks,
near-zero in the dry subtropics) rather than being a global average that is only right on the mean?

For each point it derives, from the sorted quantile heads at the given `--levels`:
  - PoP = P(tp > threshold) by CDF inversion (quantile_pop from pop_from_heads);
  - a MAGNITUDE-uncertainty proxy: predictive spread = q(hi level) - q(lo level) [mm];
  - a RAIN/NO-RAIN uncertainty: binary entropy H(PoP) = -p log2 p - (1-p) log2(1-p) (max at 0.5).

It then reports / plots:
  1. a per-LATITUDE-BAND calibration table (base rate, mean PoP, AUC, Brier, BSS, mean bias) --
     the direct "is it calibrated everywhere" check;
  2. GRIDDED maps (coarse lat/lon, aggregated over samples): observed wet frequency, mean PoP,
     calibration bias (PoP - obs), mean predictive spread, mean PoP-entropy, sample count;
  3. per-band reliability curves overlaid.

Usage:
    uv run python scripts/raina/pop_spatial.py --run-id j0tk9gxn --fstep 1 --sample-limit 16 \
        --threshold 0.1 \
        --levels 0.3,0.55,0.68,0.75,0.8,0.85,0.89,0.92,0.94,0.96,0.97,0.98,0.99,0.995,0.998,0.999 \
        --out plots/wta_diag/j0tk9gxn_spatial.png

Notes:
  - Reads DENORMALISED mm from the store, so --threshold is in mm.
  - Reuses playground/extreme_eval loading + scripts/raina/pop_from_heads (quantile_pop, _auc).
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
for _p in (str(_REPO_ROOT / "scripts" / "raina"), str(_PLAYGROUND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import extreme_eval as ee  # noqa: E402
import pop_from_heads as pfh  # noqa: E402

# signed-latitude bands (name, lo, hi) in degrees
_BANDS = [
    ("60-90 N", 60.0, 90.0),
    ("30-60 N", 30.0, 60.0),
    ("tropics", -30.0, 30.0),
    ("30-60 S", -60.0, -30.0),
    ("60-90 S", -90.0, -60.0),
]


def _binary_entropy(p: NDArray) -> NDArray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def _spread(p_sorted: NDArray, levels: NDArray, lo: float, hi: float) -> NDArray:
    """Predictive spread = q(level nearest hi) - q(level nearest lo), per point [mm]."""
    i_lo = int(np.argmin(np.abs(levels - lo)))
    i_hi = int(np.argmin(np.abs(levels - hi)))
    return p_sorted[:, i_hi] - p_sorted[:, i_lo]


def _band_metrics(pop: NDArray, y: NDArray) -> dict:
    base = float(y.mean()) if len(y) else float("nan")
    brier = float(np.mean((pop - y) ** 2)) if len(y) else float("nan")
    climo = base * (1.0 - base)
    return {
        "n": int(len(y)),
        "base": base,
        "pop_mean": float(pop.mean()) if len(y) else float("nan"),
        "auc": pfh._auc(y, pop) if len(y) > 2 else float("nan"),
        "brier": brier,
        "bss": (1.0 - brier / climo) if climo > 0 else float("nan"),
        "bias": float(pop.mean() - base) if len(y) else float("nan"),
    }


def grid_aggregate(
    lat: NDArray, lon: NDArray, fields: dict[str, NDArray], nlat: int, nlon: int
) -> dict:
    """Accumulate point fields onto a regular nlat x nlon grid; return per-cell means + count."""
    lon360 = np.mod(lon, 360.0)
    ilat = np.clip(((lat + 90.0) / 180.0 * nlat).astype(int), 0, nlat - 1)
    ilon = np.clip((lon360 / 360.0 * nlon).astype(int), 0, nlon - 1)
    cell = ilat * nlon + ilon
    count = np.bincount(cell, minlength=nlat * nlon).astype(np.float64)
    out = {"count": count.reshape(nlat, nlon)}
    for name, v in fields.items():
        s = np.bincount(cell, weights=v.astype(np.float64), minlength=nlat * nlon)
        mean = np.full(nlat * nlon, np.nan)
        nz = count > 0
        mean[nz] = s[nz] / count[nz]
        out[name] = mean.reshape(nlat, nlon)
    return out


def print_report(label: str, thr: float, glob: dict, bands: list[tuple[str, dict]]) -> None:
    print("\n" + "=" * 84)
    print(f"  Spatial PoP analysis : {label}   (P(tp > {thr} mm))")
    print("=" * 84)
    hdr = f"    {'region':>9} {'N':>10} {'base%':>7} {'PoP%':>7} {'AUC':>6} {'BSS':>7} {'bias':>7}"
    print(hdr)

    def _row(name: str, m: dict) -> str:
        return (
            f"    {name:>9} {m['n']:>10,} {m['base'] * 100:6.1f}% {m['pop_mean'] * 100:6.1f}% "
            f"{m['auc']:6.3f} {m['bss']:+7.3f} {m['bias'] * 100:+6.1f}%"
        )

    print(_row("GLOBAL", glob))
    print("    " + "-" * 60)
    for name, m in bands:
        print(_row(name, m))
    print("\n  [read] BSS should stay positive and bias near 0 in EVERY band (calibrated over")
    print("  space); AUC shows where the probability discriminates best. A band with large")
    print("  |bias| or BSS<0 is a region where the global calibration does not hold.")
    print("=" * 84)


def maybe_plot(
    grids: dict, nlat: int, nlon: int, label: str, thr: float, bands_rel: list, out: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] plotting unavailable: {exc}")
        return

    lat_edges = np.linspace(-90, 90, nlat + 1)
    lon_edges = np.linspace(0, 360, nlon + 1)

    panels = [
        ("obs", "observed wet freq", "viridis", (0, 1), False),
        ("pop", "mean forecast PoP", "viridis", (0, 1), False),
        ("bias", "calibration bias (PoP - obs)", "RdBu_r", (-0.3, 0.3), True),
        ("spread", "predictive spread q_hi-q_lo [mm]", "magma", None, False),
        ("entropy", "rain/no-rain entropy [bits]", "cividis", (0, 1), False),
        ("count", "sample count / cell", "Greys", None, False),
    ]
    geo = True
    try:
        fig, axes = plt.subplots(
            2, 3, figsize=(18, 8), subplot_kw={"projection": ccrs.Robinson()}, squeeze=False
        )
    except Exception:
        geo = False
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    axes = axes.ravel()
    for ax, (key, title, cmap, vlim, _div) in zip(axes, panels, strict=True):
        field = np.ma.masked_invalid(grids[key])
        kw = {"cmap": cmap, "shading": "auto"}
        if vlim is not None:
            kw["vmin"], kw["vmax"] = vlim
        if geo:
            kw["transform"] = ccrs.PlateCarree()
            m = ax.pcolormesh(lon_edges, lat_edges, field, **kw)
            ax.set_global()
            ax.coastlines(resolution="110m", linewidth=0.4)
        else:
            m = ax.pcolormesh(lon_edges, lat_edges, field, **kw)
        ax.set_title(title, fontsize=9)
        fig.colorbar(m, ax=ax, shrink=0.6)
    fig.suptitle(f"{label} | spatial PoP (P(tp>{thr}mm)) | {nlat}x{nlon} grid", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[info] wrote maps -> {out}")

    # per-band reliability
    if bands_rel:
        fig2, ax = plt.subplots(figsize=(5.2, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        for name, lvs, obss in bands_rel:
            ax.plot(lvs, obss, "o-", ms=3, lw=1, label=name)
        ax.set_xlabel("forecast PoP")
        ax.set_ylabel("observed wet frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label} per-band reliability (P(tp>{thr}mm))")
        ax.legend(fontsize=7)
        fig2.tight_layout()
        rel_out = out.with_name(out.stem + "_bandrel.png")
        fig2.savefig(rel_out, dpi=120, bbox_inches="tight")
        print(f"[info] wrote per-band reliability -> {rel_out}")


def _band_reliability(pop: NDArray, y: NDArray, nbins: int = 10) -> tuple[list, list]:
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(pop, edges[1:-1]), 0, nbins - 1)
    lvs, obss = [], []
    for b in range(nbins):
        sel = idx == b
        if sel.sum() > 50:
            lvs.append(float(pop[sel].mean()))
            obss.append(float(y[sel].mean()))
    return lvs, obss


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--stream", default="IMERG_ANEMOI")
    ap.add_argument("--fstep", default="1")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--sample-limit", type=int, default=16)
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--sample-ids", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.1, help="mm/6h wet threshold")
    ap.add_argument("--levels", required=True, help="comma-separated quantile levels of the heads")
    ap.add_argument("--spread-lo", type=float, default=0.75, help="lower level for spread proxy")
    ap.add_argument("--spread-hi", type=float, default=0.98, help="upper level for spread proxy")
    ap.add_argument("--nlat", type=int, default=72, help="grid rows (default 2.5 deg)")
    ap.add_argument("--nlon", type=int, default=144, help="grid cols (default 2.5 deg)")
    ap.add_argument("--out", default=None, help="optional PNG path for the spatial figures")
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
    levels = np.array([float(x) for x in args.levels.split(",") if x.strip()], dtype=np.float64)

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
    if p.ndim != 2 or p.shape[1] != len(levels):
        raise SystemExit(
            f"run has {p.shape[1] if p.ndim == 2 else 1} heads but --levels has {len(levels)}."
        )

    p_sorted = np.sort(p, axis=1)
    pop = pfh.quantile_pop(p, levels, args.threshold)
    spread = _spread(p_sorted, levels, args.spread_lo, args.spread_hi)
    entropy = _binary_entropy(pop)
    y = (t > args.threshold).astype(np.float64)
    n_samples = len(np.unique(s))
    print(f"[info] {len(t):,} points over {n_samples} samples; {len(levels)} quantile levels")

    # 1. per-band calibration table
    glob = _band_metrics(pop, y)
    bands, bands_rel = [], []
    for name, lo, hi in _BANDS:
        sel = (lat >= lo) & (lat < hi)
        if sel.sum() == 0:
            continue
        bands.append((name, _band_metrics(pop[sel], y[sel])))
        lvs, obss = _band_reliability(pop[sel], y[sel])
        if lvs:
            bands_rel.append((name, lvs, obss))
    print_report(label, args.threshold, glob, bands)

    # 2. gridded maps
    grids = grid_aggregate(
        lat, lon, {"obs": y, "pop": pop, "spread": spread, "entropy": entropy}, args.nlat, args.nlon
    )
    grids["bias"] = grids["pop"] - grids["obs"]
    if args.out:
        maybe_plot(grids, args.nlat, args.nlon, label, args.threshold, bands_rel,
                   Path(args.out).expanduser().resolve())


if __name__ == "__main__":
    main()
