#!/usr/bin/env python3

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Plot target vs prediction for CERRA output from a validation zarr zip.

Produces a 3-panel figure (target | prediction | bias) per sample/channel,
using nearest-neighbour regridding to a regular lat/lon mesh so that the
dense HEALPix scatter (~2M points) renders as a proper filled map.

Example:
    .venv/bin/python scripts/plot_cerra_output.py \\
        --zip results/tqmu9ny5/validation_chkpt00000_rank0000.zip \\
        --stream CERRA \\
        --out plots/cerra_quicklook \\
        --channels tp \\
        --samples 0 1 2 \\
        --dpi 200
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-weathergen")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/weathergen-cache")

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from weathergen.common.config import _load_private_conf
from weathergen.common.io import zarrio_reader

# Mirror cartopy path setup from plotter.py
_work_dir = Path(_load_private_conf(None)["path_shared_working_dir"]) / "assets/cartopy"
cartopy.config["data_dir"] = str(_work_dir)
cartopy.config["pre_existing_data_dir"] = str(_work_dir)
os.environ["CARTOPY_DATA_DIR"] = str(_work_dir)

# Europe bounding box (matches regions.py)
_EUROPE = dict(lat_min=35, lat_max=70, lon_min=-10, lon_max=40)

# Discrete precipitation thresholds (mm) — meteorologically meaningful
PRECIP_LEVELS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# White for trace/zero, then shades of blue → dark navy for heavy rain
_PRECIP_COLORS = [
    "#ffffff",  # 0–0.1  trace / dry  → white
    "#c6e0f5",  # 0.1–0.5             → very light blue
    "#90c2e8",  # 0.5–1               → light blue
    "#5ba3d5",  # 1–2                 → sky blue
    "#2e7fbc",  # 2–5                 → medium blue
    "#1558a0",  # 5–10                → blue
    "#0c3c80",  # 10–20               → dark blue
    "#061e54",  # 20–50               → deep navy
    "#2d004b",  # > 50                → dark purple (extreme)
]


def _make_precip_norm_cmap(vmax: float):
    """
    Build a BoundaryNorm + ListedColormap for precipitation.
    Values below the first level map to white (trace/dry).
    Values above the last level get the darkest colour.
    """
    # Keep only levels below vmax; always keep at least the first one
    levels = [l for l in PRECIP_LEVELS if l <= max(vmax * 1.05, PRECIP_LEVELS[0] + 0.01)]
    levels = sorted(set(levels)) or [PRECIP_LEVELS[0]]

    # n_intervals == len(levels) - 1 data bins, plus under/over handled by set_under/set_over
    n_intervals = len(levels) - 1
    # Slice interval colours from index 1 (skip the white "under" slot)
    interval_colors = _PRECIP_COLORS[1 : n_intervals + 1] or [_PRECIP_COLORS[1]]

    cmap = mcolors.ListedColormap(interval_colors, name="precip_blue")
    cmap.set_under("#ffffff")  # below first level → white (trace / dry)
    cmap.set_over(_PRECIP_COLORS[-1])  # above last level → darkest

    # Do NOT pass extend= to BoundaryNorm; under/over are in the cmap instead
    norm = mcolors.BoundaryNorm(levels, ncolors=len(interval_colors))
    return norm, cmap, levels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True, type=Path, help="Path to validation_chkpt*.zip")
    p.add_argument("--stream", required=True, help="Stream name, e.g. CERRA")
    p.add_argument("--out", required=True, type=Path, help="Output directory for plot images")
    p.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channel(s) to plot. Default: all channels in the store.",
    )
    p.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=None,
        help="Sample indices to plot. Default: first sample only.",
    )
    p.add_argument(
        "--fsteps",
        nargs="+",
        type=int,
        default=None,
        help="Forecast steps to plot. Default: all available.",
    )
    p.add_argument(
        "--lat-min", type=float, default=_EUROPE["lat_min"], help="Map south edge (default: 35)"
    )
    p.add_argument(
        "--lat-max", type=float, default=_EUROPE["lat_max"], help="Map north edge (default: 70)"
    )
    p.add_argument(
        "--lon-min", type=float, default=_EUROPE["lon_min"], help="Map west edge (default: -10)"
    )
    p.add_argument(
        "--lon-max", type=float, default=_EUROPE["lon_max"], help="Map east edge (default: 40)"
    )
    p.add_argument(
        "--grid-res",
        type=float,
        default=0.05,
        help="Regular-grid resolution in degrees for regridding (default: 0.05)",
    )
    p.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    p.add_argument(
        "--fmt", default="png", choices=["png", "pdf", "svg"], help="Image format (default: png)"
    )
    p.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.5,
        help="Percentile of target used to set the top of the colour scale (default: 99.5). "
        "Raise toward 100 to show more extreme values.",
    )
    p.add_argument("--no-bias", action="store_true", help="Omit the bias panel")
    return p.parse_args()


def _regrid(lats, lons, vals, lat_grid, lon_grid):
    """Nearest-neighbour scatter → regular grid via cKDTree."""
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    query_pts = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])

    tree = cKDTree(np.column_stack([lats, lons]))
    _, idx = tree.query(query_pts, k=1, workers=-1)

    return vals[idx].reshape(lat_mesh.shape), lat_mesh, lon_mesh


def _add_map_features(ax, lon_min, lon_max, lat_min, lat_max):
    """Set extent, white background, coastlines, borders, gridlines."""
    ax.set_facecolor("white")
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # White land patch drawn first so data renders on top cleanly
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "50m"),
        facecolor="white",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "50m"),
        facecolor="#f0f4f8",
        edgecolor="none",
        zorder=0,  # very light blue-grey for sea
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="#333333", zorder=3
    )
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#666666", zorder=3)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="#aaaaaa",
        alpha=0.7,
        linestyle="--",
        zorder=2,
        xlocs=range(int(lon_min) - 1, int(lon_max) + 1, 10),
        ylocs=range(int(lat_min) - 1, int(lat_max) + 1, 5),
    )
    gl.top_labels = False
    gl.right_labels = False


def plot_sample(
    lats,
    lons,
    target_vals,
    pred_vals,
    channel,
    sample,
    fstep,
    valid_time,
    lat_grid,
    lon_grid,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    vmax_percentile,
    dpi,
    fmt,
    out_dir,
    no_bias,
):
    print(f"  Regridding {len(lats):,} points to {len(lat_grid)}×{len(lon_grid)} grid...")
    tar_grid, lat_mesh, lon_mesh = _regrid(lats, lons, target_vals, lat_grid, lon_grid)
    prd_grid, _, _ = _regrid(lats, lons, pred_vals, lat_grid, lon_grid)
    bias_grid = prd_grid - tar_grid

    # ---- precipitation colour scale (shared between target and prediction) ----
    # Use the target's distribution to set the scale so both panels are comparable
    vmax_data = float(np.nanpercentile(np.maximum(target_vals, 0), vmax_percentile))
    norm_precip, cmap_precip, levels = _make_precip_norm_cmap(vmax_data)

    # ---- bias colour scale ----------------------------------------------------
    bias_abs = float(np.nanpercentile(np.abs(bias_grid), vmax_percentile))
    bias_abs = max(bias_abs, 0.01)
    norm_bias = mcolors.TwoSlopeNorm(vmin=-bias_abs, vcenter=0.0, vmax=bias_abs)

    proj = ccrs.LambertConformal(
        central_longitude=(lon_min + lon_max) / 2,
        central_latitude=(lat_min + lat_max) / 2,
    )

    ncols = 2 if no_bias else 3
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(7.5 * ncols, 6.5),
        subplot_kw={"projection": proj},
        dpi=dpi,
        facecolor="white",
    )

    panels = (
        [
            ("Target", tar_grid, norm_precip, cmap_precip),
            ("Prediction", prd_grid, norm_precip, cmap_precip),
        ]
        if no_bias
        else [
            ("Target", tar_grid, norm_precip, cmap_precip),
            ("Prediction", prd_grid, norm_precip, cmap_precip),
            ("Bias  pred − target", bias_grid, norm_bias, "RdBu_r"),
        ]
    )

    for ax, (title, grid, norm, cmap) in zip(axes, panels):
        _add_map_features(ax, lon_min, lon_max, lat_min, lat_max)
        im = ax.pcolormesh(
            lon_mesh,
            lat_mesh,
            grid,
            norm=norm,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
            rasterized=True,
            zorder=1,
        )
        is_precip = cmap is cmap_precip
        cb = plt.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.04,
            shrink=0.88,
            extend="both",
        )
        cb.set_label(f"{channel} [mm]", fontsize=9)
        if is_precip:
            cb.set_ticks(levels)
            cb.set_ticklabels([str(l) for l in levels], fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

    vt_str = str(valid_time)[:16].replace("T", " ") if valid_time is not None else "unknown"
    fig.suptitle(
        f"CERRA  {channel}  |  sample {sample}  fstep {fstep}  |  {vt_str}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{channel}_sample{sample:03d}_fstep{fstep:03d}.{fmt}"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname


def main():
    args = parse_args()
    assert args.zip.exists(), f"Zarr zip not found: {args.zip}"

    # Pre-build regular grid (shared across all samples)
    lat_grid = np.arange(args.lat_min, args.lat_max, args.grid_res)
    lon_grid = np.arange(args.lon_min, args.lon_max, args.grid_res)

    with zarrio_reader(args.zip) as zio:
        all_samples = sorted(int(s) for s in zio.samples)
        all_fsteps = sorted(int(f) for f in zio.forecast_steps)

        samples_req = args.samples if args.samples is not None else all_samples[:1]
        fsteps_req = args.fsteps if args.fsteps is not None else all_fsteps

        samples = [s for s in samples_req if s in all_samples]
        fsteps = [f for f in fsteps_req if f in all_fsteps]

        if missing_s := set(samples_req) - set(all_samples):
            print(f"WARNING: samples not in store, skipping: {sorted(missing_s)}")
        if missing_f := set(fsteps_req) - set(all_fsteps):
            print(f"WARNING: fsteps not in store, skipping: {sorted(missing_f)}")

        print(f"Store:    {args.zip}")
        print(f"Streams:  {zio.streams}")
        print(f"Samples:  {all_samples} → plotting {samples}")
        print(f"Fsteps:   {all_fsteps} → plotting {fsteps}")
        print(f"Grid:     {len(lat_grid)}lat × {len(lon_grid)}lon @ {args.grid_res}°")

        for fstep in fsteps:
            for sample in samples:
                print(f"\n--- sample={sample}  fstep={fstep} ---")
                item = zio.get_data(sample, args.stream, fstep)

                if item.target is None or item.prediction is None:
                    print("  No data, skipping.")
                    continue

                target = item.target.as_xarray().squeeze()
                pred = item.prediction.as_xarray().squeeze()

                lats = target.lat.values
                lons = target.lon.values

                # Spatial mask to plot region (avoids gridding irrelevant ocean/land outside)
                mask = (
                    (lats >= args.lat_min)
                    & (lats <= args.lat_max)
                    & (lons >= args.lon_min)
                    & (lons <= args.lon_max)
                )
                lats, lons = lats[mask], lons[mask]

                all_channels = (
                    [target.channel.values.item()]
                    if target.channel.ndim == 0
                    else list(target.channel.values)
                )
                channels = args.channels if args.channels is not None else all_channels
                channels = [c for c in channels if c in all_channels]

                if not channels:
                    print(f"  No matching channels. Available: {all_channels}")
                    continue

                valid_time = None
                if "valid_time" in target.coords:
                    vt = target.valid_time.values
                    valid_time = vt.flat[0] if vt.ndim > 0 else vt.item()

                for ch in channels:
                    print(f"  Channel: {ch}")
                    if "channel" in target.dims:
                        t_vals = target.sel(channel=ch).values[mask]
                        p_vals = pred.sel(channel=ch).values[mask]
                    else:
                        t_vals = target.values[mask]
                        p_vals = pred.values[mask]

                    plot_sample(
                        lats,
                        lons,
                        t_vals,
                        p_vals,
                        channel=ch,
                        sample=sample,
                        fstep=fstep,
                        valid_time=valid_time,
                        lat_grid=lat_grid,
                        lon_grid=lon_grid,
                        lat_min=args.lat_min,
                        lat_max=args.lat_max,
                        lon_min=args.lon_min,
                        lon_max=args.lon_max,
                        vmax_percentile=args.vmax_percentile,
                        dpi=args.dpi,
                        fmt=args.fmt,
                        out_dir=args.out,
                        no_bias=args.no_bias,
                    )

    print(f"\nDone. Plots saved under: {args.out}")


if __name__ == "__main__":
    main()
