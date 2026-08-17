#!/usr/bin/env python
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Input-data coverage figures: what actually reaches the model.

The IMERG decoder runs sit on top of a backbone fed by an analysis stream plus a set of
observation streams. These figures show, for those observation streams, *where* the data is,
*when* it is available, and *how much* of it arrives per assimilation window.

Store layout (``DataReaderObs``): each observation zarr holds a flat ``data`` array of shape
(n_rows, n_cols) with ``colnames`` in ``data.attrs``, a matching ``dates`` array, and an hourly
offset index ``idx_197001010000_1``. Hour ``h`` since 1970-01-01 occupies rows
``idx[h - 1] : idx[h]`` — verified against the ``dates`` array, and the reason ``_window_rows``
below subtracts one.

That index is the cheap way in: differencing it gives the observation count for every hour of the
whole archive without reading a single data row, which is what C03 and C04 are built on.

Usage
-----
    uv run --offline python scripts/imerg_input_coverage_figures.py --figure all
    uv run --offline python scripts/imerg_input_coverage_figures.py \
        --figure C01 --time 2023-06-02T12
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
_logger = logging.getLogger("imerg_coverage")

DATA = Path("/e/data1/slmet/ml_training")
OUTDIR = Path("plots/imerg_input_coverage")

# Reference initialisation time: the same window the evaluated IMERG forecasts start from, so the
# maps show the input state behind a forecast that appears elsewhere in the deck.
DEFAULT_TIME = "2023-06-02T12:00"
WINDOW_HOURS = 6  # the model's assimilation window
# The maps default to a single hour. The model ingests all six hourly sweeps in a window, but
# drawing them together stacks six near-identical discs and the last one drawn wins, so the
# colours would be an arbitrary mix of hours. One hour is an honest instant; --map-hours 6
# shows the full window.
MAP_HOURS = 1

BASE_EPOCH = np.datetime64("1970-01-01T00:00")
HOURLY_INDEX = "idx_197001010000_1"


# --------------------------------------------------------------------------------------------
# Stream inventory
# --------------------------------------------------------------------------------------------

# `stores` is ordered; the first entry covering the requested time is used for the maps.
# `channels` records the per-model-family channel counts from
# docs/imerg_finetune_parents_comparison.md §4, verified here against each store's `colnames`.
STREAMS: dict[str, dict] = {
    "METEOSAT_SEVIRI_IR": {
        "label": "Meteosat SEVIRI (IR)",
        "kind": "geostationary",
        "stores": ["observations-file-2014-2024-seviri-o256-wegen-v3"],
        "coords": ("lat", "lon"),
        "value": "obsvalue_rawbt_105",
        "value_label": "10.5 um brightness temperature [K]",
        "channels": "11",
        "color": "#C2603F",
    },
    "GOES_ABI_IR": {
        "label": "GOES-16 ABI (IR)",
        "kind": "geostationary",
        "stores": ["observations-file-2017-2024-abi-goes16-IR-o256-v2"],
        "coords": ("lat", "lon"),
        "value": "obsvalue_rawbt_105",
        "value_label": "10.5 um brightness temperature [K]",
        "channels": "8",
        "color": "#E08A5B",
    },
    "GOES_ABI_VIS": {
        "label": "GOES-16 ABI (VIS)",
        "kind": "geostationary",
        "stores": ["observations-file-2017-2024-abi-goes16-VIS-o256-v2"],
        "coords": ("lat", "lon"),
        "value": "obsvalue_rawbt_086",
        "value_label": "0.86 um reflectance channel",
        "channels": "2",
        "color": "#F0C27B",
    },
    "HIMAWARI_AHI_IR": {
        "label": "Himawari-8/9 AHI (IR)",
        "kind": "geostationary",
        "stores": [
            "observations-file-2015-2022-himawari8-IR-o256-v1",
            "observations-file-2022-2024-himawari9-IR-o256-v1",
        ],
        "coords": ("lat", "lon"),
        "value": "obsvalue_rawbt_105",
        "value_label": "10.5 um brightness temperature [K]",
        "channels": "8",
        "color": "#2F6F8F",
    },
    "HIMAWARI_AHI_VIS": {
        "label": "Himawari-8/9 AHI (VIS)",
        "kind": "geostationary",
        "stores": [
            "observations-file-2015-2022-himawari8-VIS-o256-v1",
            "observations-file-2022-2024-himawari9-VIS-o256-v1",
        ],
        "coords": ("lat", "lon"),
        "value": "obsvalue_rawbt_086",
        "value_label": "0.86 um reflectance channel",
        "channels": "2",
        "color": "#7FB3C8",
    },
    "AVHRR": {
        "label": "Metop AVHRR",
        "kind": "polar",
        "stores": ["observations-file-2001-2023-eumetsat-avhrr-o256-v4"],
        # This store names its coordinate columns differently from every other one.
        "coords": ("latitude", "longitude"),
        "value": "obsvalue_rawbt_1",
        "value_label": "IR window brightness temperature [K]",
        "channels": "5",
        "color": "#5B9E4C",
    },
    "METOP_IASI_PC": {
        "label": "Metop IASI (principal components)",
        "kind": "polar",
        "stores": ["observations-od-ai-0001-2011-2023-iasi-pc-bufr-v1"],
        "coords": ("lat", "lon"),
        "value": "obsvalue_pc_lw_1",
        "value_label": "long-wave principal component 1",
        "channels": "210",
        "color": "#7B4FA3",
    },
    "METOP_ABC_AVHRR_IASI": {
        "label": "Metop A/B/C IASI (radiances)",
        "kind": "polar",
        "stores": [
            "observations-ea-ofb-0001-2007-2021-metop-a-iasi-radiances-v1",
            "observations-ea-ofb-0001-2013-2023-metop-b-iasi-radiances-v1",
            "observations-ea-ofb-0001-2019-2023-metop-c-iasi-radiances-v1",
        ],
        "coords": ("lat", "lon"),
        "value": "obsvalue_rawbt_921",
        "value_label": "IASI channel 921 brightness temp [K]",
        "channels": "18",
        "color": "#A87CC4",
    },
    "SurfaceCombined": {
        "label": "Surface (SYNOP / ship / buoy)",
        "kind": "surface",
        "stores": ["observations-ea-ofb-0001-1979-2025-combined-surface-v5"],
        "coords": ("lat", "lon"),
        "value": "obsvalue_t2m_0",
        "value_label": "2 m temperature [K]",
        "channels": "9",
        "color": "#3C3C3C",
    },
}

KIND_ORDER = ["geostationary", "polar", "surface"]

# Channel counts per backbone, from docs/imerg_finetune_parents_comparison.md §4.
# "—" means the stream is not part of that model's inputs at all.
CHANNEL_TABLE_COLUMNS = ["af90zz71 (JEPA)", "fx276yn3", "cw6a4szu", "MTM x4"]
CHANNEL_TABLE = {
    "ERA5_in (analysis)": ["68 (o96)", "74 (n320)", "74 (n320)", "88 (n320)"],
    "METEOSAT_SEVIRI_IR": ["11", "11", "11", "11"],
    "GOES_ABI_IR": ["8", "8", "8", "8"],
    "GOES_ABI_VIS": ["2", "2", "2", "2"],
    "HIMAWARI_AHI_IR": ["8", "8", "8", "8"],
    "HIMAWARI_AHI_VIS": ["2", "2", "2", "2"],
    "AVHRR": ["— (in IASI stream)", "5", "5", "5"],
    "IASI": ["18 (A/B/C radiances)", "210 (PCs)", "17", "17"],
    "SurfaceCombined": ["9", "7", "7", "7"],
    "TOTAL input channels": ["126", "327", "134", "148"],
}

INK, MUTED, GRID = "#1A1A1A", "#6E6E6E", "#D8D8D8"


def apply_style() -> None:
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
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}", format=ext)
    _logger.info("wrote %s.{png,pdf}", OUTDIR / name)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Store access
# --------------------------------------------------------------------------------------------


def _open(store: str):
    import zarr

    with warnings.catch_warnings():
        # These stores keep pngs and a config.yaml next to the arrays; zarr warns about each.
        warnings.simplefilter("ignore")
        return zarr.open(str(DATA / f"{store}.zarr"), mode="r")


def store_span(store: str) -> tuple[np.datetime64, np.datetime64]:
    """First and last observation time, read from the two ends of the dates array."""
    z = _open(store)
    first = np.asarray(z["dates"][0]).ravel()[0]
    last = np.asarray(z["dates"][-1]).ravel()[0]
    return first, last


def hourly_counts(store: str) -> npt.NDArray:
    """Observations per hour for the entire archive, straight from the offset index.

    ``idx`` is cumulative row offsets per hour, so the first difference is the per-hour count —
    no data rows are touched, which makes a full multi-decade time series essentially free.
    """
    z = _open(store)
    idx = np.asarray(z[HOURLY_INDEX][:])
    return np.diff(idx, prepend=idx[0])


def _hour_of(t: np.datetime64) -> int:
    return int((t - BASE_EPOCH) / np.timedelta64(1, "h"))


def _window_rows(z, t: np.datetime64, hours: int) -> tuple[int, int]:
    """Row range covering [t, t + hours). Hour h lives in idx[h - 1] : idx[h]."""
    idx = z[HOURLY_INDEX]
    h = _hour_of(t)
    return int(idx[h - 1]), int(idx[h + hours - 1])


def store_for_time(stream: str, t: np.datetime64) -> str | None:
    """First store of this stream whose time span contains ``t``."""
    for store in STREAMS[stream]["stores"]:
        first, last = store_span(store)
        if first <= t <= last:
            return store
    return None


def read_window(stream: str, t: np.datetime64, hours: int = WINDOW_HOURS, store: str | None = None):
    """Position and observed value for every observation in one assimilation window.

    Returns ``(lat, lon, value, store)``. The value is the stream's representative channel — the
    point is to show the field the model actually ingests, not just where the pixels are.
    """
    store = store or store_for_time(stream, t)
    if store is None:
        _logger.warning("%s: no store covers %s", stream, t)
        return None, None, None, None

    z = _open(store)
    colnames = list(z["data"].attrs["colnames"])
    lat_name, lon_name = STREAMS[stream]["coords"]
    ilat, ilon = colnames.index(lat_name), colnames.index(lon_name)
    value_name = STREAMS[stream]["value"]
    ival = colnames.index(value_name) if value_name in colnames else None

    lo, hi = _window_rows(z, t, hours)
    if hi <= lo:
        _logger.warning("%s: empty window at %s", stream, t)
        return np.array([]), np.array([]), np.array([]), store

    cols = [ilat, ilon] + ([ival] if ival is not None else [])
    block = np.asarray(z["data"][lo:hi, cols], dtype=np.float32)
    lat, lon = block[:, 0], block[:, 1]
    val = block[:, 2] if ival is not None else np.full(lat.shape, np.nan, np.float32)

    good = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
    lat, lon, val = lat[good], lon[good], val[good]
    # Stores are inconsistent about longitude convention; normalise to [-180, 180).
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    _logger.info(
        "%-22s %s  %9d obs  %s in [%.1f, %.1f]",
        stream,
        str(t)[:16],
        lat.size,
        value_name,
        float(val.min()) if val.size else np.nan,
        float(val.max()) if val.size else np.nan,
    )
    return lat, lon, val, store


def _init_cartopy() -> None:
    import cartopy

    from weathergen.common.config import _load_private_conf

    work_dir = Path(_load_private_conf(None)["path_shared_working_dir"]) / "assets/cartopy"
    cartopy.config["data_dir"] = str(work_dir)
    cartopy.config["pre_existing_data_dir"] = str(work_dir)
    os.environ["CARTOPY_DATA_DIR"] = str(work_dir)


def auto_marker_size(ax, n_points: int, *, fill: float = 1.6) -> float:
    """Marker area (points^2) that makes ``n_points`` just cover the drawn part of the axis.

    A hand-picked size cannot work here: the ring carries ~330k points per hour while the surface
    network carries ~8k, over the same map. Too small and the field looks like haze (which is
    exactly what a fixed s=0.05 produced); too large and the disc edges bleed.
    """
    pos = ax.get_position()
    fig = ax.get_figure()
    area_pt2 = (pos.width * fig.get_figwidth() * 72.0) * (pos.height * fig.get_figheight() * 72.0)
    # Robinson fills roughly 80% of its bounding box.
    return fill * 0.8 * area_pt2 / max(n_points, 1)


def _thin3(lat, lon, val, limit: int, seed: int = 0):
    """Random subsample for rendering only.

    A 6 h geostationary window is several million pixels; drawing all of them makes a 100 MB
    vector file and no visible difference. Sampling is random rather than strided so no scan
    pattern is introduced, and reported observation counts always use the full array.
    """
    if lat.size <= limit:
        return lat, lon, val
    rng = np.random.default_rng(seed)
    sel = rng.choice(lat.size, limit, replace=False)
    return lat[sel], lon[sel], val[sel]


FIGURES: dict[str, object] = {}


def figure(name: str):
    def deco(fn):
        FIGURES[name] = fn
        return fn

    return deco


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------


@figure("C01")
def c01_georing(t: np.datetime64, map_hours: int = MAP_HOURS) -> None:
    """The geostationary ring as the model receives it — one 6 h window, coloured by 10.5 um BT.

    SEVIRI, GOES-16 and Himawari share the same window channel, so they go on one colour scale:
    the discs then read as a single global field rather than three unrelated images, and the
    seams and polar gaps of the ring are visible for what they are.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    _init_cartopy()
    ring = [s_ for s_, m in STREAMS.items() if m["kind"] == "geostationary" and "IR" in s_]

    fig = plt.figure(figsize=(15, 8.2))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    # One shared colour range across the ring, from the pooled 1-99th percentile so a few
    # extreme pixels do not flatten the contrast.
    pooled, data = [], {}
    for stream in ring:
        lat, lon, val, store = read_window(stream, t, hours=map_hours)
        if lat is None or lat.size == 0:
            continue
        data[stream] = (lat, lon, val)
        pooled.append(val[:: max(1, val.size // 200_000)])
    if not pooled:
        _logger.warning("no geostationary data at %s", t)
        return
    allv = np.concatenate(pooled)
    vmin, vmax = np.percentile(allv, [1, 99])

    total = sum(lat.size for lat, _, _ in data.values())
    # Size markers from the pooled count so the three discs share one density, then draw every
    # point: 300k-2M rasterised markers cost little and thinning is what washed the field out.
    size = auto_marker_size(ax, total)
    for lat, lon, val in data.values():
        mappable = ax.scatter(
            lon,
            lat,
            c=val,
            s=size,
            marker="o",
            linewidths=0,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
    ax.coastlines(linewidth=0.35, color="#9A9A9A")

    cb = fig.colorbar(
        mappable, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03, aspect=55
    )
    cb.set_label("10.5 um brightness temperature [K]  —  cold cloud tops dark, warm surface bright")

    ax.set_title(
        f"Geostationary ring — {str(t)[:16]}Z, {total / 1e6:.2f}M observations in {map_hours} h",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.005,
        0.02,
        "Meteosat SEVIRI (11 ch) + GOES-16 ABI (8 ch) + Himawari-8/9 AHI (8 ch), all on one "
        "colour scale. Gaps at the poles and between discs are the ring's real coverage "
        "limits. The model receives six such sweeps per 6 h assimilation window.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    save(fig, "C01_georing")


@figure("C02")
def c02_stream_panels(t: np.datetime64, map_hours: int = MAP_HOURS) -> None:
    """Every input stream in the same window, each on its own scale.

    Same idea as C01 but one panel per stream and per-panel colour limits, because a 2 m
    temperature, an IASI principal component and a visible reflectance share no common units.
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    _init_cartopy()
    order = sorted(STREAMS, key=lambda s_: (KIND_ORDER.index(STREAMS[s_]["kind"]), s_))
    ncols = 3
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.1 * nrows),
        subplot_kw={"projection": ccrs.Robinson()},
    )

    for ax, stream in zip(axes.flat, order, strict=False):
        meta = STREAMS[stream]
        lat, lon, val, store = read_window(stream, t, hours=map_hours)
        ax.set_global()
        ax.coastlines(linewidth=0.3, color="#9A9A9A")

        if lat is None or lat.size == 0:
            # An empty stream is a result, not a blank: say so on the panel.
            ax.text(
                0.5,
                0.5,
                "no data in this window",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#B03A2E",
                fontweight="bold",
            )
            sub = f"{meta['channels']} ch · 0 obs"
        else:
            vmin, vmax = np.percentile(val, [1, 99])
            plat, plon, pval = _thin3(lat, lon, val, 400_000)
            m = ax.scatter(
                plon,
                plat,
                c=pval,
                s=auto_marker_size(ax, plat.size),
                marker="o",
                linewidths=0,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
            cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
            cb.ax.tick_params(labelsize=8)
            cb.set_label(meta["value_label"], fontsize=8.5)
            sub = f"{meta['channels']} ch · {lat.size / 1e3:,.0f}k obs"

        ax.set_title(f"{meta['label']}\n{sub}", fontsize=11, loc="left")

    for ax in axes.flat[len(order) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"What the model ingests — {str(t)[:16]}Z, {map_hours} h",
        x=0.008,
        y=0.995,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "C02_stream_panels")


@figure("C03")
def c03_timeline(t: np.datetime64) -> None:
    """When each stream is available, and the window where all of them overlap."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    order = sorted(STREAMS, key=lambda s: (KIND_ORDER.index(STREAMS[s]["kind"]), s))
    fig, ax = plt.subplots(figsize=(13, 0.52 * sum(len(STREAMS[s]["stores"]) for s in order) + 2.2))

    y, yticks, ylabels = 0, [], []
    # Per-STREAM spans, unioned over that stream's platforms. Intersecting the individual stores
    # would be wrong: METOP_ABC_AVHRR_IASI is Metop-A OR B OR C, and Himawari is 8 then 9, so the
    # stream is available whenever any of its platforms is.
    stream_span: dict[str, tuple[np.datetime64, np.datetime64]] = {}
    for stream in order:
        meta = STREAMS[stream]
        for store in meta["stores"]:
            first, last = store_span(store)
            prev = stream_span.get(stream)
            stream_span[stream] = (
                min(first, prev[0]) if prev else first,
                max(last, prev[1]) if prev else last,
            )
            ax.barh(
                y,
                mdates.date2num(last.astype("M8[s]").astype("O"))
                - mdates.date2num(first.astype("M8[s]").astype("O")),
                left=mdates.date2num(first.astype("M8[s]").astype("O")),
                height=0.62,
                color=meta["color"],
                zorder=3,
            )
            # The satellite generation matters: name the platform, not just the stream.
            tag = store.replace("observations-", "").replace("-o256", "")
            for junk in ("file-", "ea-ofb-0001-", "od-ai-0001-"):
                tag = tag.replace(junk, "")
            yticks.append(y)
            ylabels.append(f"{meta['label']}  ·  {tag[:34]}")
            _logger.info("%-46s %s -> %s", tag[:44], str(first)[:10], str(last)[:10])
            y -= 1

    # The overlap window is what actually constrains a training or evaluation period.
    overlap_start = max(v[0] for v in stream_span.values())
    overlap_end = min(v[1] for v in stream_span.values())
    ax.axvspan(
        mdates.date2num(overlap_start.astype("M8[s]").astype("O")),
        mdates.date2num(overlap_end.astype("M8[s]").astype("O")),
        color="#DCE8EE",
        zorder=0,
    )
    ax.annotate(
        f"all streams available\n{str(overlap_start)[:10]} → {str(overlap_end)[:10]}",
        xy=(mdates.date2num(overlap_start.astype("M8[s]").astype("O")), 0.99),
        xycoords=("data", "axes fraction"),
        xytext=(6, -6),
        textcoords="offset points",
        fontsize=10,
        color="#2F6F8F",
        va="top",
        fontweight="bold",
    )

    # The window the IMERG forecasts are evaluated over — it falls PAST the end of the Metop
    # A/B/C IASI radiance stores, which is the point of drawing it here.
    ev0 = mdates.date2num(np.datetime64("2023-06-01").astype("M8[s]").astype("O"))
    ev1 = mdates.date2num(np.datetime64("2023-08-31").astype("M8[s]").astype("O"))
    ax.axvspan(ev0, ev1, color="#F2DEDA", zorder=1)
    ax.annotate(
        "IMERG evaluation\nJun–Aug 2023",
        xy=(ev1, 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9.5,
        color="#B03A2E",
        va="bottom",
        fontweight="bold",
    )

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("year")
    ax.set_title(
        "Observation stream availability — read from each store's own date array",
        loc="left",
        fontsize=15,
    )
    fig.tight_layout()
    save(fig, "C03_timeline")


@figure("C04")
def c04_counts(t: np.datetime64) -> None:
    """Observation volume per 6 h window through 2023 — duty cycle, gaps and outages."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    year_start, year_end = np.datetime64("2023-01-01T00:00"), np.datetime64("2024-01-01T00:00")
    h0, h1 = _hour_of(year_start), _hour_of(year_end)

    order = sorted(STREAMS, key=lambda s: (KIND_ORDER.index(STREAMS[s]["kind"]), s))
    fig, ax = plt.subplots(figsize=(13, 6.4))

    for stream in order:
        meta = STREAMS[stream]
        # Sum the per-store hourly counts: a stream can switch platform mid-year (Himawari 8->9).
        per_hour = np.zeros(h1 - h0)
        for store in meta["stores"]:
            counts = hourly_counts(store)
            if counts.size < h1:
                counts = np.pad(counts, (0, h1 - counts.size))
            per_hour += counts[h0:h1]
        # Aggregate to the model's 6 h assimilation window.
        n6 = (per_hour.size // WINDOW_HOURS) * WINDOW_HOURS
        per_window = per_hour[:n6].reshape(-1, WINDOW_HOURS).sum(axis=1)
        times = year_start + np.arange(per_window.size) * np.timedelta64(WINDOW_HOURS, "h")
        ax.plot(
            times.astype("M8[s]").astype("O"),
            np.where(per_window > 0, per_window, np.nan),
            color=meta["color"],
            lw=1.1,
            label=f"{meta['label']} ({meta['channels']} ch)",
        )

    ax.set_yscale("log")
    ax.set_ylabel("observations per 6 h window")
    ax.set_xlabel("2023")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=10)
    ax.set_title(
        "Observation volume per 6 h window through 2023 (log scale)", loc="left", fontsize=15
    )
    fig.text(
        0.005,
        0.005,
        "Counts come from each store's hourly offset index, so no observation values are read. "
        "Gaps are genuine outages; a missing line means the stream has no data that year.",
        fontsize=9,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    save(fig, "C04_counts_2023")


@figure("T02")
def t02_channel_inventory(t: np.datetime64) -> None:
    """Channel inventory per backbone — what the totals in the model table are made of."""
    import matplotlib.pyplot as plt

    rows = list(CHANNEL_TABLE.items())
    row_h = 0.38
    fig, ax = plt.subplots(figsize=(12.5, row_h * (len(rows) + 1) + 1.1))
    ax.axis("off")
    ax.set_position((0.005, 0.06, 0.99, 0.86))
    table = ax.table(
        cellText=[[name, *vals] for name, vals in rows],
        colLabels=["stream", *CHANNEL_TABLE_COLUMNS],
        cellLoc="left",
        bbox=[0, 0, 1, 1],
        colWidths=[0.28, 0.20, 0.17, 0.17, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#EAEAEA")
        if r == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(INK)
        else:
            if rows[r - 1][0].startswith("TOTAL"):
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#EDF3F7")
            elif r % 2 == 0:
                cell.set_facecolor("#F7F7F7")

    fig.suptitle(
        "Input channel inventory by backbone",
        x=0.008,
        y=0.99,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.005,
        0.03,
        "The IMERG runs presented use af90zz71 (JEPA, 126 channels on o96) and the c71eo6pu "
        "lineage, whose IASI stream is the 210-component fx276yn3-style feed on n320.",
        fontsize=9,
        color=MUTED,
        ha="left",
        va="top",
    )
    save(fig, "T02_channel_inventory")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figure", nargs="+", default=["all"], help="figure ids, or 'all'")
    ap.add_argument("--time", default=DEFAULT_TIME, help="window start, e.g. 2023-06-02T12:00")
    ap.add_argument(
        "--map-hours",
        type=int,
        default=MAP_HOURS,
        help="hours of data drawn on the maps (C01/C02); 6 shows a whole assimilation window",
    )
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    global OUTDIR
    if args.outdir is not None:
        OUTDIR = args.outdir

    apply_style()
    t = np.datetime64(args.time)
    wanted = sorted(FIGURES) if args.figure == ["all"] else args.figure
    unknown = [f for f in wanted if f not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; available: {sorted(FIGURES)}")

    for fid in wanted:
        _logger.info("=== %s ===", fid)
        if fid in ("C01", "C02"):
            FIGURES[fid](t, args.map_hours)
        else:
            FIGURES[fid](t)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
