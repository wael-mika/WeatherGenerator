# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Visualize masked source/target samples using the training data pipeline.

This script loads a config, builds a MultiStreamDataSampler, extracts one batch,
and plots a single variable for source and target with masking/cropping applied.
This script can run on a cpu or logging node without GPUs. 
Please activate your .venv before running.

Usage:
  python -m weathergen.utils.visualize_masking -c config/config_jepa.yml
  python -m weathergen.utils.visualize_masking -c config/config_jepa.yml --variable 2t
  python -m weathergen.utils.visualize_masking -c config/config_jepa.yml --stream ERA5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import astropy.units as u  
import astropy_healpix as hp  
import cartopy.crs as ccrs  
import cartopy.feature as cfeature  
from omegaconf import OmegaConf, open_dict  

import weathergen.common.config as wg_config  
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler  
from weathergen.train.utils import TRAIN, VAL, filter_config_by_enabled  
from weathergen.utils.utils import is_stream_diagnostic, is_stream_forcing

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize masked source/target data using the same pipeline as training. "
            "Plots a single variable for source and target with masking/cropping applied."
        ),
        allow_abbrev=False,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Config files in ascending order of precedence. The first is treated as base; "
            "subsequent configs override it."
        ),
    )
    parser.add_argument(
        "--private-config",
        type=Path,
        default=None,
        help="Path to the private configuration file that includes platform specific information.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help=(
            "Overwrite individual config options (same format as train). "
            "Example: training_config.shuffle=False"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test"],
        default="train",
        help="Which config section to use for masking (default: train).",
    )
    parser.add_argument(
        "--stream",
        type=str,
        default=None,
        help="Specific stream name to visualize. If not set, prefers era1 -> era5 -> era -> first.",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=None,
        help="Variable name to visualize (applies to both source and target when possible).",
    )
    parser.add_argument(
        "--source-variable",
        type=str,
        default=None,
        help="Optional source-only variable name (overrides --variable for source).",
    )
    parser.add_argument(
        "--target-variable",
        type=str,
        default=None,
        help="Optional target-only variable name (overrides --variable for target).",
    )
    parser.add_argument(
        "--source-sample-idx",
        type=int,
        default=0,
        help="Source sample index within the batch (default: 0).",
    )
    parser.add_argument(
        "--target-sample-idx",
        type=int,
        default=None,
        help="Target sample index within the batch (default: mapped from source).",
    )
    parser.add_argument(
        "--pair-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional list of source sample indices to plot. "
            "If set, overrides --source-sample-idx and plots these pairs."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit on number of source/target pairs to plot.",
    )
    parser.add_argument(
        "--source-step",
        type=int,
        default=0,
        help="Source time step (default: 0, most recent input step).",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=0,
        help="Target forecast step (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible masking (default: 42).",
    )
    parser.add_argument(
        "--denorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Denormalize values before plotting (default: True).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on number of points to plot (randomly sampled).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./plots/masking_preview"),
        help="Directory to save plots (default: ./plots/masking_preview).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Full output path for the figure (overrides output-dir/prefix).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="masking_preview",
        help="Filename prefix used when --output is not provided.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI (default: 150).",
    )
    parser.add_argument(
        "--include-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a full-data panel in each plot (default: True).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Colormap for data visualization (default: coolwarm).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Scatter point size (default: 2.0).",
    )
    parser.add_argument(
        "--shared-colorbar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a single shared colorbar below all panels instead of per-panel colorbars (default: False).",
    )

    return parser


def _select_stream(streams: list, preferred_name: str | None) -> Any:
    """Select a stream from the available streams.

    If a preferred name is provided, returns that stream. Otherwise, prefers
    streams containing 'era1', 'era5', or 'era' in their name (in that order).
    Falls back to the first available stream.

    Parameters
    ----------
    streams : list
        List of stream configuration objects.
    preferred_name : str or None
        Specific stream name to select. If None, uses automatic selection.

    Returns
    -------
    Any
        The selected stream configuration object.

    Raises
    ------
    ValueError
        If no streams are available or if the preferred stream is not found.
    """
    if not streams:
        raise ValueError("No streams found in configuration.")

    if preferred_name:
        for stream in streams:
            if stream.name == preferred_name:
                logger.info("Using specified stream: %s", preferred_name)
                return stream
        names = [s.name for s in streams]
        raise ValueError(f"Stream '{preferred_name}' not found. Available: {names}")

    candidate_streams = [stream for stream in streams if not stream.get("forcing", False)]
    if candidate_streams:
        streams = candidate_streams
        logger.info(
            "Restricting automatic stream selection to non-forcing streams: %s",
            [stream.name for stream in streams],
        )

    # Prefer era1 -> era5 -> era -> first
    for key in ("era1", "era5", "era"):
        for stream in streams:
            if key in stream.name.lower():
                logger.info("Selected stream: %s (matched '%s')", stream.name, key)
                return stream

    logger.info("Selected stream: %s (first available)", streams[0].name)
    return streams[0]


def _load_visualizer_config(
    base_config: Path,
    extra_configs: list[Path],
    private_config: Path | None,
    cli_overwrite,
):
    """
    Load configs for masking previews.

    Saved model JSON files may contain duplicate keys that OmegaConf.load rejects.
    For those, fall back to the standard json loader and then merge overrides on top.
    """
    if base_config.suffix == ".json":
        with base_config.open() as f:
            cf = OmegaConf.create(json.load(f))
        for extra in extra_configs:
            cf = wg_config.merge_configs(cf, OmegaConf.load(extra))
        if private_config is not None:
            cf = wg_config.merge_configs(cf, OmegaConf.load(private_config))
        return wg_config.merge_configs(cf, cli_overwrite)

    base_preview = OmegaConf.load(base_config)
    if base_preview.get("healpix_level") is None:
        return wg_config.load_merge_configs(
            private_config,
            None,
            None,
            None,
            base_config,
            *extra_configs,
            cli_overwrite,
        )

    return wg_config.load_merge_configs(
        private_config, None, None, base_config, *extra_configs, cli_overwrite
    )


def _resolve_mode_config(cf, mode: str):
    keys_to_filter = ["losses", "model_input", "target_input"]
    training_cfg = filter_config_by_enabled(cf.get("training_config"), keys_to_filter)

    if mode == "train":
        return training_cfg, TRAIN

    if mode == "val":
        val_cfg = wg_config.merge_configs(training_cfg, cf.get("validation_config", {}))
        val_cfg = filter_config_by_enabled(val_cfg, keys_to_filter)
        return val_cfg, VAL

    test_cfg = wg_config.merge_configs(training_cfg, cf.get("validation_config", {}))
    test_cfg = wg_config.merge_configs(test_cfg, cf.get("test_config", {}))
    test_cfg = filter_config_by_enabled(test_cfg, keys_to_filter)
    return test_cfg, VAL


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _wrap_lons(lons: np.ndarray) -> np.ndarray:
    return ((lons + 180.0) % 360.0) - 180.0


def _format_mask_params(params: dict) -> str:
    if not params:
        return "no mask config"
    strategy = params.get("masking_strategy", "unknown")
    cfg = params.get("masking_strategy_config", {})
    rate = cfg.get("rate", None)
    parts = [strategy]
    if isinstance(rate, (int, float)):
        parts.append(f"rate={rate:.2f}")
    if "hl_mask" in cfg:
        parts.append(f"hl={cfg['hl_mask']}")
    if "method" in cfg:
        parts.append(str(cfg["method"]))
    return ", ".join(parts)


def _map_points_visible(
    lats: np.ndarray, lons: np.ndarray, mask: np.ndarray, healpix_level: int
) -> np.ndarray:
    """Determine which data points are visible given a HEALPix cell mask.

    Maps each (lat, lon) coordinate to its corresponding HEALPix cell and
    returns a boolean array indicating whether that cell is visible (True)
    or masked out (False).

    Parameters
    ----------
    lats, lons : np.ndarray
        Latitude and longitude coordinates in degrees.
    mask : np.ndarray
        Boolean mask over HEALPix cells (True = visible).
    healpix_level : int
        HEALPix resolution level (nside = 2^level).

    Returns
    -------
    np.ndarray
        Boolean array of same length as lats/lons indicating visibility.
    """
    if len(lats) == 0:
        return np.array([], dtype=bool)
    nside = 2**healpix_level
    lon_rad = np.radians(lons) * u.rad
    lat_rad = np.radians(lats) * u.rad
    cell_indices = hp.lonlat_to_healpix(lon_rad, lat_rad, nside, order="nested")
    mask_np = _to_numpy(mask).astype(bool)
    return mask_np[cell_indices]


def _resolve_var_idx(
    channels: list[str], var_name: str | None, label: str, strict: bool
) -> tuple[int, str]:
    """Resolve a variable name to its index in the channel list.

    Parameters
    ----------
    channels : list[str]
        Available channel/variable names.
    var_name : str or None
        Requested variable name. If None, returns the first channel.
    label : str
        Label for error messages (e.g., "source", "target").
    strict : bool
        If True, raise an error when var_name is not found.
        If False, log a warning and fall back to the first channel.

    Returns
    -------
    tuple[int, str]
        The index and resolved variable name.
    """
    if not channels:
        raise ValueError(f"No channels available for {label}.")
    if var_name is None:
        return 0, channels[0]
    if var_name in channels:
        return channels.index(var_name), var_name
    if strict:
        raise ValueError(f"{label} variable '{var_name}' not found. Available: {channels}")
    logger.warning("%s variable '%s' not found. Falling back to '%s'.", label, var_name, channels[0])
    return 0, channels[0]


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _stream_has_expanded_channels(stream_cfg) -> bool:
    return any(
        stream_cfg.get(key) is not None
        for key in (
            "train_source_channels",
            "train_target_channels",
            "val_source_channels",
            "val_target_channels",
        )
    )


def _stream_is_forcing(stream_cfg, stage) -> bool:
    if stream_cfg.get("forcing", False):
        return True
    if _stream_has_expanded_channels(stream_cfg):
        return is_stream_forcing(stream_cfg, stage)
    return False


def _stream_is_diagnostic(stream_cfg, stage) -> bool:
    if stream_cfg.get("diagnostic", False) or stream_cfg.get("diagnostics", False):
        return True
    if _stream_has_expanded_channels(stream_cfg):
        return is_stream_diagnostic(stream_cfg, stage)
    return False


def _get_stream_plot_mode(stream_cfg, stage) -> str:
    """
    Determine which masked view(s) are meaningful for a stream.

    forcing    -> source only
    diagnostic -> target only
    otherwise  -> both source and target
    """
    is_forcing = _stream_is_forcing(stream_cfg, stage)
    is_diagnostic = _stream_is_diagnostic(stream_cfg, stage)

    if is_forcing and is_diagnostic:
        raise ValueError(
            f"Stream '{stream_cfg.name}' is both forcing and diagnostic; cannot infer plot mode."
        )
    if is_forcing:
        return "source_only"
    if is_diagnostic:
        return "target_only"
    return "both"


def _stream_name_sort_key(stream_cfg) -> tuple[int, str]:
    name = stream_cfg.name.lower()
    for idx, key in enumerate(("era1", "era5", "era")):
        if key in name:
            return idx, name
    return 3, name


def _select_complementary_stream(streams: list, selected_stream, stage, needed_role: str):
    role_priority = {
        "source": {"source_only": 0, "both": 1},
        "target": {"target_only": 0, "both": 1},
    }[needed_role]

    ranked_candidates = []
    for stream in streams:
        if stream.name == selected_stream.name:
            continue
        plot_mode = _get_stream_plot_mode(stream, stage)
        if plot_mode not in role_priority:
            continue
        ranked_candidates.append((role_priority[plot_mode], _stream_name_sort_key(stream), stream))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))
    return ranked_candidates[0][2]


def _resolve_stream_views(streams: list, preferred_name: str | None, stage):
    selected_stream = _select_stream(streams, preferred_name)
    selected_mode = _get_stream_plot_mode(selected_stream, stage)

    source_stream = selected_stream if selected_mode != "target_only" else None
    target_stream = selected_stream if selected_mode != "source_only" else None

    if source_stream is None:
        source_stream = _select_complementary_stream(streams, selected_stream, stage, "source")
    if target_stream is None:
        target_stream = _select_complementary_stream(streams, selected_stream, stage, "target")

    paired_streams = (
        source_stream is not None
        and target_stream is not None
        and source_stream.name != target_stream.name
    )

    return {
        "selected_stream": selected_stream,
        "selected_mode": selected_mode,
        "source_stream": source_stream,
        "target_stream": target_stream,
        "paired_streams": paired_streams,
    }


def _masked_points_from_source_view(
    dataset: MultiStreamDataSampler,
    stream_info,
    stream_name: str,
    stream_data,
    mask,
    step: int,
    var_idx: int,
    denorm: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract visible data points from a source sample after tokenization and masking.

    This function replicates the training pipeline's tokenization and masking logic
    to determine which data points are visible in the source (student) view.

    Parameters
    ----------
    dataset : MultiStreamDataSampler
        The data sampler containing tokenizer and normalization info.
    stream_info
        Stream configuration object.
    stream_name : str
        Name of the stream.
    stream_data
        Stream data object containing source_raw.
    mask : array-like
        Token-level mask (True = visible).
    step : int
        Time step index within the source sequence.
    var_idx : int
        Variable index to extract.
    denorm : bool
        Whether to denormalize the values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Latitude, longitude, and value arrays for visible points.
    """
    rdata = stream_data.source_raw[step]
    if rdata is None or rdata.data is None or len(rdata.data) == 0:
        return np.array([]), np.array([]), np.array([])

    data = np.asarray(rdata.data).copy()
    if denorm:
        data = dataset.denormalize_source_channels(stream_name, data)

    # Tokenize and apply the same mask as in training.
    tokens = dataset.tokenizer.get_tokens_windows(stream_info, [rdata], pad_tokens=True)
    idxs_cells, idxs_cells_lens = tokens[0]
    mask_np = _to_numpy(mask).astype(bool)
    mask_tokens, _ = dataset.tokenizer.cell_to_token_mask(idxs_cells, idxs_cells_lens, mask_np)
    if mask_tokens is None or len(mask_tokens) == 0:
        return np.array([]), np.array([]), np.array([])

    idxs_tokens = [i for t in idxs_cells for i in t]
    idxs_data = [t for t, m in zip(idxs_tokens, mask_tokens, strict=True) if m]
    if len(idxs_data) == 0:
        return np.array([]), np.array([]), np.array([])

    idxs_data = torch.cat(idxs_data).cpu().numpy()
    idxs_data = idxs_data[idxs_data > 0] - 1  # remove padding offset
    if idxs_data.size == 0:
        return np.array([]), np.array([]), np.array([])

    coords = _to_numpy(rdata.coords)[idxs_data]
    lats = coords[:, 0]
    lons = _wrap_lons(coords[:, 1])
    vals = data[idxs_data, var_idx]
    return lats, lons, vals


def _full_points_from_source(
    dataset: MultiStreamDataSampler,
    stream_name: str,
    stream_data,
    step: int,
    var_idx: int,
    denorm: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract all data points from a source sample (no masking applied).

    Parameters
    ----------
    dataset : MultiStreamDataSampler
        The data sampler containing normalization info.
    stream_name : str
        Name of the stream.
    stream_data
        Stream data object containing source_raw.
    step : int
        Time step index within the source sequence.
    var_idx : int
        Variable index to extract.
    denorm : bool
        Whether to denormalize the values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Latitude, longitude, and value arrays for all points.
    """
    rdata = stream_data.source_raw[step]
    if rdata is None or rdata.data is None or len(rdata.data) == 0:
        return np.array([]), np.array([]), np.array([])
    data = np.asarray(rdata.data).copy()
    if denorm:
        data = dataset.denormalize_source_channels(stream_name, data)
    coords = _to_numpy(rdata.coords)
    lats = coords[:, 0]
    lons = _wrap_lons(coords[:, 1])
    vals = data[:, var_idx]
    return lats, lons, vals


def _masked_points_from_target_values(
    dataset: MultiStreamDataSampler,
    stream_name: str,
    stream_data,
    step: int,
    var_idx: int,
    denorm: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract visible data points from a target sample using target tokens.

    Uses the pre-computed target_tokens and target_coords_raw which contain
    only the visible (non-masked) target points.

    Parameters
    ----------
    dataset : MultiStreamDataSampler
        The data sampler containing normalization info.
    stream_name : str
        Name of the stream.
    stream_data
        Stream data object containing target_tokens and target_coords_raw.
    step : int
        Forecast step index.
    var_idx : int
        Variable index to extract.
    denorm : bool
        Whether to denormalize the values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Latitude, longitude, and value arrays for visible target points.
    """
    if step >= len(stream_data.target_tokens):
        return np.array([]), np.array([]), np.array([])

    values = _to_numpy(stream_data.target_tokens[step])
    coords = _to_numpy(stream_data.target_coords_raw[step])
    if values.size == 0 or coords.size == 0:
        return np.array([]), np.array([]), np.array([])

    if denorm:
        values = dataset.denormalize_target_channels(stream_name, np.array(values, copy=True))

    lats = coords[:, 0]
    lons = _wrap_lons(coords[:, 1])
    vals = values[:, var_idx]
    return lats, lons, vals


def _downsample(
    rng: np.random.Generator,
    lats: np.ndarray,
    lons: np.ndarray,
    vals: np.ndarray,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly downsample points if they exceed the maximum limit.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducible sampling.
    lats, lons, vals : np.ndarray
        Coordinate and value arrays to downsample.
    max_points : int or None
        Maximum number of points to keep. If None or if data is already
        smaller, returns the original arrays unchanged.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Downsampled latitude, longitude, and value arrays.
    """
    if max_points is None or len(vals) <= max_points:
        return lats, lons, vals
    idxs = rng.choice(len(vals), size=max_points, replace=False)
    return lats[idxs], lons[idxs], vals[idxs]


def _plot_cartopy_panels(
    panels: list[dict[str, Any]],
    title: str,
    out_path: Path,
    dpi: int,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "coolwarm",
    point_size: float = 2.0,
    shared_colorbar: bool = False,
) -> None:
    if not panels:
        raise ValueError("No panels to plot.")

    ref_values = []
    for panel in panels:
        vals_full = panel.get("vals_full", np.array([]))
        vals_plot = panel.get("vals_plot", np.array([]))
        if len(vals_full):
            ref_values.append(np.asarray(vals_full))
        elif len(vals_plot):
            ref_values.append(np.asarray(vals_plot))

    if not ref_values:
        raise ValueError("No data points to plot after masking.")

    if vmin is None or vmax is None:
        vals_ref = np.concatenate(ref_values)
        vmin = np.nanpercentile(vals_ref, 2)
        vmax = np.nanpercentile(vals_ref, 98)

    proj = ccrs.Robinson()
    ncols = len(panels)
    fig = plt.figure(figsize=(8 * ncols, 7.5), dpi=dpi)
    bg_point_size = max(1.0, point_size * 0.5)

    def _setup_axis(ax, title_text: str):
        ax.set_global()
        ax.coastlines(resolution="110m", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.set_title(title_text, fontsize=11, fontweight="bold")

    scatters = []

    for idx, panel in enumerate(panels, start=1):
        ax = fig.add_subplot(1, ncols, idx, projection=proj)
        _setup_axis(ax, panel["label"])

        lats_full = panel.get("lats_full", np.array([]))
        lons_full = panel.get("lons_full", np.array([]))
        vals_full = panel.get("vals_full", np.array([]))
        visible = panel.get("visible", np.array([], dtype=bool))

        if panel.get("show_background", True) and len(lats_full) and len(visible):
            ax.scatter(
                lons_full[~visible],
                lats_full[~visible],
                c="#d3d3d3",
                s=bg_point_size,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )

        lats_plot = panel.get("lats_plot", np.array([]))
        lons_plot = panel.get("lons_plot", np.array([]))
        vals_plot = panel.get("vals_plot", np.array([]))

        if len(vals_plot) == 0 and len(lats_full) and len(visible) and len(vals_full):
            lats_plot = lats_full[visible]
            lons_plot = lons_full[visible]
            vals_plot = vals_full[visible]

        sc = ax.scatter(
            lons_plot,
            lats_plot,
            c=vals_plot,
            s=point_size,
            cmap=cmap,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
        scatters.append(sc)

        if not shared_colorbar:
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, orientation="horizontal")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    if shared_colorbar:
        fig.subplots_adjust(top=0.92, bottom=0.18, wspace=0.06)
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
        fig.colorbar(scatters[0], cax=cbar_ax, orientation="horizontal")
    else:
        fig.subplots_adjust(top=0.96, bottom=0.08, wspace=0.06)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_cartopy_three_panel(
    lats_full: np.ndarray,
    lons_full: np.ndarray,
    vals_full: np.ndarray,
    src_visible: np.ndarray,
    tgt_visible: np.ndarray,
    lats_src: np.ndarray,
    lons_src: np.ndarray,
    vals_src: np.ndarray,
    lats_tgt: np.ndarray,
    lons_tgt: np.ndarray,
    vals_tgt: np.ndarray,
    title: str,
    src_label: str,
    tgt_label: str,
    out_path: Path,
    dpi: int,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "coolwarm",
    point_size: float = 2.0,
    shared_colorbar: bool = False,
    show_source: bool = True,
    show_target: bool = True,
) -> None:
    """Create a 2 or 3 panel visualization of masked source/target data.

    Generates a figure with Robinson projection showing:
    - Panel 1 (optional): Full unmasked data
    - Panel 2: Source/Student view with mask applied
    - Panel 3: Target/Teacher view with mask applied

    Masked-out regions are shown as light gray background points.

    Parameters
    ----------
    lats_full, lons_full, vals_full : np.ndarray
        Coordinates and values for the full (unmasked) data. If empty, the full
        panel is omitted and a 2-panel layout is used.
    src_visible, tgt_visible : np.ndarray
        Boolean arrays indicating which full-data points are visible in source/target.
    lats_src, lons_src, vals_src : np.ndarray
        Coordinates and values for the masked source view.
    lats_tgt, lons_tgt, vals_tgt : np.ndarray
        Coordinates and values for the masked target view.
    title : str
        Overall figure title.
    src_label, tgt_label : str
        Labels for source and target panels.
    out_path : Path
        Output file path for the saved figure.
    dpi : int
        Output image resolution.
    vmin, vmax : float, optional
        Color scale limits. If None, computed from 2nd-98th percentile of data.
    cmap : str
        Matplotlib colormap name.
    point_size : float
        Scatter point size.
    shared_colorbar : bool
        If True, use a single colorbar below all panels instead of per-panel colorbars.
    """
    if len(vals_full) == 0 and len(vals_src) == 0 and len(vals_tgt) == 0:
        raise ValueError("No data points to plot after masking.")
    if not show_source and not show_target:
        raise ValueError("At least one masked panel must be shown.")

    panels = []
    if len(vals_full):
        panels.append(
            {
                "label": "Full Data (No Masking)",
                "lats_plot": lats_full,
                "lons_plot": lons_full,
                "vals_plot": vals_full,
                "show_background": False,
            }
        )
    if show_source:
        panels.append(
            {
                "label": src_label,
                "lats_full": lats_full,
                "lons_full": lons_full,
                "vals_full": vals_full,
                "visible": src_visible,
                "lats_plot": lats_src,
                "lons_plot": lons_src,
                "vals_plot": vals_src,
            }
        )
    if show_target:
        panels.append(
            {
                "label": tgt_label,
                "lats_full": lats_full,
                "lons_full": lons_full,
                "vals_full": vals_full,
                "visible": tgt_visible,
                "lats_plot": lats_tgt,
                "lons_plot": lons_tgt,
                "vals_plot": vals_tgt,
            }
        )

    _plot_cartopy_panels(
        panels,
        title,
        out_path,
        dpi,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        point_size=point_size,
        shared_colorbar=shared_colorbar,
    )




def main(args=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = get_parser()
    parsed = parser.parse_args(args)

    config_paths = parsed.config
    base_config = config_paths[0]
    extra_configs = config_paths[1:]

    cli_overwrite = wg_config.from_cli_arglist(parsed.options)
    cf = _load_visualizer_config(base_config, extra_configs, parsed.private_config, cli_overwrite)

    with open_dict(cf):
        cf.rank = 0
        cf.world_size = 1
        cf.local_rank = 0
        cf.with_ddp = False
        cf.data_loading.rng_seed = parsed.seed if parsed.seed is not None else int(time.time())
        if cf.get("general") is None:
            cf.general = OmegaConf.create()
        if cf.general.get("run_id") is None:
            cf.general.run_id = "preview"

    # Always load streams from streams_directory to match training behavior.
    cf.streams = wg_config.load_streams(Path(cf.streams_directory))

    mode_cfg, stage = _resolve_mode_config(cf, parsed.mode)
    stream_view = _resolve_stream_views(cf.streams, parsed.stream, stage)
    selected_stream = stream_view["selected_stream"]
    source_stream = stream_view["source_stream"]
    target_stream = stream_view["target_stream"]
    paired_streams = stream_view["paired_streams"]

    if source_stream is None and target_stream is None:
        logger.error("No source or target stream could be resolved for visualization.")
        return 1

    active_streams = []
    active_stream_names = set()
    for stream in (source_stream, target_stream):
        if stream is not None and stream.name not in active_stream_names:
            active_streams.append(stream)
            active_stream_names.add(stream.name)
    cf.streams = active_streams

    source_stream_name = source_stream.name if source_stream is not None else None
    target_stream_name = target_stream.name if target_stream is not None else None

    logger.info(
        "Using source stream '%s' and target stream '%s' (selected '%s').",
        source_stream_name,
        target_stream_name,
        selected_stream.name,
    )
    if paired_streams and parsed.include_full:
        logger.info(
            "Paired forcing/diagnostic visualization renders two masked panels; ignoring --include-full."
        )

    dataset = MultiStreamDataSampler(cf, mode_cfg, stage=stage)
    batch = next(iter(dataset))

    if batch.len_sources() == 0:
        logger.error("No source samples in batch. Check model_input configuration.")
        return 1

    # Resolve variable indices
    source_var_name = parsed.source_variable or parsed.variable
    target_var_name = parsed.target_variable or parsed.variable

    source_var_idx = None
    if source_stream_name is not None:
        source_ds = dataset.streams_datasets[source_stream_name][0]
        source_var_idx, source_var_name = _resolve_var_idx(
            source_ds.source_channels,
            source_var_name,
            "source",
            strict=parsed.source_variable is not None,
        )

    pair_indices = parsed.pair_indices or list(range(batch.len_sources()))
    for idx in pair_indices:
        if idx >= batch.len_sources():
            raise ValueError(f"source-sample-idx {idx} out of range [0, {batch.len_sources()}).")
    if parsed.max_pairs is not None:
        pair_indices = pair_indices[: parsed.max_pairs]
    if parsed.target_sample_idx is not None and len(pair_indices) > 1:
        raise ValueError("--target-sample-idx can only be used with a single pair.")

    include_full = parsed.include_full and not paired_streams

    rng = np.random.default_rng(parsed.seed)
    pairs_data = []

    for source_idx in pair_indices:
        if parsed.target_sample_idx is None:
            target_idx = int(batch.source2target_matching_idxs[source_idx])
            if target_idx < 0 or target_idx >= batch.len_targets():
                logger.warning("Mapped target index is invalid; falling back to 0.")
                target_idx = 0
        else:
            target_idx = parsed.target_sample_idx
            if target_idx >= batch.len_targets():
                raise ValueError(
                    f"target-sample-idx {target_idx} out of range [0, {batch.len_targets()})."
                )

        source_sample = batch.get_source_sample(source_idx)
        target_sample = batch.get_target_sample(target_idx)

        source_stream_data = (
            source_sample.get_stream_data(source_stream_name) if source_stream_name is not None else None
        )
        target_stream_data = (
            target_sample.get_stream_data(target_stream_name) if target_stream_name is not None else None
        )

        if source_stream_name is not None and source_stream_data is None:
            logger.error("Source stream data missing for stream '%s'.", source_stream_name)
            return 1
        if target_stream_name is not None and target_stream_data is None:
            logger.error("Target stream data missing for stream '%s'.", target_stream_name)
            return 1

        if source_stream_name is not None and parsed.source_step >= len(source_stream_data.source_raw):
            raise ValueError(
                f"source-step {parsed.source_step} out of range [0, {len(source_stream_data.source_raw)})."
            )
        if target_stream_name is not None and parsed.target_step >= len(target_stream_data.target_tokens):
            raise ValueError(
                f"target-step {parsed.target_step} out of range [0, {len(target_stream_data.target_tokens)})."
            )

        source_meta = source_sample.meta_info[source_stream_name] if source_stream_name is not None else None
        target_meta = target_sample.meta_info[target_stream_name] if target_stream_name is not None else None

        target_view = None
        target_var_idx = None

        if target_stream_name is not None:
            target_ds = dataset.streams_datasets[target_stream_name][0]
            # Decide target view: use target values if available, else network input view.
            target_view = "target_values"
            if len(target_stream_data.target_tokens) == 0 or target_stream_data.target_tokens[
                parsed.target_step
            ].numel() == 0:
                target_view = "network_input"

            if target_view == "target_values":
                target_var_idx, target_var_name = _resolve_var_idx(
                    target_ds.target_channels,
                    target_var_name,
                    "target",
                    strict=parsed.target_variable is not None,
                )
            else:
                target_var_idx, target_var_name = _resolve_var_idx(
                    target_ds.source_channels,
                    target_var_name,
                    "target(network_input)",
                    strict=parsed.target_variable is not None,
                )
                if parsed.target_step >= len(target_stream_data.source_raw):
                    raise ValueError(
                        f"target-step {parsed.target_step} out of range for network_input "
                        f"[0, {len(target_stream_data.source_raw)})."
                    )

        lats_full = np.array([])
        lons_full = np.array([])
        vals_full = np.array([])
        lats_src = np.array([])
        lons_src = np.array([])
        vals_src = np.array([])
        lats_tgt_full = np.array([])
        lons_tgt_full = np.array([])
        vals_tgt_full = np.array([])
        lats_tgt = np.array([])
        lons_tgt = np.array([])
        vals_tgt = np.array([])
        src_visible = np.array([], dtype=bool)
        tgt_visible = np.array([], dtype=bool)

        if source_stream_name is not None:
            lats_full, lons_full, vals_full = _full_points_from_source(
                dataset,
                source_stream_name,
                source_stream_data,
                parsed.source_step,
                source_var_idx,
                parsed.denorm,
            )
            lats_full, lons_full, vals_full = _downsample(
                rng, lats_full, lons_full, vals_full, parsed.max_points
            )

            lats_src, lons_src, vals_src = _masked_points_from_source_view(
                dataset,
                source_stream,
                source_stream_name,
                source_stream_data,
                source_meta.mask,
                parsed.source_step,
                source_var_idx,
                parsed.denorm,
            )
            lats_src, lons_src, vals_src = _downsample(
                rng, lats_src, lons_src, vals_src, parsed.max_points
            )
            src_visible = _map_points_visible(
                lats_full, lons_full, source_meta.mask, cf.healpix_level
            )

        if target_stream_name is not None:
            if target_view == "target_values":
                lats_tgt, lons_tgt, vals_tgt = _masked_points_from_target_values(
                    dataset,
                    target_stream_name,
                    target_stream_data,
                    parsed.target_step,
                    target_var_idx,
                    parsed.denorm,
                )
            else:
                lats_tgt_full, lons_tgt_full, vals_tgt_full = _full_points_from_source(
                    dataset,
                    target_stream_name,
                    target_stream_data,
                    parsed.target_step,
                    target_var_idx,
                    parsed.denorm,
                )
                lats_tgt_full, lons_tgt_full, vals_tgt_full = _downsample(
                    rng, lats_tgt_full, lons_tgt_full, vals_tgt_full, parsed.max_points
                )

                lats_tgt, lons_tgt, vals_tgt = _masked_points_from_source_view(
                    dataset,
                    target_stream,
                    target_stream_name,
                    target_stream_data,
                    target_meta.mask,
                    parsed.target_step,
                    target_var_idx,
                    parsed.denorm,
                )

            lats_tgt, lons_tgt, vals_tgt = _downsample(
                rng, lats_tgt, lons_tgt, vals_tgt, parsed.max_points
            )

            if source_stream_name == target_stream_name and len(lats_full):
                tgt_visible = _map_points_visible(
                    lats_full, lons_full, target_meta.mask, cf.healpix_level
                )
            elif len(lats_tgt_full):
                tgt_visible = _map_points_visible(
                    lats_tgt_full, lons_tgt_full, target_meta.mask, cf.healpix_level
                )

        src_label = (
            f"Source {source_stream_name} ({source_var_name})\n"
            f"{_format_mask_params(_to_dict(source_meta.params))}\n"
            f"points={len(vals_src)}"
            if source_stream_name is not None
            else ""
        )
        tgt_label = (
            f"Target {target_stream_name} ({target_var_name}, {target_view})\n"
            f"{_format_mask_params(_to_dict(target_meta.params))}\n"
            f"points={len(vals_tgt)}"
            if target_stream_name is not None
            else ""
        )

        pairs_data.append(
            {
                "pair_idx": source_idx,
                "source_idx": source_idx,
                "target_idx": target_idx,
                "source_stream_name": source_stream_name,
                "target_stream_name": target_stream_name,
                "source_var_name": source_var_name,
                "target_var_name": target_var_name,
                "source_params": _to_dict(source_meta.params) if source_meta is not None else {},
                "target_params": _to_dict(target_meta.params) if target_meta is not None else {},
                "target_view": target_view,
                "lats_full": lats_full,
                "lons_full": lons_full,
                "vals_full": vals_full,
                "lats_tgt_full": lats_tgt_full,
                "lons_tgt_full": lons_tgt_full,
                "vals_tgt_full": vals_tgt_full,
                "src_visible": src_visible,
                "tgt_visible": tgt_visible,
                "lats_src": lats_src,
                "lons_src": lons_src,
                "vals_src": vals_src,
                "lats_tgt": lats_tgt,
                "lons_tgt": lons_tgt,
                "vals_tgt": vals_tgt,
                "src_label": src_label,
                "tgt_label": tgt_label,
                "show_source": source_stream_name is not None,
                "show_target": target_stream_name is not None,
            }
        )

    if source_stream_name is not None:
        base_idx = batch.get_source_sample(pair_indices[0]).get_stream_data(source_stream_name).sample_idx
    else:
        first_target_idx = pairs_data[0]["target_idx"]
        base_idx = batch.get_target_sample(first_target_idx).get_stream_data(target_stream_name).sample_idx
    time_win = dataset.time_window_handler.window(base_idx)

    if source_stream_name is not None and target_stream_name is not None:
        stream_label = (
            source_stream_name
            if source_stream_name == target_stream_name
            else f"{source_stream_name}_to_{target_stream_name}"
        )
    else:
        stream_label = source_stream_name or target_stream_name

    title = (
        f"Masked data preview | stream={stream_label} | mode={parsed.mode} | "
        f"time={time_win.start}..{time_win.end}"
    )

    outputs = []
    for p in pairs_data:
        if parsed.output is not None:
            base = parsed.output
            stem = base.stem
            suffix = base.suffix or ".png"
            out_path = base.with_name(f"{stem}_pair{p['pair_idx']}{suffix}")
        else:
            if p["show_source"] and p["show_target"]:
                view_name = (
                    f"{_safe_name(p['source_var_name'])}_vs_{_safe_name(p['target_var_name'])}"
                )
            elif p["show_source"]:
                view_name = f"source_{_safe_name(p['source_var_name'])}"
            else:
                view_name = f"target_{_safe_name(p['target_var_name'])}"

            if p["source_stream_name"] and p["target_stream_name"]:
                stream_name_tag = (
                    p["source_stream_name"]
                    if p["source_stream_name"] == p["target_stream_name"]
                    else f"{p['source_stream_name']}_to_{p['target_stream_name']}"
                )
            else:
                stream_name_tag = p["source_stream_name"] or p["target_stream_name"]
            out_name = f"{parsed.prefix}_{stream_name_tag}_pair{p['pair_idx']}_{view_name}.png"
            out_path = parsed.output_dir / out_name

        if paired_streams:
            panels = []
            if p["show_source"]:
                panels.append(
                    {
                        "label": p["src_label"],
                        "lats_full": p["lats_full"],
                        "lons_full": p["lons_full"],
                        "vals_full": p["vals_full"],
                        "visible": p["src_visible"],
                        "lats_plot": p["lats_src"],
                        "lons_plot": p["lons_src"],
                        "vals_plot": p["vals_src"],
                    }
                )
            if p["show_target"]:
                panels.append(
                    {
                        "label": p["tgt_label"],
                        "lats_full": p["lats_tgt_full"],
                        "lons_full": p["lons_tgt_full"],
                        "vals_full": p["vals_tgt_full"],
                        "visible": p["tgt_visible"],
                        "lats_plot": p["lats_tgt"],
                        "lons_plot": p["lons_tgt"],
                        "vals_plot": p["vals_tgt"],
                    }
                )
            _plot_cartopy_panels(
                panels,
                title,
                out_path,
                parsed.dpi,
                cmap=parsed.cmap,
                point_size=parsed.point_size,
                shared_colorbar=parsed.shared_colorbar,
            )
        elif include_full:
            _plot_cartopy_three_panel(
                p["lats_full"],
                p["lons_full"],
                p["vals_full"],
                p["src_visible"],
                p["tgt_visible"],
                p["lats_src"],
                p["lons_src"],
                p["vals_src"],
                p["lats_tgt"],
                p["lons_tgt"],
                p["vals_tgt"],
                title,
                p["src_label"],
                p["tgt_label"],
                out_path,
                parsed.dpi,
                cmap=parsed.cmap,
                point_size=parsed.point_size,
                shared_colorbar=parsed.shared_colorbar,
                show_source=p["show_source"],
                show_target=p["show_target"],
            )
        else:
            # Reuse three-panel plotter but feed empty full panel data to skip it visually.
            _plot_cartopy_three_panel(
                np.array([]),
                np.array([]),
                np.array([]),
                p["src_visible"],
                p["tgt_visible"],
                p["lats_src"],
                p["lons_src"],
                p["vals_src"],
                p["lats_tgt"],
                p["lons_tgt"],
                p["vals_tgt"],
                title,
                p["src_label"],
                p["tgt_label"],
                out_path,
                parsed.dpi,
                cmap=parsed.cmap,
                point_size=parsed.point_size,
                shared_colorbar=parsed.shared_colorbar,
                show_source=p["show_source"],
                show_target=p["show_target"],
            )
        outputs.append(out_path)

    meta = {
        "config": [str(p) for p in config_paths],
        "mode": parsed.mode,
        "stream": stream_label,
        "selected_stream": selected_stream.name,
        "source_stream": source_stream_name,
        "target_stream": target_stream_name,
        "paired_streams": paired_streams,
        "include_full": include_full,
        "time_window": {"start": str(time_win.start), "end": str(time_win.end)},
        "plot_style": "cartopy",
        "pairs": [
            {
                "source": (
                    {
                        "sample_idx": p["source_idx"],
                        "stream": p["source_stream_name"],
                        "step": parsed.source_step,
                        "variable": p["source_var_name"],
                        "mask_params": p["source_params"],
                        "points": int(len(p["vals_src"])),
                    }
                    if p["show_source"]
                    else None
                ),
                "target": (
                    {
                        "sample_idx": p["target_idx"],
                        "stream": p["target_stream_name"],
                        "step": parsed.target_step,
                        "variable": p["target_var_name"],
                        "view": p["target_view"],
                        "mask_params": p["target_params"],
                        "points": int(len(p["vals_tgt"])),
                    }
                    if p["show_target"]
                    else None
                ),
            }
            for p in pairs_data
        ],
        "outputs": [str(p) for p in outputs],
    }

    meta_path = outputs[0].with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(meta), f, indent=2)

    logger.info("Saved plot(s): %s", ", ".join(map(str, outputs)))
    logger.info("Saved metadata: %s", meta_path)
    return 0


def _to_serializable(obj):
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _to_dict(params) -> dict:
    if params is None:
        return {}
    serial = _to_serializable(params)
    return serial if isinstance(serial, dict) else dict(serial)


if __name__ == "__main__":
    raise SystemExit(main())
