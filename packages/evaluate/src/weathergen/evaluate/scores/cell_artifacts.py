# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Detect decode-cell artifacts: is the error a per-cell offset (the "diamond" tiling)?

**What this measures.** A point decoder gives every target point inside one HEALPix cell the same
few latent vectors, so its natural failure mode is an error that is roughly CONSTANT WITHIN a cell
and jumps BETWEEN cells. Rendered as a map that is the familiar tiling / diamond artifact. This
module decomposes the error variance along the model's own tessellation and reports how much of it
is that per-cell offset.

**The one thing that makes this trustworthy: `dc` ALONE CANNOT TELL A TILING ARTIFACT FROM A
SMOOTH LARGE-SCALE BIAS.** Any error that varies slowly across the domain -- a warm bias in the
north, a seasonal drift -- also varies between cells and lands in the same number. Observed in
practice: a run whose z_850 `dc_share` reached 0.998 turned out to have a smoothly-varying bias,
not tile edges, and its boundary discontinuity was simultaneously FALLING. Reading `dc` alone
would have reported a catastrophic artifact that did not exist.

`cell_dc_roughness` below is the discriminator, and it is why this module returns a verdict rather
than a score:

    dc high + roughness high  -> TRUE TILING   (adjacent cells disagree; hard edges)
    dc high + roughness low   -> SMOOTH BIAS   (cell means vary slowly; a different bug)
    dc low                    -> CLEAN         (whatever the error is, it is not cell-shaped)

**Three further properties you must respect when using these numbers:**

1. **The share moves when the denominator moves.** `cell_dc_share` rose 0.957 -> 0.992 on a run
   whose ABSOLUTE per-cell variance FELL 29% -- everything else simply improved faster. Compare
   arms on `cell_dc_var`, read the share only as a within-run diagnostic.
2. **There is a sampling-noise floor that depends on decode density.** Even a perfect model with a
   purely white residual produces a non-zero between-cell variance, because a cell mean over n
   points has sampling variance sigma^2/n. The floor is ~ `within_var * (n_cells - 1) / n_points`,
   so it GROWS as points-per-cell falls. It is estimated and subtracted here
   (`cell_dc_var_debiased`), and `mean_points_per_cell` is returned so cross-density comparisons
   are at least visible. Do not compare raw shares across different decode densities.
3. **The cell labels must be the model's own tessellation.** Scoring a level-6 model on a level-5
   grid measures a blurred version of the real thing -- measured: the same tiled field scored 1.00
   with the correct labels and 0.26 through a mismatched HEALPix level. The caller supplies
   `cell_ids`; nothing here guesses them.

The between-cell term is POPULATION-WEIGHTED (the law-of-total-variance form), not a plain
variance over the cell-mean array. On a regional domain the boundary cells are partially covered
and hold very few points, so their means are the noisiest; an unweighted variance would weight
exactly those the most.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

_logger = logging.getLogger(__name__)

__all__ = ["cell_error_decomposition", "healpix_cell_ids"]


def healpix_cell_ids(lat_deg, lon_deg, healpix_level: int, nest: bool = True):
    """Map lat/lon (degrees) to HEALPix cell indices at ``healpix_level``.

    Imported lazily: ``astropy_healpix`` is NOT a declared dependency of weathergen-evaluate, so a
    hard import would break every other score for users who do not need this one. If you rely on
    this path, add ``astropy-healpix`` to packages/evaluate/pyproject.toml.

    You can avoid it entirely by passing ``cell_ids`` to :func:`cell_error_decomposition`
    directly, which is preferable when the run's output already carries a cell index -- that is
    the model's OWN assignment and cannot disagree with it.
    """
    try:
        from astropy import units as u
        from astropy_healpix import HEALPix
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "cell-artifact scores need HEALPix cell ids. Either pass `cell_ids` explicitly, or "
            "install astropy-healpix (not currently a weathergen-evaluate dependency)."
        ) from exc

    hp = HEALPix(nside=2**healpix_level, order="nested" if nest else "ring")
    return np.asarray(
        hp.lonlat_to_healpix(np.asarray(lon_deg) * u.deg, np.asarray(lat_deg) * u.deg)
    )


def _unit_vectors(lat_deg, lon_deg):
    """lat/lon degrees -> unit vectors, so cell centroids average correctly across the dateline."""
    lat, lon = np.radians(np.asarray(lat_deg)), np.radians(np.asarray(lon_deg))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _cell_roughness(cell_mean, cell_vec, between_var, k_neighbours: int = 6) -> float:
    """How much do ADJACENT cells disagree, relative to the overall spread of cell means?

    This is the tiling-vs-bias discriminator. For each cell take its ``k_neighbours`` nearest
    cells (by centroid) and average ``(mean_a - mean_b)^2``. If the cell means were spatially
    uncorrelated -- a true tiling artifact -- that expectation is ``2 * between_var``, so the
    normalised value is ~1. If they vary smoothly, neighbours are similar and it tends to 0.

    Returns NaN when there are too few cells to have neighbours.
    """
    n = len(cell_mean)
    if n < k_neighbours + 2 or not np.isfinite(between_var) or between_var <= 0.0:
        return float("nan")
    tree = cKDTree(cell_vec)
    # k+1 because the first hit is the cell itself
    _, idx = tree.query(cell_vec, k=min(k_neighbours + 1, n))
    nb = idx[:, 1:]
    diffs = (cell_mean[:, None] - cell_mean[nb]) ** 2
    return float(np.nanmean(diffs) / (2.0 * between_var))


def cell_error_decomposition(
    prediction,
    ground_truth,
    cell_ids,
    lat_deg=None,
    lon_deg=None,
    min_points_per_cell: int = 8,
    k_neighbours: int = 6,
    roughness_tiling_threshold: float = 0.5,
    dc_share_threshold: float = 0.15,
) -> dict:
    """Split the error variance into a per-cell offset and the rest, and classify the artifact.

    Parameters
    ----------
    prediction, ground_truth : array-like, same shape, flattened over points.
    cell_ids : array-like of int, one decode-cell index per point. Must be the tessellation the
        MODEL used -- see the module docstring.
    lat_deg, lon_deg : optional, needed only for ``cell_dc_roughness`` (cell centroids). Without
        them the decomposition is still returned and roughness is NaN -- which means the
        tiling-vs-bias question CANNOT be answered, so the verdict degrades to
        ``dc_high_unclassified``.
    min_points_per_cell : cells with fewer points are dropped. Their means are dominated by
        sampling noise and they are usually partially-covered domain-boundary cells.

    Returns
    -------
    dict with absolute variances, shares, the estimated noise floor, the debiased values,
    ``cell_dc_roughness``, ``mean_points_per_cell`` and a ``verdict`` string.
    """
    p = np.asarray(prediction, dtype=np.float64).ravel()
    t = np.asarray(ground_truth, dtype=np.float64).ravel()
    c = np.asarray(cell_ids).ravel()
    if not (p.shape == t.shape == c.shape):
        raise ValueError(
            f"prediction {p.shape}, ground_truth {t.shape} and cell_ids {c.shape} must match"
        )

    resid = p - t
    good = np.isfinite(resid)
    if good.sum() < 2:
        return {"verdict": "insufficient_data"}
    resid, c = resid[good], c[good]

    # dense-pack the cell labels so bincount stays small even for sparse global tessellations
    _, inv = np.unique(c, return_inverse=True)
    counts = np.bincount(inv)

    keep = counts >= min_points_per_cell
    if keep.sum() < 2:
        return {"verdict": "insufficient_cells", "n_cells": int(keep.sum())}

    # restrict to well-populated cells, then recompute the point-level quantities on that subset
    keep_pt = keep[inv]
    resid_k, inv_k = resid[keep_pt], inv[keep_pt]
    _, inv_k = np.unique(inv_k, return_inverse=True)
    counts_k = np.bincount(inv_k)
    cell_mean = np.bincount(inv_k, weights=resid_k) / counts_k
    n_pts, n_cells = int(counts_k.sum()), int(len(counts_k))

    total_var = float(np.var(resid_k))
    grand = float(np.mean(resid_k))
    w = counts_k / n_pts
    between_var = float(np.sum(w * (cell_mean - grand) ** 2))  # law of total variance
    within_var = max(total_var - between_var, 0.0)

    # A cell mean over n points has sampling variance within/n even with NO real cell structure,
    # so a fraction of `between_var` is guaranteed noise. This floor GROWS as density falls.
    noise_floor = float(within_var * (n_cells - 1) / n_pts) if n_pts > n_cells else float("nan")
    between_debiased = (
        float(max(between_var - noise_floor, 0.0)) if np.isfinite(noise_floor) else float("nan")
    )

    share = (lambda x: float(x / total_var)) if total_var > 0 else (lambda x: float("nan"))

    roughness = float("nan")
    if lat_deg is not None and lon_deg is not None:
        vecs = _unit_vectors(np.asarray(lat_deg).ravel()[good], np.asarray(lon_deg).ravel()[good])
        vk = vecs[keep_pt]
        centroid = np.stack(
            [np.bincount(inv_k, weights=vk[:, d]) / counts_k for d in range(3)], axis=-1
        )
        norm = np.linalg.norm(centroid, axis=1, keepdims=True)
        centroid = centroid / np.maximum(norm, 1e-12)
        roughness = _cell_roughness(cell_mean, centroid, between_var, k_neighbours)

    dc_share = share(between_var)
    if not np.isfinite(dc_share) or dc_share < dc_share_threshold:
        verdict = "clean"
    elif not np.isfinite(roughness):
        verdict = "dc_high_unclassified"  # no coordinates -> cannot separate tiling from bias
    elif roughness >= roughness_tiling_threshold:
        verdict = "tiling"
    else:
        verdict = "smooth_bias"

    return {
        "resid_var": total_var,
        "cell_dc_var": between_var,
        "cell_within_var": within_var,
        "cell_dc_share": dc_share,
        "cell_dc_noise_floor_var": noise_floor,
        "cell_dc_var_debiased": between_debiased,
        "cell_dc_share_debiased": (
            share(between_debiased) if np.isfinite(between_debiased) else float("nan")
        ),
        "cell_dc_roughness": roughness,
        "n_cells": n_cells,
        "n_points": n_pts,
        "mean_points_per_cell": float(n_pts / n_cells),
        "verdict": verdict,
    }
