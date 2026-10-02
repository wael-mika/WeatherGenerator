# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CPU tests for the cell-artifact decomposition.

The load-bearing test is `test_smooth_bias_is_not_reported_as_tiling`: without it this metric
would flag every large-scale bias as a decode artifact, which is the exact false positive that
made a real run's z_850 look like catastrophic tiling when it had none.
"""

import numpy as np

from weathergen.evaluate.scores.cell_artifacts import cell_error_decomposition


def _grid(n_side: int = 12, per_cell: int = 40, seed: int = 0):
    """A toy tessellation: n_side^2 square cells on a lat/lon patch, `per_cell` points in each."""
    rng = np.random.default_rng(seed)
    lat, lon, cell = [], [], []
    for i in range(n_side):
        for j in range(n_side):
            cid = i * n_side + j
            lat.append(45.0 + i + rng.uniform(0, 1, per_cell))
            lon.append(5.0 + j + rng.uniform(0, 1, per_cell))
            cell.append(np.full(per_cell, cid))
    return (
        np.concatenate(lat),
        np.concatenate(lon),
        np.concatenate(cell).astype(int),
        rng,
    )


def test_white_noise_error_is_clean():
    """No cell structure at all -> the share must sit near the sampling floor, verdict clean."""
    lat, lon, cell, rng = _grid()
    truth = rng.normal(size=lat.size)
    # iid error, no cell structure at all
    pred = truth + rng.normal(scale=1.0, size=lat.size)
    r = cell_error_decomposition(pred, truth, cell, lat, lon)
    assert r["verdict"] == "clean", r
    # with 40 points per cell the floor is ~1/40; allow generous headroom but not 0.15
    assert r["cell_dc_share"] < 0.15, r["cell_dc_share"]
    assert r["cell_dc_var_debiased"] < r["cell_dc_var"], "the floor must be subtracted"


def test_per_cell_offsets_are_reported_as_tiling():
    """A constant offset per cell, uncorrelated between cells -- the target artifact."""
    lat, lon, cell, rng = _grid()
    truth = rng.normal(size=lat.size)
    offsets = rng.normal(scale=3.0, size=cell.max() + 1)  # independent per cell
    pred = truth + offsets[cell] + rng.normal(scale=0.2, size=lat.size)
    r = cell_error_decomposition(pred, truth, cell, lat, lon)
    assert r["verdict"] == "tiling", r
    assert r["cell_dc_share"] > 0.9, r["cell_dc_share"]
    # independent cell means -> E[(a-b)^2] = 2*var -> roughness ~ 1
    assert 0.6 < r["cell_dc_roughness"] < 1.5, r["cell_dc_roughness"]


def test_smooth_bias_is_not_reported_as_tiling():
    """*** THE ONE THAT MATTERS. ***

    A smooth large-scale bias also puts nearly all the error variance BETWEEN cells, so `dc_share`
    is high and, read alone, screams "tiling". It is not tiling -- adjacent cells agree closely.
    A real run hit exactly this: z_850 at dc_share 0.998 with a FALLING boundary discontinuity.
    """
    lat, lon, cell, rng = _grid()
    truth = rng.normal(size=lat.size)
    pred = truth + 0.6 * (lat - lat.mean())  # smooth ramp, no discontinuity anywhere
    r = cell_error_decomposition(pred, truth, cell, lat, lon)
    assert r["cell_dc_share"] > 0.9, "a smooth ramp is still almost entirely between-cell"
    assert r["verdict"] == "smooth_bias", r
    assert r["cell_dc_roughness"] < 0.5, r["cell_dc_roughness"]


def test_roughness_separates_the_two_at_equal_dc_share():
    """Both cases can have ~the same dc_share; only roughness tells them apart."""
    lat, lon, cell, rng = _grid()
    truth = rng.normal(size=lat.size)
    tiled = truth + rng.normal(scale=3.0, size=cell.max() + 1)[cell]
    smooth = truth + 0.6 * (lat - lat.mean())
    a = cell_error_decomposition(tiled, truth, cell, lat, lon)
    b = cell_error_decomposition(smooth, truth, cell, lat, lon)
    assert a["cell_dc_share"] > 0.9 and b["cell_dc_share"] > 0.9, (a, b)
    assert a["cell_dc_roughness"] > 3 * b["cell_dc_roughness"], (
        a["cell_dc_roughness"],
        b["cell_dc_roughness"],
    )


def test_without_coordinates_the_verdict_refuses_to_classify():
    """No lat/lon -> tiling and smooth bias are indistinguishable, and it must SAY so."""
    lat, lon, cell, rng = _grid()
    truth = rng.normal(size=lat.size)
    pred = truth + rng.normal(scale=3.0, size=cell.max() + 1)[cell]
    r = cell_error_decomposition(pred, truth, cell)
    assert r["verdict"] == "dc_high_unclassified", r
    assert np.isnan(r["cell_dc_roughness"])


def test_noise_floor_grows_as_points_per_cell_falls():
    """The floor is density-dependent -- why raw shares do not compare across densities."""
    dense = _grid(per_cell=200, seed=1)
    sparse = _grid(per_cell=10, seed=1)
    out = []
    for lat, lon, cell, rng in (dense, sparse):
        truth = rng.normal(size=lat.size)
        pred = truth + rng.normal(size=lat.size)  # pure noise in both cases
        out.append(cell_error_decomposition(pred, truth, cell, lat, lon, min_points_per_cell=5))
    dense_r, sparse_r = out
    assert sparse_r["cell_dc_noise_floor_var"] > dense_r["cell_dc_noise_floor_var"], (
        "a sparser decode must have a LARGER guaranteed-noise floor"
    )
    assert sparse_r["cell_dc_share"] > dense_r["cell_dc_share"], (
        "the raw share inflates as density falls -- the reason to report mean_points_per_cell"
    )


def test_sparse_cells_are_dropped_not_averaged_in():
    """Boundary cells hold few points and the noisiest means; they must not dominate."""
    lat, lon, cell, rng = _grid(per_cell=40)
    # bolt on a handful of 2-point cells with wild errors
    extra_cell = cell.max() + 1 + np.repeat(np.arange(20), 2)
    extra_lat = 45.0 + rng.uniform(0, 12, extra_cell.size)
    extra_lon = 5.0 + rng.uniform(0, 12, extra_cell.size)
    truth = rng.normal(size=lat.size)
    pred = truth + rng.normal(scale=0.1, size=lat.size)
    r_clean = cell_error_decomposition(pred, truth, cell, lat, lon, min_points_per_cell=8)
    r_with = cell_error_decomposition(
        np.concatenate([pred, rng.normal(scale=50.0, size=extra_cell.size)]),
        np.concatenate([truth, np.zeros(extra_cell.size)]),
        np.concatenate([cell, extra_cell]),
        np.concatenate([lat, extra_lat]),
        np.concatenate([lon, extra_lon]),
        min_points_per_cell=8,
    )
    assert r_with["n_cells"] == r_clean["n_cells"], "2-point cells must be excluded"


def test_mismatched_shapes_fail_loudly():
    lat, lon, cell, rng = _grid(n_side=4, per_cell=10)
    try:
        cell_error_decomposition(np.zeros(5), np.zeros(5), cell)
    except ValueError:
        return
    raise AssertionError("mismatched shapes must raise, not broadcast silently")
