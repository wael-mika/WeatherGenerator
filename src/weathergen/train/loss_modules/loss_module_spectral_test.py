# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the WFCL loss module's config guards and coordinate recovery."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.train.loss_modules.loss_module_spectral import LossSpectralWFCL

CERRA_ZARR = "/e/data1/slmet/ml_training/cerra-rr-an-oper-se-al-ec-mars-5p5km-1985-2023-3h-v2.zarr"

BASE = {
    "target_stream": "CERRA",
    "grid_width": 1069,
    "grid_height": 1069,
    "patch_size": 64,
    "coords_zarr": CERRA_ZARR,
    "total_steps": 4096,
}


def _make(**overrides):
    cfg = OmegaConf.create({**BASE, **overrides})
    return LossSpectralWFCL(
        OmegaConf.create({"general": {"istep": 0}, "streams": {}}),
        OmegaConf.create({}),
        "train",
        "cpu",
        wfcl=cfg,
    )


def test_total_steps_must_be_explicit():
    """An inflated horizon pins P_t near 1 and the amplitude phase never arrives --
    a silent failure, so refuse to guess."""
    cfg = {k: v for k, v in BASE.items() if k != "total_steps"}
    with pytest.raises(ValueError, match="explicit `total_steps`"):
        LossSpectralWFCL(
            OmegaConf.create({"general": {"istep": 0}}),
            OmegaConf.create({}),
            "train",
            "cpu",
            wfcl=OmegaConf.create(cfg),
        )


def test_patch_size_must_suit_the_wavelet_depth():
    with pytest.raises(ValueError, match="edge-replicate"):
        _make(patch_size=100, J=3)


def test_gamma_length_is_checked():
    with pytest.raises(ValueError, match="gamma has"):
        _make(gamma=[1.0, 1.0])


def test_facl_mode_is_validated():
    with pytest.raises(ValueError, match="facl_mode"):
        _make(facl_mode="both")


@pytest.mark.skipif(not __import__("os").path.isdir(CERRA_ZARR), reason="CERRA zarr unavailable")
def test_coord_lookup_recovers_native_indices_on_the_real_grid():
    """The load-bearing assumption of the whole patch scheme: a target point's
    lat/lon identifies its cell in the source raster exactly."""
    import zarr

    z = zarr.open(CERRA_ZARR, mode="r")
    lat = np.asarray(z["latitudes"][:])
    lon = np.asarray(z["longitudes"][:])

    # *** Query with what the READER delivers, not the raw zarr arrays. ***
    # data_reader_anemoi applies _clip_lat/_clip_lon, and _clip_lon re-bases longitude
    # to [-180,180) AND casts to float32. An earlier version of this test used the raw
    # float64 columns, matched 100%, and sailed past the bug that starved runs
    # hlhzsm6b and ztpf511o at a 99.7% match rate.
    from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon

    reader_lat, reader_lon = _clip_lat(lat), _clip_lon(lon)
    assert reader_lon.dtype == np.float32
    assert reader_lon.min() < 0.0, "expected the reader to re-base longitude to [-180,180)"

    mod = _make()
    want = np.random.default_rng(0).choice(lat.size, 20_000, replace=False)
    coords = torch.from_numpy(np.stack([reader_lat[want], reader_lon[want]], axis=-1))
    got = mod._native_idx(coords)

    assert (got >= 0).all(), f"{int((got < 0).sum())} of 20000 points failed to match"
    assert torch.equal(got, torch.from_numpy(want.astype("int64")))


@pytest.mark.skipif(not __import__("os").path.isdir(CERRA_ZARR), reason="CERRA zarr unavailable")
def test_raw_zarr_coords_would_have_starved_the_loss():
    """Pins the regression: keying the table off raw float64 coords really does miss
    enough points to leave no complete patches, so the _clip_* reuse is load-bearing."""
    import zarr

    from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon

    z = zarr.open(CERRA_ZARR, mode="r")
    lat, lon = np.asarray(z["latitudes"][:]), np.asarray(z["longitudes"][:])

    raw_keys = LossSpectralWFCL._coord_key(torch.from_numpy(lat), torch.from_numpy(lon))
    reader_keys = LossSpectralWFCL._coord_key(
        torch.from_numpy(_clip_lat(lat)), torch.from_numpy(_clip_lon(lon))
    )
    disagree = int((raw_keys != reader_keys).sum())
    frac = disagree / lat.size
    assert disagree > 0, "raw and reader coords agree -- the guard below is meaningless"
    # a 64x64 patch needs all 4096 points
    survival = (1.0 - frac) ** 4096
    assert survival < 0.01, f"expected patch survival to collapse, got {survival}"


@pytest.mark.skipif(not __import__("os").path.isdir(CERRA_ZARR), reason="CERRA zarr unavailable")
def test_coord_keys_are_collision_free_over_the_whole_grid():
    import zarr

    z = zarr.open(CERRA_ZARR, mode="r")
    keys = LossSpectralWFCL._coord_key(
        torch.from_numpy(np.asarray(z["latitudes"][:])),
        torch.from_numpy(np.asarray(z["longitudes"][:])),
    )
    assert torch.unique(keys).numel() == keys.numel(), "coordinate quantisation collides"


def test_wfcl_scalar_is_scheduled_between_phase_and_amplitude():
    """At P_t=1 the loss is pure correlation; at P_t=0 pure amplitude. A field that
    is correctly placed but under-amplified must therefore be cheap early and
    expensive late -- that ordering is the entire mechanism."""
    mod = _make()
    torch.manual_seed(0)
    target = torch.rand(4, 2, 64, 64)
    pred = target * 0.3  # right phase everywhere, badly wrong amplitude

    loss_phase, logs_p = mod._wfcl(pred, target, p_t=1.0)
    loss_amp, logs_a = mod._wfcl(pred, target, p_t=0.0)

    assert loss_phase < 1e-4, f"correlation term should be ~0 for a scaled copy: {loss_phase}"
    assert loss_amp > loss_phase * 100
    assert logs_p["FCL"] < 1e-5
    assert logs_a["FAL"] > 0


def test_per_orientation_logs_are_emitted():
    mod = _make()
    _loss, logs = mod._wfcl(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64), p_t=0.5)
    for level in (1, 2, 3):
        assert f"WAL_L{level}" in logs and f"WCL_L{level}" in logs
        for k in range(1, 7):
            assert f"WCL_L{level}_k{k}" in logs


def test_fourier_and_wavelet_halves_can_be_ablated():
    t, p = torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64)
    fourier_only = _make(use_wavelet=False)._wfcl(p, t, 0.5)[1]
    wavelet_only = _make(use_fourier=False)._wfcl(p, t, 0.5)[1]
    assert "FACL" in fourier_only and "WACL" not in fourier_only
    assert "WACL" in wavelet_only and "FACL" not in wavelet_only


def test_schedule_start_step_spans_the_finetune_window():
    """On a continuation the absolute istep is already large. Without an offset the
    ramp is over before the finetune starts; with it, P_t begins at 1 again."""
    from weathergen.train.loss_modules.spectral_utils import phase_weight

    parent_end, run_end = 1280, 1600
    naive = phase_weight(parent_end, run_end, 0.1)
    offset = phase_weight(parent_end - parent_end, run_end - parent_end, 0.1)
    assert naive < 0.15, "the trap: schedule nearly exhausted at continuation start"
    assert offset == 1.0

    mod = _make(total_steps=run_end, schedule_start_step=parent_end)
    assert mod.schedule_start_step == parent_end


def test_schedule_start_step_must_precede_total_steps():
    with pytest.raises(ValueError, match="schedule would be empty"):
        _make(total_steps=1000, schedule_start_step=1000)
