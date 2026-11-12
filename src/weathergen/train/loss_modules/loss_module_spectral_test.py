# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf

from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon
from weathergen.train.loss_modules.loss_module_spectral import LossSpectralAMSE, LossSpectralWFCL

STREAM = "IMERG_ANEMOI"


@pytest.fixture(scope="module")
def grid_zarr(tmp_path_factory):
    """A small reduced Gaussian grid stored like the anemoi N320 stores."""
    nlat = 48
    mu, _ = np.polynomial.legendre.leggauss(nlat)
    row_lat = np.degrees(np.arcsin(mu[::-1]))
    counts = [max(40, 2 * int(np.ceil(nlat * np.cos(np.radians(la))))) for la in row_lat]
    lat = np.concatenate([np.full(n, la) for la, n in zip(row_lat, counts, strict=True)])
    lon = np.concatenate([np.arange(n) * 360.0 / n for n in counts])
    path = tmp_path_factory.mktemp("grid") / "small_reduced_gaussian.zarr"
    g = zarr.open_group(str(path), mode="w")
    g.create_array("latitudes", data=lat)
    g.create_array("longitudes", data=lon)
    return str(path), lat, lon


def _batch(target, pred, lat, lon, perm, n_steps=2):
    """preds / targets / metadata in the layout LossCalculator hands to a loss term."""
    coords = torch.from_numpy(np.stack([_clip_lat(lat), _clip_lon(lon)], 1))[perm]
    phys_p, phys_t = [], []
    for _ in range(n_steps):
        phys_p.append({STREAM: [pred[:, perm].reshape(pred.shape[0], -1)]})
        phys_t.append(
            {
                STREAM: {
                    "target": [target[perm]],
                    "target_coords": [coords],
                    "target_times": [None],
                    "target_metda_data": [{STREAM: SimpleNamespace(global_params={"idx": 0})}],
                    "is_spoof": [False],
                }
            }
        )
    meta = (None, [SimpleNamespace(global_params={"correspondence": 0})], None, None)
    return SimpleNamespace(physical=phys_p), SimpleNamespace(physical=phys_t), meta


def _fields(lat, lon, seed=0):
    rng = np.random.default_rng(seed)
    la, lo = np.radians(lat), np.radians(lon)
    x = np.cos(la) ** 2 * (np.cos(3 * lo) + 0.5 * np.sin(5 * lo)) + 0.3 * np.sin(2 * la)
    x = x + 0.05 * rng.normal(size=x.shape)
    return torch.from_numpy(x).float()[:, None]


def _cf(istep=0):
    return OmegaConf.create({"general": {"istep": istep}})


def test_amse_module(grid_zarr):
    path, lat, lon = grid_zarr
    mod = LossSpectralAMSE(
        _cf(), None, "train", "cpu", amse={"target_stream": STREAM, "grid_zarr": path, "lmax": 23}
    )
    target = _fields(lat, lon)
    perm = torch.randperm(len(lat))
    p, t, m = _batch(target, target[None].clone(), lat, lon, perm)
    lv = mod.compute_loss(p, t, m, istep=0)
    assert float(lv.loss) < 1e-5
    assert set(lv.losses_all[STREAM]["amse"]) >= {
        "amse",
        "spectral_mse",
        "power_ratio",
        "num_fields",
    }

    pred = (0.3 * target + 0.1 * _fields(lat, lon, seed=1))[None].requires_grad_(True)
    t_nan = target.clone()
    t_nan[:5] = torch.nan  # polar-cap style NaNs are tolerated
    p, t, m = _batch(t_nan, pred, lat, lon, perm)
    lv = mod.compute_loss(p, t, m, istep=0)
    assert float(lv.loss) > 0.0
    lv.loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0

    p, t, m = _batch(target, target[None], lat, lon, perm[:-10])  # a partial field must be loud
    with pytest.raises(RuntimeError, match="max_num_targets=-1"):
        mod.compute_loss(p, t, m, istep=0)


def test_wfcl_module(grid_zarr):
    path, lat, lon = grid_zarr
    cfg = {
        "target_stream": STREAM,
        "grid_zarr": path,
        "patch_size": 8,
        "J": 2,
        "lat_max": 60.0,
        "gamma": [1.0, 1.0],
        "total_steps": 100,
    }
    mod = LossSpectralWFCL(_cf(istep=10), None, "train", "cpu", wfcl=cfg)
    target = _fields(lat, lon)
    perm = torch.randperm(len(lat))
    p, t, m = _batch(target, target[None].clone(), lat, lon, perm)
    lv = mod.compute_loss(p, t, m, istep=10)
    assert abs(float(lv.loss)) < 1e-4
    logs = lv.losses_all[STREAM]["wfcl"]
    assert set(logs) >= {"FAL", "FCL", "WAL_L1", "WCL_L2", "P_t", "num_fields"}
    assert 0.0 < float(logs["P_t"]["avg"]) < 1.0

    pred = (0.5 * target + 0.2 * _fields(lat, lon, seed=3))[None].requires_grad_(True)
    p, t, m = _batch(target, pred, lat, lon, perm)
    lv = mod.compute_loss(p, t, m, istep=10)
    assert float(lv.loss) > 0.0
    lv.loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_wfcl_requires_explicit_total_steps(grid_zarr):
    path, _, _ = grid_zarr
    with pytest.raises(ValueError, match="total_steps"):
        LossSpectralWFCL(
            _cf(), None, "train", "cpu", wfcl={"target_stream": STREAM, "grid_zarr": path}
        )


def test_wfcl_p_min_floor(grid_zarr):
    path, lat, lon = grid_zarr
    cfg = {
        "target_stream": STREAM,
        "grid_zarr": path,
        "patch_size": 8,
        "J": 2,
        "gamma": [1.0, 1.0],
        "total_steps": 100,
        "p_min": 0.3,
    }
    # istep far past the ramp: without the floor P_t would be 0
    mod = LossSpectralWFCL(_cf(istep=500), None, "train", "cpu", wfcl=cfg)
    target = _fields(lat, lon)
    pred = (0.5 * target + 0.2 * _fields(lat, lon, seed=3))[None]
    p, t, m = _batch(target, pred, lat, lon, torch.randperm(len(lat)))
    logs = mod.compute_loss(p, t, m, istep=500).losses_all[STREAM]["wfcl"]
    assert float(logs["P_t"]["avg"]) == pytest.approx(0.3)
