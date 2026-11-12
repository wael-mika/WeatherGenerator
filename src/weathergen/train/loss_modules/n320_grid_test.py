# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import numpy as np
import pytest
import torch

from weathergen.train.loss_modules.n320_grid import (
    BilinearPatches,
    PointIndexer,
    ReducedGaussianGrid,
    ReducedGaussianSHT,
    _normalised_legendre,
    amse_from_coefficients,
    assemble_full_field,
)

N320 = Path("/e/data1/slmet/ml_training/aifs-ea-an-oper-0001-mars-n320-1979-2023-6h-v8.zarr")
NLAT, LMAX = 64, 20


def _small_grid(nlat=NLAT, min_points=48):
    """Reduced Gaussian grid stored like the anemoi N320 stores (row by row, north to south)."""
    mu, _ = np.polynomial.legendre.leggauss(nlat)
    row_lat = np.degrees(np.arcsin(mu[::-1]))
    counts = [max(min_points, 2 * int(np.ceil(nlat * np.cos(np.radians(la))))) for la in row_lat]
    lat = np.concatenate([np.full(n, la) for la, n in zip(row_lat, counts, strict=True)])
    lon = np.concatenate([np.arange(n) * 360.0 / n for n in counts])
    return ReducedGaussianGrid(lat, lon)


@pytest.fixture(scope="module")
def grid():
    return _small_grid()


@pytest.fixture(scope="module")
def sht(grid):
    return ReducedGaussianSHT(grid, lmax=LMAX)


def _harmonic_field(grid, coeffs, lmax=LMAX):
    """Real band-limited field sum_{l,m} c_lm P_lm(mu) cos(m lon) (+ sin parts), on the grid."""
    mu = np.sin(np.radians(grid.row_lat))
    p = _normalised_legendre(mu, lmax)
    w = np.polynomial.legendre.leggauss(grid.n_rows)[1][::-1]
    p = p / np.sqrt((p * p * w).sum(-1) / 2.0)[..., None].clip(1e-300)
    x = np.zeros(grid.n_points)
    for (ll, m, kind), c in coeffs.items():
        for r in range(grid.n_rows):
            s, n = grid.starts[r], grid.counts[r]
            lon = np.radians(grid.lon[s : s + n])
            trig = np.cos(m * lon) if kind == "c" else np.sin(m * lon)
            x[s : s + n] += c * p[m, ll, r] * trig
    return torch.from_numpy(x).float()


def _random_coeffs(seed, lmax=LMAX):
    rng = np.random.default_rng(seed)
    coeffs = {}
    for ll in range(lmax + 1):
        for m in range(ll + 1):
            coeffs[(ll, m, "c")] = rng.normal() / (1 + ll)
            if m > 0:
                coeffs[(ll, m, "s")] = rng.normal() / (1 + ll)
    return coeffs


def _random_field(grid, seed, lmax=LMAX):
    return _harmonic_field(grid, _random_coeffs(seed, lmax), lmax)


def _phase_scrambled(coeffs, seed):
    """Same |a_lm| (hence the same PSD at every l), random phase per (l, m)."""
    rng = np.random.default_rng(seed)
    out = {}
    for (ll, m, kind), c in coeffs.items():
        if kind != "c":
            continue
        if m == 0:
            out[(ll, 0, "c")] = c * rng.choice([-1.0, 1.0])
            continue
        s, th = coeffs[(ll, m, "s")], rng.uniform(0, 2 * np.pi)
        out[(ll, m, "c")] = c * np.cos(th) - s * np.sin(th)
        out[(ll, m, "s")] = c * np.sin(th) + s * np.cos(th)
    return out


def test_legendre_orthonormal_under_gauss_quadrature():
    mu, w = np.polynomial.legendre.leggauss(NLAT)
    p = _normalised_legendre(mu, LMAX)
    p = p / np.sqrt((p * p * w).sum(-1) / 2.0)[..., None]
    for m in range(LMAX + 1):
        gram = (p[m, m:] * w) @ p[m, m:].T
        np.testing.assert_allclose(gram, 2.0 * np.eye(LMAX + 1 - m), atol=1e-10)


def test_parseval_and_spectral_mse(grid, sht):
    x, y = _random_field(grid, 0), _random_field(grid, 1)
    a = sht(torch.stack([x, y]))
    (psd,) = ReducedGaussianSHT.spectra(a)
    torch.testing.assert_close(
        psd.sum(-1), sht.sphere_mean_square(torch.stack([x, y])), rtol=1e-4, atol=0
    )
    _, logs = amse_from_coefficients(a[:1], a[1:])
    torch.testing.assert_close(
        logs["spectral_mse"], sht.sphere_mean_square((x - y)[None])[0], rtol=1e-4, atol=0
    )


def test_single_harmonic_lands_at_its_wavenumber(grid, sht):
    x = _harmonic_field(grid, {(7, 3, "c"): 1.0})
    (psd,) = ReducedGaussianSHT.spectra(sht(x[None]))
    assert int(psd[0].argmax()) == 7
    assert float(psd[0, 7] / psd[0].sum()) > 0.9999


def test_amse_properties(grid, sht):
    x = _random_field(grid, 2)
    a_x = sht(x[None])
    amse, logs = amse_from_coefficients(a_x, a_x)
    power = float(ReducedGaussianSHT.spectra(a_x)[0].sum())
    assert float(amse) < 1e-5 * power  # float32 round-off in 1 - rho only
    # a longitude shift keeps PSD(l) at every l -> AMSE == MSE; otherwise AMSE >= MSE
    k = 5
    shifted = torch.cat(
        [
            torch.roll(x[s : s + n], (k * n) // 64)
            for s, n in zip(grid.starts, grid.counts, strict=True)
        ]
    )
    _, logs = amse_from_coefficients(sht(shifted[None]), a_x)
    torch.testing.assert_close(logs["amse"], logs["spectral_mse"], rtol=2e-2, atol=1e-6)
    y = _random_field(grid, 3)
    _, logs = amse_from_coefficients(sht(y[None]), a_x)
    assert float(logs["amse"]) >= float(logs["spectral_mse"])


def test_amse_rewards_correct_power_where_mse_rewards_damping(grid, sht):
    coeffs = _random_coeffs(4)
    target = _harmonic_field(grid, coeffs)
    indep = _harmonic_field(grid, _phase_scrambled(coeffs, 5))  # same PSD(l), decorrelated
    a_t = sht(target[None])
    alphas = [0.25, 0.5, 1.0, 2.0]
    amse = [float(amse_from_coefficients(sht((al * indep)[None]), a_t)[0]) for al in alphas]
    mse = [
        float(amse_from_coefficients(sht((al * indep)[None]), a_t)[1]["spectral_mse"])
        for al in alphas
    ]
    assert int(np.argmin(mse)) == 0  # MSE: shrink an uncorrelated prediction towards zero
    assert alphas[int(np.argmin(amse))] == 1.0  # AMSE: keep the target's power


def test_amse_gradient_finite_at_zero_prediction(grid, sht):
    target = _random_field(grid, 6)
    pred = torch.zeros(grid.n_points, requires_grad=True)
    amse, _ = amse_from_coefficients(sht(pred[None]), sht(target[None]))
    amse.sum().backward()
    assert torch.isfinite(pred.grad).all()


def test_bilinear_patches_reproduce_constant_and_linear_fields(grid):
    bp = BilinearPatches(grid, patch_size=8, lat_max=60.0)
    const = bp(torch.full((grid.n_points, 1), 3.0))
    torch.testing.assert_close(const, torch.full_like(const, 3.0))
    lat_field = torch.from_numpy(grid.lat).float()[:, None]
    patches = bp(lat_field)
    for k, (clat, _) in enumerate(bp.patch_centres):
        col = patches[k, 0, :, 0]
        assert abs(float(col.mean()) - clat) < 0.6
        assert float(col[0]) > float(col[-1])  # north at the top
    assert all(abs(c[0]) + 8 * 180.0 / grid.n_rows / 2 <= 60.0 + 1e-6 for c in bp.patch_centres)


def test_assemble_full_field(grid):
    from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon

    idx = PointIndexer(grid)
    n = grid.n_points
    coords = torch.from_numpy(np.stack([_clip_lat(grid.lat), _clip_lon(grid.lon)], 1))
    perm = torch.randperm(n)
    target = torch.arange(n, dtype=torch.float32)[:, None]
    pred = 2 * target[None]
    tgt, prd = assemble_full_field(target[perm], pred[:, perm], coords[perm], idx)
    torch.testing.assert_close(tgt, target)
    torch.testing.assert_close(prd, pred)
    assert (
        assemble_full_field(target[perm][1:], pred[:, perm][:, 1:], coords[perm][1:], idx) is None
    )
    dup = torch.cat([perm[:-1], perm[:1]])
    assert assemble_full_field(target[dup], pred[:, dup], coords[dup], idx) is None
    t_nan = target.clone()
    t_nan[0] = torch.nan
    tgt, _ = assemble_full_field(t_nan, pred, coords, idx)
    assert float(tgt[0, 0]) == float(pred[0, 0, 0])


@pytest.mark.skipif(not N320.exists(), reason="N320 store not available")
def test_real_n320_grid():
    g = ReducedGaussianGrid.from_zarr(str(N320))
    assert g.n_points == 542080 and g.n_rows == 640
    sht = ReducedGaussianSHT(g, lmax=31)  # also checks the rows are the Gaussian latitudes
    from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon

    idx = PointIndexer(g)
    coords = torch.from_numpy(np.stack([_clip_lat(g.lat), _clip_lon(g.lon)], 1))
    assert torch.equal(idx(coords), torch.arange(g.n_points))
    x = torch.from_numpy(np.cos(np.radians(g.lat)) ** 2 * np.cos(2 * np.radians(g.lon))).float()
    (psd,) = ReducedGaussianSHT.spectra(sht(x[None]))
    torch.testing.assert_close(psd.sum(-1), sht.sphere_mean_square(x[None]), rtol=1e-4, atol=0)
