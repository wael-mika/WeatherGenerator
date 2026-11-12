# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Full-field tools for losses on a reduced Gaussian grid (the anemoi N320 stores).

Three pieces, all built once from the grid's own coordinates:

* :class:`PointIndexer` maps the target points a loss receives (lat/lon as the reader delivers
  them, in tokenised order) back to grid indices, so a loss can rebuild the COMPLETE field in
  store order. Spectral losses are meaningless on a partial field, so callers must check that
  every grid point arrived exactly once.
* :class:`ReducedGaussianSHT` is a spherical-harmonic analysis on the reduced Gaussian grid:
  an FFT along each latitude row, then a Gauss-Legendre transform with fully normalised
  associated Legendre functions. Real fields only.
* :class:`BilinearPatches` resamples the full field onto fixed, regular, approximately square
  ``S x S`` rasters (for 2-D Fourier/wavelet losses), by bilinear interpolation along and
  between rows. It is a fixed sparse matrix, hence differentiable and cheap.
* :class:`RasterPatches` is the limited-area counterpart: plain S x S tiles of a field stored
  row-major on a regular H x W raster (CERRA). Only :class:`PointIndexer` is needed with it, so
  the reduced-Gaussian convention below does not apply there.

Grid convention (verified for every anemoi N320 store on Jupiter): points are stored row by row
from north to south, each row has constant latitude and ``n_r`` points equally spaced in
longitude starting at 0 degrees; the row latitudes are the Gaussian latitudes.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Coordinates are quantised to this many degrees to look a point up (as in LossSpectralWFCL):
# far coarser than float32 noise (~2e-5 deg) and far finer than the N320 spacing (~0.28 deg).
_COORD_QUANT = 1e-3


def coord_key(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Collision-free int64 key for (lat, lon) in degrees.

    Both sides go through float32 first, because that is the precision the reader delivers; a
    float64 value and its float32 cast can round to different bins, and a spectral loss needs
    EVERY point, so even a 0.1% miss rate is fatal.
    """
    lat_q = torch.round((lat.float().double() + 90.0) / _COORD_QUANT).to(torch.int64)
    lon_q = torch.round((lon.float().double() % 360.0) / _COORD_QUANT).to(torch.int64)
    return lat_q * 400_000 + lon_q


class ReducedGaussianGrid:
    """Row structure of a reduced Gaussian grid stored row by row, north to south."""

    def __init__(self, lat: NDArray, lon: NDArray):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        self.lat, self.lon = lat, lon
        self.n_points = len(lat)
        self.starts = np.flatnonzero(np.r_[True, lat[1:] != lat[:-1]])
        self.counts = np.diff(np.r_[self.starts, self.n_points])
        self.row_lat = lat[self.starts]
        if len(np.unique(self.row_lat)) != len(self.row_lat) or not np.all(
            np.diff(self.row_lat) < 0
        ):
            raise ValueError("Grid is not stored row by row from north to south.")
        for r in range(len(self.starts)):
            row_lon = lon[self.starts[r] : self.starts[r] + self.counts[r]] % 360.0
            expect = (np.arange(self.counts[r]) * 360.0 / self.counts[r]) % 360.0
            d = np.abs(((row_lon - expect + 180.0) % 360.0) - 180.0).max()
            if d > 1e-6:
                raise ValueError(f"Row {r} is not equally spaced from 0 deg (max offset {d:.2e}).")
        self.n_rows = len(self.row_lat)

    @classmethod
    def from_zarr(cls, path: str) -> ReducedGaussianGrid:
        import zarr

        z = zarr.open(str(path), mode="r")
        return cls(z["latitudes"][:], z["longitudes"][:])


class GridPoints:
    """Raw point coordinates of a stream's zarr, in storage order, with no structure assumed.

    Enough to index points (:class:`PointIndexer`); :class:`ReducedGaussianGrid` adds the row
    structure the bilinear patches and the spherical-harmonic transform need.
    """

    def __init__(self, lat: NDArray, lon: NDArray):
        self.lat = np.asarray(lat, dtype=np.float64)
        self.lon = np.asarray(lon, dtype=np.float64)
        self.n_points = len(self.lat)

    @classmethod
    def from_zarr(cls, path: str) -> GridPoints:
        import zarr

        z = zarr.open(str(path), mode="r")
        return cls(z["latitudes"][:], z["longitudes"][:])


class PointIndexer:
    """Map reader coordinates (lat/lon in degrees, as delivered) to grid indices."""

    def __init__(self, grid: GridPoints | ReducedGaussianGrid, device: str | torch.device = "cpu"):
        # The reader's own transforms, so the keys cannot drift from what arrives at the loss.
        from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon

        lat = torch.from_numpy(_clip_lat(grid.lat))
        lon = torch.from_numpy(_clip_lon(grid.lon))
        keys = coord_key(lat, lon)
        order = torch.argsort(keys)
        if torch.any(keys[order][1:] == keys[order][:-1]):
            raise ValueError("Coordinate keys collide; quantisation too coarse for this grid.")
        self.keys = keys[order].to(device)
        self.index = order.to(torch.int64).to(device)
        self.n_points = grid.n_points

    def to(self, device) -> PointIndexer:
        self.keys, self.index = self.keys.to(device), self.index.to(device)
        return self

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """(P, 2) lat/lon -> (P,) grid indices; -1 where a point is not on the grid."""
        q = coord_key(coords[:, 0], coords[:, 1]).to(self.keys.device)
        pos = torch.searchsorted(self.keys, q).clamp_max(self.keys.numel() - 1)
        hit = self.keys[pos] == q
        return torch.where(hit, self.index[pos], torch.full_like(pos, -1))


def _normalised_legendre(mu: NDArray, lmax: int) -> NDArray:
    """Fully normalised associated Legendre functions, shape (lmax+1 [m], lmax+1 [l], nlat).

    Standard stable recursion (column-wise in l for each m); entries with l < m are zero.
    Normalisation is fixed afterwards by Gauss quadrature (see ReducedGaussianSHT).
    """
    nlat = len(mu)
    sin_t = np.sqrt(np.clip(1.0 - mu * mu, 0.0, None))
    p = np.zeros((lmax + 1, lmax + 1, nlat))
    pmm = np.ones(nlat)
    for m in range(lmax + 1):
        if m > 0:
            # sectoral start; any m-dependent constant is removed by the quadrature normalisation
            pmm = pmm * np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sin_t
        p[m, m] = pmm
        if m + 1 <= lmax:
            p[m, m + 1] = np.sqrt(2.0 * m + 3.0) * mu * pmm
        for ll in range(m + 2, lmax + 1):
            a = np.sqrt((2.0 * ll - 1.0) * (2.0 * ll + 1.0) / ((ll - m) * (ll + m)))
            b = np.sqrt(
                (2.0 * ll + 1.0)
                * (ll + m - 1.0)
                * (ll - m - 1.0)
                / ((ll - m) * (ll + m) * (2.0 * ll - 3.0))
            )
            p[m, ll] = a * mu * p[m, ll - 1] - b * p[m, ll - 2]
    return p


class ReducedGaussianSHT(nn.Module):
    """Spherical-harmonic analysis on a reduced Gaussian grid.

    For a real field x, returns complex coefficients a[l, m] (m = 0..lmax) with

        a_lm = (1/2) sum_r w_r P_lm(mu_r) f_m(r),   f_m(r) = (1/n_r) sum_k x_rk exp(-i m lon_rk)

    where P_lm is normalised so that sum_r w_r P_lm(mu_r)^2 = 2 (Gauss weights w_r sum to 2).
    Then (1/4pi) int |Y_lm|^2 = 1 and Parseval holds as

        mean over the sphere of x^2 = sum_l PSD(l),  PSD(l) = |a_l0|^2 + 2 sum_{m>=1} |a_lm|^2.

    Row coefficients with m above a row's Nyquist (n_r // 2) are set to zero: on a reduced grid
    those rows cannot carry them, and the Legendre functions are negligible there anyway.
    The Legendre table is (lmax+1)^2 x nlat floats (~1 GB float32 for N320 at lmax 639).
    """

    def __init__(self, grid: ReducedGaussianGrid, lmax: int | None = None):
        super().__init__()
        nlat = grid.n_rows
        self.lmax = int(nlat - 1 if lmax is None else lmax)
        mu_g, w_g = np.polynomial.legendre.leggauss(nlat)  # ascending in mu = south -> north
        mu_g, w_g = mu_g[::-1].copy(), w_g[::-1].copy()  # north -> south, as the rows are
        mu_grid = np.sin(np.radians(grid.row_lat))
        dev = np.abs(mu_grid - mu_g).max()
        if dev > 1e-6:
            raise ValueError(f"Row latitudes are not the Gaussian latitudes (max |dmu| {dev:.2e}).")
        plm = _normalised_legendre(mu_g, self.lmax)
        # fix the normalisation exactly with the quadrature: sum_r w_r P^2 = 2
        norm = np.sqrt((plm * plm * w_g).sum(-1) / 2.0)
        plm = np.where(norm[..., None] > 0, plm / np.where(norm > 0, norm, 1.0)[..., None], 0.0)
        # fold the 0.5 * w_r of the analysis into the table: (m, l, r)
        self.register_buffer("plm_w", torch.from_numpy(0.5 * plm * w_g).to(torch.float32))
        self.register_buffer("w", torch.from_numpy(w_g).to(torch.float32))
        self.starts = [int(s) for s in grid.starts]
        self.counts = [int(c) for c in grid.counts]
        self.n_points = grid.n_points
        # rows grouped by length, so all rows of one length are transformed together
        groups: dict[int, list[int]] = {}
        for r, n in enumerate(self.counts):
            groups.setdefault(n, []).append(r)
        self._groups = [
            (n, torch.tensor(rows), torch.tensor([self.starts[r] for r in rows]))
            for n, rows in sorted(groups.items())
        ]

    def row_fourier(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_points) real -> (B, nlat, lmax+1) complex row coefficients f_m(r)."""
        b = x.shape[0]
        out = torch.zeros(
            (b, len(self.counts), self.lmax + 1), dtype=torch.complex64, device=x.device
        )
        for n, rows, starts in self._groups:
            idx = (
                starts.to(x.device)[:, None] + torch.arange(n, device=x.device)[None, :]
            ).reshape(-1)
            seg = x[:, idx].reshape(b, len(rows), n)
            f = torch.fft.rfft(seg.float(), dim=-1) / n  # (B, rows, n//2+1)
            mmax = min(self.lmax, (n - 1) // 2)  # drop the Nyquist / beyond-Nyquist bins
            out[:, rows.to(x.device), : mmax + 1] = f[..., : mmax + 1]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_points) real -> (B, lmax+1 [l], lmax+1 [m]) complex coefficients a[l, m]."""
        f = self.row_fourier(x)  # (B, r, m)
        # a[b, l, m] = sum_r plm_w[m, l, r] * f[b, r, m]; real table, so real and imaginary parts
        # are transformed separately instead of materialising a complex copy of the table
        a_re = torch.einsum("mlr,brm->blm", self.plm_w, f.real.contiguous())
        a_im = torch.einsum("mlr,brm->blm", self.plm_w, f.imag.contiguous())
        return torch.complex(a_re, a_im)

    @staticmethod
    def spectra(a: torch.Tensor, b: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        """PSD(l) of a (and of b, plus the cross-spectrum Re<a, b>(l) if b is given)."""
        wm = torch.full((a.shape[-1],), 2.0, device=a.device)
        wm[0] = 1.0
        psd_a = (a.real**2 + a.imag**2) @ wm
        if b is None:
            return (psd_a,)
        psd_b = (b.real**2 + b.imag**2) @ wm
        csd = (a.real * b.real + a.imag * b.imag) @ wm
        return psd_a, psd_b, csd

    def sphere_mean_square(self, x: torch.Tensor) -> torch.Tensor:
        """Quadrature mean of x^2 over the sphere, (B, n_points) -> (B,)."""
        out = torch.zeros(x.shape[0], device=x.device)
        for r, (s, n) in enumerate(zip(self.starts, self.counts, strict=True)):
            out = out + 0.5 * self.w[r] * (x[:, s : s + n].float() ** 2).mean(-1)
        return out


def amse_from_coefficients(
    a_pred: torch.Tensor, a_target: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, dict]:
    """Adjusted MSE of Subich et al. (2025) from spherical-harmonic coefficients (B, l, m).

    Per total wavenumber l, with prediction/target power P_x, P_y and coherence
    rho = CSD / sqrt(P_x P_y):

        AMSE = sum_l [ (sqrt(P_x) - sqrt(P_y))^2 + 2 max(P_x, P_y) (1 - rho) ]

    MSE is the same sum with sqrt(P_x P_y) instead of max(P_x, P_y), so AMSE keeps the full
    penalty on decorrelation but no longer rewards damping power to hedge against it.
    Returns the per-field AMSE (B,) and diagnostics. eps clamps sit inside the square roots,
    so all-zero spectral bands keep finite gradients.
    """
    p_x, p_y, csd = ReducedGaussianSHT.spectra(a_pred, a_target)
    s_x = p_x.clamp_min(eps).sqrt()
    s_y = p_y.clamp_min(eps).sqrt()
    rho = csd / (s_x * s_y)
    amp = (s_x - s_y) ** 2
    dec = 2.0 * torch.maximum(p_x, p_y) * (1.0 - rho)
    amse = (amp + dec).sum(-1)
    mse = (p_x + p_y - 2.0 * csd).sum(-1)
    logs = {
        "amse": amse.mean().detach(),
        "amplitude_term": amp.sum(-1).mean().detach(),
        "decorrelation_term": dec.sum(-1).mean().detach(),
        "spectral_mse": mse.mean().detach(),
        "power_ratio": (p_x.sum(-1) / p_y.sum(-1).clamp_min(eps)).mean().detach(),
    }
    return amse, logs


class BilinearPatches(nn.Module):
    """Fixed bilinear resampling of a full reduced-Gaussian field onto S x S patches.

    Patches tile the band |lat| <= lat_max in rows of patches of S pixels; pixel spacing is the
    grid's latitude spacing (180 / n_rows degrees, ~31 km for N320) in both directions, the
    longitude spacing being stretched by 1 / cos(lat_centre) so pixels stay roughly square.
    (Within a patch the physical pixel width still varies with cos(lat), by up to ~25% at the
    highest band -- acceptable for the 2-D spectral terms, and the poles are excluded.)
    """

    def __init__(self, grid: ReducedGaussianGrid, patch_size: int = 64, lat_max: float = 60.0):
        super().__init__()
        s = int(patch_size)
        dlat = 180.0 / grid.n_rows
        height = s * dlat
        n_bands = int(np.floor(2.0 * lat_max / height))
        centres_lat = (np.arange(n_bands) - (n_bands - 1) / 2.0) * height
        rows_idx, cols_idx, vals = [], [], []
        pix = 0
        patch_meta = []
        offs = (np.arange(s) - (s - 1) / 2.0) * dlat
        for clat in centres_lat:
            dlon = dlat / np.cos(np.radians(clat))
            width = s * dlon
            n_lon = int(np.floor(360.0 / width))
            for clon in np.arange(n_lon) * (360.0 / n_lon):
                lat_px = (clat - offs)[:, None] * np.ones((1, s))  # north at the top
                lon_px = (clon + (np.arange(s) - (s - 1) / 2.0) * dlon)[None, :] * np.ones((s, 1))
                idx, w = self._bilinear(grid, lat_px.ravel(), lon_px.ravel())
                rows_idx.append(np.repeat(np.arange(pix, pix + s * s), 4))
                cols_idx.append(idx.ravel())
                vals.append(w.ravel())
                pix += s * s
                patch_meta.append((clat, clon))
        mat = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([np.concatenate(rows_idx), np.concatenate(cols_idx)])),
            torch.from_numpy(np.concatenate(vals)).to(torch.float32),
            size=(pix, grid.n_points),
        ).coalesce()
        self.register_buffer("mat", mat)
        self.patch_size = s
        self.n_patches = len(patch_meta)
        self.patch_centres = patch_meta

    @staticmethod
    def _bilinear(grid: ReducedGaussianGrid, lat: NDArray, lon: NDArray):
        """4 grid indices and weights per (lat, lon): linear in lon within the two bracketing rows,
        then linear in lat between them."""
        rl = grid.row_lat
        # bracketing rows: j (north) with rl[j] >= lat > rl[j+1]
        j = np.clip(np.searchsorted(-rl, -lat, side="right") - 1, 0, grid.n_rows - 2)
        s_lat = (rl[j] - lat) / (rl[j] - rl[j + 1])
        idx = np.zeros((len(lat), 4), dtype=np.int64)
        w = np.zeros((len(lat), 4))
        for c, (jj, wl) in enumerate(((j, 1.0 - s_lat), (j + 1, s_lat))):
            n = grid.counts[jj]
            pos = (lon % 360.0) / (360.0 / n)
            k0 = np.floor(pos).astype(np.int64)
            t = pos - k0
            idx[:, 2 * c] = grid.starts[jj] + (k0 % n)
            idx[:, 2 * c + 1] = grid.starts[jj] + ((k0 + 1) % n)
            w[:, 2 * c] = wl * (1.0 - t)
            w[:, 2 * c + 1] = wl * t
        return idx, w

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """(n_points, C) full field -> (K, C, S, S) patches."""
        out = torch.sparse.mm(self.mat, field.float())  # (K*S*S, C)
        k, s = self.n_patches, self.patch_size
        return out.view(k, s, s, field.shape[-1]).permute(0, 3, 1, 2).contiguous()


class RasterPatches(nn.Module):
    """Non-overlapping S x S tiles of a field stored row-major on a regular H x W raster.

    For limited-area grids such as CERRA (1069 x 1069 Lambert conformal, 5.5 km): the patches are
    the native grid cells, no resampling. The tiling is centred; the ``H % S`` rows and ``W % S``
    columns it leaves over are split between the two edges and not scored.
    """

    def __init__(self, height: int, width: int, patch_size: int):
        super().__init__()
        h, w, s = int(height), int(width), int(patch_size)
        if s > min(h, w):
            raise ValueError(f"patch_size={s} exceeds the {h} x {w} raster")
        self.height, self.width, self.patch_size = h, w, s
        self.n_rows, self.n_cols = h // s, w // s
        self.row0 = (h - self.n_rows * s) // 2
        self.col0 = (w - self.n_cols * s) // 2
        self.n_patches = self.n_rows * self.n_cols

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """(H * W, C) full field in storage order -> (K, C, S, S) patches."""
        h, w, s, c = self.height, self.width, self.patch_size, field.shape[-1]
        if field.shape[0] != h * w:
            raise ValueError(f"field has {field.shape[0]} points, raster is {h} x {w} = {h * w}")
        x = field.float().reshape(h, w, c)
        x = x[self.row0 : self.row0 + self.n_rows * s, self.col0 : self.col0 + self.n_cols * s]
        x = x.reshape(self.n_rows, s, self.n_cols, s, c).permute(0, 2, 4, 1, 3)
        return x.reshape(self.n_patches, c, s, s)


def assemble_full_field(
    target: torch.Tensor,
    pred: torch.Tensor,
    coords: torch.Tensor,
    indexer: PointIndexer,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Rebuild complete fields in grid order from a loss's target points.

    target (P, C), pred (E, P, C), coords (P, 2). Returns (target_full (N, C), pred_full (E, N, C))
    or None unless every grid point arrived exactly once. NaN targets (e.g. IMERG's polar caps)
    are filled with the detached prediction there, i.e. zero error at those points, so they
    neither inject spurious spectral power nor carry gradient.
    """
    idx = indexer(coords)
    n = indexer.n_points
    if idx.numel() != n or bool((idx < 0).any()):
        return None
    if int(torch.bincount(idx, minlength=n).max()) != 1:
        return None
    tgt = torch.empty((n, target.shape[-1]), dtype=target.dtype, device=target.device)
    tgt[idx] = target
    prd = torch.empty((pred.shape[0], n, pred.shape[-1]), dtype=pred.dtype, device=pred.device)
    prd[:, idx] = pred
    nan = torch.isnan(tgt)
    if bool(nan.any()):
        tgt = torch.where(nan, prd.mean(0).detach(), tgt)
    return tgt, prd
