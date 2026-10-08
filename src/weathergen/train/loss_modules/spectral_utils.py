# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spectral primitives for the Wavelet-Fourier Composite Loss (WFCL).

Reference
---------
An, Kim, Cho & Ham (2026), *Toward spatially sharper precipitation prediction via
global-local frequency guidance*, npj Clim Atmos Sci, doi:10.1038/s41612-026-01472-y.
Prerequisite: Yan et al. (2024), *Fourier Amplitude and Correlation Loss*, NeurIPS,
arXiv:2410.23159.

Everything here is pure tensor math on a **regular 2-D grid** ``(N, C, H, W)``.
Getting that grid out of WeatherGenerator's scattered target points is a separate
problem handled in :mod:`weathergen.train.loss_modules.spectral_patches`.

The DTCWT band math (:func:`wal_wcl`) deliberately takes the sub-band tensors
rather than a field, so it can be tested without ``pytorch_wavelets`` installed.
"""

from __future__ import annotations

import torch

# Guards |z| at the origin. Precipitation fields are mostly exact zeros, so the
# undefined-gradient case is reached constantly rather than occasionally.
EPS = 1e-12


def amplitude(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    """``|z|`` with a gradient that stays finite at ``z = 0``.

    ``sqrt(re**2 + im**2)`` has an infinite derivative at the origin, which
    autograd turns into NaN for every exactly-zero coefficient. Note that
    ``torch.abs()`` on a *complex* tensor is in fact safe on torch >= 2.x
    (``sgn(0 + 0j) == 0``), contrary to what is often assumed -- but building a
    complex tensor just to take its modulus costs a copy, so we work on the real
    and imaginary parts directly throughout.
    """
    return torch.sqrt(re * re + im * im + EPS)


def _corr(
    t_re: torch.Tensor,
    t_im: torch.Tensor,
    p_re: torch.Tensor,
    p_im: torch.Tensor,
    dims: tuple[int, ...],
) -> torch.Tensor:
    """``1 - <t, p> / (||t|| ||p||)`` over ``dims``, in [0, 2].

    Both norms are clamped *inside* the square root. Clamping outside it (i.e.
    ``sqrt(x) + eps``) still routes a zero through ``d/dx sqrt(x)`` and yields
    NaN gradients whenever a whole sub-band is zero.
    """
    num = (t_re * p_re + t_im * p_im).sum(dim=dims)
    n_t = (t_re * t_re + t_im * t_im).sum(dim=dims).clamp_min(EPS).sqrt()
    n_p = (p_re * p_re + p_im * p_im).sum(dim=dims).clamp_min(EPS).sqrt()
    return 1.0 - num / (n_t * n_p)


def fal_fcl(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourier amplitude loss and Fourier correlation loss.

    Args:
        pred, target: ``(N, C, H, W)`` real float32 fields on a regular grid.

    Returns:
        ``(fal, fcl)``, each ``(N, C)`` -- reduced per sample and per channel so
        that channels are never normalised against one another.

    ``norm="ortho"`` is load-bearing: it makes Parseval hold as
    ``sum|x|^2 == sum|F|^2``, which puts FAL on the scale of the field values.
    Without it FAL is inflated by a factor of ``H*W`` (~1e5 on any realistic
    grid) relative to FCL, and the composite degenerates to FAL-only -- the
    collapse mode Yan et al. describe in their Appendix C.
    """
    f_p = torch.fft.fft2(pred, norm="ortho")
    f_t = torch.fft.fft2(target, norm="ortho")

    fal = (amplitude(f_t.real, f_t.imag) - amplitude(f_p.real, f_p.imag)).pow(2).mean(dim=(-2, -1))
    fcl = _corr(f_t.real, f_t.imag, f_p.real, f_p.imag, dims=(-2, -1))
    return fal, fcl


def wal_wcl(
    yh_pred: list[torch.Tensor],
    yh_target: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Wavelet amplitude and correlation losses, per level and orientation.

    Args:
        yh_pred, yh_target: lists of length ``J``; element ``l`` has shape
            ``(N, C, 6, H_l, W_l, 2)`` with the trailing axis holding
            ``(real, imag)`` -- the layout ``DTCWTForward`` returns.

    Returns:
        ``(wal, wcl)``, two lists of length ``J``, each element ``(N, C, 6)``.

    Only the six directional high-pass bands enter; the low-pass band is excluded
    because FACL already constrains coarse structure and including it double-counts.
    """
    wal_levels, wcl_levels = [], []
    for h_p, h_t in zip(yh_pred, yh_target, strict=True):
        p_re, p_im = h_p[..., 0], h_p[..., 1]
        t_re, t_im = h_t[..., 0], h_t[..., 1]

        wal = (amplitude(t_re, t_im) - amplitude(p_re, p_im)).pow(2).mean(dim=(-2, -1))
        wcl = _corr(t_re, t_im, p_re, p_im, dims=(-2, -1))

        wal_levels.append(wal)
        wcl_levels.append(wcl)
    return wal_levels, wcl_levels


def phase_weight(step: int, total_steps: int, alpha: float = 0.1) -> float:
    """``P_t``: 1 -> 0 linearly over the first ``(1 - alpha)`` of training, then 0.

    Training therefore starts as pure phase/correlation matching and ends as pure
    amplitude matching. ``step`` must be the **global** optimiser step; in
    WeatherGenerator that is ``cf.general.istep``, which does survive
    ``train_continue`` (``trainer.py`` derives ``mini_epoch_base`` from it).
    """
    if total_steps <= 0:
        return 0.0
    # Clamped at BOTH ends. A `schedule_start_step` a little above the parent's true
    # final istep makes `step` negative for the first few steps, and an unclamped
    # value then exceeds 1 -- which flips (1 - P_t) negative and briefly *rewards*
    # amplitude error. Measured at 1.0008 on run idgem1hg's first 12 steps: harmless
    # there, but only by luck.
    return min(1.0, max(0.0, 1.0 - step / ((1.0 - alpha) * total_steps)))
