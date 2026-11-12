# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Structured (spectral) losses on full reduced-Gaussian fields (the anemoi N320 stores).

* :class:`LossSpectralAMSE` -- Adjusted MSE of Subich et al. (2025), "Fixing the double penalty
  in data-driven weather forecasting through a modified spherical harmonic loss function":
  per total spherical-harmonic wavenumber, an amplitude term plus a decorrelation term weighted
  by max(P_pred, P_target) instead of MSE's sqrt(P_pred * P_target). Used ALONE, as in the paper.
* :class:`LossSpectralWFCL` -- Wavelet-Fourier Composite Loss (An et al. 2026, building on Yan et
  al. 2024). The 2-D Fourier and DTCWT terms run on fixed S x S patches of the full field:
  ``grid: reduced_gaussian`` (default) resamples them bilinearly from a global reduced Gaussian
  grid (O96, N320) over |lat| <= lat_max; ``grid: raster`` tiles a regular limited-area raster
  stored row-major (CERRA, ``grid_shape: [1069, 1069]``) at native resolution. It is a
  correction term: add it to MSE.

Both need the COMPLETE field at every forecast step: set ``streams.<target>.max_num_targets=-1``.
A partial field raises -- a spectral loss on a random point subset would be meaningless, and with
MSE at weight 0 a silently skipped term would train nothing at all.

Config (one key under ``loss_fcts``, holding the parameters)::

    amse:
      type: LossSpectralAMSE
      loss_fcts:
        amse: {target_stream: IMERG_ANEMOI, grid_zarr: /e/data1/.../...-n320-....zarr, lmax: 639}
    wfcl:
      type: LossSpectralWFCL
      weight: 0.1
      loss_fcts:
        wfcl: {target_stream: IMERG_ANEMOI, grid_zarr: ..., patch_size: 64, lat_max: 60,
               J: 3, total_steps: 1530, schedule_start_step: 0}

``channels: [...]`` (optional, both losses) restricts the term to those target channels.
See config/config_cerra_wfcl_finetune.yml for a documented CERRA example.
"""

from __future__ import annotations

import functools
import logging
from collections import defaultdict

import torch
from omegaconf import DictConfig, OmegaConf

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.loss_modules.n320_grid import (
    BilinearPatches,
    GridPoints,
    PointIndexer,
    RasterPatches,
    ReducedGaussianGrid,
    ReducedGaussianSHT,
    amse_from_coefficients,
    assemble_full_field,
)
from weathergen.train.loss_modules.spectral_patches import DTCWTBands, check_patch_size
from weathergen.train.loss_modules.spectral_utils import fal_fcl, phase_weight, wal_wcl
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _grid(path: str) -> ReducedGaussianGrid:
    return ReducedGaussianGrid.from_zarr(path)


@functools.lru_cache(maxsize=4)
def _points(path: str) -> GridPoints:
    return GridPoints.from_zarr(path)


@functools.lru_cache(maxsize=4)
def _sht(path: str, lmax: int, device: str) -> ReducedGaussianSHT:
    """One transform (and its ~1 GB Legendre table) per process and device, shared by the
    training and validation loss calculators."""
    return ReducedGaussianSHT(_grid(path), lmax=lmax).to(device)


@functools.lru_cache(maxsize=4)
def _indexer(path: str, device: str) -> PointIndexer:
    return PointIndexer(_points(path), device=device)


class _FullFieldLoss(LossModuleBase):
    """Shared plumbing: config, sample/target matching, full-field assembly, logging."""

    name = "LossSpectral"
    term = "spectral"

    def __init__(
        self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **loss_fcts
    ):
        LossModuleBase.__init__(self)
        self.cf, self.mode_cfg, self.stage, self.device = cf, mode_cfg, stage, device
        if len(loss_fcts) != 1:
            raise ValueError(
                f"{self.name}: expected exactly one entry under loss_fcts, got {list(loss_fcts)}"
            )
        cfg = next(iter(loss_fcts.values()))
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        self.cfg = dict(cfg or {})
        self.target_stream = str(self.cfg["target_stream"])
        self.grid_zarr = str(self.cfg["grid_zarr"])
        self.indexer = _indexer(self.grid_zarr, str(device))
        chans = self.cfg.get("channels")
        self.channels: list[str] | None = [str(ch) for ch in chans] if chans else None
        self._channel_idx: list[int] | None = None

    def _resolve_channel_idx(self) -> list[int] | None:
        """Indices of ``channels`` among the stream's target channels, resolved on first use
        (the channel lists are filled in on the stream config at runtime)."""
        if self.channels is None:
            return None
        if self._channel_idx is None:
            info = self.cf.streams[self.target_stream]
            names = list(
                info.val_target_channels if self.stage == "val" else info.train_target_channels
            )
            missing = [ch for ch in self.channels if ch not in names]
            if missing:
                raise ValueError(
                    f"{self.name}: channels {missing} are not target channels of stream "
                    f"'{self.target_stream}' ({names})"
                )
            self._channel_idx = [names.index(ch) for ch in self.channels]
        return self._channel_idx

    def _log_keys(self) -> list[str]:
        raise NotImplementedError

    def _field_loss(
        self, target_full: torch.Tensor, pred_full: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """target (N, C), pred (E, N, C) on the full grid -> scalar loss, logs."""
        raise NotImplementedError

    def compute_loss(self, preds, targets, metadata, **kwargs) -> LossValues:
        _s2t, output_info, _t2s, _tinfo = metadata
        stream = self.target_stream
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        agg: dict[str, list[torch.Tensor]] = defaultdict(list)
        ctr_steps = 0

        for preds_cur, target_cur in zip(preds.physical, targets.physical, strict=True):
            if stream not in target_cur:
                continue
            preds_batch = preds_cur.get(stream, [])
            if not preds_batch:
                continue
            targets_batch = target_cur[stream]["target"]
            coords_batch = target_cur[stream]["target_coords"]
            targets_params = target_cur[stream]["target_metda_data"]
            is_spoof = target_cur[stream]["is_spoof"]

            loss_ts = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_b = 0
            for pred, pred_params in zip(preds_batch, output_info, strict=True):
                corr = pred_params.global_params.get("correspondence", -1)
                match = [
                    i
                    for i, t in enumerate(targets_params)
                    if t[stream].global_params["idx"] == corr
                ]
                if len(match) != 1 or is_spoof[match[0]]:
                    continue
                t_idx = match[0]
                target = targets_batch[t_idx]
                if target.shape[0] == 0 or pred.shape[0] == 0:
                    continue
                pred = pred.reshape([pred.shape[0], *target.shape])  # (E, P, C)
                ch_idx = self._resolve_channel_idx()
                if ch_idx is not None:
                    target, pred = target[:, ch_idx], pred[..., ch_idx]
                coords = coords_batch[t_idx].to(target.device, non_blocking=True)
                full = assemble_full_field(target, pred, coords, self.indexer)
                if full is None:
                    raise RuntimeError(
                        f"{self.name}: stream '{stream}' delivered {target.shape[0]} target "
                        f"points, not the complete {self.indexer.n_points}-point grid of "
                        f"{self.grid_zarr} (each exactly once). Spectral losses need full "
                        f"fields: set streams.{stream}.max_num_targets=-1 for training AND "
                        "validation."
                    )
                with torch.autocast(device_type=target.device.type, enabled=False):
                    lo, logs = self._field_loss(full[0].float(), full[1].float())
                loss_ts = loss_ts + lo
                ctr_b += 1
                for k, v in logs.items():
                    agg[k].append(v)
            if ctr_b > 0:
                loss = loss + loss_ts / ctr_b
                ctr_steps += 1

        if ctr_steps > 0:
            loss = loss / ctr_steps
        out = defaultdict(dict)
        out[stream] = defaultdict(lambda: defaultdict(dict))
        # the same key set on every rank and step: ddp_average all_reduces per key
        for key in self._log_keys():
            vals = agg.get(key)
            out[stream][self.term][key]["avg"] = (
                torch.stack(vals).mean() if vals else torch.zeros((), device=self.device)
            )
        out[stream][self.term]["num_fields"]["avg"] = torch.tensor(
            float(len(agg.get(self._log_keys()[0], []))), device=self.device
        )
        # peak GPU memory so far, GiB: full-field targets are the memory driver of these runs
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        out[stream][self.term]["peak_mem_gib"]["avg"] = torch.tensor(peak, device=self.device)
        return LossValues(loss=loss, losses_all=out, stddev_all=None)


class LossSpectralAMSE(_FullFieldLoss):
    """Subich et al. (2025) Adjusted MSE on the full N320 field, per member and channel."""

    name = "LossSpectralAMSE"
    term = "amse"

    def __init__(self, cf, mode_cfg, stage, device, **loss_fcts):
        super().__init__(cf, mode_cfg, stage, device, **loss_fcts)
        self.lmax = int(self.cfg.get("lmax", 639))
        self.sht = _sht(self.grid_zarr, self.lmax, str(device))
        _logger.info(
            f"{self.name}: stream={self.target_stream} lmax={self.lmax} grid={self.grid_zarr}"
        )

    def _log_keys(self):
        return ["amse", "amplitude_term", "decorrelation_term", "spectral_mse", "power_ratio"]

    def _field_loss(self, target_full, pred_full):
        e, n, c = pred_full.shape
        # members and channels are independent fields: (E*C, N) and (C, N)
        a_p = self.sht(pred_full.permute(0, 2, 1).reshape(e * c, n))
        a_t = self.sht(target_full.t().contiguous())
        a_t = a_t.repeat(e, 1, 1)
        amse, logs = amse_from_coefficients(a_p, a_t)
        return amse.mean(), logs


class LossSpectralWFCL(_FullFieldLoss):
    """WFCL (An et al. 2026) on fixed S x S patches of the full field: bilinear patches of a
    reduced Gaussian grid over |lat| <= lat_max, or native tiles of a regular raster (CERRA).

    Schedule as on wm/fm-residual-flow: P_t ramps 1 -> 0 over the first (1 - alpha) of
    ``total_steps`` counted from ``schedule_start_step`` (both REQUIRED to be explicit, because
    ``istep`` is absolute on continuations); training starts phase/correlation-only and ends
    amplitude-only, or at ``p_min`` (default 0) to keep the correlation terms when WFCL is used
    without MSE.
    """

    name = "LossSpectralWFCL"
    term = "wfcl"

    def __init__(self, cf, mode_cfg, stage, device, **loss_fcts):
        super().__init__(cf, mode_cfg, stage, device, **loss_fcts)
        c = self.cfg
        self.patch_size = int(c.get("patch_size", 64))
        self.grid = str(c.get("grid", "reduced_gaussian"))
        self.lat_max = float(c.get("lat_max", 60.0))
        self.j = int(c.get("J", 3))
        self.beta = float(c.get("beta", 1.0))
        self.alpha = float(c.get("alpha", 0.1))
        self.p_min = float(c.get("p_min", 0.0))
        if not 0.0 <= self.p_min <= 1.0:
            raise ValueError(f"p_min must be in [0, 1], got {self.p_min}")
        self.gamma = [float(g) for g in c.get("gamma", [1.0] * self.j)]
        if len(self.gamma) != self.j:
            raise ValueError(f"gamma has {len(self.gamma)} entries, expected J={self.j}")
        if c.get("total_steps") in (None, "???"):
            raise ValueError(
                "LossSpectralWFCL requires an explicit `total_steps` (num_mini_epochs is an "
                "absolute stop point on continuations, so it cannot be derived)."
            )
        if c.get("schedule_start_step") == "???":
            raise ValueError(
                "LossSpectralWFCL: set `schedule_start_step` to the parent run's final istep "
                "(istep continues on train_continue; left at 0 the schedule would already be over)."
            )
        self.total_steps = int(c["total_steps"])
        self.schedule_start_step = int(c.get("schedule_start_step", 0))
        if self.schedule_start_step >= self.total_steps:
            raise ValueError("schedule_start_step must be before total_steps")
        check_patch_size(self.patch_size, self.j)
        if self.grid == "reduced_gaussian":
            self.patches = BilinearPatches(_grid(self.grid_zarr), self.patch_size, self.lat_max).to(
                device
            )
            geometry = f"|lat|<={self.lat_max}"
        elif self.grid == "raster":
            if "grid_shape" not in c:
                raise ValueError("grid: raster needs `grid_shape: [height, width]`")
            h, w = (int(v) for v in c["grid_shape"])
            if h * w != self.indexer.n_points:
                raise ValueError(
                    f"grid_shape {h} x {w} = {h * w} does not match the "
                    f"{self.indexer.n_points} points of {self.grid_zarr}"
                )
            self.patches = RasterPatches(h, w, self.patch_size)
            geometry = f"raster {h}x{w}"
        else:
            raise ValueError(f"grid must be 'reduced_gaussian' or 'raster', got '{self.grid}'")
        self.bands = DTCWTBands(
            self.j, biort=str(c.get("biort", "near_sym_b")), qshift=str(c.get("qshift", "qshift_b"))
        ).to(device)
        _logger.info(
            f"{self.name}: stream={self.target_stream} patches={self.patches.n_patches}x"
            f"{self.patch_size}^2 {geometry} channels={self.channels} J={self.j} beta={self.beta} "
            f"gamma={self.gamma} "
            f"alpha={self.alpha} p_min={self.p_min} total_steps={self.total_steps} "
            f"start={self.schedule_start_step}"
        )

    def _log_keys(self):
        keys = ["loss", "FAL", "FCL", "FACL", "WACL", "P_t"]
        for level in range(1, self.j + 1):
            keys += [f"WAL_L{level}", f"WCL_L{level}"]
        return keys

    def _field_loss(self, target_full, pred_full):
        p_t = phase_weight(
            int(self.cf.general.istep) - self.schedule_start_step,
            self.total_steps - self.schedule_start_step,
            self.alpha,
            self.p_min,
        )
        tp = self.patches(target_full)  # (K, C, S, S)
        tb = self.bands(tp)
        losses, logs_m = [], []
        for member in pred_full:
            pp = self.patches(member)
            fal, fcl = fal_fcl(pp, tp)
            fal, fcl = fal.mean(), fcl.mean()
            facl = (1.0 - p_t) * fal + p_t * fcl
            wal_l, wcl_l = wal_wcl(self.bands(pp), tb)
            wav = pp.new_zeros(())
            logs = {"FAL": fal.detach(), "FCL": fcl.detach(), "FACL": facl.detach()}
            for level in range(self.j):
                wal, wcl = wal_l[level].mean(), wcl_l[level].mean()
                wav = wav + self.gamma[level] * ((1.0 - p_t) * wal + p_t * wcl)
                logs[f"WAL_L{level + 1}"], logs[f"WCL_L{level + 1}"] = wal.detach(), wcl.detach()
            lo = self.beta * facl + wav
            logs |= {
                "WACL": wav.detach(),
                "loss": lo.detach(),
                "P_t": torch.tensor(p_t, device=pp.device),
            }
            losses.append(lo)
            logs_m.append(logs)
        merged = {k: torch.stack([m[k] for m in logs_m]).mean() for k in logs_m[0]}
        return torch.stack(losses).mean(), merged
