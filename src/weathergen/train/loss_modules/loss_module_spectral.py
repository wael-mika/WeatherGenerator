# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Wavelet-Fourier Composite Loss (WFCL).

An, Kim, Cho & Ham (2026), *Toward spatially sharper precipitation prediction via
global-local frequency guidance*, npj Clim Atmos Sci, doi:10.1038/s41612-026-01472-y,
building on Yan et al. (2024) NeurIPS (arXiv:2410.23159).

    L = beta * FACL + sum_L gamma_L * (1/6) * sum_k WACL(L, k)
    FACL = (1 - P_t) * FAL + P_t * FCL
    WACL = (1 - P_t) * WAL + P_t * WCL
    P_t  = max(0, 1 - t / ((1 - alpha) * T))

**Why the schedule is on both terms.** The npj paper writes FACL and WACL with an
expectation superscript. In the original FACL, FAL and FCL are not summed -- each
step draws p ~ U[0,1] and takes FAL if p > P(t) else FCL. The composite is the
*expectation* of that Bernoulli alternation, i.e. the convex combination above.
A fixed `FAL + FCL` is a different objective, and the FACL authors report they
could not make fixed weighting work. `facl_mode: sampled` recovers the original
stochastic form for ablation.

So training starts as pure phase/correlation matching and ends as pure amplitude
matching. That order matters: FAL alone has no gradient path to the target's phase
(Yan et al. Appendix C) and collapses if it dominates before coarse structure is learned.

**What this constrains that a structure function does not.** A variogram constrains
real-space increment statistics; WCL constrains scale- *and orientation*-resolved
phase, per (level, orientation) band. The two are complementary, not redundant --
see the campaign note in the config header.

**Ensembles.** Scores members, never the ensemble mean. Asking the mean of a
generative ensemble to be sharp is the double-penalty trap in a new costume; the
mean of a well-calibrated ensemble *should* be smooth. Reuses
`LossStructureFunction.ensemble_fields`.

**A ResidualFlow training pack is NOT an ensemble.** That decoder's leading axis holds
heterogeneous slots (see `PACK_*` in `weathergen.model.flow_math`), and `reduce: members`
sorts it POINTWISE, so what gets scored is a per-pixel chimera of all four -- two of which
carry no gradient at all. Set `pack_member: x1` on such a decoder; the constructor refuses
to build without it. Run `prr1iy4u` is the measurement of getting this wrong.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig

from weathergen.model.flow_math import PACK_MEMBERS
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.loss_modules.loss_module_structure import LossStructureFunction
from weathergen.train.loss_modules.spectral_patches import (
    DTCWTBands,
    check_patch_size,
    group_into_patches,
    points_to_patches,
)
from weathergen.train.loss_modules.spectral_utils import fal_fcl, phase_weight, wal_wcl
from weathergen.train.utils import TRAIN, Stage

_logger = logging.getLogger(__name__)

# Coordinates are quantised to this many degrees to look a point up in the source
# raster. Far coarser than float32 noise on a ~340 deg longitude (~2e-5 deg) and far
# finer than the grid spacing (>= 0.045 deg for CERRA), so the mapping is exact and
# collision-free.
_COORD_QUANT = 1e-3


class LossSpectralWFCL(LossModuleBase):
    """WFCL over contiguous native-resolution patches of a stream's source grid.

    Config (under ``loss_fcts``)::

        loss_fcts:
          "wfcl":
            target_stream: CERRA
            grid_width: 1069
            grid_height: 1069
            patch_size: 64
            coords_zarr: /e/data1/slmet/ml_training/cerra-....zarr
            channels: ['tp', '10si']
            total_steps: 4096          # REQUIRED; see below
            J: 3
            beta: 1.0
            gamma: [1.0, 1.0, 1.0]
            alpha: 0.1
            facl_mode: expected
            reduce: members     # real ensembles only
            pack_member: x1     # REQUIRED on a ResidualFlow decoder; see below

    ``total_steps`` must be set explicitly. Deriving it from
    ``num_mini_epochs * samples_per_mini_epoch`` is wrong on this codebase, because
    ``num_mini_epochs`` is an absolute stop point rather than a count -- on a
    continuation that inflates the horizon, P_t barely moves, and the run trains in
    permanently phase-only mode without ever reaching the amplitude phase.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.mode_cfg = mode_cfg
        self.stage = stage
        self.device = device
        self.name = "LossSpectralWFCL"

        cfg_key = next(iter(loss_fcts.keys()))
        cfg = loss_fcts[cfg_key]

        self.target_stream: str = cfg["target_stream"]
        self.grid_width: int = int(cfg["grid_width"])
        self.grid_height: int = int(cfg["grid_height"])
        self.patch_size: int = int(cfg["patch_size"])
        self.coords_zarr: str = str(cfg["coords_zarr"])

        self.j: int = int(cfg.get("J", 3))
        self.beta: float = float(cfg.get("beta", 1.0))
        self.alpha: float = float(cfg.get("alpha", 0.1))
        gamma = cfg.get("gamma", [1.0] * self.j)
        self.gamma: list[float] = [float(g) for g in gamma]
        if len(self.gamma) != self.j:
            raise ValueError(f"gamma has {len(self.gamma)} entries, expected J={self.j}")

        if "total_steps" not in cfg:
            raise ValueError(
                "LossSpectralWFCL requires an explicit `total_steps`. Deriving it from "
                "num_mini_epochs is wrong here (num_mini_epochs is an absolute stop point, "
                "not a count), and an inflated horizon silently pins P_t near 1 so the run "
                "never reaches the amplitude phase."
            )
        self.total_steps: int = int(cfg["total_steps"])
        # `istep` is absolute and survives train_continue, so on a finetune it already sits at
        # the parent's final step. Left alone, the ramp would be nearly over before this run
        # takes its first gradient step (a parent at 128/160 mini-epochs starts at P_t = 0.11,
        # i.e. amplitude-only -- the exact opposite of the intended schedule). Set this to the
        # parent's final istep so the ramp spans the finetune window instead.
        self.schedule_start_step: int = int(cfg.get("schedule_start_step", 0))
        if self.schedule_start_step >= self.total_steps:
            raise ValueError(
                f"schedule_start_step={self.schedule_start_step} is not before "
                f"total_steps={self.total_steps}; the schedule would be empty."
            )

        self.facl_mode: str = str(cfg.get("facl_mode", "expected"))
        if self.facl_mode not in ("expected", "sampled"):
            raise ValueError(f"facl_mode must be 'expected' or 'sampled', got {self.facl_mode}")
        self.use_fourier: bool = bool(cfg.get("use_fourier", True))
        self.use_wavelet: bool = bool(cfg.get("use_wavelet", True))

        self.channels: list[str] | None = (
            [str(c) for c in cfg["channels"]] if cfg.get("channels") else None
        )
        self._channel_idx: list[int] | None = None
        red = cfg.get("reduce", "members")
        self.reduce = red if red in ("mean", "median", "members") else int(red)

        # *** THE PACK. See `weathergen.model.flow_math` PACK_* and run `prr1iy4u`. ***
        # A ResidualFlow decoder hands the TRAINING loss a pack of heterogeneous slots, not an
        # ensemble. `reduce` is meaningless on it and `members` is destructive. `pack_member`
        # names the slot to score; `x1` is the only correct answer for a spectral loss, because
        # the CFM slot carries the raw source y0 at full amplitude and so is noise-dominated in
        # exactly the fine bands this loss constrains (1.63x the true residual at 11-22 km).
        # At VALIDATION the same decoder returns real samples ([flow_ens_size, N, C]), so the
        # slot index is meaningless there and `reduce` applies as usual. The term reaches the
        # validation calculator by config merge whether or not the arm file mentions it, so this
        # is not optional: kc5oigof's first attempt died in its first validation pass with
        # `IndexError: index 3 is out of bounds for dimension 0 with size 2`, before any
        # training step, on all three chained jobs.
        pm = cfg.get("pack_member", None)
        self.pack_member: int | None = (
            None if pm is None else (PACK_MEMBERS[pm] if isinstance(pm, str) else int(pm))
        )
        if self.pack_member is not None and self.stage != TRAIN:
            _logger.info(
                "LossSpectralWFCL: pack_member=%s ignored at stage=%s (real samples there; "
                "reduce=%s applies)",
                pm,
                self.stage,
                red,
            )
            self.pack_member = None
        decoder_type = str(cf.get("decoder_type", ""))
        if decoder_type == "ResidualFlow" and self.stage == TRAIN and self.pack_member is None:
            raise ValueError(
                "LossSpectralWFCL on a ResidualFlow decoder must set `pack_member` (use "
                "`pack_member: x1`). Its training prediction is a heterogeneous pack -- "
                f"{list(PACK_MEMBERS)} -- and `reduce: {red}` would score a pointwise sort of "
                "all four slots, two of which carry no gradient (mu is frozen under "
                "flow_freeze_base, r_scale is a buffer). That is what run prr1iy4u measured."
            )
        if self.pack_member is not None and decoder_type != "ResidualFlow":
            raise ValueError(
                f"`pack_member` is only meaningful for a ResidualFlow decoder, got "
                f"decoder_type='{decoder_type}'. Use `reduce` for a real ensemble."
            )

        check_patch_size(self.patch_size, self.j)
        self.bands = DTCWTBands(
            self.j,
            biort=str(cfg.get("biort", "near_sym_b")),
            qshift=str(cfg.get("qshift", "qshift_b")),
        ).to(device)

        self._coord_lut: tuple[torch.Tensor, torch.Tensor] | None = None
        self._warned_match = False

        _logger.info(
            "LossSpectralWFCL: stream=%s grid=%dx%d patch=%d J=%d beta=%.3g gamma=%s "
            "alpha=%.3g total_steps=%d facl_mode=%s reduce=%s pack_member=%s channels=%s",
            self.target_stream,
            self.grid_height,
            self.grid_width,
            self.patch_size,
            self.j,
            self.beta,
            self.gamma,
            self.alpha,
            self.total_steps,
            self.facl_mode,
            self.reduce,
            self.pack_member,
            self.channels,
        )

    # ------------------------------------------------------------------
    # native-index recovery
    # ------------------------------------------------------------------

    @staticmethod
    def _coord_key(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """Collision-free int64 key for a (lat, lon) pair on the source raster.

        Both sides are forced through float32 before quantising, because that is the
        precision the reader delivers. Quantising a float64 and its float32 cast can
        land either side of a bin edge, and the consequence is out of all proportion
        to the size of the error: a patch needs all 4096 of its points, so even a
        0.3% miss rate leaves essentially no complete patches (0.997^4096 ~ 4e-6)
        and the loss quietly sees nothing at all.

        The inputs must already have been through the reader's own lat/lon transform
        -- see :meth:`_lut`. Getting that wrong is exactly how run hlhzsm6b/ztpf511o
        ended up at a 99.7% match rate and starved.
        """
        lat_q = torch.round((lat.float().double() + 90.0) / _COORD_QUANT).to(torch.int64)
        lon_q = torch.round((lon.float().double() % 360.0) / _COORD_QUANT).to(torch.int64)
        return lat_q * 400_000 + lon_q

    def _lut(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sorted (key, native_index) table for the whole source raster, built once.

        Recovering the native index from coordinates keeps the patch scheme entirely
        inside the loss -- no extra field has to be threaded through the tokenizer,
        the model and the batch structure just to survive the round trip.

        *** The zarr's raw coordinates are NOT what arrives at the loss. ***
        ``data_reader_anemoi`` stores ``_clip_lat(ds.latitudes)`` and
        ``_clip_lon(ds.longitudes)``, and ``_clip_lon`` both re-bases longitude from
        [0, 360) to [-180, 180) AND casts to float32. Keying this table off the raw
        float64 arrays therefore disagrees with the query by up to a float32 ulp
        (~3e-5 deg at lon 342), which straddles the quantisation bin for ~0.3% of
        points. We reuse the reader's own functions rather than reimplementing them
        so the two cannot drift apart again.
        """
        if self._coord_lut is None:
            import zarr

            from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon

            z = zarr.open(self.coords_zarr, mode="r")
            lat = torch.from_numpy(_clip_lat(np.asarray(z["latitudes"][:])))
            lon = torch.from_numpy(_clip_lon(np.asarray(z["longitudes"][:])))
            n = lat.numel()
            if n != self.grid_width * self.grid_height:
                raise ValueError(
                    f"{self.coords_zarr} has {n} points, config says "
                    f"{self.grid_height}x{self.grid_width}={self.grid_width * self.grid_height}"
                )
            keys = self._coord_key(lat, lon)
            order = torch.argsort(keys)
            self._coord_lut = (
                keys[order].to(self.device),
                order.to(torch.int64).to(self.device),
            )
            _logger.info("LossSpectralWFCL: built coord->index table for %d points", n)
        return self._coord_lut

    def _native_idx(self, coords: torch.Tensor) -> torch.Tensor:
        """Map (P, 2) lat/lon onto flat source-raster indices; -1 where unmatched."""
        keys_sorted, idx_sorted = self._lut()
        q = self._coord_key(coords[:, 0], coords[:, 1])
        pos = torch.searchsorted(keys_sorted, q).clamp_max(keys_sorted.numel() - 1)
        hit = keys_sorted[pos] == q
        return torch.where(hit, idx_sorted[pos], torch.full_like(pos, -1))

    def _log_keys(self) -> list[str]:
        """The exact key set logged every step, on every rank, regardless of content."""
        keys = ["loss"]
        if self.use_fourier:
            keys += ["FAL", "FCL", "FACL"]
        if self.use_wavelet:
            keys += ["WACL"]
            for level in range(1, self.j + 1):
                keys += [f"WAL_L{level}", f"WCL_L{level}"]
                keys += [f"WCL_L{level}_k{k}" for k in range(1, 7)]
        return keys

    @staticmethod
    def _time_key(times, num_points: int, device) -> torch.Tensor:
        """Per-point integer timestamp, for splitting a window's snapshots apart."""
        if times is None:
            return torch.zeros(num_points, dtype=torch.long, device=device)
        if isinstance(times, torch.Tensor):
            t = times.reshape(times.shape[0], -1)[:, 0] if times.ndim > 1 else times
            return t.to(torch.long).to(device)
        arr = np.asarray(times)
        if arr.dtype.kind == "M":
            arr = arr.astype("datetime64[s]").astype("int64")
        arr = arr.reshape(arr.shape[0], -1)[:, 0] if arr.ndim > 1 else arr
        return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.long).to(device)

    def _resolve_channel_idx(self) -> list[int] | None:
        if self.channels is None:
            return None
        if self._channel_idx is None:
            stream_info = self.cf.streams[self.target_stream]
            names = list(
                stream_info.val_target_channels
                if self.stage == "val"
                else stream_info.train_target_channels
            )
            self._channel_idx = [names.index(c) for c in self.channels]
        return self._channel_idx

    # ------------------------------------------------------------------
    # the loss itself
    # ------------------------------------------------------------------

    def _wfcl(
        self, pred_patches: torch.Tensor, target_patches: torch.Tensor, p_t: float
    ) -> tuple[torch.Tensor, dict]:
        """Composite loss over ``(K, C, S, S)`` patch rasters."""
        loss = pred_patches.new_zeros(())
        logs: dict[str, torch.Tensor] = {}

        if self.use_fourier:
            fal, fcl = fal_fcl(pred_patches, target_patches)
            fal, fcl = fal.mean(), fcl.mean()
            if self.facl_mode == "sampled":
                # One draw for the whole step. Under DDP every rank must take the same
                # branch or the gradients being averaged come from different objectives.
                use_fcl = float(torch.rand((), device=pred_patches.device)) <= p_t
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    flag = torch.tensor([1.0 if use_fcl else 0.0], device=pred_patches.device)
                    torch.distributed.broadcast(flag, src=0)
                    use_fcl = bool(flag.item())
                facl = fcl if use_fcl else fal
            else:
                facl = (1.0 - p_t) * fal + p_t * fcl
            loss = loss + self.beta * facl
            logs |= {"FAL": fal.detach(), "FCL": fcl.detach(), "FACL": facl.detach()}

        if self.use_wavelet:
            wal_l, wcl_l = wal_wcl(self.bands(pred_patches), self.bands(target_patches))
            wav = pred_patches.new_zeros(())
            for level in range(self.j):
                # mean over (K, C, 6) is exactly the paper's (1/6) sum over orientations
                wal, wcl = wal_l[level].mean(), wcl_l[level].mean()
                wacl = (1.0 - p_t) * wal + p_t * wcl
                wav = wav + self.gamma[level] * wacl
                logs |= {
                    f"WAL_L{level + 1}": wal.detach(),
                    f"WCL_L{level + 1}": wcl.detach(),
                }
                # per-orientation phase error: the diagnostic a scalar SF ratio cannot give
                for k in range(6):
                    logs[f"WCL_L{level + 1}_k{k + 1}"] = wcl_l[level][..., k].mean().detach()
            loss = loss + wav
            logs["WACL"] = wav.detach()

        logs["loss"] = loss.detach()
        return loss, logs

    def compute_loss(self, preds, targets, metadata) -> LossValues:
        _source2target_idxs, output_info, _target2source_idxs, _target_info = metadata

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        agg: dict[str, list[torch.Tensor]] = defaultdict(list)
        per_step: dict[str, torch.Tensor] = {}
        ctr_steps = 0
        n_patches_seen = 0

        stream_name = self.target_stream
        p_t = phase_weight(
            int(self.cf.general.istep) - self.schedule_start_step,
            self.total_steps - self.schedule_start_step,
            self.alpha,
        )
        ch_idx = self._resolve_channel_idx()

        for timestep_idx, (preds_cur, target_cur) in enumerate(
            zip(preds.physical, targets.physical, strict=True)
        ):
            if stream_name not in target_cur:
                continue
            preds_batch = preds_cur.get(stream_name, [])
            if not preds_batch:
                continue

            targets_batch = target_cur[stream_name]["target"]
            coords_batch = target_cur[stream_name]["target_coords"]
            times_batch = target_cur[stream_name]["target_times"]
            targets_params = target_cur[stream_name]["target_metda_data"]
            targets_is_spoof = target_cur[stream_name]["is_spoof"]

            loss_ts = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_b = 0

            for pred, pred_params in zip(preds_batch, output_info, strict=True):
                target_idx_native = pred_params.global_params.get("correspondence", -1)
                match = [
                    i
                    for i, t in enumerate(targets_params)
                    if t[stream_name].global_params["idx"] == target_idx_native
                ]
                if len(match) != 1 or targets_is_spoof[match[0]]:
                    continue
                t_idx = match[0]

                target = targets_batch[t_idx]
                coords = coords_batch[t_idx]
                if target.shape[0] == 0 or pred.shape[0] == 0:
                    continue

                pred = pred.reshape([pred.shape[0], *target.shape])  # (ens, P, C)
                coords = coords.to(target.device, non_blocking=True)

                valid = torch.isfinite(target).all(-1) & torch.isfinite(coords).all(-1)
                if int(valid.sum()) < self.patch_size**2:
                    continue

                native = self._native_idx(coords[valid])
                found = native >= 0
                # A few unmatched points destroy whole patches, so a degraded match
                # rate must be loud rather than showing up as "no patches this batch".
                rate = float(found.float().mean())
                if rate < 0.999 and not self._warned_match:
                    self._warned_match = True
                    _logger.error(
                        "LossSpectralWFCL: only %.3f%% of target coordinates matched the "
                        "source raster in %s. Patches need every point, so this will starve "
                        "the loss. Check grid_width/grid_height and that coords_zarr is the "
                        "same dataset the stream reads.",
                        100.0 * rate,
                        self.coords_zarr,
                    )
                if int(found.sum()) < self.patch_size**2:
                    continue

                # *** Split by timestamp, not just by grid cell. ***
                # tokenize_spacetime puts every snapshot of the target window in the
                # same array, so a grid cell appears once PER TIME. Grouping on the
                # cell alone gives 8192-point "patches" that fail the completeness
                # test; grouping on (time, cell) gives one clean raster per snapshot.
                tkey = self._time_key(times_batch[t_idx], target.shape[0], target.device)
                gather, _pids = group_into_patches(
                    native[found],
                    self.grid_width,
                    self.patch_size,
                    group_key=tkey[valid][found],
                )
                if gather.shape[0] == 0:
                    continue
                n_patches_seen += gather.shape[0]

                sel = torch.nonzero(valid, as_tuple=True)[0][found]
                target_sel = target[sel]
                fields = LossStructureFunction.ensemble_fields(
                    pred[:, sel], self.reduce, pack_member=self.pack_member
                )
                if ch_idx is not None:
                    target_sel = target_sel[:, ch_idx]
                    fields = [f[:, ch_idx] for f in fields]

                # never run FFT/DTCWT under autocast: fp16 underflows the normalised
                # correlation denominators and torch.fft is useless there anyway
                with torch.autocast(device_type=target.device.type, enabled=False):
                    tp_grid = points_to_patches(target_sel.float(), gather)
                    member_losses, member_logs = [], []
                    for field in fields:
                        pp_grid = points_to_patches(field.float(), gather)
                        lo, lg = self._wfcl(pp_grid, tp_grid, p_t)
                        member_losses.append(lo)
                        member_logs.append(lg)

                loss_ts = loss_ts + torch.stack(member_losses).mean()
                ctr_b += 1
                for key in member_logs[0]:
                    agg[key].append(torch.stack([m[key] for m in member_logs]).mean())

            if ctr_b > 0:
                loss = loss + loss_ts / ctr_b
                per_step[str(timestep_idx)] = (loss_ts / ctr_b).detach()
                ctr_steps += 1

        if ctr_steps > 0:
            loss = loss / ctr_steps
        else:
            _logger.warning(
                "LossSpectralWFCL: no complete %dx%d patches for stream '%s' in this batch. "
                "Is `target_patches` set on the stream, with a matching patch_size?",
                self.patch_size,
                self.patch_size,
                stream_name,
            )

        reordered = defaultdict(dict)
        reordered[stream_name] = defaultdict(lambda: defaultdict(dict))
        for step_str, val in per_step.items():
            reordered[stream_name]["wfcl"]["loss"][step_str] = val
        # A rank that saw no usable patches would otherwise emit fewer keys than its
        # peers, and prepare_losses_for_logging all_reduces per key -- mismatched key
        # sets across ranks deadlock or crash the collective. Emit the full set always.
        for key in self._log_keys():
            vals = agg.get(key)
            reordered[stream_name]["wfcl"][key]["avg"] = (
                torch.stack(vals).mean() if vals else torch.zeros((), device=self.device)
            )
        # P_t is the whole story behind a moving loss value; log it every step.
        # These are all_reduced by ddp_average, which is NCCL: a CPU tensor raises
        # "No backend type associated with device type cpu" and kills the run at the
        # first validation log.
        reordered[stream_name]["wfcl"]["P_t"]["avg"] = torch.tensor(p_t, device=self.device)
        reordered[stream_name]["wfcl"]["num_patches"]["avg"] = torch.tensor(
            float(n_patches_seen), device=self.device
        )
        if per_step:
            reordered[stream_name]["wfcl"]["avg"] = loss.detach()

        return LossValues(loss=loss, losses_all=reordered, stddev_all=None)
