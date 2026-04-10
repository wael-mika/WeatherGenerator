# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Wavelet-Fourier Composite Loss (WFCL) module for precipitation fields.

Combines Fourier-domain amplitude/correlation losses with DTCWT wavelet-domain
multi-scale losses.  Operates by reconstructing a regular lat/lon grid from the
flat HEALPix/O96 point arrays that the WeatherGenerator uses internally.

Reference: the loss formulation follows the WFCL paper for precipitation
downscaling, adapted for spherical grids.
"""

import logging
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_wavelets import DTCWTForward

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.loss_modules.spectral_utils import (
    build_grid_mapping,
    compute_pt,
    dtcwt_decompose,
    flat_to_grid,
    fourier_amplitude_loss,
    fourier_correlation_loss,
    wavelet_amplitude_loss,
    wavelet_correlation_loss,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


class LossSpectralWFCL(LossModuleBase):
    """Wavelet-Fourier Composite Loss for precipitation fields.

    This loss module reconstructs a regular 2-D grid from the flat point arrays
    used by the WeatherGenerator, then computes Fourier and wavelet spectral
    losses that encourage the model to produce predictions with realistic
    spatial structure at all scales.

    Configuration is passed via the ``loss_fcts`` mechanism in the YAML config.
    Expected config structure (under a key like ``"wfcl"``):

    .. code-block:: yaml

        loss_fcts:
          "wfcl":
            target_stream: "IMERG_ANEMOI"
            grid_H: 181
            grid_W: 360
            beta: 1.0
            alpha: 0.1
            total_steps: 16384
            wavelet_levels: 3
            gamma_levels: [1, 1, 1]
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

        # Extract config -- there should be exactly one key in loss_fcts
        cfg_key = list(loss_fcts.keys())[0]
        cfg = loss_fcts[cfg_key]

        self.target_stream: str = cfg["target_stream"]
        self.grid_h: int = cfg.get("grid_H", 181)
        self.grid_w: int = cfg.get("grid_W", 360)
        self.beta: float = cfg.get("beta", 1.0)
        self.alpha: float = cfg.get("alpha", 0.1)
        self.wavelet_levels: int = cfg.get("wavelet_levels", 3)

        # Per-level weights, normalised to sum to 1
        gamma = np.array(cfg.get("gamma_levels", [1.0] * self.wavelet_levels), dtype=np.float64)
        gamma = gamma / gamma.sum()
        self.gamma_levels = gamma.tolist()

        # Total training steps for P_t schedule.  Prefer explicit config;
        # fall back to num_mini_epochs * samples_per_mini_epoch (≈ total samples,
        # a reasonable proxy when batch_size is not accessible here).
        if "total_steps" in cfg:
            self.total_steps: int = int(cfg["total_steps"])
        else:
            n_epochs = mode_cfg.get("num_mini_epochs", 1)
            spe = mode_cfg.get("samples_per_mini_epoch", 1)
            self.total_steps = int(n_epochs * spe)
            _logger.info(
                "LossSpectralWFCL: total_steps not set explicitly, "
                "using num_mini_epochs * samples_per_mini_epoch = %d",
                self.total_steps,
            )

        # Initialise DTCWT forward transform (stateless, no learnable params)
        self.dtcwt_fwd = DTCWTForward(J=self.wavelet_levels, biort="near_sym_b", qshift="qshift_b")
        # Move DTCWT filters to device
        self.dtcwt_fwd = self.dtcwt_fwd.to(device)

        # Grid mapping cache -- built lazily on first forward pass
        self._grid_mapping: tuple | None = None

        _logger.info(
            "LossSpectralWFCL initialised: stream=%s, grid=(%d,%d), "
            "wavelet_levels=%d, beta=%.2f, alpha=%.2f, total_steps=%d",
            self.target_stream,
            self.grid_h,
            self.grid_w,
            self.wavelet_levels,
            self.beta,
            self.alpha,
            self.total_steps,
        )

    # ------------------------------------------------------------------
    # Grid mapping
    # ------------------------------------------------------------------

    def _get_or_build_grid_mapping(self, coords: torch.Tensor):
        """Return cached grid mapping or build it from coordinates."""
        if self._grid_mapping is None:
            row, col, counts = build_grid_mapping(coords, self.grid_h, self.grid_w)
            self._grid_mapping = (
                row.to(self.device),
                col.to(self.device),
                counts.to(self.device),
            )
        return self._grid_mapping

    # ------------------------------------------------------------------
    # Core WFCL computation
    # ------------------------------------------------------------------

    def _compute_wfcl(
        self,
        target_grid: torch.Tensor,
        pred_grid: torch.Tensor,
        current_step: int,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the full WFCL loss on regular grids.

        Args:
            target_grid: (C, H, W) float32 tensor.
            pred_grid: (C, H, W) float32 tensor.
            current_step: current global training step.

        Returns:
            loss: scalar tensor (with grad).
            details: dict of component loss values (detached) for logging.
        """
        # -- Fourier losses --
        fal = fourier_amplitude_loss(pred_grid, target_grid)
        fcl = fourier_correlation_loss(pred_grid, target_grid)
        facl = fal + fcl

        # -- Wavelet losses --
        # DTCWT requires even spatial dimensions; pad if necessary
        _, h, w = target_grid.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            target_grid = torch.nn.functional.pad(target_grid, (0, pad_w, 0, pad_h))
            pred_grid = torch.nn.functional.pad(pred_grid, (0, pad_w, 0, pad_h))

        p_t = compute_pt(current_step, self.total_steps, self.alpha)

        # Wavelet losses only if wavelet_levels > 0
        wavelet_loss = torch.tensor(0.0, device=target_grid.device, requires_grad=True)
        if self.wavelet_levels > 0:
            h_target = dtcwt_decompose(target_grid, self.dtcwt_fwd)
            h_pred = dtcwt_decompose(pred_grid, self.dtcwt_fwd)

            for level_idx in range(self.wavelet_levels):
                wal_bands = wavelet_amplitude_loss(h_pred[level_idx], h_target[level_idx])  # (6,)
                wcl_bands = wavelet_correlation_loss(h_pred[level_idx], h_target[level_idx])  # (6,)

                # Scheduled combination per band
                wacl_bands = (1.0 - p_t) * wal_bands + p_t * wcl_bands  # (6,)

                # Average over 6 directional bands, weight by gamma
                wavelet_loss = wavelet_loss + self.gamma_levels[level_idx] * wacl_bands.mean()

        # -- Final composite --
        total_loss = self.beta * facl + wavelet_loss

        details = {
            "FAL": fal.detach(),
            "FCL": fcl.detach(),
            "FACL": facl.detach(),
            "wavelet": wavelet_loss.detach(),
            "P_t": p_t,
            "total": total_loss.detach(),
        }

        return total_loss, details

    # ------------------------------------------------------------------
    # LossModuleBase interface
    # ------------------------------------------------------------------

    def compute_loss(self, preds, targets, metadata) -> LossValues:
        """Compute spectral loss over the target stream for all forecast steps.

        Follows the same outer iteration pattern as ``LossPhysical``:
        iterate over timesteps, find target/prediction pairs via the
        correspondence mechanism, reconstruct grids, and compute WFCL.
        """
        source2target_idxs, output_info, target2source_idxs, target_info = metadata

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses_all = defaultdict(dict)
        ctr_timesteps = 0
        current_step = self.cf.general.istep

        stream_name = self.target_stream

        for timestep_idx, (preds_cur, target_cur) in enumerate(
            zip(preds.physical, targets.physical, strict=True)
        ):
            # Check if this stream has data for this timestep
            if stream_name not in target_cur:
                continue
            preds_batch = preds_cur.get(stream_name, [])
            if not preds_batch:
                continue

            targets_batch = target_cur[stream_name]["target"]
            targets_coords_batch = target_cur[stream_name]["target_coords"]
            targets_params = target_cur[stream_name]["target_metda_data"]
            targets_is_spoof = target_cur[stream_name]["is_spoof"]

            loss_timestep = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_batch = 0

            for pred, pred_params in zip(preds_batch, output_info, strict=True):
                # Match prediction to target via correspondence
                target_idx_native = pred_params.global_params.get("correspondence", -1)
                target_idx = [
                    i
                    for i, t in enumerate(targets_params)
                    if t[stream_name].global_params["idx"] == target_idx_native
                ]
                if len(target_idx) == 0:
                    continue
                assert len(target_idx) == 1
                target_idx = target_idx[0]

                # Skip spoofed inputs
                if targets_is_spoof[target_idx]:
                    continue

                target = targets_batch[target_idx]
                coords = targets_coords_batch[target_idx]

                # Skip if no data points
                if target.shape[0] == 0 or pred.shape[0] == 0:
                    continue

                # Reshape pred: (ens_dim, ...) -> (ens_dim, N, C) -> (N, C)
                pred = pred.reshape([pred.shape[0], *target.shape])
                pred_mean = pred[0] if pred.shape[0] == 0 else pred.mean(0)

                # Handle NaN
                mask_nan = ~torch.isnan(target)
                target_clean = torch.where(mask_nan, target, torch.zeros_like(target))
                pred_clean = torch.where(mask_nan, pred_mean, torch.zeros_like(pred_mean))

                # Cast to float32 for spectral operations
                target_f32 = target_clean.float()
                pred_f32 = pred_clean.float()

                # Build grid mapping (cached after first call)
                row, col, counts = self._get_or_build_grid_mapping(coords)

                # Reconstruct regular grids
                target_grid = flat_to_grid(
                    target_f32, row, col, counts, self.grid_h, self.grid_w
                )
                pred_grid = flat_to_grid(
                    pred_f32, row, col, counts, self.grid_h, self.grid_w
                )

                # Compute WFCL
                wfcl_loss, details = self._compute_wfcl(target_grid, pred_grid, current_step)

                loss_timestep = loss_timestep + wfcl_loss
                ctr_batch += 1

                # Store per-timestep details for logging
                losses_all[stream_name][str(timestep_idx)] = {
                    k: v for k, v in details.items() if k != "P_t"
                }

            if ctr_batch > 0:
                loss = loss + loss_timestep / ctr_batch
                ctr_timesteps += 1

        if ctr_timesteps > 0:
            loss = loss / ctr_timesteps

        if loss == 0.0:
            _logger.warning(
                "LossSpectralWFCL: loss is 0.0 -- check that stream '%s' "
                "has data in the current batch.",
                stream_name,
            )

        # Reorder losses_all to match LossPhysical's expected format:
        # [stream_name][loss_fct_name][metric][output_step]
        reordered = defaultdict(dict)
        reordered[stream_name] = defaultdict(lambda: defaultdict(dict))
        for step_str, step_details in losses_all.get(stream_name, {}).items():
            for metric_name, val in step_details.items():
                reordered[stream_name]["wfcl"][metric_name][step_str] = val

        # Add averages
        if "wfcl" in reordered[stream_name]:
            reordered[stream_name]["wfcl"]["avg"] = loss.detach()

        return LossValues(loss=loss, losses_all=reordered, stddev_all=None)
