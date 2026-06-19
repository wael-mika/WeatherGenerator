# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Structure-function (variogram) loss for precipitation fields.

This is the grid-free real-space twin of a power-spectrum / WFCL loss. It penalises a
mismatch between the predicted and target *second-order structure function*

    S(r) = E[ ( y(x) - y(x+r) )^p ]   as a function of separation distance r,

estimated from random pairs of target points binned by great-circle distance. Matching S(r)
across distance bins is, via Wiener-Khinchin, equivalent to matching the spatial power
spectrum -- but with no FFT and no regular grid, so it fits the WeatherGenerator decoder's
scattered per-point output natively and avoids the failure modes of the gridded WFCL
(resolution ceiling, zero-fill artifacts -- see docs/decoder_assessment.md §7 and the WFCL
post-mortem). A blurry / over-smoothed prediction has too-small short-range increments, which
this loss detects and penalises directly; MSE (a first-order, pointwise statistic) cannot.

Pairs with the log-ratio objective per distance bin so the *shape* of S(r) is matched and
short-range (high-frequency) bins are not drowned out by the large-scale bins.

Plug-in pattern mirrors LossPhysical: a loss *module* (not a loss-fct) because it needs the
raw point coordinates. Select via the training config with ``target_and_aux_calc: Physical``.

Expected config (under ``training_config.losses``):

.. code-block:: yaml

    "structure": {
        type: LossStructureFunction,
        weight: 0.1,
        target_and_aux_calc: Physical,
        loss_fcts: {
          "struct": {
            target_stream: CERRA,        # apply on a FINE stream (CERRA tp 5.5km / native IMERG)
            num_pairs: 8192,
            bin_edges_km: [10, 25, 50, 100, 200],
            increment_power: 2.0,        # 2 = standard structure function; 1 = robust (heavy tails)
            reduce: mean,                # ensemble reduction: "mean" (== member when ens=1) or int
          },
        },
      }

Note on scale: choose ``bin_edges_km`` to straddle the artifact scale of the *target* stream;
on a ~1° stream (IMERG_ANEMOI) the sub-cell scales are unresolved -- prefer a fine stream.
"""

import logging
from collections import defaultdict

import torch
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


def _haversine_km(coords: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
    """Great-circle distance (km) between point pairs given (lat, lon) in degrees.

    Args:
        coords: (N, 2) tensor of (latitude, longitude) in degrees.
        idx_i, idx_j: (P,) long index tensors selecting the two endpoints of each pair.
    Returns:
        (P,) distances in km.
    """
    earth_radius_km = 6371.0
    lat = torch.deg2rad(coords[:, 0])
    lon = torch.deg2rad(coords[:, 1])
    lat1, lat2 = lat[idx_i], lat[idx_j]
    dlat = lat2 - lat1
    dlon = lon[idx_j] - lon[idx_i]
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2.0 * earth_radius_km * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def structure_function_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    coords: torch.Tensor,
    bin_edges_km: list[float],
    num_pairs: int = 8192,
    increment_power: float = 2.0,
    eps: float = 1e-6,
    min_pairs_per_bin: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Per-channel grid-free structure-function loss for one (reduced) field.

    Samples ``num_pairs`` random point pairs, bins them by great-circle distance into the bins
    defined by ``bin_edges_km``, and matches the predicted vs target mean increment per bin in
    log space: loss = mean_{bin, channel} ( log S_pred - log S_target )^2.

    Args:
        target: (N, C) target values (already restricted to valid finite points).
        pred:   (N, C) predicted values (single field; ensemble already reduced by the caller).
        coords: (N, 2) (lat, lon) in degrees for the same N points.
        bin_edges_km: ascending list of B+1 edges; bins are (e0, e1], ..., (e_{B-1}, e_B].
        num_pairs: number of random pairs to sample.
        increment_power: p in |y_i - y_j|^p (2 = standard; 1 = robust for heavy tails).
        eps: stabiliser inside the log.
        min_pairs_per_bin: skip bins with fewer surviving pairs than this.
        generator: optional RNG for reproducible pair sampling.
    Returns:
        (C,) per-channel loss, or None if too few points/pairs to form any bin.
    """
    n = target.shape[0]
    if n < 2:
        return None

    device = target.device
    idx_i = torch.randint(0, n, (num_pairs,), device=device, generator=generator)
    idx_j = torch.randint(0, n, (num_pairs,), device=device, generator=generator)

    dist = _haversine_km(coords, idx_i, idx_j)
    lo, hi = float(bin_edges_km[0]), float(bin_edges_km[-1])
    keep = (dist > lo) & (dist <= hi) & (idx_i != idx_j)
    if int(keep.sum()) < min_pairs_per_bin:
        return None
    idx_i, idx_j, dist = idx_i[keep], idx_j[keep], dist[keep]

    # increments (target side carries no gradient; pred side does, via the index gather)
    inc_t = (target[idx_i] - target[idx_j]).abs().pow(increment_power)  # [P, C]
    inc_p = (pred[idx_i] - pred[idx_j]).abs().pow(increment_power)  # [P, C]

    losses = []
    for b in range(len(bin_edges_km) - 1):
        in_bin = (dist > float(bin_edges_km[b])) & (dist <= float(bin_edges_km[b + 1]))
        if int(in_bin.sum()) < min_pairs_per_bin:
            continue
        s_t = inc_t[in_bin].mean(0)  # [C]
        s_p = inc_p[in_bin].mean(0)  # [C]
        losses.append((torch.log(s_p + eps) - torch.log(s_t + eps)).pow(2))

    if not losses:
        return None
    return torch.stack(losses, 0).mean(0)  # [C]


class LossStructureFunction(LossModuleBase):
    """Grid-free structure-function (variogram) loss module. See module docstring."""

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
        self.name = "LossStructureFunction"

        # exactly one config key (e.g. "struct"), mirroring LossSpectralWFCL
        cfg_key = next(iter(loss_fcts.keys()))
        cfg = loss_fcts[cfg_key]

        self.target_stream: str = cfg["target_stream"]
        self.num_pairs: int = int(cfg.get("num_pairs", 8192))
        self.min_pairs_per_bin: int = int(cfg.get("min_pairs_per_bin", 8))
        default_bins = [10, 25, 50, 100, 200]
        self.bin_edges_km: list[float] = [float(x) for x in cfg.get("bin_edges_km", default_bins)]
        self.increment_power: float = float(cfg.get("increment_power", 2.0))
        self.eps: float = float(cfg.get("eps", 1e-6))
        red = cfg.get("reduce", "mean")
        self.reduce = red if red == "mean" else int(red)

        _logger.info(
            "LossStructureFunction: stream=%s, bins(km)=%s, num_pairs=%d, power=%.1f, reduce=%s",
            self.target_stream,
            self.bin_edges_km,
            self.num_pairs,
            self.increment_power,
            self.reduce,
        )

    def _reduce_ensemble(self, pred: torch.Tensor) -> torch.Tensor:
        """pred: (ens, N, C) -> (N, C). For ens_size=1 the mean equals the single member."""
        if self.reduce == "mean":
            return pred.mean(0)
        return pred[self.reduce]

    def compute_loss(self, preds, targets, metadata) -> LossValues:
        _source2target_idxs, output_info, _target2source_idxs, _target_info = metadata

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        per_step = {}
        ctr_steps = 0
        stream_name = self.target_stream

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
            targets_params = target_cur[stream_name]["target_metda_data"]
            targets_is_spoof = target_cur[stream_name]["is_spoof"]

            loss_ts = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_b = 0
            for pred, pred_params in zip(preds_batch, output_info, strict=True):
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

                if targets_is_spoof[target_idx]:
                    continue

                target = targets_batch[target_idx]
                coords = coords_batch[target_idx]
                if target.shape[0] == 0 or pred.shape[0] == 0:
                    continue

                pred = pred.reshape([pred.shape[0], *target.shape])  # [ens, N, C]
                pred_red = self._reduce_ensemble(pred)  # [N, C]

                valid = torch.isfinite(target).all(-1) & torch.isfinite(coords).all(-1)
                if int(valid.sum()) < 2:
                    continue

                loss_ch = structure_function_loss(
                    target[valid],
                    pred_red[valid],
                    coords[valid].to(self.device),
                    bin_edges_km=self.bin_edges_km,
                    num_pairs=self.num_pairs,
                    increment_power=self.increment_power,
                    eps=self.eps,
                    min_pairs_per_bin=self.min_pairs_per_bin,
                )
                if loss_ch is None:
                    continue

                loss_ts = loss_ts + loss_ch.mean()
                ctr_b += 1

            if ctr_b > 0:
                loss = loss + loss_ts / ctr_b
                per_step[str(timestep_idx)] = (loss_ts / ctr_b).detach()
                ctr_steps += 1

        if ctr_steps > 0:
            loss = loss / ctr_steps
        elif _logger.isEnabledFor(logging.WARNING):
            _logger.warning(
                "LossStructureFunction: no data for stream '%s' in this batch.", stream_name
            )

        # log layout mirrors LossPhysical: [stream][loss_fct][metric][step]
        reordered = defaultdict(dict)
        reordered[stream_name] = defaultdict(lambda: defaultdict(dict))
        for step_str, val in per_step.items():
            reordered[stream_name]["structure"]["loss"][step_str] = val
        if "structure" in reordered[stream_name]:
            reordered[stream_name]["structure"]["avg"] = loss.detach()

        return LossValues(loss=loss, losses_all=reordered, stddev_all=None)
