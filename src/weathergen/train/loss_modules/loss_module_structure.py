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
            channels: ['tp'],            # optional: restrict to these target channels (by name);
                                         # omit for all channels. Circular channels (10wdir) give
                                         # spurious increments under |.|^p -- exclude them.
            num_pairs: 262144,
            bin_edges_km: [10, 25, 50, 100, 200],
            increment_power: 2.0,        # 2 = standard structure function; 1 = robust (heavy tails)
            reduce: mean,                # ensemble handling, see below
            pair_sampling: uniform,      # 'uniform' (default, comparable) | 'stratified'
          },
        },
      }

``reduce`` (ensemble handling; all identical for ens_size=1):
  - ``mean``    : ensemble mean field. For quantile heads this is the conditional-mean product
                  (re-blurred) -- right for a comparability METRIC, wrong for ens>1 TRAINING.
  - ``median``  : per-point sorted middle (mean of the two central sorted members for even K).
                  The correct single-field product of a quantile head -- raw head indices carry
                  no quantile identity (heads do not self-order under the pinball sort).
  - ``members`` : per-point sort, then the SF loss is computed for EVERY sorted member and
                  averaged -- pushes realistic spatial texture into each quantile field.
  - int         : a single RAW member index (pre-sort). Only meaningful when members have fixed
                  identities (e.g. genuine ensemble runs), not for quantile heads.

Note on scale: choose ``bin_edges_km`` to straddle the artifact scale of the *target* stream;
on a ~1° stream (IMERG_ANEMOI) the sub-cell scales are unresolved -- prefer a fine stream.

Note on ``num_pairs`` and ``pair_sampling`` (IMPORTANT): with ``pair_sampling: uniform`` pairs are
sampled uniformly over points, so their distance distribution follows the domain's pair-distance
density -- short separations are RARE. Measured on the CERRA Europe domain (~6.9e6 km^2 of points,
1.14e6 points per time slice), ``num_pairs: 262144`` yields roughly **6 / 22 / 89 / 358** pairs for
the default 10-25 / 25-50 / 50-100 / 100-200 km bins. The finest bin therefore falls below
``min_pairs_per_bin`` and is dropped, and the next one is estimated from ~20 heavy-tailed
increments, i.e. it is noise. Any conclusion drawn from ``ratio_10_25km`` or ``ratio_25_50km`` of a
uniform-sampled run -- including the "25-50 km anomaly" seen across the CERRA campaign -- has to be
re-checked before it is believed.

``pair_sampling: stratified`` removes the problem by filling every bin to ``num_pairs``
independently; it costs a sort per bin and is the recommended setting for new runs. ``uniform``
remains the default so that every run since vbm9r3om stays comparable. The realised per-bin pair
count is logged next to each ratio (``npairs_<lo>_<hi>km``) when ``log_bin_ratios`` is on.

Validation-only usage: since the loss calculator skips terms with ``weight: 0`` and the
validation config is merged ON TOP of the training config, adding this block under
``validation_config.losses`` (with any weight > 0) computes it as a validation metric without
affecting training gradients -- the clean way to score a pure-MSE A/B for sharpness.
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


def _bucket_pairs(
    coords: torch.Tensor,
    lo: float,
    hi: float,
    num_pairs: int,
    generator: torch.Generator | None,
    max_rounds: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Point pairs whose separation lands in (lo, hi] km, without a spatial index.

    Uniform pair sampling is hopeless at short separations: on a continental domain the fraction
    of point pairs closer than 25 km is ~1e-5, so a 262144-pair draw yields single digits. This
    instead bins points onto a planar km grid of side ``hi/1.5``, pairs each anchor with a random
    point from a randomly chosen neighbouring bucket, and keeps whatever lands in the bin. Every
    step is a sort or a gather, so it runs on the GPU next to the model with no new dependency.

    The planar approximation (degrees -> km with a per-point cos(lat) factor) is only used to
    assign buckets; the distances that decide membership are exact haversine.
    """
    lat, lon = coords[:, 0], coords[:, 1]
    side = hi / 1.5
    by = torch.floor(lat * 111.0 / side).long()
    bx = torch.floor(lon * 111.0 * torch.cos(torch.deg2rad(lat)) / side).long()
    by = by - by.min()
    bx = bx - bx.min()
    width = int(bx.max().item()) + 1
    key = by * width + bx

    order = torch.argsort(key)
    key_sorted = key[order]
    uniq, counts = torch.unique_consecutive(key_sorted, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts

    keep_i, keep_j, found = [], [], 0
    for _ in range(max_rounds):
        n = num_pairs
        i = torch.randint(0, coords.shape[0], (n,), device=coords.device, generator=generator)
        # a random neighbouring bucket, including the anchor's own
        dy = torch.randint(-1, 2, (n,), device=coords.device, generator=generator)
        dx = torch.randint(-1, 2, (n,), device=coords.device, generator=generator)
        want = key[i] + dy * width + dx
        loc = torch.searchsorted(uniq, want.clamp(min=0))
        loc = loc.clamp(max=len(uniq) - 1)
        hit = uniq[loc] == want
        off = (
            torch.rand(n, device=coords.device, generator=generator) * counts[loc].to(torch.float32)
        ).long()
        j = order[(starts[loc] + off.clamp(max=counts[loc] - 1)).clamp(0, len(order) - 1)]
        dist = _haversine_km(coords, i, j)
        ok = hit & (dist > lo) & (dist <= hi) & (i != j)
        keep_i.append(i[ok])
        keep_j.append(j[ok])
        found += int(ok.sum())
        if found >= num_pairs:
            break
    return torch.cat(keep_i)[:num_pairs], torch.cat(keep_j)[:num_pairs]


def _bin_term(
    target: torch.Tensor,
    pred: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    increment_power: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squared log-ratio of the predicted and target mean increment for one distance bin."""
    inc_t = (target[idx_i] - target[idx_j]).abs().pow(increment_power)  # [P, C]
    inc_p = (pred[idx_i] - pred[idx_j]).abs().pow(increment_power)  # [P, C]
    s_t = inc_t.mean(0)  # [C]
    s_p = inc_p.mean(0)  # [C]
    return (torch.log(s_p + eps) - torch.log(s_t + eps)).pow(2), s_p, s_t


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
    return_spectra: bool = False,
    pair_sampling: str = "uniform",
) -> torch.Tensor | None | tuple:
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
        return_spectra: also return the per-bin S_pred, S_target and pair count. The scalar loss
            is a SQUARED LOG RATIO and is therefore blind to the sign of the mismatch -- too much
            and too little fine-scale variance score identically. Only the ratio S_pred/S_target
            tells you which, and that is what decides whether a fix should add or remove
            variance. See playground/docs/flow_matching_decoder.md.
        pair_sampling: ``uniform`` (default) draws pairs uniformly over points, which is what
            every run since vbm9r3om used -- keep it for comparability. It severely under-samples
            short separations: on a continental domain 262144 pairs put ~6 into a 10-25 km bin,
            below any usable ``min_pairs_per_bin``, so the finest bins are noise or are skipped.
            ``stratified`` fills each bin to ``num_pairs`` independently (see ``_bucket_pairs``),
            which is what the offline tool playground/scripts/decoder_forensics.py does.
    Returns:
        (C,) per-channel loss, or None if too few points/pairs to form any bin.
        With ``return_spectra``: ``(loss, s_pred, s_target, counts)`` where the spectra are
        ``(B, C)`` with NaN in bins that had too few pairs and ``counts`` is ``(B,)``.
    """
    empty = (None, None, None, None) if return_spectra else None
    n = target.shape[0]
    if n < 2:
        return empty

    device = target.device
    num_bins, num_ch = len(bin_edges_km) - 1, target.shape[1]

    # per-bin (idx_i, idx_j); the uniform branch draws once and masks, so its RNG consumption
    # and its arithmetic are unchanged from the original implementation
    bin_pairs: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    if pair_sampling == "uniform":
        idx_i = torch.randint(0, n, (num_pairs,), device=device, generator=generator)
        idx_j = torch.randint(0, n, (num_pairs,), device=device, generator=generator)
        dist = _haversine_km(coords, idx_i, idx_j)
        lo, hi = float(bin_edges_km[0]), float(bin_edges_km[-1])
        keep = (dist > lo) & (dist <= hi) & (idx_i != idx_j)
        if int(keep.sum()) < min_pairs_per_bin:
            return empty
        idx_i, idx_j, dist = idx_i[keep], idx_j[keep], dist[keep]
        for b in range(num_bins):
            in_bin = (dist > float(bin_edges_km[b])) & (dist <= float(bin_edges_km[b + 1]))
            bin_pairs.append((idx_i[in_bin], idx_j[in_bin]))
    elif pair_sampling == "stratified":
        for b in range(num_bins):
            bin_pairs.append(
                _bucket_pairs(
                    coords,
                    float(bin_edges_km[b]),
                    float(bin_edges_km[b + 1]),
                    num_pairs,
                    generator,
                )
            )
    else:
        raise ValueError(f"unknown pair_sampling '{pair_sampling}' (use 'uniform'/'stratified')")

    losses = []
    counts = torch.zeros(num_bins, dtype=torch.int64)
    if return_spectra:
        nan = float("nan")
        sp_all = torch.full((num_bins, num_ch), nan, device=device, dtype=torch.float32)
        st_all = torch.full((num_bins, num_ch), nan, device=device, dtype=torch.float32)

    for b, pair in enumerate(bin_pairs):
        i_b, j_b = pair
        counts[b] = len(i_b)
        if len(i_b) < min_pairs_per_bin:
            # a starved bin is a measurement failure, not a no-op: the loss silently stops
            # covering that scale. Say so rather than dropping it.
            _logger.warning(
                "LossStructureFunction: bin %g-%g km got %d pairs (< %d) and is not scored; "
                "raise num_pairs or set pair_sampling: stratified",
                bin_edges_km[b],
                bin_edges_km[b + 1],
                len(i_b),
                min_pairs_per_bin,
            )
            continue
        loss_b, s_p, s_t = _bin_term(target, pred, i_b, j_b, increment_power, eps)
        losses.append(loss_b)
        if return_spectra:
            sp_all[b] = s_p.detach().float()
            st_all[b] = s_t.detach().float()

    if not losses:
        return empty
    loss = torch.stack(losses, 0).mean(0)  # [C]
    return (loss, sp_all, st_all, counts) if return_spectra else loss


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
        # optional list of target-channel NAMES to restrict to (e.g. ['tp']); None = all
        self.channels: list[str] | None = (
            [str(c) for c in cfg["channels"]] if cfg.get("channels") else None
        )
        self._channel_idx: list[int] | None = None  # resolved lazily from stream channel names
        # see module docstring: uniform pair sampling starves the short-distance bins on large
        # domains, so the default is deliberately high (cost is negligible)
        self.num_pairs: int = int(cfg.get("num_pairs", 262144))
        self.min_pairs_per_bin: int = int(cfg.get("min_pairs_per_bin", 8))
        default_bins = [10, 25, 50, 100, 200]
        self.bin_edges_km: list[float] = [float(x) for x in cfg.get("bin_edges_km", default_bins)]
        self.increment_power: float = float(cfg.get("increment_power", 2.0))
        self.eps: float = float(cfg.get("eps", 1e-6))
        red = cfg.get("reduce", "mean")
        self.reduce = red if red in ("mean", "median", "members") else int(red)
        # Emit per-bin S_pred/S_target alongside the scalar loss. Off by default so existing
        # runs log exactly what they always did. Diagnostic only -- it never affects the loss.
        self.log_bin_ratios: bool = bool(cfg.get("log_bin_ratios", False))
        self._bin_ratio_sum: torch.Tensor | None = None
        self._bin_ratio_cnt: torch.Tensor | None = None
        self._bin_pair_sum: torch.Tensor | None = None
        # 'uniform' reproduces every run since vbm9r3om exactly; 'stratified' fills each distance
        # bin independently so the short-separation bins are actually measured. See the function.
        self.pair_sampling: str = str(cfg.get("pair_sampling", "uniform"))

        # Results are keyed by self.name in LossCalculator.compute_loss, so two
        # LossStructureFunction terms in one config would overwrite each other's log entry (the
        # loss values are still both summed -- only the reported breakdown is lost). Suffixing by
        # reduce mode lets e.g. a mean-SF and a members-SF coexist, which is how the generative
        # arm reads coherence. 'mean' deliberately keeps the bare name so the campaign-comparable
        # metric key is byte-identical to every run since vbm9r3om.
        if self.reduce != "mean":
            self.name = f"LossStructureFunction_{self.reduce}"

        _logger.info(
            "LossStructureFunction: stream=%s, channels=%s, bins(km)=%s, num_pairs=%d, "
            "power=%.1f, reduce=%s, pair_sampling=%s",
            self.target_stream,
            self.channels if self.channels is not None else "all",
            self.bin_edges_km,
            self.num_pairs,
            self.increment_power,
            self.reduce,
            self.pair_sampling,
        )

    def _resolve_channel_idx(self) -> list[int] | None:
        """Map configured channel names to column indices of the target stream (cached)."""
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

    @staticmethod
    def ensemble_fields(pred: torch.Tensor, reduce) -> list[torch.Tensor]:
        """Resolve the ensemble dim into the field(s) the SF loss scores.

        pred: (ens, N, C). Returns a list of (N, C) fields; the loss is averaged over them.
        For ens_size=1 every mode returns the single member. See the module docstring for the
        semantics of "mean" / "median" / "members" / int.
        """
        k = pred.shape[0]
        if k == 1:
            return [pred[0]]
        if reduce == "mean":
            return [pred.mean(0)]
        if reduce == "median":
            srt = pred.sort(0).values
            if k % 2 == 0:
                return [srt[k // 2 - 1 : k // 2 + 1].mean(0)]
            return [srt[k // 2]]
        if reduce == "members":
            srt = pred.sort(0).values
            return [srt[m] for m in range(k)]
        return [pred[reduce]]

    def compute_loss(self, preds, targets, metadata) -> LossValues:
        _source2target_idxs, output_info, _target2source_idxs, _target_info = metadata

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        per_step = {}
        ctr_steps = 0
        stream_name = self.target_stream

        if self.log_bin_ratios:
            nb = len(self.bin_edges_km) - 1
            self._bin_ratio_sum = torch.zeros(nb, device=self.device)
            self._bin_ratio_cnt = torch.zeros(nb, device=self.device)
            self._bin_pair_sum = torch.zeros(nb, device=self.device)

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
                fields = self.ensemble_fields(pred, self.reduce)  # list of [N, C]

                # target_coords_raw is produced on CPU by the dataloader; move it first
                coords = coords.to(target.device, non_blocking=True)
                valid = torch.isfinite(target).all(-1) & torch.isfinite(coords).all(-1)
                if int(valid.sum()) < 2:
                    continue

                target_sel = target[valid]
                ch_idx = self._resolve_channel_idx()
                if ch_idx is not None:
                    target_sel = target_sel[:, ch_idx]

                # average the SF loss over the resolved field(s); one independent pair sample
                # per field keeps the same-pair pred/target cancellation within each call
                loss_fields = []
                for field in fields:
                    pred_sel = field[valid]
                    if ch_idx is not None:
                        pred_sel = pred_sel[:, ch_idx]
                    out = structure_function_loss(
                        target_sel,
                        pred_sel,
                        coords[valid],
                        bin_edges_km=self.bin_edges_km,
                        num_pairs=self.num_pairs,
                        increment_power=self.increment_power,
                        eps=self.eps,
                        min_pairs_per_bin=self.min_pairs_per_bin,
                        return_spectra=self.log_bin_ratios,
                        pair_sampling=self.pair_sampling,
                    )
                    if self.log_bin_ratios:
                        loss_ch, s_p, s_t, n_pairs = out
                        if s_p is not None:
                            # channel-mean ratio per bin; accumulated so the logged value is an
                            # average over the whole validation pass, not one batch
                            ratio = (s_p / s_t.clamp_min(self.eps)).nanmean(-1)  # [B]
                            fin = torch.isfinite(ratio)
                            self._bin_ratio_sum[fin] += ratio[fin]
                            self._bin_ratio_cnt[fin] += 1
                            self._bin_pair_sum += n_pairs.to(self._bin_pair_sum)
                    else:
                        loss_ch = out
                    if loss_ch is not None:
                        loss_fields.append(loss_ch.mean())
                if not loss_fields:
                    continue

                loss_ts = loss_ts + torch.stack(loss_fields).mean()
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

        # Per-bin S_pred/S_target. The scalar loss above is a squared log ratio and cannot say
        # whether the prediction has too much or too little variance at a given scale; this can.
        # > 1 = too much fine-scale variance (over-sharp / noisy), < 1 = too smooth (blurred).
        if self.log_bin_ratios and float(self._bin_ratio_cnt.sum()) > 0:
            ratios = self._bin_ratio_sum / self._bin_ratio_cnt.clamp_min(1)
            pairs = self._bin_pair_sum / self._bin_ratio_cnt.clamp_min(1)
            for b in range(len(self.bin_edges_km) - 1):
                if float(self._bin_ratio_cnt[b]) == 0:
                    continue
                lo, hi = int(self.bin_edges_km[b]), int(self.bin_edges_km[b + 1])
                reordered[stream_name]["structure"][f"ratio_{lo}_{hi}km"] = ratios[b].detach()
                # a ratio estimated from a handful of pairs is noise, not a measurement; log the
                # sample size next to it so the two are never read apart
                reordered[stream_name]["structure"][f"npairs_{lo}_{hi}km"] = pairs[b].detach()

        return LossValues(loss=loss, losses_all=reordered, stddev_all=None)
