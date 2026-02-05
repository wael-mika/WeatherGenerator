# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch
import torch.nn.functional as F

stat_loss_fcts = ["stats", "kernel_crps"]  # Names of loss functions that need std computed

LOSS_CONFIG: dict = {}


def set_loss_config(cfg: dict | None) -> None:
    global LOSS_CONFIG
    LOSS_CONFIG = cfg or {}


def _get_loss_cfg(name: str) -> dict:
    cfg = LOSS_CONFIG.get(name, {})
    return cfg if isinstance(cfg, dict) else {}


def _get_thresholds_tensor(target: torch.Tensor, cfg: dict) -> torch.Tensor:
    thresholds = cfg.get("thresholds", None)
    thresholds_mm = cfg.get("thresholds_mm", None)
    if thresholds is None and thresholds_mm is None:
        thresholds = [10.0]
    if thresholds_mm is not None:
        x = torch.tensor(thresholds_mm, device=target.device, dtype=target.dtype)
        transform_type = cfg.get("transform_type", "none")
        alpha = float(cfg.get("transform_alpha", 0.15))
        eps = float(cfg.get("transform_eps", 2.39e-7))
        offset = float(cfg.get("transform_offset", 1.0))
        mu = cfg.get("transform_mu", None)
        sigma = cfg.get("transform_sigma", None)

        if transform_type == "arcsinh":
            y = torch.asinh(x / alpha)
        elif transform_type == "log10":
            y = torch.log10(x + offset)
        elif transform_type == "log_eps":
            y = torch.log((x + eps) / eps)
        elif transform_type == "none":
            y = x
        else:
            y = x

        if mu is not None and sigma is not None and float(sigma) > 0:
            y = (y - float(mu)) / float(sigma)

        return y

    return torch.tensor(thresholds, device=target.device, dtype=target.dtype)


def _intensity_weights_from_target(
    target: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    mode = cfg.get("mode", "log1p")
    alpha = float(cfg.get("alpha", 1.0))
    power = float(cfg.get("power", 1.0))
    scale = float(cfg.get("scale", 1.0))
    min_weight = float(cfg.get("min_weight", 1.0))
    max_weight = cfg.get("max_weight", None)

    vals = torch.abs(target)
    vals = vals.mean(dim=-1)  # [num_points]
    vals = vals * scale

    if mode == "log1p":
        w = torch.log1p(vals)
        if power != 1.0:
            w = w**power
    elif mode == "power":
        w = vals**power
    elif mode == "linear":
        w = vals
    else:
        w = vals

    w = 1.0 + alpha * w
    if max_weight is not None:
        w = torch.clamp(w, min=min_weight, max=float(max_weight))
    else:
        w = torch.clamp(w, min=min_weight)

    return w


def gaussian(x, mu=0.0, std_dev=1.0):
    # unnormalized Gaussian where maximum is one
    return torch.exp(-0.5 * (x - mu) * (x - mu) / (std_dev * std_dev))


def normalized_gaussian(x, mu=0.0, std_dev=1.0):
    return (1 / (std_dev * np.sqrt(2.0 * np.pi))) * torch.exp(
        -0.5 * (x - mu) * (x - mu) / (std_dev * std_dev)
    )


def erf(x, mu=0.0, std_dev=1.0):
    c1 = torch.sqrt(torch.tensor(0.5 * np.pi))
    c2 = torch.sqrt(1.0 / torch.tensor(std_dev * std_dev))
    c3 = torch.sqrt(torch.tensor(2.0))
    val = c1 * (1.0 / c2 - std_dev * torch.special.erf((mu - x) / (c3 * std_dev)))
    return val


def gaussian_crps(target, ens, mu, stddev):
    # see Eq. A2 in S. Rasp and S. Lerch. Neural networks for postprocessing ensemble weather
    # forecasts. Monthly Weather Review, 146(11):3885 – 3900, 2018.
    c1 = np.sqrt(1.0 / np.pi)
    t1 = 2.0 * erf((target - mu) / stddev) - 1.0
    t2 = 2.0 * normalized_gaussian((target - mu) / stddev)
    val = stddev * ((target - mu) / stddev * t1 + t2 - c1)
    return torch.mean(val)  # + torch.mean( torch.sqrt( stddev) )


def stats(target, ens, mu, stddev):
    diff = gaussian(target, mu, stddev) - 1.0
    return torch.mean(diff * diff) + torch.mean(torch.sqrt(stddev))


def stats_normalized(target, ens, mu, stddev):
    a = normalized_gaussian(target, mu, stddev)
    max = 1 / (np.sqrt(2 * np.pi) * stddev)
    d = a - max
    return torch.mean(d * d) + torch.mean(torch.sqrt(stddev))


def stats_normalized_erf(target, ens, mu, stddev):
    delta = -torch.abs(target - mu)
    d = 0.5 + torch.special.erf(delta / (np.sqrt(2.0) * stddev))
    return torch.mean(d * d)  # + torch.mean( torch.sqrt( stddev) )


def mse(target, ens, mu, *kwargs):
    return torch.nn.functional.mse_loss(target, mu)


def mse_ens(target, ens, mu, stddev):
    mse_loss = torch.nn.functional.mse_loss
    return torch.stack([mse_loss(target, mem) for mem in ens], 0).mean()


def kernel_crps(
    targets,
    preds,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    fair=True,
):
    """
    Compute kernel CRPS

    Params:
    target : shape ( num_data_points , num_channels )
    pred : shape ( ens_dim , num_data_points , num_channels)
    weights_channels : shape = (num_channels,)
    weights_points : shape = (num_data_points)

    Returns:
    loss: scalar - overall weighted CRPS
    loss_chs: [C] - per-channel CRPS (location-weighted, not channel-weighted)
    """

    ens_size = preds.shape[0]
    assert ens_size > 1, "Ensemble size has to be greater than 1 for kernel CRPS."
    assert len(preds.shape) == 3, "if data has batch dimension, remove unsqueeze() below"

    # replace NaN by 0
    mask_nan = ~torch.isnan(targets)
    targets = torch.where(mask_nan, targets, 0)
    preds = torch.where(mask_nan, preds, 0)

    # permute to enable/simply broadcasting and contractions below
    preds = preds.permute([2, 1, 0]).unsqueeze(0).to(torch.float32)
    targets = targets.permute([1, 0]).unsqueeze(0).to(torch.float32)

    mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)

    ens_n = -1.0 / (ens_size * (ens_size - 1)) if fair else -1.0 / (ens_size**2)
    abs = torch.abs
    ens_var = torch.zeros(size=preds.shape[:-1], device=preds.device)
    # loop to reduce memory usage
    for i in range(ens_size):
        ens_var += torch.sum(ens_n * abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)

    kcrps_locs_chs = mae + ens_var

    # apply point weighting
    if weights_points is not None:
        kcrps_locs_chs = kcrps_locs_chs * weights_points
    # apply channel weighting
    kcrps_chs = torch.mean(torch.mean(kcrps_locs_chs, 0), -1)
    if weights_channels is not None:
        kcrps_chs = kcrps_chs * weights_channels

    return torch.mean(kcrps_chs), kcrps_chs


def mse_channel_location_weighted(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Compute weighted MSE loss for one window or step

    The function implements:

    loss = Mean_{channels}( weight_channels * Mean_{data_pts}( (target - pred) * weights_points ))

    Geometrically,

        ------------------------     -
        |                      |    |  |
        |                      |    |  |
        |                      |    |  |
        |     target - pred    | x  |wp|
        |                      |    |  |
        |                      |    |  |
        |                      |    |  |
        ------------------------     -
                    x
        ------------------------
        |          wc          |
        ------------------------

    where wp = weights_points and wc = weights_channels and "x" denotes row/col-wise multiplication.

    The computations are:
    1. weight the rows of (target - pred) by wp = weights_points
    2. take the mean over the row
    3. weight the collapsed cols by wc = weights_channels
    4. take the mean over the channel-weighted cols

    Params:
        target : shape ( num_data_points , num_channels )
        target : shape ( ens_dim , num_data_points , num_channels)
        weights_channels : shape = (num_channels,)
        weights_points : shape = (num_data_points)

    Return:
        loss : weight loss for gradient computation
        loss_chs : losses per channel with location weighting but no channel weighting
    """

    mask_nan = ~torch.isnan(target)
    pred = pred[0] if pred.shape[0] == 0 else pred.mean(0)

    diff2 = torch.square(torch.where(mask_nan, target, 0) - torch.where(mask_nan, pred, 0))
    if weights_points is not None:
        diff2 = (diff2.transpose(1, 0) * weights_points).transpose(1, 0)
    loss_chs = diff2.mean(0)
    loss = torch.mean(loss_chs * weights_channels if weights_channels is not None else loss_chs)

    return loss, loss_chs


def mse_intensity_weighted(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Compute intensity-weighted MSE loss for one window or step.

    Weights are computed from target intensity (in transformed space) to
    emphasize extremes. Optional location weights are combined multiplicatively.
    """
    cfg = _get_loss_cfg("intensity_weighted_mse")

    mask_nan = ~torch.isnan(target)
    pred = pred[0] if pred.shape[0] == 0 else pred.mean(0)

    target_filled = torch.where(mask_nan, target, 0)
    pred_filled = torch.where(mask_nan, pred, 0)

    weights_intensity = _intensity_weights_from_target(target_filled, cfg)
    if weights_points is not None:
        weights_intensity = weights_intensity * weights_points

    diff2 = torch.square(target_filled - pred_filled)
    diff2 = (diff2.transpose(1, 0) * weights_intensity).transpose(1, 0)

    loss_chs = diff2.mean(0)
    loss = torch.mean(loss_chs * weights_channels if weights_channels is not None else loss_chs)

    return loss, loss_chs


def soft_exceedance(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Differentiable exceedance loss using logistic smoothing.

    Supports threshold-specific weights via config:
        threshold_weights: [w1, w2, ...] matching thresholds_mm order
    Higher thresholds (rare events) can get higher weight to prioritize them.
    """
    cfg = _get_loss_cfg("soft_exceedance")
    temperature = float(cfg.get("temperature", 1.0))
    thresholds = _get_thresholds_tensor(target, cfg)

    # Threshold-specific weights (optional)
    threshold_weights_list = cfg.get("threshold_weights", None)
    if threshold_weights_list is not None:
        threshold_weights = torch.tensor(
            threshold_weights_list, device=target.device, dtype=target.dtype
        )
    else:
        threshold_weights = None

    mask_nan = ~torch.isnan(target)
    pred = pred[0] if pred.shape[0] == 0 else pred.mean(0)

    target_filled = torch.where(mask_nan, target, 0)
    pred_filled = torch.where(mask_nan, pred, 0)

    losses_chs = torch.zeros(target.shape[-1], device=target.device, dtype=target.dtype)
    total_weight = 0.0

    for i, thr in enumerate(thresholds):
        w = threshold_weights[i].item() if threshold_weights is not None else 1.0
        total_weight += w

        labels = (target_filled >= thr).to(dtype=target.dtype)
        logits = (pred_filled - thr) / temperature
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        if weights_points is not None:
            bce = (bce.transpose(1, 0) * weights_points).transpose(1, 0)

        losses_chs += w * bce.mean(0)

    losses_chs /= total_weight
    loss = torch.mean(losses_chs * weights_channels if weights_channels is not None else losses_chs)

    return loss, losses_chs


def quantile_upper(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
):
    """
    Pinball loss for upper quantiles to capture distribution tail.

    Config options (via loss_config.quantile_upper):
        quantiles: [0.9, 0.95, 0.99] - which quantiles to optimize
        weights: [1.0, 2.0, 4.0] - weight per quantile (higher = more extreme)

    Pinball loss formula: q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)
    This asymmetrically penalizes underprediction for high quantiles.
    """
    cfg = _get_loss_cfg("quantile_upper")
    quantiles = cfg.get("quantiles", [0.9, 0.95, 0.99])
    quantile_weights = cfg.get("weights", [1.0] * len(quantiles))

    mask_nan = ~torch.isnan(target)
    pred_mean = pred[0] if pred.shape[0] == 0 else pred.mean(0)

    target_filled = torch.where(mask_nan, target, 0)
    pred_filled = torch.where(mask_nan, pred_mean, 0)
    mask_float = mask_nan.float()

    # diff > 0 means target > pred (underprediction)
    diff = target_filled - pred_filled

    total_loss = torch.zeros(target.shape[-1], device=target.device, dtype=target.dtype)
    total_weight = 0.0

    for q, w in zip(quantiles, quantile_weights):
        # Pinball: q * ReLU(diff) + (1-q) * ReLU(-diff)
        pinball = q * torch.clamp(diff, min=0) + (1 - q) * torch.clamp(-diff, min=0)

        if weights_points is not None:
            pinball = (pinball.transpose(1, 0) * weights_points).transpose(1, 0)

        # Mask invalid points and average
        pinball = pinball * mask_float
        denom = mask_float.sum(0).clamp(min=1)
        total_loss += w * (pinball.sum(0) / denom)
        total_weight += w

    total_loss /= total_weight
    loss = torch.mean(total_loss * weights_channels if weights_channels is not None else total_loss)

    return loss, total_loss


def centroid_shift(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights_channels: torch.Tensor | None,
    weights_points: torch.Tensor | None,
    coords: torch.Tensor | None,
):
    """
    Centroid shift loss based on soft exceedance weights at specified thresholds.

    Config options (via loss_config.centroid_shift):
        thresholds_mm: [20.0] - thresholds in mm (transformed to model space)
        threshold_weights: [1.0] - optional weights per threshold
        temperature: 1.0 - sigmoid temperature for soft exceedance
        min_points: 5 - minimum hard exceedances required for both target/pred
        scale_km: 1000.0 - divide distance (km) by this to normalize magnitude
        earth_radius_km: 6371.0 - radius used for distance
    """
    cfg = _get_loss_cfg("centroid_shift")
    thresholds = _get_thresholds_tensor(target, cfg)
    temperature = float(cfg.get("temperature", 1.0))
    min_points = int(cfg.get("min_points", 5))
    scale_km = float(cfg.get("scale_km", 1000.0))
    earth_r = float(cfg.get("earth_radius_km", 6371.0))

    threshold_weights_list = cfg.get("threshold_weights", None)
    if threshold_weights_list is not None:
        threshold_weights = torch.tensor(
            threshold_weights_list, device=target.device, dtype=target.dtype
        )
    else:
        threshold_weights = None

    if coords is None or coords.numel() == 0:
        loss_chs = torch.zeros(target.shape[-1], device=target.device, dtype=target.dtype)
        loss = torch.mean(loss_chs * weights_channels if weights_channels is not None else loss_chs)
        return loss, loss_chs

    # coords are [num_points, 2] => lat, lon in degrees
    coords = coords.to(device=target.device, dtype=target.dtype)
    lat = coords[:, 0] * np.pi / 180.0
    lon = coords[:, 1] * np.pi / 180.0
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    vec = torch.stack([x, y, z], dim=-1)  # [N, 3]

    mask_nan = ~torch.isnan(target)
    pred_mean = pred[0] if pred.shape[0] == 0 else pred.mean(0)
    target_filled = torch.where(mask_nan, target, 0)
    pred_filled = torch.where(mask_nan, pred_mean, 0)

    n_ch = target.shape[-1]
    loss_chs = torch.zeros(n_ch, device=target.device, dtype=target.dtype)
    weight_chs = torch.zeros(n_ch, device=target.device, dtype=target.dtype)

    for i, thr in enumerate(thresholds):
        w_thr = threshold_weights[i].item() if threshold_weights is not None else 1.0
        for c in range(n_ch):
            t = target_filled[:, c]
            p = pred_filled[:, c]
            valid = mask_nan[:, c]

            if valid.sum() < min_points:
                continue

            # Hard exceedance counts for stability
            if (t[valid] >= thr).sum() < min_points or (p[valid] >= thr).sum() < min_points:
                continue

            wt = torch.sigmoid((t - thr) / temperature) * valid
            wp = torch.sigmoid((p - thr) / temperature) * valid

            if weights_points is not None:
                wt = wt * weights_points
                wp = wp * weights_points

            sum_wt = wt.sum()
            sum_wp = wp.sum()
            if sum_wt <= 0 or sum_wp <= 0:
                continue

            ct = (wt[:, None] * vec).sum(0) / (sum_wt + 1e-6)
            cp = (wp[:, None] * vec).sum(0) / (sum_wp + 1e-6)

            ct = ct / (ct.norm() + 1e-6)
            cp = cp / (cp.norm() + 1e-6)

            dot = (ct * cp).sum().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            angle = torch.acos(dot)
            dist = earth_r * angle / scale_km

            loss_chs[c] += w_thr * dist
            weight_chs[c] += w_thr

    weight_chs = torch.where(weight_chs > 0, weight_chs, torch.ones_like(weight_chs))
    loss_chs = loss_chs / weight_chs
    loss = torch.mean(loss_chs * weights_channels if weights_channels is not None else loss_chs)

    return loss, loss_chs


# Signals to the loss calculator that coords are required.
centroid_shift.requires_coords = True


def cosine_latitude(stream_data, forecast_offset, fstep, min_value=1e-3, max_value=1.0):
    latitudes_radian = stream_data.target_coords_raw[forecast_offset + fstep][:, 0] * np.pi / 180
    return (max_value - min_value) * np.cos(latitudes_radian) + min_value


def gamma_decay(forecast_steps, gamma):
    fsteps = np.arange(forecast_steps)
    weights = gamma**fsteps
    return weights * (len(fsteps) / np.sum(weights))
