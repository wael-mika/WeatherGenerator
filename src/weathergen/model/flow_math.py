# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pure-torch pieces of the flow-matching decoders.

Everything here is deliberately free of ``flash_attn`` (and of any import that pulls it in), so
the CPU unit-test suite can exercise the algebra that the decoders rest on. ``engines.py`` and
``decode_residual_flow.py`` re-export from here; nothing in this module knows about the model.
"""

import math

import torch


def sample_flow_time(
    n: int,
    device,
    dtype,
    mode: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Draw the flow time ``t`` for conditional flow-matching training -> ``[n, 1]`` in (0, 1).

    ``uniform`` is the textbook choice and the default. ``logit_normal`` --
    ``t = sigmoid(N(mean, std))``, Esser et al. 2024 (SD3) -- concentrates training on the middle
    of the path, where the velocity is hardest to predict and where sample quality is decided,
    and spends less on the two ends where the task is nearly trivial. It is one of the
    best-established quality wins in the flow-matching literature and costs nothing at inference.
    """
    if mode == "uniform":
        return torch.rand((n, 1), device=device, dtype=dtype)
    if mode == "logit_normal":
        z = torch.randn((n, 1), device=device, dtype=dtype) * logit_std + logit_mean
        return torch.sigmoid(z)
    raise ValueError(f"unknown flow_time_sampling '{mode}' (use 'uniform' or 'logit_normal')")


def sample_flow_time_for_points(
    num_points: int,
    output_lens: torch.Tensor,
    device,
    dtype,
    per_cell: bool = False,
    mode: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Flow time for the training path -> ``[num_points, 1]``.

    Default (``per_cell=False``) draws an independent ``t`` for every target point. That is
    **inconsistent with the sampler**, which advances every point of a cell together on one shared
    ``t``. Two consequences, both bad for a decoder whose purpose is to emit a joint sample:

    * the states the ODE visits (all coordinates at the same ``t``) have vanishing probability
      under a training distribution of iid per-point times, so the velocity field is evaluated
      far outside the region it was fitted on;
    * during training a point's cell-mates sit at unrelated noise levels, some of them nearly
      clean, so the cheapest way to predict a velocity is to copy from a neighbour that has
      already resolved. No such neighbour exists at sampling time, and the learned coupling is
      then inert.

    With ``per_cell=True``, ``t`` is drawn once per decode cell and broadcast over that cell's
    points, which is exactly the state the sampler produces. Cells stay independent -- they are
    independent in the network too (varlen groups, per-cell KV), and one draw per cell keeps
    hundreds of distinct times per step, so the time marginal is still well covered.
    """
    if not per_cell:
        return sample_flow_time(
            num_points, device, dtype, mode=mode, logit_mean=logit_mean, logit_std=logit_std
        )
    # output_lens is a per-group count vector with a leading 0 (the attention cumsums it)
    counts = output_lens[1:].to(torch.long)
    assert int(counts.sum()) == num_points, (
        f"flow_time_per_cell: decode-group counts sum to {int(counts.sum())} but there are "
        f"{num_points} target points"
    )
    t_cell = sample_flow_time(
        counts.shape[0], device, dtype, mode=mode, logit_mean=logit_mean, logit_std=logit_std
    )
    return t_cell.repeat_interleave(counts, dim=0)


def flow_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the flow time ``t`` in [0, 1].

    Args:
        t   : ``[N, 1]`` flow time per point.
        dim : embedding width (even).

    Returns:
        ``[N, dim]``
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    ang = t.float() * freqs.unsqueeze(0) * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(t.dtype)


def physical_to_residual(y1: torch.Tensor, mu: torch.Tensor, r_scale: torch.Tensor) -> torch.Tensor:
    """The flow-matching target: the residual of ``y1`` about ``mu``, in units of ``r_scale``.

    **Deliberately unclamped.** The reference implementation this is adapted from clamps
    ``y1 - mu`` before scaling, which silently breaks the identity in ``residual_to_physical``:
    the probability path would be built from a clamped residual while the MSE regresses against
    the *unclamped* ``y1``, training the network toward a target it never sees on the path.
    """
    return (y1 - mu) / r_scale


def residual_to_physical(
    mu: torch.Tensor, y0: torch.Tensor, v: torch.Tensor, r_scale: torch.Tensor
) -> torch.Tensor:
    """Pack the velocity prediction so that a plain MSE against ``y1`` *is* the CFM objective.

    With ``r := (y1 - mu) / r_scale`` the conditional-flow-matching target velocity on the linear
    path is ``u = r - y0``. Returning ``mu + r_scale * (y0 + v)`` gives

        ||y1 - (mu + r_scale*(y0 + v))||^2  =  ||(y1 - mu) - r_scale*(y0 + v)||^2
                                           =  r_scale^2 * ||(r - y0) - v||^2

    i.e. the CFM loss scaled by ``r_scale^2``, with no new loss function and with the existing
    channel/point weighting and NaN masking applied for free. Callers pass ``mu.detach()`` here so
    the corrector can never blur the deterministic branch.

    ``r_scale`` broadcasts as ``[C]`` over ``[N, C]``. The identity holds per channel, so a
    per-channel ``r_scale`` weights each channel's CFM term by its residual variance -- which is
    exactly what makes the two loss terms commensurate (see ``ResidualScale``).
    """
    return mu + r_scale * (y0 + v)


class ResidualScale(torch.nn.Module):
    """Per-channel residual scale, calibrated online and carried in the ``state_dict``.

    Setting ``r_scale_c = std(residual_c)`` is what makes the deterministic and flow loss terms
    commensurate: the flow term is ``Var(residual_c) * CFM_c`` while the deterministic term is
    ``~Var(residual_c)`` at convergence, so their ratio -- the effective ``alpha`` -- is the same
    for every channel, and the existing channel weighting keeps its meaning.

    A single global scalar does not survive this data. Measured CERRA residual stds span 0.32 (2t)
    to 0.85 (tp); a scalar tuned for tp leaves 2t's normalised residual at std ~0.4, which is the
    regime where the target is small against the unit-variance noise and the velocity degenerates
    toward ``-y0``.

    A buffer beats a config value because it self-calibrates, because it rides in the checkpoint
    so evaluation integrates with exactly the scale training used, and because DDP's
    ``broadcast_buffers=True`` keeps ranks identical with no all-reduce.

    It is its own module because ``load_model_state`` re-initialises missing checkpoint keys by
    calling ``to_empty()`` + ``reset_parameters()`` on the highest-level module covering them; a
    bare buffer on the branch would make that root the whole branch, which has no
    ``reset_parameters``.
    """

    def __init__(
        self,
        num_channels: int,
        fixed: float | None = None,
        ema: float = 0.999,
        freeze_after: int = 1000,
        floor: float = 1e-2,
        ceil: float = 1e2,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.fixed = fixed
        self.ema = ema
        self.freeze_after = freeze_after
        self.floor = floor
        self.ceil = ceil
        self.register_buffer("scale", torch.ones(num_channels))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.scale.fill_(1.0)
        self.count.zero_()

    @torch.no_grad()
    def update(self, residual: torch.Tensor, finite_mask: torch.Tensor) -> None:
        """EMA the per-channel RMS of the *physical* residual over finite points only.

        Frozen after ``freeze_after`` updates so the loss weighting is stationary and the training
        curves are readable.
        """
        if self.fixed is not None or int(self.count) >= self.freeze_after:
            return
        counts = finite_mask.sum(0).clamp(min=1)
        rms = ((residual * finite_mask).pow(2).sum(0) / counts).sqrt()
        # ignore channels with no finite points this step
        seen = finite_mask.any(0)
        obs = torch.where(seen, rms, self.scale)
        # start from the first observation rather than crawling up from 1.0
        w = 0.0 if int(self.count) == 0 else self.ema
        self.scale.mul_(w).add_(obs * (1.0 - w))
        self.count.add_(1)

    def value(self) -> torch.Tensor:
        """The scale to divide residuals by -> ``[C]``."""
        if self.fixed is not None:
            return torch.full_like(self.scale, self.fixed)
        return self.scale.clamp(self.floor, self.ceil)
