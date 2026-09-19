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
    scope: str | None = None,
) -> torch.Tensor:
    """Flow time for the training path -> ``[num_points, 1]``.

    ``scope`` selects how widely one draw of ``t`` is shared. ``per_cell`` is the older boolean
    spelling and maps to ``"cell"`` / ``"point"``; pass ``scope`` to reach ``"global"``.

    **point** -- an independent ``t`` for every target point. **Falsified** (density arm
    ``f8as896p``): it is inconsistent with the sampler, which advances every point of a cell
    together on one shared ``t``. Two consequences, both bad for a decoder whose purpose is to emit
    a joint sample:

    * the states the ODE visits (all coordinates at the same ``t``) have vanishing probability
      under a training distribution of iid per-point times, so the velocity field is evaluated
      far outside the region it was fitted on;
    * during training a point's cell-mates sit at unrelated noise levels, some of them nearly
      clean, so the cheapest way to predict a velocity is to copy from a neighbour that has
      already resolved. No such neighbour exists at sampling time, and the learned coupling is
      then inert.

    **cell** -- one draw per decode cell, broadcast over that cell's points. Exactly the state the
    sampler produces *within* a cell. Its stated precondition is that **cells stay independent**:
    they are independent in the network (varlen groups, per-cell KV), so a jump in ``t`` across a
    boundary is invisible. One draw per cell keeps hundreds of distinct times per step.

    **global** -- one draw for the whole batch. Required by a spatially correlated source
    (``correlated_source``), which breaks the precondition above: with a coherent ``y0`` the state
    ``y_t = (1-t) y0 + t r`` is continuous across a boundary only if ``t`` is, so a per-cell ``t``
    puts a discontinuity into the training state itself and the network learns to reproduce it.
    Global is also strictly safer than per-cell w.r.t. the neighbour leak that sank ``point``,
    since every point shares one time, and it is what ``sample`` already does at inference.
    """
    if scope is None:
        scope = "cell" if per_cell else "point"
    assert scope in ("point", "cell", "global"), (
        f"unknown flow_time_scope '{scope}' (use 'point', 'cell' or 'global')"
    )
    if scope == "point":
        return sample_flow_time(
            num_points, device, dtype, mode=mode, logit_mean=logit_mean, logit_std=logit_std
        )
    if scope == "global":
        t = sample_flow_time(
            1, device, dtype, mode=mode, logit_mean=logit_mean, logit_std=logit_std
        )
        return t.expand(num_points, 1)
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


EARTH_RADIUS_KM = 6371.0


def latlon_to_unit_vectors(coords: torch.Tensor) -> torch.Tensor:
    """``[N, >=2]`` of (latitude, longitude) in DEGREES -> ``[N, 3]`` unit vectors.

    The column convention is the readers': ``coords[:, 0]`` is latitude and ``coords[:, 1]``
    longitude, as ``theta_phi_to_standard_coords`` in ``tokenizer_utils`` assumes. Extra columns
    are ignored.

    Unit vectors rather than degrees because the source's covariance has to be isotropic in real
    distance: a longitude difference is worth ``cos(lat)`` of a latitude difference, so a field
    built on raw degrees would be stretched east-west and would stretch differently at every
    latitude.
    """
    assert coords.ndim == 2 and coords.shape[1] >= 2, (
        f"expected [N, >=2] (lat, lon) in degrees, got {tuple(coords.shape)}"
    )
    lat = torch.deg2rad(coords[:, 0].to(torch.float32))
    lon = torch.deg2rad(coords[:, 1].to(torch.float32))
    cos_lat = torch.cos(lat)
    return torch.stack([cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)], -1)


def correlated_source(
    coords_xyz: torch.Tensor,
    num_channels: int,
    length_km: float,
    num_modes: int,
    generator: torch.Generator | None = None,
    mix: float = 1.0,
) -> torch.Tensor:
    """Flow-matching source ``y0`` as a SPATIALLY COHERENT Gaussian field -> ``[N, num_channels]``.

    **Why this exists.** The corrector's only texture mechanism is self-attention *within one
    HEALPix cell* (``cu_seqlens`` makes the mask strictly block-diagonal), so with an iid ``y0``
    neighbouring cells generate statistically independent textures. The only way to make
    independent draws agree at a shared border is to average them -- which is what soft blend does,
    and it costs 31-69% of the small-scale amplitude. Drawing ``y0`` from a field that is a smooth
    function of POSITION makes the two sides of a boundary share their noise instead, so they agree
    by construction and nothing is averaged. Measured target: 50-89% of the tiling-aligned per-cell
    artifact is the sampled component (``FINDINGS_DC_NULL_2026-09-06.md`` §2.1).

    **Construction.** Random Fourier features,
    ``y0(x) = sqrt(2/M) * sum_j cos(k_j . x + phi_j)`` with ``k_j ~ N(0, sigma^2 I_3)`` and
    ``phi_j ~ U[0, 2pi)``, evaluated on the unit sphere.

    * The marginal is **exactly** unit variance for any ``M`` (``E[cos^2] = 1/2``), which is what
      the rest of the algebra needs: ``ResidualScale`` normalises the residual to unit std, the
      ``y0 + v`` loss identity (see ``residual_to_physical``) is untouched, and the closed-form
      ``v* = (2t-1) y_t / ((1-t)^2 + t^2)`` still integrates to variance 1. CFM itself is valid for
      any source you can sample: the path and the target ``u = r - y0`` are unchanged in form.
    * The covariance is ``exp(-d^2 / (2 l^2))`` in the large-``M`` limit, ``d`` the chord distance,
      so ``length_km`` sets the coherence scale directly.
    * It carries **no tiling of its own** -- a piecewise-constant HEALPix-grid noise field would
      import a second grid, which is the artifact this is meant to remove.

    ``mix`` interpolates ``sqrt(1-a) * iid + sqrt(a) * field``, preserving unit variance;
    ``mix=0`` is the old iid behaviour exactly. Channels get independent fields.

    Args:
        coords_xyz : ``[N, 3]`` unit vectors, the points' TRUE positions (never the host-cell
                     frame -- a per-cell frame would make the field discontinuous at boundaries,
                     which is the bug this avoids).
        length_km  : coherence length. Too long starves the sample of degrees of freedom and
                     blurs it; ~2-3x the point spacing is the intended regime.
        generator  : draw the modes per member / per forward. Do NOT cache them in a buffer --
                     buffers reach checkpoints through EMA, which has already cost this campaign
                     a whole campaign (see ``ema.py``).
    """
    assert coords_xyz.ndim == 2 and coords_xyz.shape[1] == 3, (
        f"correlated_source expects [N, 3] unit vectors, got {tuple(coords_xyz.shape)}"
    )
    assert 0.0 <= mix <= 1.0, f"flow_source_mix must be in [0, 1], got {mix}"
    assert length_km > 0.0 and num_modes >= 1

    n = coords_xyz.shape[0]
    device, dtype = coords_xyz.device, torch.float32
    kw = {"device": device, "dtype": dtype, "generator": generator}

    iid = torch.randn((n, num_channels), **kw)
    if mix == 0.0:
        return iid

    # sigma is the inverse coherence length expressed on the UNIT sphere
    sigma = EARTH_RADIUS_KM / float(length_km)
    x = coords_xyz.to(dtype)
    field = torch.empty((n, num_channels), device=device, dtype=dtype)
    for c in range(num_channels):
        k = torch.randn((3, num_modes), **kw) * sigma
        phi = torch.rand((1, num_modes), **kw) * (2.0 * math.pi)
        field[:, c] = math.sqrt(2.0 / num_modes) * torch.cos(x @ k + phi).sum(-1)

    if mix == 1.0:
        return field
    return math.sqrt(1.0 - mix) * iid + math.sqrt(mix) * field


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


def dominant_replica(idx: torch.Tensor, w: torch.Tensor, n_state: int) -> torch.Tensor:
    """Row of each original point's highest-weight soft-blend replica -> ``[n_state]`` long.

    Soft blend replicates a point into several host cells and, as shipped, averages their
    velocities at every ODE step. That average is what makes the sample smooth: below the cell
    scale the replicas generate nearly independent texture, so folding them is a k-member
    ensemble mean and small-scale amplitude falls by ~1/sqrt(k). Measured on `ir77afjz` against
    its matched control, member SF ratio at 10-25 km retains 31-69% -- for 10si, r_850 and u_850
    that is BELOW the deterministic base, i.e. worse than not correcting at all.

    *** MEASURED, AND IT IS NOT THE FIX. *** The hypothesis was that the conditional mean needs
    the averaging while the fluctuation does not, so selecting one replica's velocity would keep
    the seam benefit and return the texture. Three matched inference arms off one snapshot and one
    checkpoint (a0ivju54 / vh8takpq / ycei1xd5), member SF ratio at 10-25 km on 10si:

        no blend            0.607        mu only, no blend   0.420
        blend k=3           0.355        mu only, blend k=3  0.334
        blend k=3, no fold  0.362

    Selecting instead of folding recovers 0.007 of the 0.252 that blending costs. The corrector's
    contribution over bare `mu` falls from +0.187 to +0.021 (folded) or +0.028 (selected), i.e.
    replication disables the corrector whatever the fold does. Blending `mu` accounts for only
    about 20%; the rest is the corrector going out of distribution, since its only texture
    mechanism is within-cell self-attention and replication inflates every varlen group by ~1.6x
    with duplicated points it never saw in training (blend is eval-only, `model.py:1032`).

    Kept because the flag is the experiment's record and costs nothing at the default. Do not
    reach for it as a seam fix.

    The highest weight is the nearest cell CENTRE, which is the containing cell for only ~91% of
    points -- a HEALPix cell is a rhombus, not a Voronoi cell. Nearest-centre is the right choice
    anyway: it is the most in-distribution host for the point's coordinate frame.
    """
    best = torch.full((n_state,), -1.0, device=w.device, dtype=w.dtype)
    best = best.scatter_reduce(0, idx, w, reduce="amax", include_self=False)
    hit = w >= best[idx] - 1e-6
    pick = torch.zeros(n_state, dtype=torch.long, device=w.device)
    rows = torch.arange(idx.numel(), device=w.device)
    return pick.scatter_(0, idx[hit], rows[hit])


def physical_to_residual(y1: torch.Tensor, mu: torch.Tensor, r_scale: torch.Tensor) -> torch.Tensor:
    """The flow-matching target: the residual of ``y1`` about ``mu``, in units of ``r_scale``.

    **Deliberately unclamped.** The reference implementation this is adapted from clamps
    ``y1 - mu`` before scaling, which silently breaks the identity in ``residual_to_physical``:
    the probability path would be built from a clamped residual while the MSE regresses against
    the *unclamped* ``y1``, training the network toward a target it never sees on the path.
    """
    return (y1 - mu) / r_scale


# ---------------------------------------------------------------------------------------------
# THE TRAINING PACK, AND WHY ITS SLOTS ARE NAMED.
#
# ``ResidualFlowPointDecoder`` returns a stack of HETEROGENEOUS quantities during training, not
# an ensemble. Every consumer must address a slot BY INDEX. A loss that treats this axis as an
# ensemble is not merely imprecise, it is scoring something that does not exist: ``reduce:
# members`` sorts POINTWISE (``LossStructureFunction.ensemble_fields``), so each "member" it
# returns is a per-pixel chimera of all four slots. Measured on a 3-wide pack with a corrector
# that had learned 90% of the velocity, the sorted fields drew
#     srt[0]  18.5% mu / 48.6% cfm / 32.8% r_scale
#     srt[1]  79.4% mu / 15.3% cfm /  5.3% r_scale
#     srt[2]   2.1% mu / 36.1% cfm / 61.9% r_scale
# and WFCL's FAL came out 253x, WAL_L1 118x the value of scoring the intended member -- while
# being nearly INSENSITIVE to the corrector, because PACK_MU is frozen under
# ``flow_freeze_base`` and PACK_RSCALE is a buffer. That is run `prr1iy4u` (+93% RMSE on 10si,
# WAL_L1 +108%): a measurement of the pack layout, not of the loss.
#
# Slots, and the one consumer each is for:
PACK_MU = 0  # mu                                -- mse_det
PACK_CFM = 1  # mu + r_scale*(y0 + v)            -- mse_flow
PACK_RSCALE = 2  # r_scale, broadcast (a buffer) -- mse_flow divides it out
PACK_X1 = 3  # mu + r_scale*(y_t + (1-t)*v)      -- every spectral / structure term
PACK_WIDTH = 4

# Config-facing names, so a config says `pack_member: x1` rather than a bare integer.
PACK_MEMBERS = {"mu": PACK_MU, "cfm": PACK_CFM, "r_scale": PACK_RSCALE, "x1": PACK_X1}


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


def residual_to_physical_x1(
    mu: torch.Tensor,
    y_t: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    r_scale: torch.Tensor,
) -> torch.Tensor:
    """The x1-prediction of the same velocity -- ``mu + r_scale * (y_t + (1 - t) * v)``.

    On the linear path ``y_t = (1-t)*y0 + t*r`` with CFM target ``u = r - y0`` we have
    ``r = y_t + (1-t)*u`` exactly, so this and ``residual_to_physical`` agree at ``v == u``
    and share a minimiser. They differ entirely in what they carry when ``v != u``:

        residual_to_physical     mu + r_scale*(y0 + v)          carries y0 at FULL amplitude
        residual_to_physical_x1  mu + r_scale*(y_t + (1-t)*v)   carries it as (1 - t)

    That is irrelevant to MSE, which is why ``mse_flow`` uses the first form. It is decisive for
    a SPECTRAL loss, which reads the field's amplitude and phase spectrum directly: the raw
    source dominates exactly the fine-scale bands such a loss exists to constrain. Measured on
    a 64x64 patch at CERRA's 5.5 km with ``r_scale`` 0.312 (10si), against the true residual:

        band              |r_scale*y0| / |y1 - mu|
        L1  11-22 km               1.63
        L2  22-44 km               0.97
        L3  44-88 km               0.57

    i.e. the band the sharpness campaign is about is more noise than signal in the first form.
    Under this form the same contamination falls off as ``(1-t)``: 0.19 of the true residual at
    L1 for t=0.25, 0.06 at t=0.75, 0.03 at t=0.9.

    So: ``mse_flow`` scores member ``PACK_CFM``, every spectral/structure term scores
    ``PACK_X1``, and both move the same weights through the same ``v``.
    """
    return mu + r_scale * (y_t + (1.0 - t) * v)


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
        gain: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.fixed = fixed
        self.ema = ema
        self.freeze_after = freeze_after
        self.floor = floor
        self.ceil = ceil
        self.gain = gain
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
        """The scale to divide residuals by -> ``[C]``.

        ``gain`` is a MULTIPLIER on the calibrated per-channel scale; ``fixed`` REPLACES it with
        one global scalar. They are not interchangeable and the difference matters:

            C4's calibrated r_scale  [10si .36  2t .109  r_850 .371  t_850 .089
                                      tp .69   u_850 .289  v_850 .324  z_850 .080]

        so ``fixed: 0.5`` is a 5.6x INCREASE on 2t and a cut on tp -- a recalibration, not a
        dilution sweep. ``gain: 0.5`` halves the injected residual on every channel and leaves
        the relative calibration alone, which is what sweeping the sharpness/placement trade
        actually requires.

        Defaults to 1.0, so training is untouched (and at the CFM optimum a gain cancels between
        ``physical_to_residual`` and ``residual_to_physical`` anyway). It is meant to be set in
        an INFERENCE config: at sampling it multiplies the integrated residual directly, in
        ``mu + r_scale * y``.
        """
        base = (
            torch.full_like(self.scale, self.fixed)
            if self.fixed is not None
            else self.scale.clamp(self.floor, self.ceil)
        )
        return base * self.gain if self.gain != 1.0 else base
