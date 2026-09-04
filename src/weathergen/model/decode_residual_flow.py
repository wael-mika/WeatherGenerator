# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Residual flow-matching corrector.

``FlowMatchingPointDecoder`` replaces the deterministic readout and transports noise straight to
the field. This decoder keeps the deterministic readout and flow-matches only the **residual**
``r = y - mu``, so one checkpoint carries both the RMSE-optimal blur and a sharp sample.
"""

import logging

import torch

from weathergen.model.engines import (
    FlowNullConditioning,
    TargetPredictionEngineClassic,
    flow_time_embedding,
    sample_flow_time_for_points,
)
from weathergen.model.flow_math import (
    ResidualScale,
    physical_to_residual,
    residual_to_physical,
)

logger = logging.getLogger(__name__)

FLOW_COND_MODES = ("mu", "mu+tokens", "mu+tokens+latent")


class ResidualFlowBranch(TargetPredictionEngineClassic):
    """The corrector itself: a small second decode stack predicting the residual velocity.

    It is a ``TargetPredictionEngineClassic`` so it inherits the varlen cross-attention to the
    cell's latents *and* the within-cell self-attention. That self-attention is the only mechanism
    coupling a cell's target points to each other, and therefore the only thing that can produce
    spatially coherent texture rather than independent per-point noise.

    **Every parameter and buffer must live in a named child module with a ``reset_parameters()``.**
    ``load_model_state`` re-initialises missing checkpoint keys by calling ``to_empty()`` +
    ``reset_parameters()`` on the highest-level module covering them. A bare parameter here would
    make *this* module the root, and ``TargetPredictionEngineClassic`` has no ``reset_parameters``
    -- warm starting would die with an ``AttributeError``. That is why ``null_cond`` and
    ``r_scale`` are modules rather than a raw parameter and a raw buffer.
    """

    def __init__(
        self,
        cf,
        dims_embed,
        dim_coord_in,
        tr_dim_head_proj,
        tr_mlp_hidden_factor,
        softcap,
        stream_config: dict,
        num_channels: int,
    ):
        dim_time = int(cf.get("flow_dim_time", 32))
        assert dim_time % 2 == 0, f"flow_dim_time must be even, got {dim_time}"
        # the time embedding rides along with the coordinate frame in the AdaLN conditioning,
        # so this stack -- unlike the inherited deterministic one -- is built for the widened aux
        super().__init__(
            cf,
            dims_embed,
            dim_coord_in + dim_time,
            tr_dim_head_proj,
            tr_mlp_hidden_factor,
            softcap,
            stream_config=stream_config,
        )
        self.name = f"ResidualFlowBranch_{stream_config['name']}"

        self.dim_time = dim_time
        self.num_channels = num_channels
        self.num_steps = int(cf.get("flow_num_steps", 24))
        self.time_sampling = str(cf.get("flow_time_sampling", "uniform"))
        self.time_logit_mean = float(cf.get("flow_time_logit_mean", 0.0))
        self.time_logit_std = float(cf.get("flow_time_logit_std", 1.0))
        self.time_per_cell = bool(cf.get("flow_time_per_cell", False))
        self.solver = str(cf.get("flow_solver", "euler"))
        assert self.solver in ("euler", "heun"), f"unknown flow_solver '{self.solver}'"

        self.cond_mode = str(cf.get("flow_cond", "mu+tokens+latent"))
        assert self.cond_mode in FLOW_COND_MODES, (
            f"unknown flow_cond '{self.cond_mode}' (use one of {FLOW_COND_MODES})"
        )
        self.cond_on_latent = self.cond_mode == "mu+tokens+latent"
        self.cond_on_tokens = self.cond_mode != "mu"
        self.detach_cond = bool(cf.get("flow_detach_cond", True))

        self.cond_dropout = float(cf.get("flow_cond_dropout", 0.0))
        self.guidance = float(cf.get("flow_guidance", 1.0))
        # Both are no-ops unless the latent is actually the KV: with flow_cond in
        # {mu, mu+tokens} the KV is ALREADY the null token, so dropping it changes nothing and
        # v_uncond == v_cond makes any guidance weight the identity.
        assert self.cond_on_latent or (self.cond_dropout == 0.0 and self.guidance == 1.0), (
            f"flow_cond_dropout/flow_guidance require flow_cond='mu+tokens+latent'; "
            f"got flow_cond='{self.cond_mode}'"
        )
        self.null_cond = FlowNullConditioning(cf.ae_global_dim_embed)

        residual_scale = cf.get("flow_residual_scale", None)
        self.r_scale = ResidualScale(
            num_channels,
            fixed=None if residual_scale is None else float(residual_scale),
            ema=float(cf.get("flow_residual_scale_ema", 0.999)),
            freeze_after=int(cf.get("flow_residual_scale_steps", 1000)),
        )

        # Diagnostic only, off by default and free when off. Answers the two questions that
        # cannot be settled from validation aggregates: is the velocity learned at all (does it
        # correlate with its own target u = r - y0?), and does the Euler integration preserve the
        # unit variance the linear path guarantees, or does it blow up?
        self.debug_trace = bool(cf.get("flow_debug_trace", False))
        self._trace_n = 0

        self.embed_state = torch.nn.Linear(num_channels, dims_embed[0])
        self.embed_mu = torch.nn.Linear(num_channels, dims_embed[0])
        # Default init on purpose -- do NOT zero-init. This head is the only path from the block
        # stack to the loss, so zero weights give the whole stack, embed_state and embed_mu
        # exactly zero gradient on the first step. (The reference implementation zero-inits its
        # output because there the corrector feeds the single scored quantity and a random
        # corrector at step 0 would corrupt it. Here mu and the corrected field are separate loss
        # members, so a random corrector cannot perturb the deterministic term at all.)
        self.vel_head = torch.nn.Linear(dims_embed[-1], num_channels)

    def kv_for(self, latent):
        """Cross-attention KV: the real latent, or the learned null token for variant 1.

        Substituting the null token rather than skipping the cross-attention blocks keeps
        ``latent_lens`` and all varlen bookkeeping unchanged, leaves no dead parameters, and lets
        both conditioning variants be A/B'd from one checkpoint. It also severs the encoder
        gradient *structurally* in the no-latent modes, which is exactly variant 1's semantics.
        """
        if not self.cond_on_latent:
            return self.null_cond.expand_to(latent)
        return latent

    def velocity(self, y_t, t, mu, latent, base, latent_lens, output_lens, coordinates):
        """One evaluation of ``v_theta(y_t, t, mu, cond)`` -> ``[N, num_channels]``."""
        q = base + self.embed_mu(mu.to(base.dtype)) + self.embed_state(y_t.to(base.dtype))
        aux = torch.cat([coordinates, flow_time_embedding(t, self.dim_time)], dim=-1)
        tokens = super().forward(latent, q, latent_lens, output_lens, aux)
        return self.vel_head(tokens)

    def guided_velocity(self, y_t, t, mu, latent, base, latent_lens, output_lens, coordinates):
        """Velocity with classifier-free guidance applied (sampling only)."""
        v_c = self.velocity(y_t, t, mu, latent, base, latent_lens, output_lens, coordinates)
        if self.guidance == 1.0:
            return v_c
        v_u = self.velocity(
            y_t,
            t,
            mu,
            self.null_cond.expand_to(latent),
            base,
            latent_lens,
            output_lens,
            coordinates,
        )
        return v_u + self.guidance * (v_c - v_u)

    def sample(self, mu, latent, base, latent_lens, output_lens, coordinates, ens_size):
        """Integrate the residual ODE and add it back -> ``[ens_size, N, num_channels]``.

        ``mu`` is computed once by the caller; members differ only in the base noise, so the
        deterministic stack is never re-run.
        """
        n = base.shape[0]
        rs = self.r_scale.value().to(mu.dtype)
        dt = 1.0 / self.num_steps
        preds = []
        with torch.no_grad():
            for _ in range(ens_size):
                y = torch.randn((n, self.num_channels), device=base.device, dtype=torch.float32)
                for i_step in range(self.num_steps):
                    t = torch.full((n, 1), i_step * dt, device=base.device, dtype=torch.float32)
                    v1 = self.guided_velocity(
                        y, t, mu, latent, base, latent_lens, output_lens, coordinates
                    ).float()
                    if self.solver == "heun":
                        t_next = torch.full(
                            (n, 1), (i_step + 1) * dt, device=base.device, dtype=torch.float32
                        )
                        v2 = self.guided_velocity(
                            y + dt * v1,
                            t_next,
                            mu,
                            latent,
                            base,
                            latent_lens,
                            output_lens,
                            coordinates,
                        ).float()
                        y = y + dt * 0.5 * (v1 + v2)
                    else:
                        y = y + dt * v1
                    if self.debug_trace and self._trace_n < 30 and i_step % 6 == 0:
                        self._trace_n += 1
                        logger.info(
                            "FLOWTRACE sample step=%02d | y.std=%s | v.std=%s",
                            i_step,
                            [round(x, 3) for x in y.std(0).tolist()],
                            [round(x, 3) for x in v1.std(0).tolist()],
                        )
                # no output clamp: targets are z-normalised here, not in [0, 1]
                preds.append(mu.float() + rs.float() * y)
        return torch.stack(preds, 0)

    def training_pack(self, mu, target, latent, base, latent_lens, output_lens, coordinates):
        """Build member 1 of the training pack -> ``[N, num_channels]``.

        Returns ``mu.detach() + r_scale*(y0 + v)``, so a plain MSE against the target is the CFM
        objective scaled by ``r_scale^2``. See ``residual_to_physical``.
        """
        mu_c = mu.detach()
        y1 = target.float()
        rs = self.r_scale.value().to(torch.float32)

        r = physical_to_residual(y1, mu_c.float(), rs)
        finite = torch.isfinite(r)
        y0 = torch.randn_like(r)
        # masked / spoofed targets are NaN. Substituting y0 gives them zero velocity instead of
        # poisoning y_t; the loss masks these points anyway, so they contribute no gradient.
        r = torch.where(finite, r, y0)

        if self.training:
            self.r_scale.update((y1 - mu_c.float()).nan_to_num(), finite.to(y1.dtype))

        t = sample_flow_time_for_points(
            r.shape[0],
            output_lens,
            r.device,
            r.dtype,
            per_cell=self.time_per_cell,
            mode=self.time_sampling,
            logit_mean=self.time_logit_mean,
            logit_std=self.time_logit_std,
        )
        y_t = (1.0 - t) * y0 + t * r

        # Conditioning dropout for classifier-free guidance. Written as a blend rather than an
        # if/else on purpose: DDP runs with _set_static_graph(), which requires the same set of
        # participating parameters every iteration. A branch would leave null_latent unused on
        # most steps and trip that. With mask == 0 the expression is exactly `latent`.
        if self.cond_dropout > 0.0:
            mask = (torch.rand((), device=latent.device) < self.cond_dropout).to(latent.dtype)
            latent = (1.0 - mask) * latent + mask * self.null_cond.expand_to(latent)

        v = self.velocity(y_t, t, mu_c, latent, base, latent_lens, output_lens, coordinates)

        if self.debug_trace and self._trace_n < 3:
            self._trace_n += 1
            with torch.no_grad():
                u = (r - y0).float()  # the CFM target velocity on the linear path
                vf = v.float()
                fin = torch.isfinite(u).all(-1) & torch.isfinite(vf).all(-1)
                uc, vc = u[fin], vf[fin]
                uc = uc - uc.mean(0)
                vc = vc - vc.mean(0)
                corr = (uc * vc).mean(0) / (uc.std(0) * vc.std(0)).clamp_min(1e-12)
                logger.info(
                    "FLOWTRACE train | r.std=%s | u.std=%s | v.std=%s | corr(v,u)=%s",
                    [round(x, 3) for x in r.float().std(0).tolist()],
                    [round(x, 3) for x in u.std(0).tolist()],
                    [round(x, 3) for x in vf.std(0).tolist()],
                    [round(x, 3) for x in corr.tolist()],
                )

        return residual_to_physical(mu_c.float(), y0, v.float(), rs)


class ResidualFlowPointDecoder(TargetPredictionEngineClassic):
    """Deterministic decode plus a flow-matching corrector on its residual.

    **Why a residual and not the field.** A pointwise-proper loss drives a decoder toward the
    conditional mean, which for a downscaling task is the blur: ``skill = 2*A*rho - A^2`` is
    maximised at amplitude ``A = rho``. Generating the *field* (``FlowMatchingPointDecoder``)
    abandons that optimum entirely. Generating the *residual* keeps it: ``mu`` is still trained by
    its own MSE, so the same checkpoint yields both the RMSE-optimal prediction and a sharp member.

    **Warm start is the reason for the class layout.** ``super().__init__`` is called with the
    *un-widened* ``dim_coord_in``, so the inherited ``self.tte`` has byte-identical parameter
    shapes and names to a plain ``PerceiverIOCoordConditioning`` decoder, and ``mu`` is still
    produced by ``Model.pred_heads``. Loading a deterministic parent therefore matches every
    inherited key and reports missing keys only under ``...flow.*``, which the reinit path fills
    in. Warm starting from a ``FlowMatching`` run instead **crashes**: ``strict=False`` still
    raises on *size* mismatch, and that decoder widens the AdaLN aux.

    Returns:
        training : ``[3, N, C]`` -- member 0 is ``mu`` (undetached, trains the base via
            ``mse_det``), member 1 is ``mu.detach() + r_scale*(y0+v)``, and member 2 is the
            per-channel ``r_scale`` that ``mse_flow`` divides out so every channel's CFM term
            carries the same weight. **The training loss must use ``mse_det``/``mse_flow``,
            never plain ``mse``**, which would average the members into nonsense.
        eval : ``[ens_size, N, C]`` of corrected fields, or ``[1, N, C]`` of bare ``mu`` when
            ``flow_ens_size == 0`` -- which evaluates the same checkpoint as a purely
            deterministic model.
    """

    needs_target_in_forward = True

    def __init__(
        self,
        cf,
        dims_embed,
        dim_coord_in,
        tr_dim_head_proj,
        tr_mlp_hidden_factor,
        softcap,
        stream_config: dict,
        num_channels: int,
    ):
        # NOTE: un-widened dim_coord_in -- this keeps `tte.*` checkpoint-compatible with a
        # deterministic parent. The corrector's own stack widens its aux internally.
        super().__init__(
            cf,
            dims_embed,
            dim_coord_in,
            tr_dim_head_proj,
            tr_mlp_hidden_factor,
            softcap,
            stream_config=stream_config,
        )
        self.name = f"ResidualFlowPointDecoder_{stream_config['name']}"
        self.num_channels = num_channels

        assert len(set(dims_embed)) == 1, (
            "the residual corrector assumes a uniform decoder width; "
            f"got dims_embed={list(dims_embed)}"
        )
        num_layers = int(cf.get("flow_num_layers", 2))
        self.flow = ResidualFlowBranch(
            cf,
            [dims_embed[0]] * (num_layers + 1),
            dim_coord_in,
            tr_dim_head_proj,
            tr_mlp_hidden_factor,
            softcap,
            stream_config=stream_config,
            num_channels=num_channels,
        )

        self.freeze_base = bool(cf.get("flow_freeze_base", False))
        if self.freeze_base:
            self.tte.requires_grad_(False)

        ens = cf.get("flow_ens_size", None)
        self.flow_ens_size = None if ens is None else int(ens)

    def train(self, mode: bool = True):
        """Keep a frozen deterministic stack in eval mode.

        ``requires_grad_(False)`` alone is not enough: the decode blocks carry live
        ``nn.Dropout(0.1)`` gated on ``self.training``, and the trainer calls ``model.train()``
        every epoch. A "frozen" base left in training mode emits a *stochastic* ``mu``, and the
        corrector would spend its capacity learning to undo dropout noise.
        """
        super().train(mode)
        if self.freeze_base:
            self.tte.eval()
        return self

    def det_forward(self, latent, output, latent_lens, output_lens, coordinates):
        """The plain deterministic decode -> tokens for ``Model.pred_heads``."""
        if self.freeze_base:
            # also drops the det-stack activation graph, which is the memory headroom the
            # corrector needs at level 6
            with torch.no_grad():
                return super().forward(latent, output, latent_lens, output_lens, coordinates)
        return super().forward(latent, output, latent_lens, output_lens, coordinates)

    def correct(
        self,
        mu,
        det_tokens,
        coord_tokens,
        latent,
        latent_lens,
        output_lens,
        coordinates,
        target=None,
        ens_size=1,
    ):
        """Apply the corrector to a deterministic prediction ``mu`` of shape ``[N, C]``."""
        base = det_tokens if self.flow.cond_on_tokens else coord_tokens
        kv = self.flow.kv_for(latent)
        if self.flow.detach_cond:
            # the flow loss then moves flow parameters only, and `mse_det` stays provably
            # identical to a standalone deterministic run
            mu_cond, base, kv = mu.detach(), base.detach(), kv.detach()
        else:
            mu_cond = mu

        if self.flow_ens_size is not None:
            ens_size = self.flow_ens_size

        if not self.training:
            if ens_size == 0:
                # evaluate this checkpoint as a purely deterministic model
                return mu.unsqueeze(0)
            return self.flow.sample(
                mu_cond, kv, base, latent_lens, output_lens, coordinates, ens_size
            )

        assert target is not None, (
            "ResidualFlowPointDecoder needs the target during training to build the "
            "probability path; predict_decoders must pass it."
        )
        member = self.flow.training_pack(
            mu_cond, target, kv, base, latent_lens, output_lens, coordinates
        )
        # Member 2 carries the per-channel r_scale so `mse_flow` can divide it out and score the
        # CFM objective with EQUAL weight per channel. Without it the loss weight is r_scale_c^2,
        # which starves exactly the channels the base already predicts well -- see the table in
        # loss_functions.mse_flow. It is a loss-contract slot, not a prediction; `mse_det` still
        # slices member 0 and validation never sees this pack at all.
        r_scale = self.flow.r_scale.value().to(mu.dtype).expand_as(mu)
        return torch.stack([mu.float(), member, r_scale.float()], 0)
