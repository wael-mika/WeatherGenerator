# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Callable
import logging
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from weathergen.model.norms import AdaLayerNorm, RMSNorm

logger = logging.getLogger(__name__)


class NamedLinear(torch.nn.Module):
    def __init__(self, name: str | None = None, **kwargs):
        super(NamedLinear, self).__init__()
        self.linear = nn.Linear(**kwargs)
        if name is not None:
            self.name = name

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class MLP(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_layers=2,
        hidden_factor=2,
        pre_layer_norm=True,
        dropout_rate=0.0,
        nonlin=torch.nn.GELU,
        with_residual=False,
        norm_type="LayerNorm",
        dim_aux=None,
        norm_eps=1e-5,
        name: str | None = None,
    ):
        """Multi-layer perceptron with optional pre-LayerNorm, residual connection,
        and auxiliary-conditioned adaptive normalisation (AdaLayerNorm).

        Args:
            dim_in: Input feature dimension.
            dim_out: Output feature dimension.
            num_layers: Total linear layers (must be >= 2).
            hidden_factor: Multiplier applied to *dim_in* to obtain hidden width.
            pre_layer_norm: If ``True``, prepend a normalisation layer.
            dropout_rate: Dropout probability after each activation.
            nonlin: Activation constructor (default :class:`torch.nn.GELU`).
            with_residual: Add a skip connection from input to output.
            norm_type: ``"LayerNorm"`` or ``"RMSNorm"``.
            dim_aux: If not ``None``, the first norm becomes an
                :class:`AdaLayerNorm` conditioned on an auxiliary tensor of this
                dimension.
            norm_eps: Epsilon for the normalisation layer.
            name: Optional name attached as an attribute for debugging.
        """

        super(MLP, self).__init__()

        if name is not None:
            self.name = name

        assert num_layers >= 2

        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        dim_hidden = int(dim_in * hidden_factor)

        self.layers = torch.nn.ModuleList()

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        if pre_layer_norm:
            self.layers.append(
                norm(dim_in, eps=norm_eps)
                if dim_aux is None
                else AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps)
            )

        self.layers.append(torch.nn.Linear(dim_in, dim_hidden))
        self.layers.append(nonlin())
        self.layers.append(torch.nn.Dropout(p=dropout_rate))

        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nonlin())
            self.layers.append(torch.nn.Dropout(p=dropout_rate))

        self.layers.append(torch.nn.Linear(dim_hidden, dim_out))

    def forward(
        self,
        x: torch.Tensor,
        *args,
        aux: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply MLP to `x` with optional auxiliary conditioning.

        Calling conventions kept for backward compatibility:
        - `mlp(x)` for non-aux MLPs.
        - `mlp(x, ..., aux_tensor)` for aux MLPs, where aux is passed as the last positional arg.
        - `mlp(x, aux=aux_tensor)` for aux MLPs.
        """
        del kwargs
        x_in = x

        aux_value = aux
        if self.with_aux and aux_value is None:
            if len(args) == 0 or not torch.is_tensor(args[-1]):
                raise ValueError(
                    "MLP with aux conditioning expects a tensor `aux` as keyword "
                    "argument or as the last positional argument."
                )
            aux_value = args[-1]

        for i, layer in enumerate(self.layers):
            x = layer(x, aux_value) if (i == 0 and self.with_aux) else layer(x)

        if self.with_residual:
            if x.shape[-1] == x_in.shape[-1]:
                x = x_in + x
            else:
                assert x.shape[-1] % x_in.shape[-1] == 0
                x = x + x_in.repeat([*[1 for _ in x.shape[:-1]], x.shape[-1] // x_in.shape[-1]])

        return x


class LoadBalancingLoss(torch.nn.Module):
    """Auxiliary load-balancing loss from Switch Transformers
    (`Fedus et al., 2022 <https://arxiv.org/abs/2101.03961>`_).

    The loss encourages uniform expert utilisation by penalising the
    dot-product between two per-expert statistics:

    * **mean router probability** — the average softmax score assigned to
      each expert across all (valid) tokens.
    * **mean assignment weight** — the average realised gate weight that each
      expert received via top-k selection.

    .. math::

        L_{\\text{balance}} = N_E \\sum_{i=1}^{N_E} \\bar{p}_i \\cdot \\bar{w}_i

    The final value is scaled by *weight* before being returned.
    """

    def __init__(self, num_experts: int, weight: float = 0.01):
        """
        Args:
            num_experts: Number of experts (*E*).
            weight: Scalar multiplier applied to the raw loss.
        """
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight

    def forward(
        self,
        router_probs: torch.Tensor,
        assignment_weights: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the scaled load-balancing loss.

        Args:
            router_probs: ``[N, E]`` softmax probabilities from the router.
            assignment_weights: ``[N, E]`` per-token gate weights that were
                realised through top-k selection (rows sum to 1 for each token).
            token_mask: Optional ``[N]`` boolean mask.  When provided, only
                ``True`` positions contribute to the per-expert averages.

        Returns:
            A scalar tensor: ``weight * N_E * sum(mean_prob * mean_assignment)``.
        """
        if router_probs.ndim != 2 or assignment_weights.ndim != 2:
            raise ValueError(
                "LoadBalancingLoss expects [N, E] tensors for router_probs and assignment_weights"
            )

        if token_mask is not None:
            token_mask_f = token_mask.to(router_probs.dtype).unsqueeze(-1)
            denom = token_mask_f.sum().clamp_min(1.0)
            mean_prob = (router_probs * token_mask_f).sum(dim=0) / denom
            mean_assignment = (assignment_weights * token_mask_f).sum(dim=0) / denom
        else:
            mean_prob = router_probs.mean(dim=0)
            mean_assignment = assignment_weights.mean(dim=0)

        loss = self.num_experts * (mean_prob * mean_assignment).sum()
        return self.weight * loss


class RouterZLoss(torch.nn.Module):
    """Router z-loss from ST-MoE (`Zoph et al., 2022
    <https://arxiv.org/abs/2202.08906>`_).

    Penalises large router logits to prevent the softmax distribution
    from collapsing to a single expert.  The loss is the mean squared
    log-partition function of the router logits:

    .. math::

        L_z = \\frac{1}{N} \\sum_{n=1}^{N}
              \\bigl(\\log \\sum_{e=1}^{E} \\exp z_{n,e}\\bigr)^2

    Scaled by *weight* before being returned.
    """

    def __init__(self, weight: float = 0.001):
        """
        Args:
            weight: Scalar multiplier applied to the raw z-loss.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        router_logits: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the scaled router z-loss.

        Args:
            router_logits: ``[N, E]`` raw (pre-softmax) router logits.
            token_mask: Optional ``[N]`` boolean mask.  When provided, only
                ``True`` positions contribute to the mean.

        Returns:
            A scalar tensor: ``weight * mean(logsumexp(logits)^2)``.
        """
        log_z = torch.logsumexp(router_logits, dim=-1)
        if token_mask is not None:
            mask_f = token_mask.to(log_z.dtype)
            denom = mask_f.sum().clamp_min(1.0)
            loss = (log_z ** 2 * mask_f).sum() / denom
        else:
            loss = (log_z ** 2).mean()
        return self.weight * loss


class MoERouter(torch.nn.Module):
    """Dense top-k router for mixture-of-experts blocks.

    Each token is projected to ``num_experts`` logits via a learned
    projection (optionally a 2-layer MLP), followed by softmax and top-k
    selection.  Multiplicative jitter noise can be applied to the input
    during training to encourage exploration.
    """

    def __init__(
        self,
        dim_in: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        router_bias: bool = False,
        router_hidden_dim: int = 0,
    ):
        """
        Args:
            dim_in: Token embedding dimension.
            num_experts: Number of experts to score.
            top_k: How many experts each token is routed to.
            jitter_noise: Half-width of the uniform multiplicative jitter
                applied to router inputs during training (0 disables).
            router_bias: Whether the final router linear layer has a bias.
            router_hidden_dim: If >0, use a 2-layer MLP with this hidden
                size instead of a single linear projection.
        """
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        if router_hidden_dim and router_hidden_dim > 0:
            self.router_weights = nn.Sequential(
                nn.Linear(dim_in, router_hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(router_hidden_dim, num_experts, bias=router_bias),
            )
        else:
            self.router_weights = nn.Linear(dim_in, num_experts, bias=router_bias)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route each token to its top-k experts.

        Args:
            x: Token features ``[N, D]`` or ``[B, T, D]`` (flattened internally).
            position_ids: Unused — accepted for interface compatibility with
                :class:`SpatialMoERouter`.
            token_mask: Optional ``[N]`` boolean mask.  Masked tokens receive
                zero router probabilities and will not influence expert
                selection.

        Returns:
            router_probs: ``[N, E]`` softmax probabilities over all experts.
            expert_indices: ``[N, K]`` indices of the chosen experts per token.
            expert_weights: ``[N, K]`` gate weights normalised to sum to 1
                across the *K* selected experts for each token.
            router_logits: ``[N, E]`` raw pre-softmax logits (used for z-loss).
        """
        del position_ids
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim != 2:
            raise ValueError(f"MoERouter expects [N, D] or [B, T, D], got {tuple(x.shape)}")

        if self.training and self.jitter_noise > 0:
            x = x * torch.empty_like(x).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        router_logits = self.router_weights(x)

        if token_mask is not None:
            token_mask = token_mask.to(torch.bool)
            router_probs = torch.zeros_like(router_logits)
            valid_idx = token_mask.nonzero(as_tuple=True)[0]
            if valid_idx.numel() > 0:
                router_probs[valid_idx] = F.softmax(router_logits[valid_idx], dim=-1)
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return router_probs, expert_indices, expert_weights, router_logits


class SpatialMoERouter(torch.nn.Module):
    """Dual-head gated router that combines content-based and spatial
    expert scores via a learned per-token gate.

    Two independent scoring heads produce expert logits:

    * **Content head** — projects token features to expert scores,
      optionally through a 2-layer MLP.
    * **Spatial head** — projects a sinusoidal position embedding to
      expert scores, capturing geographic priors (e.g. tropics vs poles).

    A learned gate ``alpha = sigmoid(gate_proj(x))`` blends the two::

        router_logits = alpha * content_logits + (1 - alpha) * spatial_logits

    This lets the model learn *when* to trust geography (quiet regions,
    stable spatial priors) versus content (extreme events, unusual
    atmospheric states).

    The position embedding table is a **register_buffer** (not
    ``nn.Embedding`` / ``nn.Parameter``) to avoid the FSDP2 DTensor
    all-gather issue.  It is initialised with sinusoidal ``(theta, phi)``
    HEALPix coordinates via :meth:`initialize_from_coordinates`.

    Special tokens (register / class tokens) use ``position_id = -1``
    and receive a zero spatial embedding.
    """

    def __init__(
        self,
        dim_in: int,
        num_experts: int,
        num_positions: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        router_bias: bool = False,
        position_embed_dim: int = 128,
        router_hidden_dim: int = 0,
    ):
        """
        Args:
            dim_in: Token embedding dimension.
            num_experts: Number of experts to score.
            num_positions: Size of the spatial embedding table (typically the
                number of HEALPix cells).
            top_k: How many experts each token is routed to.
            jitter_noise: Half-width of the uniform multiplicative jitter
                applied to content features during training (0 disables).
            router_bias: Whether the final content-head linear has a bias.
            position_embed_dim: Dimension of each spatial position embedding.
            router_hidden_dim: If >0, use a 2-layer MLP for the content head.
        """
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        self.num_experts = num_experts
        self.num_positions = num_positions
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.position_embed_dim = position_embed_dim

        # --- Spatial embedding buffer (FSDP2-safe) ---
        # nn.Embedding weight is sharded as a DTensor by FSDP2, and
        # F.embedding with a DTensor weight triggers an implicit
        # full_tensor() all-gather on every forward.  Using a plain buffer
        # keeps the table replicated and avoids the issue.
        self.register_buffer(
            "position_embed_weight", torch.zeros(num_positions, position_embed_dim)
        )

        # --- Content head: token features → expert scores ---
        if router_hidden_dim and router_hidden_dim > 0:
            self.content_head = nn.Sequential(
                nn.Linear(dim_in, router_hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(router_hidden_dim, num_experts, bias=router_bias),
            )
        else:
            self.content_head = nn.Linear(dim_in, num_experts, bias=router_bias)

        # --- Spatial head: position embedding → expert scores ---
        self.spatial_head = nn.Linear(position_embed_dim, num_experts, bias=False)

        # --- Per-token gate blending content vs spatial ---
        # Initialised to output 0 → sigmoid(0) = 0.5 → equal blend at start.
        self.gate_proj = nn.Linear(dim_in, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def initialize_from_coordinates(self, theta: torch.Tensor, phi: torch.Tensor):
        """Initialise the embedding table with sinusoidal features derived
        from HEALPix ``(theta, phi)`` coordinates.

        The embedding vector for each cell is built as::

            embed[:half]  = interleaved sin/cos of theta at log-spaced frequencies
            embed[half:]  = interleaved sin/cos of phi   at log-spaced frequencies

        This provides a smooth, continuous spatial signal so that
        neighbouring cells receive similar router inputs.

        Args:
            theta: ``[num_positions]`` co-latitude angles (radians).
            phi: ``[num_positions]`` longitude angles (radians).

        Returns:
            ``self``, for convenience chaining.
        """
        assert len(theta) == self.num_positions, (len(theta), self.num_positions)
        assert len(phi) == self.num_positions, (len(phi), self.num_positions)

        with torch.no_grad():
            embed_dim = self.position_embed_dim
            embeddings = torch.zeros(self.num_positions, embed_dim)
            half_dim = embed_dim // 2

            theta_cpu = theta.cpu().float()
            phi_cpu = phi.cpu().float()

            freqs_theta = torch.exp(
                torch.arange(0, half_dim, 2).float()
                * -(np.log(10000.0) / max(half_dim, 1))
            )
            embeddings[:, 0:half_dim:2] = torch.sin(theta_cpu.unsqueeze(1) * freqs_theta)
            embeddings[:, 1:half_dim:2] = torch.cos(theta_cpu.unsqueeze(1) * freqs_theta)

            freqs_phi = torch.exp(
                torch.arange(0, half_dim, 2).float()
                * -(np.log(10000.0) / max(half_dim, 1))
            )
            embeddings[:, half_dim + 0 :: 2] = torch.sin(phi_cpu.unsqueeze(1) * freqs_phi)
            embeddings[:, half_dim + 1 :: 2] = torch.cos(phi_cpu.unsqueeze(1) * freqs_phi)

            self.position_embed_weight.copy_(
                embeddings.to(
                    device=self.position_embed_weight.device,
                    dtype=self.position_embed_weight.dtype,
                )
            )

        return self

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens using dual-head gated content + spatial scoring.

        Args:
            x: Token features ``[N, D]`` or ``[B, T, D]`` (flattened internally).
            position_ids: ``[N]`` integer IDs indexing the spatial embedding
                table.  Negative values receive a zero embedding (used for
                special tokens).  If ``None``, a sequential modular fallback
                ``arange(N) % num_positions`` is used.
            token_mask: Optional ``[N]`` boolean mask.  Masked tokens get
                zero router probabilities.

        Returns:
            router_probs: ``[N, E]`` softmax probabilities over all experts.
            expert_indices: ``[N, K]`` indices of the chosen experts per token.
            expert_weights: ``[N, K]`` gate weights normalised per token.
            router_logits: ``[N, E]`` raw pre-softmax logits (used for z-loss).
        """
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim != 2:
            raise ValueError(
                f"SpatialMoERouter expects [N, D] or [B, T, D], got {tuple(x.shape)}"
            )

        num_tokens = x.shape[0]
        if position_ids is None:
            position_ids = (
                torch.arange(num_tokens, device=x.device, dtype=torch.long) % self.num_positions
            )
        else:
            if position_ids.ndim != 1:
                position_ids = position_ids.reshape(-1)
            if position_ids.numel() != num_tokens:
                raise ValueError(
                    "position_ids must match number of tokens after flattening: "
                    f"{position_ids.numel()} vs {num_tokens}"
                )
            position_ids = position_ids.to(device=x.device, dtype=torch.long)

        # Multiplicative jitter on content features (before content head).
        if self.training and self.jitter_noise > 0:
            x = x * torch.empty_like(x).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # --- Content scores ---
        content_logits = self.content_head(x)  # [N, E]

        # --- Spatial scores ---
        # Clamp to valid range; zero out special tokens (id < 0) via
        # elementwise multiply — avoids GPU-CPU sync.
        clipped_ids = position_ids.clamp(min=0) % self.num_positions
        pos_embed = F.embedding(clipped_ids, self.position_embed_weight).to(dtype=x.dtype)
        pos_embed = pos_embed * (position_ids >= 0).to(dtype=x.dtype).unsqueeze(-1)
        spatial_logits = self.spatial_head(pos_embed)  # [N, E]

        # --- Gated combination ---
        alpha = torch.sigmoid(self.gate_proj(x))  # [N, 1]
        router_logits = alpha * content_logits + (1.0 - alpha) * spatial_logits

        if token_mask is not None:
            token_mask = token_mask.to(torch.bool)
            router_probs = torch.zeros_like(router_logits)
            valid_idx = token_mask.nonzero(as_tuple=True)[0]
            if valid_idx.numel() > 0:
                router_probs[valid_idx] = F.softmax(router_logits[valid_idx], dim=-1)
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return router_probs, expert_indices, expert_weights, router_logits


class MoEBlock(torch.nn.Module):
    """Sparse mixture-of-experts block that wraps an arbitrary expert factory.

    **Forward pass (high-level)**:

    1. Flatten inputs to token-major ``[N, D]``.
    2. Route every token to its *top_k* experts via a learned router.
    3. Build an assignment edge list ``(token_idx, expert_idx, gate_weight)``.
    4. Remove edges for masked-out tokens.
    5. Enforce per-expert *capacity* — drop lowest-gate edges when an expert
       receives more assignments than its budget.
    6. Optionally re-normalise gate weights so that remaining edges per
       token still sum to 1.
    7. Dispatch routed tokens to each active expert and combine the
       weighted outputs with :func:`index_add_`.
    8. Add the residual connection (if enabled).
    9. During training, compute an auxiliary load-balancing loss.

    The block stores diagnostic tensors (``last_expert_indices``,
    ``last_expert_weights``, ``last_aux_loss``) that can be inspected
    after each forward pass for logging or debugging.
    """

    def __init__(
        self,
        expert_fn: Callable[[], nn.Module],
        dim_in: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.0,
        router_bias: bool = False,
        load_balance_weight: float = 0.01,
        renormalize_gates: bool = True,
        with_residual: bool = True,
        name: str | None = None,
        use_spatial_router: bool = False,
        num_positions: int | None = None,
        position_embed_dim: int = 128,
        router_hidden_dim: int = 0,
        router_z_loss_weight: float = 0.0,
        debug_enabled: bool = False,
        debug_interval: int = 100,
        debug_top_experts: int = 3,
        debug_name: str | None = None,
    ):
        """
        Args:
            expert_fn: Zero-argument callable that returns a fresh expert
                module.  Called ``num_experts`` times.
            dim_in: Token embedding dimension (must match expert input dim).
            num_experts: Number of expert copies to instantiate.
            top_k: Experts selected per token.
            capacity_factor: Per-expert capacity as a fraction of
                ``top_k * valid_tokens / num_experts``.  Set to ``0`` or
                ``None`` to disable capacity enforcement.
            jitter_noise: Half-width of uniform multiplicative jitter
                applied to router inputs during training (0 disables).
            router_bias: Whether the router projection has a bias term.
            load_balance_weight: Scalar multiplier for the auxiliary
                :class:`LoadBalancingLoss`.
            renormalize_gates: If ``True``, re-normalise gate weights per
                token after capacity-based edge dropping.
            with_residual: If ``True``, add a skip connection ``output += x``.
            name: Optional name attached as a module attribute.
            use_spatial_router: Use :class:`SpatialMoERouter` instead of the
                plain :class:`MoERouter`.
            num_positions: Spatial embedding table size (required when
                *use_spatial_router* is ``True``).
            position_embed_dim: Dimension of the spatial position embedding.
            router_hidden_dim: If >0, use a 2-layer MLP router with this
                hidden size instead of a single linear projection.
            router_z_loss_weight: Scalar multiplier for the
                :class:`RouterZLoss`.  Set to ``0`` to disable.
            debug_enabled: Emit periodic diagnostic log lines during training.
            debug_interval: Log every *N*-th forward pass (rank 0 only).
            debug_top_experts: Number of top experts shown in the log line.
            debug_name: Label used in diagnostic log lines.
        """
        super().__init__()
        if name is not None:
            self.name = name

        self.dim_in = dim_in
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.with_residual = with_residual
        self.use_spatial_router = use_spatial_router
        self.renormalize_gates = renormalize_gates
        self.debug_enabled = debug_enabled
        self.debug_interval = max(int(debug_interval), 1)
        self.debug_top_experts = max(int(debug_top_experts), 1)
        self.debug_name = debug_name or name or "MoEBlock"
        self._debug_forward_counter = 0

        self.experts = nn.ModuleList([expert_fn() for _ in range(num_experts)])
        if use_spatial_router:
            if num_positions is None:
                raise ValueError("num_positions is required when use_spatial_router=True")
            self.router = SpatialMoERouter(
                dim_in=dim_in,
                num_experts=num_experts,
                num_positions=num_positions,
                top_k=top_k,
                jitter_noise=jitter_noise,
                router_bias=router_bias,
                position_embed_dim=position_embed_dim,
                router_hidden_dim=router_hidden_dim,
            )
        else:
            self.router = MoERouter(
                dim_in=dim_in,
                num_experts=num_experts,
                top_k=top_k,
                jitter_noise=jitter_noise,
                router_bias=router_bias,
                router_hidden_dim=router_hidden_dim,
            )

        self.load_balance_loss = LoadBalancingLoss(
            num_experts=num_experts,
            weight=load_balance_weight,
        )
        self.router_z_loss: RouterZLoss | None = None
        if router_z_loss_weight and router_z_loss_weight > 0:
            self.router_z_loss = RouterZLoss(weight=router_z_loss_weight)
        self.last_aux_loss: torch.Tensor | None = None
        self.last_expert_indices: torch.Tensor | None = None
        self.last_expert_weights: torch.Tensor | None = None
        self.register_buffer("position_ids", torch.empty(0, dtype=torch.long), persistent=False)

    def _should_log_debug(self) -> bool:
        """Return `True` when this forward pass should emit a debug line."""
        if not (self.debug_enabled and self.training):
            return False
        self._debug_forward_counter += 1
        if self._debug_forward_counter % self.debug_interval != 0:
            return False
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return False
        return True

    def _format_top_experts(self, expert_indices_flat: torch.Tensor) -> str:
        """Format the top expert usage fractions for compact logging."""
        if expert_indices_flat.numel() == 0:
            return "-"

        counts = torch.bincount(expert_indices_flat, minlength=self.num_experts).to(torch.float32)
        total = counts.sum().clamp_min(1.0)
        fractions = counts / total
        top_n = min(self.debug_top_experts, self.num_experts)
        values, indices = torch.topk(fractions, k=top_n, sorted=True)
        return ", ".join(
            f"{int(idx)}:{float(val):.3f}" for idx, val in zip(indices.tolist(), values.tolist(), strict=True)
        )

    def _log_debug_stats(
        self,
        n_tokens: int,
        token_mask_flat: torch.Tensor | None,
        expert_indices_pre_capacity: torch.Tensor,
        expert_indices_post_capacity: torch.Tensor,
        expert_weights: torch.Tensor,
        capacity: int | None,
    ) -> None:
        """Emit one MoE diagnostic line for the current forward pass.

        The output format is::

            MoE[<name>] fwd=<step> valid_tokens=<v>/<n> cap=<c>
            dropped=<d>(<pct>%) mean_top1_gate=<g>
            experts_pre=<...> experts_post=<...> aux=<a>

        See ``docs/moe_parameters_and_debugging.md`` for field definitions.
        """
        valid_tokens = n_tokens if token_mask_flat is None else int(token_mask_flat.sum().item())

        top1_weights = expert_weights[:, 0]
        if token_mask_flat is not None:
            top1_weights = top1_weights[token_mask_flat]
        mean_top1_gate = float(top1_weights.mean().item()) if top1_weights.numel() > 0 else 0.0

        num_assignments_before = int(expert_indices_pre_capacity.numel())
        num_assignments_after = int(expert_indices_post_capacity.numel())
        dropped_assignments = num_assignments_before - num_assignments_after
        dropped_ratio = dropped_assignments / max(num_assignments_before, 1)

        pre_capacity_summary = self._format_top_experts(expert_indices_pre_capacity)
        post_capacity_summary = self._format_top_experts(expert_indices_post_capacity)
        if self.last_aux_loss is None:
            aux_loss_value = float("nan")
        elif hasattr(self.last_aux_loss, "full_tensor"):
            aux_loss_value = float(self.last_aux_loss.full_tensor().item())
        else:
            aux_loss_value = float(self.last_aux_loss.item())

        logger.info(
            "MoE[%s] fwd=%d valid_tokens=%d/%d cap=%s dropped=%d(%.2f%%) "
            "mean_top1_gate=%.4f experts_pre=%s experts_post=%s aux=%.4e",
            self.debug_name,
            self._debug_forward_counter,
            valid_tokens,
            n_tokens,
            "none" if capacity is None else capacity,
            dropped_assignments,
            dropped_ratio * 100.0,
            mean_top1_gate,
            pre_capacity_summary,
            post_capacity_summary,
            aux_loss_value,
        )

    def set_position_ids(self, position_ids: torch.Tensor | None) -> None:
        """
        Set default per-token position IDs used when `forward(..., position_ids=...)`
        is not provided by the caller.
        """
        if position_ids is None:
            # Determine device, guarding against the meta device that is present
            # before the model is materialised (e.g. during FSDP2 init).
            dev = self.position_ids.device
            if dev.type == "meta":
                dev = torch.device("cpu")
            self.position_ids = torch.empty(0, dtype=torch.long, device=dev)
        else:
            # Use the incoming tensor's device directly; callers (initialize_moe_position_ids)
            # already place position_ids on the correct CUDA device.  Inferring the
            # target device from self.position_ids would propagate "meta" when the
            # buffer has not yet been materialised (train_continue path).
            self.position_ids = position_ids.to(dtype=torch.long)

    def _flatten_tokens(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        token_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[int, int, int] | None,
    ]:
        """Flatten optional batched inputs to token-major ``[N, ...]`` format.

        Handles both 2-D ``[N, D]`` (already flat) and 3-D ``[B, T, D]``
        (batched) inputs.  Auxiliary tensors, position IDs, and token masks
        are broadcast / reshaped to match.

        Args:
            x: Token features ``[N, D]`` or ``[B, T, D]``.
            aux: Optional auxiliary conditioning ``[N, A]`` or ``[B, T, A]``.
            position_ids: Optional ``[T]``, ``[N]``, or ``[B, T]`` position IDs.
            token_mask: Optional ``[T]``, ``[N]``, or ``[B, T]`` boolean mask.

        Returns:
            x_flat: ``[N, D]``
            aux_flat: ``[N, A]`` or ``None``
            pos_ids_flat: ``[N]`` or ``None``
            token_mask_flat: ``[N]`` or ``None``
            shape_info: ``(B, T, D)`` tuple for un-flattening output, or
                ``None`` when input was already 2-D.
        """
        is_batched = x.ndim == 3
        if x.ndim == 2:
            n, dim = x.shape
            batch_size = None
            seq_len = None
        elif x.ndim == 3:
            batch_size, seq_len, dim = x.shape
            x = x.reshape(-1, dim)
        else:
            raise ValueError(f"MoEBlock expects [N, D] or [B, T, D], got {tuple(x.shape)}")

        aux_flat = None
        if aux is not None:
            if not torch.is_tensor(aux):
                raise ValueError("MoEBlock expects aux to be a torch.Tensor or None")
            if aux.device != x.device:
                aux = aux.to(x.device)
            if (not is_batched) and aux.ndim == 2 and aux.shape[0] == x.shape[0]:
                aux_flat = aux
            elif is_batched and aux.ndim == 3 and aux.shape[0] == batch_size and aux.shape[1] == seq_len:
                aux_flat = aux.reshape(-1, aux.shape[-1])
            else:
                raise ValueError(
                    "MoEBlock aux must match x leading dimensions (per-token conditioning)."
                )

        pos_ids_flat = None
        if position_ids is not None:
            if position_ids.ndim == 1:
                if is_batched and position_ids.numel() == seq_len:
                    pos_ids_flat = position_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                else:
                    pos_ids_flat = position_ids.reshape(-1)
            elif position_ids.ndim == 2 and is_batched:
                if position_ids.shape[0] != batch_size or position_ids.shape[1] != seq_len:
                    raise ValueError("position_ids shape must match [B, T]")
                pos_ids_flat = position_ids.reshape(-1)
            else:
                pos_ids_flat = position_ids.reshape(-1)

        token_mask_flat = None
        if token_mask is not None:
            token_mask = token_mask.to(device=x.device, dtype=torch.bool)
            if token_mask.ndim == 1:
                if is_batched and token_mask.numel() == seq_len:
                    token_mask_flat = token_mask.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                else:
                    token_mask_flat = token_mask.reshape(-1)
            elif token_mask.ndim == 2 and is_batched:
                if token_mask.shape[0] != batch_size or token_mask.shape[1] != seq_len:
                    raise ValueError("token_mask shape must match [B, T]")
                token_mask_flat = token_mask.reshape(-1)
            else:
                token_mask_flat = token_mask.reshape(-1)

        shape_info = None
        if batch_size is not None:
            shape_info = (batch_size, seq_len, dim)
        return x, aux_flat, pos_ids_flat, token_mask_flat, shape_info

    def _dispatch_to_experts(
        self,
        x_flat: torch.Tensor,
        output: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices_flat: torch.Tensor,
        expert_weights_flat: torch.Tensor,
        aux_flat: torch.Tensor | None,
        extra_args: tuple,
        extra_kwargs: dict,
    ) -> None:
        """Dispatch routed tokens to experts and accumulate weighted outputs.

        Rather than scanning the full assignment vector with a boolean mask
        per expert (O(E * num_edges)), this method sorts edges by expert
        index once and then slices contiguous groups, reducing Python-loop
        overhead.

        Experts that receive no routed tokens are called with a single
        zero-weight dummy token so that their parameters participate in the
        autograd graph.  This is essential for DDP: without it,
        ``prepare_for_backward`` would never see these parameters, making
        ``find_unused_parameters=True`` ineffective and causing an
        allreduce deadlock.

        .. note::

           Each expert is still called sequentially.  For higher throughput
           at scale consider grouped-GEMM / Megablocks-style batching.

        Args:
            x_flat: ``[N, D]`` token embeddings.
            output: ``[N, D]`` pre-allocated output tensor (modified in-place
                via :func:`index_add_`).
            token_indices: ``[A]`` token index for each assignment edge.
            expert_indices_flat: ``[A]`` expert index for each assignment edge.
            expert_weights_flat: ``[A]`` gate weight for each assignment edge.
            aux_flat: Optional ``[N, Aux]`` auxiliary conditioning; matching
                rows are gathered for each routed token.
            extra_args: Additional positional args forwarded to the expert.
            extra_kwargs: Additional keyword args forwarded to the expert.
        """
        # Determine which experts are active (received at least one token).
        if token_indices.numel() > 0:
            # Group assignments by expert in one pass to reduce Python-side masking overhead.
            sort_idx = torch.argsort(expert_indices_flat)
            expert_sorted = expert_indices_flat[sort_idx]
            token_sorted = token_indices[sort_idx]
            weight_sorted = expert_weights_flat[sort_idx]

            active_experts, counts = torch.unique_consecutive(expert_sorted, return_counts=True)
            active_experts_set = set(active_experts.tolist())
            offsets = counts.cumsum(0).tolist()

            start = 0
            for expert_idx, end in zip(active_experts.tolist(), offsets, strict=True):
                tok_idx = token_sorted[start:end]
                expert_input = x_flat[tok_idx]
                if aux_flat is not None:
                    expert_output = self.experts[expert_idx](
                        expert_input,
                        aux_flat[tok_idx],
                        *extra_args,
                        **extra_kwargs,
                    )
                else:
                    expert_output = self.experts[expert_idx](
                        expert_input,
                        *extra_args,
                        **extra_kwargs,
                    )

                weighted = expert_output * weight_sorted[start:end].unsqueeze(-1).to(expert_output.dtype)
                output.index_add_(0, tok_idx, weighted.to(output.dtype))
                start = end
        else:
            active_experts_set = set()

        # DDP safety: call idle experts with a dummy token so their
        # parameters are in the autograd graph visible to
        # prepare_for_backward.  The zero multiplier ensures no
        # numerical contribution.
        if self.training:
            for expert_idx in range(self.num_experts):
                if expert_idx not in active_experts_set:
                    dummy_in = x_flat[:1].detach()
                    if aux_flat is not None:
                        dummy_out = self.experts[expert_idx](
                            dummy_in,
                            aux_flat[:1].detach(),
                            *extra_args,
                            **extra_kwargs,
                        )
                    else:
                        dummy_out = self.experts[expert_idx](
                            dummy_in,
                            *extra_args,
                            **extra_kwargs,
                        )
                    # Zero contribution — only creates the autograd edge.
                    output[0] = output[0] + dummy_out.sum() * 0.0

    def forward(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
        *extra_args,
        position_ids: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        **extra_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the full MoE forward: route, dispatch, combine, residual.

        Args:
            x: Token tensor ``[N, D]`` or ``[B, T, D]``.
            aux: Optional per-token conditioning tensor whose leading
                dimensions match *x*.  Passed through to each expert.
            position_ids: Optional per-token position IDs for the spatial
                router.  If ``None`` and default IDs have been registered
                via :meth:`set_position_ids`, those are used instead.
            token_mask: Optional boolean mask of valid tokens.  Masked
                tokens are excluded from routing, capacity accounting, and
                the auxiliary loss.

        Returns:
            output: Same shape as *x*.
            aux_loss: Scalar load-balancing loss when in training mode,
                ``None`` otherwise.
        """
        x_in = x
        pos_ids = position_ids
        if pos_ids is None and self.position_ids.numel() > 0:
            pos_ids = self.position_ids

        x_flat, aux_flat, pos_ids_flat, token_mask_flat, shape_info = self._flatten_tokens(
            x, aux, pos_ids, token_mask
        )
        should_log_debug = self._should_log_debug()

        router_probs, expert_indices, expert_weights, router_logits = self.router(
            x_flat, position_ids=pos_ids_flat, token_mask=token_mask_flat
        )

        n_tokens, dim = x_flat.shape
        output = torch.zeros_like(x_flat)

        token_indices = torch.arange(n_tokens, device=x_flat.device).repeat_interleave(self.top_k)
        expert_indices_flat = expert_indices.reshape(-1)
        expert_weights_flat = expert_weights.reshape(-1)
        # Parallel arrays of length N*K: each entry is one assignment edge
        # (token_idx, expert_idx, gate_weight) from the top-k selection.

        if token_mask_flat is not None:
            valid_assign = token_mask_flat[token_indices]
            token_indices = token_indices[valid_assign]
            expert_indices_flat = expert_indices_flat[valid_assign]
            expert_weights_flat = expert_weights_flat[valid_assign]
        # Snapshot before capacity clipping (used for debug logging).
        expert_indices_pre_capacity = expert_indices_flat

        capacity = None
        if self.capacity_factor is not None and self.capacity_factor > 0:
            num_tokens_for_capacity = n_tokens
            if token_mask_flat is not None:
                num_tokens_for_capacity = int(token_mask_flat.sum().item())
            capacity = int(
                math.ceil(self.capacity_factor * self.top_k * max(num_tokens_for_capacity, 1) / self.num_experts)
            )
            capacity = max(capacity, 1)
            # capacity = max number of assignment edges retained *per expert*.
            # Multiplied by top_k so that CF=1.25 means 25% headroom above
            # the perfectly-balanced case regardless of top_k.

        # --- Capacity enforcement: keep at most `capacity` edges per expert,
        # preferring edges with the highest gate weights. ---
        if capacity is not None and token_indices.numel() > 0:
            keep_mask = torch.zeros_like(expert_indices_flat, dtype=torch.bool)
            for expert_idx in range(self.num_experts):
                mask = expert_indices_flat == expert_idx
                if not mask.any():
                    continue
                idxs = mask.nonzero(as_tuple=True)[0]
                if idxs.numel() > capacity:
                    weights = expert_weights_flat[idxs]
                    topk = torch.topk(weights, capacity, sorted=False).indices
                    idxs = idxs[topk]
                keep_mask[idxs] = True

            token_indices = token_indices[keep_mask]
            expert_indices_flat = expert_indices_flat[keep_mask]
            expert_weights_flat = expert_weights_flat[keep_mask]

            if self.renormalize_gates and token_indices.numel() > 0:
                # Re-normalize the remaining gate mass per token after dropping edges.
                token_weight_sum = torch.zeros(n_tokens, dtype=expert_weights_flat.dtype, device=expert_weights_flat.device)
                token_weight_sum.index_add_(0, token_indices, expert_weights_flat)
                expert_weights_flat = expert_weights_flat / token_weight_sum[token_indices].clamp_min(1e-9)

        self._dispatch_to_experts(
            x_flat,
            output,
            token_indices,
            expert_indices_flat,
            expert_weights_flat,
            aux_flat,
            extra_args,
            extra_kwargs,
        )

        if shape_info is not None:
            batch_size, seq_len, _ = shape_info
            output = output.reshape(batch_size, seq_len, dim)
            if self.with_residual:
                output = output + x_in
            self.last_expert_indices = expert_indices.reshape(batch_size, seq_len, self.top_k).detach()
            self.last_expert_weights = expert_weights.reshape(batch_size, seq_len, self.top_k).detach()
        else:
            if self.with_residual:
                output = output + x_in
            self.last_expert_indices = expert_indices.detach()
            self.last_expert_weights = expert_weights.detach()

        # --- Auxiliary load-balancing loss (training only). ---
        # Uses the *original* router assignments (before capacity clipping) so
        # that the loss penalises the router's raw routing decisions.
        # Accumulates across forecast steps; call reset_aux_loss() before each
        # training step.
        if self.training:
            assignment_weights = torch.zeros(
                (n_tokens, self.num_experts),
                device=x_flat.device,
                dtype=torch.float32,
            )
            assignment_weights.scatter_add_(1, expert_indices, expert_weights.to(assignment_weights.dtype))
            step_aux = self.load_balance_loss(
                router_probs,
                assignment_weights,
                token_mask_flat,
            )
            if self.router_z_loss is not None:
                step_aux = step_aux + self.router_z_loss(router_logits, token_mask_flat)
            if self.last_aux_loss is not None:
                self.last_aux_loss = self.last_aux_loss + step_aux
            else:
                self.last_aux_loss = step_aux

        if should_log_debug:
            self._log_debug_stats(
                n_tokens,
                token_mask_flat,
                expert_indices_pre_capacity,
                expert_indices_flat,
                expert_weights,
                capacity,
            )

        return output, self.last_aux_loss

    def get_aux_loss(self) -> torch.Tensor | None:
        """Return the auxiliary load-balancing loss from the most recent
        forward pass, or ``None`` if the model was in eval mode."""
        return self.last_aux_loss

    def reset_aux_loss(self) -> None:
        """Reset the accumulated auxiliary loss. Call before each training step."""
        self.last_aux_loss = None
