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
        """Constructor"""

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

    def forward(self, *args):
        x, x_in, aux = args[0], args[0], args[-1]

        for i, layer in enumerate(self.layers):
            x = layer(x, aux) if (i == 0 and self.with_aux) else layer(x)

        if self.with_residual:
            if x.shape[-1] == x_in.shape[-1]:
                x = x_in + x
            else:
                assert x.shape[-1] % x_in.shape[-1] == 0
                x = x + x_in.repeat([*[1 for _ in x.shape[:-1]], x.shape[-1] // x_in.shape[-1]])

        return x


class LoadBalancingLoss(torch.nn.Module):
    """
    Auxiliary load-balancing loss from Switch Transformers.
    """

    def __init__(self, num_experts: int, weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight

    def forward(
        self,
        router_probs: torch.Tensor,
        assignment_weights: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            router_probs: [N, E] router probabilities.
            assignment_weights: [N, E] per-token expert assignment weights (sum to 1 per token).
            token_mask: [N] optional mask for valid tokens.
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


class MoERouter(torch.nn.Module):
    """
    Top-k router for mixture-of-experts blocks.
    """

    def __init__(
        self,
        dim_in: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        router_bias: bool = False,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.router_weights = nn.Linear(dim_in, num_experts, bias=router_bias)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del position_ids
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim != 2:
            raise ValueError(f"MoERouter expects [N, D] or [B, T, D], got {tuple(x.shape)}")

        router_logits = self.router_weights(x)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

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
        return router_probs, expert_indices, expert_weights


class SpatialMoERouter(torch.nn.Module):
    """
    Router that augments token features with learned spatial position embeddings.
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
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        self.num_experts = num_experts
        self.num_positions = num_positions
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.position_embed_dim = position_embed_dim
        self.position_embed = nn.Embedding(num_positions, position_embed_dim)
        self.router_weights = nn.Linear(dim_in + position_embed_dim, num_experts, bias=router_bias)

    def initialize_from_coordinates(self, theta: torch.Tensor, phi: torch.Tensor):
        assert len(theta) == self.num_positions, (len(theta), self.num_positions)
        assert len(phi) == self.num_positions, (len(phi), self.num_positions)

        device = self.position_embed.weight.device
        theta = theta.to(device)
        phi = phi.to(device)

        with torch.no_grad():
            embed_dim = self.position_embed_dim
            embeddings = torch.zeros(self.num_positions, embed_dim, device=device)
            half_dim = embed_dim // 2

            freqs_theta = torch.exp(
                torch.arange(0, half_dim, 2, device=device).float()
                * -(np.log(10000.0) / max(half_dim, 1))
            )
            embeddings[:, 0:half_dim:2] = torch.sin(theta.unsqueeze(1) * freqs_theta)
            embeddings[:, 1:half_dim:2] = torch.cos(theta.unsqueeze(1) * freqs_theta)

            freqs_phi = torch.exp(
                torch.arange(0, half_dim, 2, device=device).float()
                * -(np.log(10000.0) / max(half_dim, 1))
            )
            embeddings[:, half_dim + 0 :: 2] = torch.sin(phi.unsqueeze(1) * freqs_phi)
            embeddings[:, half_dim + 1 :: 2] = torch.cos(phi.unsqueeze(1) * freqs_phi)
            self.position_embed.weight.copy_(embeddings)

        return self

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        pos_embed = torch.zeros(
            (num_tokens, self.position_embed_dim), device=x.device, dtype=x.dtype
        )
        valid_pos = position_ids >= 0
        if valid_pos.any():
            pos_embed[valid_pos] = self.position_embed(position_ids[valid_pos] % self.num_positions)

        x_with_pos = torch.cat([x, pos_embed], dim=-1)

        router_logits = self.router_weights(x_with_pos)
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

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

        return router_probs, expert_indices, expert_weights


class MoEBlock(torch.nn.Module):
    """
    General mixture-of-experts wrapper around any expert factory.
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
        debug_enabled: bool = False,
        debug_interval: int = 100,
        debug_top_experts: int = 3,
        debug_name: str | None = None,
    ):
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
            )
        else:
            self.router = MoERouter(
                dim_in=dim_in,
                num_experts=num_experts,
                top_k=top_k,
                jitter_noise=jitter_noise,
                router_bias=router_bias,
            )

        self.load_balance_loss = LoadBalancingLoss(
            num_experts=num_experts,
            weight=load_balance_weight,
        )
        self.last_aux_loss: torch.Tensor | None = None
        self.last_expert_indices: torch.Tensor | None = None
        self.last_expert_weights: torch.Tensor | None = None
        self.register_buffer("position_ids", torch.empty(0, dtype=torch.long), persistent=False)

    def _should_log_debug(self) -> bool:
        if not (self.debug_enabled and self.training):
            return False
        self._debug_forward_counter += 1
        if self._debug_forward_counter % self.debug_interval != 0:
            return False
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return False
        return True

    def _format_top_experts(self, expert_indices_flat: torch.Tensor) -> str:
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
        if position_ids is None:
            self.position_ids = torch.empty(0, dtype=torch.long, device=self.position_ids.device)
        else:
            self.position_ids = position_ids.to(device=self.position_ids.device, dtype=torch.long)

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

    def forward(
        self,
        x: torch.Tensor,
        aux: torch.Tensor | None = None,
        *extra_args,
        position_ids: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        **extra_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_in = x
        pos_ids = position_ids
        if pos_ids is None and self.position_ids.numel() > 0:
            pos_ids = self.position_ids

        x_flat, aux_flat, pos_ids_flat, token_mask_flat, shape_info = self._flatten_tokens(
            x, aux, pos_ids, token_mask
        )
        should_log_debug = self._should_log_debug()

        router_probs, expert_indices, expert_weights = self.router(
            x_flat, position_ids=pos_ids_flat, token_mask=token_mask_flat
        )

        n_tokens, dim = x_flat.shape
        output = torch.zeros_like(x_flat)

        token_indices = torch.arange(n_tokens, device=x_flat.device).repeat_interleave(self.top_k)
        expert_indices_flat = expert_indices.reshape(-1)
        expert_weights_flat = expert_weights.reshape(-1)

        if token_mask_flat is not None:
            valid_assign = token_mask_flat[token_indices]
            token_indices = token_indices[valid_assign]
            expert_indices_flat = expert_indices_flat[valid_assign]
            expert_weights_flat = expert_weights_flat[valid_assign]
        expert_indices_pre_capacity = expert_indices_flat

        capacity = None
        if self.capacity_factor is not None and self.capacity_factor > 0:
            num_tokens_for_capacity = n_tokens
            if token_mask_flat is not None:
                num_tokens_for_capacity = int(token_mask_flat.sum().item())
            capacity = int(
                math.ceil(self.capacity_factor * max(num_tokens_for_capacity, 1) / self.num_experts)
            )
            capacity = max(capacity, 1)

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
                token_weight_sum = x_flat.new_zeros(n_tokens)
                token_weight_sum.index_add_(0, token_indices, expert_weights_flat)
                expert_weights_flat = expert_weights_flat / token_weight_sum[token_indices].clamp_min(1e-9)

        if token_indices.numel() > 0:
            for expert_idx in range(self.num_experts):
                mask = expert_indices_flat == expert_idx
                if not mask.any():
                    continue
                tok_idx = token_indices[mask]
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

                weighted = expert_output * expert_weights_flat[mask].unsqueeze(-1)
                output.index_add_(0, tok_idx, weighted)

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

        self.last_aux_loss = None
        if self.training:
            assignment_weights = torch.zeros(
                (n_tokens, self.num_experts),
                device=x_flat.device,
                dtype=torch.float32,
            )
            assignment_weights.scatter_add_(1, expert_indices, expert_weights)
            self.last_aux_loss = self.load_balance_loss(
                router_probs,
                assignment_weights,
                token_mask_flat,
            )

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
        return self.last_aux_loss
