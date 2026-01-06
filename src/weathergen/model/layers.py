# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple

from weathergen.model.norms import AdaLayerNorm, RMSNorm


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
    Load balancing loss for MoE routing to encourage uniform expert utilization.

    Implements the auxiliary loss from "Switch Transformers" (Fedus et al., 2021):
    L_balance = num_experts * sum_i(f_i * P_i)

    where:
        f_i = fraction of tokens routed to expert i
        P_i = average router probability for expert i

    This encourages balanced load across experts.
    """

    def __init__(self, num_experts: int, weight: float = 0.01):
        """
        Args:
            num_experts: Number of experts in the mixture
            weight: Weight coefficient for the auxiliary loss (default: 0.01)
        """
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight

    def forward(self, router_probs: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss.

        Args:
            router_probs: Router probabilities [batch_size, seq_len, num_experts]
            expert_mask: Binary mask of selected experts [batch_size, seq_len, num_experts]

        Returns:
            Scalar loss value
        """
        # Average router probability per expert across all tokens
        # P_i in the formula
        mean_prob = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Fraction of tokens assigned to each expert
        # f_i in the formula
        mean_assignment = expert_mask.float().mean(dim=[0, 1])  # [num_experts]

        # Load balancing loss: encourages P_i * f_i to be uniform
        loss = self.num_experts * (mean_prob * mean_assignment).sum()

        return self.weight * loss


class MoERouter(torch.nn.Module):
    """
    Router module for Mixture of Experts.

    Learns to route tokens to experts based on token representations.
    Supports top-k routing where each token is processed by k experts.
    """

    def __init__(
        self,
        dim_in: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        router_bias: bool = False,
    ):
        """
        Args:
            dim_in: Input dimension
            num_experts: Number of experts in the mixture
            top_k: Number of experts to route each token to (default: 2)
            jitter_noise: Standard deviation of noise added during training (default: 0.0)
            router_bias: Whether to use bias in router linear layer (default: False)
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

        # Router learns to map input to expert scores
        self.router_weights = nn.Linear(dim_in, num_experts, bias=router_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: Input tensor [batch_size, seq_len, dim_in]

        Returns:
            router_probs: Softmax probabilities over all experts [batch_size, seq_len, num_experts]
            expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            expert_weights: Normalized weights for selected experts [batch_size, seq_len, top_k]
        """
        # Compute router logits
        router_logits = self.router_weights(x)  # [batch_size, seq_len, num_experts]

        # Add jitter noise during training for exploration
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

        # Compute probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]

        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # [batch_size, seq_len, top_k]

        # Normalize weights of selected experts to sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return router_probs, expert_indices, expert_weights


class MoEBlock(torch.nn.Module):
    """
    Mixture of Experts (MoE) block with load-balanced routing.

    This is a general-purpose MoE implementation that can wrap any expert module
    (MLP, attention, custom layers, etc.) and route tokens to experts based on
    learned routing decisions.

    Key features:
    - Flexible expert creation via factory function
    - Top-k routing with load balancing
    - Auxiliary loss for uniform expert utilization
    - Support for residual connections
    - Compatible with auxiliary conditioning (e.g., AdaLayerNorm)

    Usage example:
        # MoE with MLP experts
        moe = MoEBlock(
            expert_fn=lambda: MLP(dim_in=512, dim_out=512, hidden_factor=2),
            dim_in=512,
            num_experts=8,
            top_k=2,
        )

        # Forward pass
        output, aux_loss = moe(x)
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
        with_residual: bool = True,
        name: str | None = None,
    ):
        """
        Args:
            expert_fn: Factory function that creates a single expert module.
                      Called num_experts times to create all experts.
                      Example: lambda: MLP(dim_in=512, dim_out=512)
            dim_in: Input dimension for routing
            num_experts: Number of experts in the mixture (default: 8)
            top_k: Number of experts each token is routed to (default: 2)
            capacity_factor: Expert capacity factor for handling token overflow (default: 1.25)
            jitter_noise: Noise added to router logits during training (default: 0.0)
            router_bias: Whether router uses bias (default: False)
            load_balance_weight: Weight for load balancing auxiliary loss (default: 0.01)
            with_residual: Whether to add residual connection (default: True)
            name: Optional name for the module
        """
        super().__init__()

        if name is not None:
            self.name = name

        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.with_residual = with_residual

        # Create experts using the factory function
        self.experts = nn.ModuleList([expert_fn() for _ in range(num_experts)])

        # Router for expert selection
        self.router = MoERouter(
            dim_in=dim_in,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
            router_bias=router_bias,
        )

        # Load balancing loss
        self.load_balance_loss = LoadBalancingLoss(
            num_experts=num_experts,
            weight=load_balance_weight,
        )

        # Track auxiliary loss for this block
        self.last_aux_loss = None

    def forward(self, *args) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with expert routing.

        Args:
            *args: Variable arguments to support auxiliary inputs.
                   First argument is always the input tensor x.
                   Last argument may be auxiliary conditioning (e.g., for AdaLayerNorm).

        Returns:
            output: Processed tensor with same shape as input
            aux_loss: Load balancing auxiliary loss (None if not training)
        """
        x = args[0]  # Input tensor
        x_in = x  # Save for residual connection

        batch_size, seq_len, dim = x.shape

        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, dim)  # [batch_size * seq_len, dim]

        # Route tokens to experts
        router_probs, expert_indices, expert_weights = self.router(x)
        # router_probs: [batch_size, seq_len, num_experts]
        # expert_indices: [batch_size, seq_len, top_k]
        # expert_weights: [batch_size, seq_len, top_k]

        # Initialize output
        output = torch.zeros_like(x_flat)  # [batch_size * seq_len, dim]

        # Flatten indices and weights for processing
        expert_indices_flat = expert_indices.view(-1, self.top_k)  # [batch_size * seq_len, top_k]
        expert_weights_flat = expert_weights.view(-1, self.top_k)  # [batch_size * seq_len, top_k]

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            # Create mask for all top_k positions
            expert_mask = (expert_indices_flat == expert_idx)  # [batch_size * seq_len, top_k]
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]  # Indices of tokens using this expert

            if len(token_indices) == 0:
                continue  # No tokens for this expert

            # Get tokens for this expert
            expert_input = x_flat[token_indices]  # [num_tokens, dim]

            # Pass through expert (handle both simple and auxiliary-conditioned experts)
            if len(args) > 1:
                # Expert may need auxiliary input (e.g., MLP with AdaLayerNorm)
                expert_output = self.experts[expert_idx](*[expert_input] + list(args[1:]))
            else:
                expert_output = self.experts[expert_idx](expert_input)

            # Weight expert outputs and accumulate
            for k in range(self.top_k):
                # Find which tokens use this expert in position k
                k_mask = expert_mask[:, k]  # [batch_size * seq_len]
                k_token_indices = k_mask.nonzero(as_tuple=True)[0]

                if len(k_token_indices) == 0:
                    continue

                # Map from expert's output to original tokens
                expert_output_indices = torch.isin(token_indices, k_token_indices).nonzero(as_tuple=True)[0]

                # Get weights for these tokens
                weights = expert_weights_flat[k_token_indices, k].unsqueeze(-1)  # [num_k_tokens, 1]

                # Accumulate weighted output
                output[k_token_indices] += weights * expert_output[expert_output_indices]

        # Reshape output
        output = output.view(batch_size, seq_len, dim)

        # Add residual connection if requested
        if self.with_residual:
            output = output + x_in

        # Compute auxiliary load balancing loss during training
        aux_loss = None
        if self.training:
            # Create binary mask of expert assignments for load balancing
            expert_mask = torch.zeros(
                batch_size, seq_len, self.num_experts,
                device=x.device, dtype=torch.float32
            )
            expert_mask.scatter_(2, expert_indices, 1.0)

            aux_loss = self.load_balance_loss(router_probs, expert_mask)
            self.last_aux_loss = aux_loss

        return output, aux_loss

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        """
        Get the last computed auxiliary loss.
        Useful for accumulating losses across multiple MoE blocks.

        Returns:
            Last auxiliary loss or None
        """
        return self.last_aux_loss

    def get_routing_stats(self, x: torch.Tensor) -> dict:
        """
        Get routing statistics for analysis.

        Args:
            x: Input tensor [batch_size, seq_len, dim]

        Returns:
            Dictionary with routing statistics:
                - expert_utilization: Fraction of tokens per expert
                - router_entropy: Entropy of router distribution
                - top1_expert_distribution: Distribution of top-1 expert choices
        """
        with torch.no_grad():
            router_probs, expert_indices, expert_weights = self.router(x)

            # Expert utilization
            expert_mask = torch.zeros(
                x.shape[0], x.shape[1], self.num_experts,
                device=x.device, dtype=torch.float32
            )
            expert_mask.scatter_(2, expert_indices, 1.0)
            expert_utilization = expert_mask.mean(dim=[0, 1])  # [num_experts]

            # Router entropy (higher = more uncertain/balanced)
            entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()

            # Top-1 expert distribution
            top1_experts = expert_indices[:, :, 0]  # [batch_size, seq_len]
            top1_distribution = torch.bincount(
                top1_experts.flatten(),
                minlength=self.num_experts
            ).float() / top1_experts.numel()

            return {
                "expert_utilization": expert_utilization.cpu().numpy(),
                "router_entropy": entropy.item(),
                "top1_expert_distribution": top1_distribution.cpu().numpy(),
            }
