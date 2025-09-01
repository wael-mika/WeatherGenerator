# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
from torch import nn
from weathergen.model.norms import AdaLayerNorm, RMSNorm


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

class _DenseBlock(nn.Module):
    """A tiny FFN that mirrors the structure of your MLP stack."""
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers=2,
                 nonlin=nn.GELU, dropout_rate=0.0):
        super().__init__()
        layers = [nn.Linear(dim_in, dim_hidden), nonlin(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(dim_hidden, dim_hidden), nonlin(), nn.Dropout(dropout_rate)]
        layers += [nn.Linear(dim_hidden, dim_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MoEMLP(nn.Module):
    """
    Drop-in MoE MLP:
    - Same call pattern as your MLP: forward(*args) where args=(x, ...) and optional aux at the end
    - Supports residual add exactly like MLP
    - Optional AdaLayerNorm when dim_aux is provided
    - Simple top-k router; computes all experts and mixes with gated weights (small-E friendly)
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        num_layers=2,
        hidden_factor=2,
        pre_layer_norm=True,
        dropout_rate=0.0,
        nonlin=nn.GELU,
        with_residual=False,
        norm_type="LayerNorm",
        dim_aux=None,
        norm_eps=1e-5,
        name: str | None = None,
        # MoE bits
        num_experts: int = 8,
        top_k: int = 4,
        router_noisy_std: float = 0.0,  # set >0 to add noise to router logits
    ):
        super().__init__()
        if name is not None:
            self.name = name

        assert num_layers >= 2
        assert 1 <= top_k <= num_experts

        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        self.pre_layer_norm = pre_layer_norm
        self.top_k = top_k
        self.num_experts = num_experts

        dim_hidden = int(dim_in * hidden_factor)

        # Norm (match your MLP behavior)
        Norm = nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm
        if pre_layer_norm:
            self.norm = (
                Norm(dim_in, eps=norm_eps)
                if dim_aux is None
                else AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps)
            )
        else:
            self.norm = None  # no pre-norm

        # Router
        self.router = nn.Linear(dim_in, num_experts)
        self.router_noisy_std = router_noisy_std

        # Experts (identical shape)
        self.experts = nn.ModuleList(
            [
                _DenseBlock(
                    dim_in=dim_in,
                    dim_hidden=dim_hidden,
                    dim_out=dim_out,
                    num_layers=num_layers,
                    nonlin=nonlin,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_experts)
            ]
        )

        # For optional aux loss (load-balancing); not used unless you read it
        self.register_buffer("last_aux_loss", torch.zeros((), dtype=torch.float32))

    def _gate(self, x_norm):
        # x_norm: [*, D]. Router works on the last dim.
        logits = self.router(x_norm)
        if self.router_noisy_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noisy_std

        if self.top_k == self.num_experts:
            # softmax over all experts
            weights = torch.softmax(logits, dim=-1)
            top_idx = None  # not needed
        else:
            # top-k softmax
            top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)  # [*, k]
            weights = torch.softmax(top_vals, dim=-1)                      # [*, k]
        return weights, top_idx

    @torch.no_grad()
    def _compute_load_balance_aux(self, weights, top_idx, num_experts):
        """
        Simple load-balancing penalty from Switch/MoE papers:
        Encourage uniform expert probability and uniform usage.
        Works with both full-softmax (top_idx None) and top-k.
        """
        # weights: [B..., K] or [B..., E], top_idx: [B..., K] or None
        # Collapse batch/time dims:
        if top_idx is None:
            # weights over E
            probs = weights.mean(dim=tuple(range(weights.dim() - 1)))  # [E]
            usage = probs  # identical here
        else:
            # Build usage over experts from top-k selection
            *prefix, K = weights.shape
            flat_w = weights.reshape(-1, K)         # [N, K]
            flat_i = top_idx.reshape(-1, K)         # [N, K]
            E = num_experts
            usage = torch.zeros(E, device=weights.device, dtype=weights.dtype)
            # Sum weights per expert
            usage.scatter_add_(0, flat_i.reshape(-1), flat_w.reshape(-1))
            usage = usage / usage.sum().clamp_min(1e-6)  # normalize
            probs = usage  # proxy
        # Target is uniform 1/E
        E = num_experts
        target = torch.full_like(probs, 1.0 / E)
        # KL-like penalty (symmetric-ish, simple)
        aux = (probs * (probs.add(1e-6).log() - target.add(1e-6).log())).sum()
        return aux

    def forward(self, *args):
        # Match your MLP(*args) calling convention
        x = args[0]
        x_in = x
        aux = args[-1] if self.with_aux else None

        # Optional pre-norm (possibly adaptive)
        if self.norm is not None:
            if self.with_aux:
                x = self.norm(x, aux)
            else:
                x = self.norm(x)

        # Router
        weights, top_idx = self._gate(x)

        # Compute all experts once (simple and fine for small E)
        expert_outs = [exp(x) for exp in self.experts]  # each [*, D_out]
        # Stack to [E, *, D_out] then combine
        y_stack = torch.stack(expert_outs, dim=0)  # [E, ..., D_out]

        if top_idx is None:
            # Full-softmax combine
            # weights: [..., E] -> reshape to [1, ..., E, 1] to broadcast
            combine_w = weights.unsqueeze(0).transpose(0, -1).transpose(0, -1)  # legacy-safe
            # Simpler: align dims
            # Want [E, ..., 1]
            combine_w = weights.transpose(-1, -2) if weights.dim() == 2 else weights
            # robust way:
            combine_w = weights.unsqueeze(0)              # [1, ..., E]
            combine_w = combine_w.movedim(-1, 0)          # [E, ...]
            combine_w = combine_w.unsqueeze(-1)           # [E, ..., 1]
            y = (y_stack * combine_w).sum(dim=0)          # [..., D_out]
        else:
            # Top-k combine: build gated sum via gather
            # y_stack: [E, ..., D_out] -> expand gather index to expert dim
            # Create zeros output and add selected experts weighted
            y = torch.zeros_like(y_stack[0])
            # Flatten prefix for simple loop (top_k usually tiny)
            for k in range(self.top_k):
                idx_k = top_idx[..., k]                   # [...]
                w_k = weights[..., k]                     # [...]
                # Gather expert outputs: for each position pick expert idx_k
                # Convert [E, ..., D] -> [..., D] by indexing expert dim
                sel = y_stack.index_select(0, idx_k.reshape(-1)).reshape(*idx_k.shape, *y_stack.shape[2:])
                # Weight and accumulate
                y = y + sel * w_k.unsqueeze(-1)

        # Residual (same logic as your MLP)
        if self.with_residual:
            if y.shape[-1] == x_in.shape[-1]:
                y = x_in + y
            else:
                assert y.shape[-1] % x_in.shape[-1] == 0
                y = y + x_in.repeat([*[1 for _ in y.shape[:-1]], y.shape[-1] // x_in.shape[-1]])

        # Optional: update aux loss (not returned; read if you want)
        with torch.no_grad():
            self.last_aux_loss = self._compute_load_balance_aux(weights, top_idx, self.num_experts)

        return y
