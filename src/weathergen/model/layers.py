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


class StructuredCoordEmbedding(nn.Module):
    """
    Embeds the heterogeneous target-coord vector by processing each semantic
    group independently before fusing, instead of treating the full vector as flat.

    The coord vector is a concatenation of five distinct groups:

        [stream_id(1) | time(5) | geoinfo(G) | 5×vertex_block(75) | 8×center(24)]

    where G = dim_coord_in - 105 and is 0 for grid streams (ERA5, FESOM)
    and 1–3 for observational streams.

    Each group is projected to a shared latent width `d_group`, the five
    vertex blocks and eight center offsets are mean-pooled (permutation-
    equivariant across their respective elements), all group embeddings are
    summed, and the result passes through LayerNorm → Linear to produce the
    final embedding of size dim_embed.

    Key properties vs. a flat MLP:
    - Shared vertex / center projections give 5× / 8× more gradient signal
      per parameter update for the geometric groups.
    - Sum-of-groups fusion lets each group contribute independently; a flat
      MLP must discover the group boundaries from data.
    - stream_id is treated as a continuous 1-D feature (safe when IDs are
      non-contiguous integers such as 0, 2, …); upgrade to nn.Embedding once
      IDs are confirmed to be dense and 0-indexed.

    Config keys (under embed_target_coords):
        net      : "structured"
        dim_embed: <int>   output embedding dimension (required)
        d_group  : <int>   internal group projection width (default 64)

    Example stream config::

        embed_target_coords:
          net      : structured
          dim_embed: 256
          d_group  : 64   # optional
    """

    # Fixed sizes derived from get_target_coords_local() in tokenizer_utils.py
    _STREAM_ID_SIZE: int = 1
    _TIME_SIZE: int = 5
    _VERTEX_DIM: int = 15  # 3D cell-centre delta + 12D local frame, per vertex
    _NUM_VERTICES: int = 5
    _CENTER_DIM: int = 3  # 3D offset per HEALPix neighbour centre
    _NUM_CENTERS: int = 8
    _FIXED_SIZE: int = (
        _STREAM_ID_SIZE
        + _TIME_SIZE
        + _NUM_VERTICES * _VERTEX_DIM  # 75
        + _NUM_CENTERS * _CENTER_DIM  # 24
    )  # = 105; geoinfo_size is appended on top

    def __init__(
        self,
        dim_coord_in: int,
        dim_embed: int,
        d_group: int = 64,
        dropout_rate: float = 0.0,
        name: str | None = None,
    ):
        super().__init__()

        if name is not None:
            self.name = name

        self.geoinfo_size: int = dim_coord_in - self._FIXED_SIZE
        assert self.geoinfo_size >= 0, (
            f"StructuredCoordEmbedding: dim_coord_in={dim_coord_in} is smaller than "
            f"the minimum fixed size {self._FIXED_SIZE}. "
            "Check that dim_coord_in comes from get_targets_coords_size()."
        )

        # One linear per semantic group (bias=False; fusion LayerNorm provides bias)
        self.stream_proj = nn.Linear(self._STREAM_ID_SIZE, d_group, bias=False)
        self.time_proj = nn.Linear(self._TIME_SIZE, d_group, bias=False)
        # Shared projection applied independently to each of the 5 vertex blocks
        self.vertex_proj = nn.Linear(self._VERTEX_DIM, d_group, bias=False)
        # Shared projection applied independently to each of the 8 centre offsets
        self.center_proj = nn.Linear(self._CENTER_DIM, d_group, bias=False)
        # Geoinfo projection is optional: skipped when geoinfo_size == 0 (e.g. ERA5)
        self.geo_proj: nn.Linear | None = (
            nn.Linear(self.geoinfo_size, d_group, bias=False) if self.geoinfo_size > 0 else None
        )

        self.fusion_norm = nn.LayerNorm(d_group)
        self.out_proj = nn.Linear(d_group, dim_embed)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, dim_coord_in) — flat target-coord vector as produced by
               get_target_coords_local() in tokenizer_utils.py.
        Returns:
            (N, dim_embed) embedding ready to be used as cross-attention queries.
        """
        n = x.shape[0]
        g = self.geoinfo_size
        # Offset where the five vertex blocks start (after stream_id + time + geoinfo)
        v_start = self._STREAM_ID_SIZE + self._TIME_SIZE + g

        stream_id = x[:, :1]  # (n, 1)
        time_enc = x[:, 1 : 1 + self._TIME_SIZE]  # (n, 5)
        vertices = x[:, v_start : v_start + self._NUM_VERTICES * self._VERTEX_DIM]
        vertices = vertices.view(n, self._NUM_VERTICES, self._VERTEX_DIM)  # (n, 5, 15)
        c_start = v_start + self._NUM_VERTICES * self._VERTEX_DIM
        centers = x[:, c_start : c_start + self._NUM_CENTERS * self._CENTER_DIM]
        centers = centers.view(n, self._NUM_CENTERS, self._CENTER_DIM)  # (n, 8, 3)

        # Project each group and accumulate into a single d_group vector
        fused = (
            self.stream_proj(stream_id)  # (n, d_group)
            + self.time_proj(time_enc)  # (n, d_group)
            + self.vertex_proj(vertices).mean(1)  # (n, d_group) — mean over 5 vertices
            + self.center_proj(centers).mean(1)  # (n, d_group) — mean over 8 centres
        )

        if g > 0:
            geoinfo = x[:, 1 + self._TIME_SIZE : 1 + self._TIME_SIZE + g]  # (n, g)
            fused = fused + self.geo_proj(geoinfo)  # type: ignore[misc]

        return self.out_proj(self.fusion_norm(fused))  # (n, dim_embed)
