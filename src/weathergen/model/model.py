# ruff: noqa: T201
# (C) Copyright 2025 WeatherGenerator contributors.

#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
import typing
import warnings

import astropy_healpix as hp
import astropy_healpix.healpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.encoder import EncoderModule
from weathergen.model.engines import (
    BilinearDecoder,
    EnsPredictionHead,
    ForecastingEngine,
    IdentityEngine,
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
    LatentState,
    TargetPredictionEngine,
    TargetPredictionEngineClassic,
    TargetPredictionEngineMLP,
)
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.utils import _resolve_variable_groups, get_num_parameters
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import get_dtype, is_stream_forcing

logger = logging.getLogger(__name__)


# Number of cells in a 2-ring HEALPix neighbourhood (self + 8 ring-1 + 16 ring-2).
# Interior cells have exactly 25; boundary cells are padded with self-cell.
_RING2_SIZE = 25


def _pick_tte_cls(decoder_type: str):
    """Return the TTE class that corresponds to decoder_type.

    Used for both the shared-TTE path and per-group Option-B TTEs so the
    same decoder_type applies consistently across all routing paths.
    """
    if decoder_type == "MLPDecoder":
        return TargetPredictionEngineMLP
    if decoder_type == "PerceiverIOCoordConditioning":
        return TargetPredictionEngineClassic
    return TargetPredictionEngine


type StreamName = str


class ModelOutput:
    """
    Representation of model output
    """

    physical: list[dict[StreamName, torch.Tensor]]
    latent: list[dict[str, torch.Tensor | LatentState]]

    def __init__(self, len_output: int) -> None:
        self.physical = [{} for _ in range(len_output)]
        self.latent = [{} for _ in range(len_output)]

    def add_physical_prediction(
        self, fstep: int, stream_name: StreamName, pred: torch.Tensor
    ) -> None:
        self.physical[fstep][stream_name] = pred

    def add_latent_prediction(self, fstep: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[fstep][latent_name] = pred

    def get_physical_prediction(
        self, fstep: int, stream_name: StreamName | None = None, sample_idx: int | None = None
    ):
        pred = self.physical[fstep]
        if stream_name is not None:
            pred = pred.get(stream_name, None)
            if sample_idx is not None:
                assert sample_idx < len(pred), "Invalid sample index."
                pred = pred[sample_idx]
        return pred

    def get_latent_prediction(self, fstep: int):
        return self.latent[fstep]


class ModelParams(torch.nn.Module):
    """Creation of query and embedding parameters of the model."""

    def __init__(self, cf) -> None:
        super(ModelParams, self).__init__()

        self.cf = cf

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**cf.healpix_level
        self.dtype = get_dtype(cf.attention_dtype)

        # Positional embeddings
        self.max_tokens_local_per_cell = cf.get("ae_local_max_tokens_per_cell", 64)
        self.pe_embed = torch.nn.Parameter(
            torch.zeros(self.max_tokens_local_per_cell, cf.ae_local_dim_embed, dtype=self.dtype),
            requires_grad=False,
        )

        pe = torch.zeros(
            self.num_healpix_cells,
            cf.ae_local_num_queries,
            cf.ae_global_dim_embed,
            dtype=self.dtype,
        )
        self.pe_global = torch.nn.Parameter(pe, requires_grad=False)

        # RoPE coordinates
        self.rope_2D = cf.get("rope_2D", False)
        if self.rope_2D:
            self.num_extra_tokens = cf.num_register_tokens + cf.num_class_tokens
            total_tokens = (
                self.num_healpix_cells + self.num_extra_tokens
            ) * cf.ae_local_num_queries
            self.register_buffer(
                "rope_coords",
                torch.zeros(
                    1,
                    total_tokens,
                    2,
                    dtype=self.dtype,
                ),
            )
            self.register_buffer(
                "rope_cell_coords",
                torch.zeros(
                    self.num_healpix_cells,
                    2,
                    dtype=self.dtype,
                ),
            )
        else:
            self.rope_coords = None
            self.rope_cell_coords = None

        # HEALPix neighbours
        hlc = self.healpix_level
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(
                np.arange(self.num_healpix_cells), 2**hlc, order="nested"
            ).transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        self.hp_nbours = torch.nn.Parameter(
            torch.empty((temp.shape[0], (temp.shape[1] + 1)), dtype=torch.int32),
            requires_grad=False,
        )
        # 2-ring neighbourhood for Decoder B1 (see _RING2_SIZE).
        self.hp_nbours_2ring = torch.nn.Parameter(
            torch.empty((self.num_healpix_cells, _RING2_SIZE), dtype=torch.int32),
            requires_grad=False,
        )

        self.q_cells_lens = torch.nn.Parameter(
            torch.ones(self.num_healpix_cells + 1, dtype=torch.int32), requires_grad=False
        )
        self.q_cells_lens.data[0] = 0

    def create(self, cf: Config) -> "ModelParams":
        self.reset_parameters(cf)
        return self

    def reset_parameters(self, cf: Config) -> "ModelParams":
        """Creates positional embedding for each grid point for each stream used after stream
        embedding, positional embedding for all stream assimilated cell-level local embedding,
        initializing queries for local-to-global adapters, HEALPix neighbourhood based parameter
        initializing for target prediction.

        Sinusoidal positional encoding: Harmonic positional encoding based upon sine and cosine for
            both per stream after stream embedding and per cell level for local assimilation.

        HEALPix neighbourhood structure: Determine the neighbors for each cell and initialize each
            with its own cell number as well as the cell numbers of its neighbors. If a cell has
            fewer than eight neighbors, use its own cell number to fill the remaining slots.

        Query len based parameter creation: Calculate parameters for the calculated token length at
            each cell after local assimilation.

        Args:
            cf : Configuration
        """

        # positional encodings

        dim_embed = cf.ae_local_dim_embed
        token_idx_bias = 16
        freq_bias = 8
        self.pe_embed.data.fill_(0.0)
        position = torch.arange(
            token_idx_bias,
            token_idx_bias + self.max_tokens_local_per_cell,
            device=self.pe_embed.device,
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(freq_bias, freq_bias + dim_embed, 2, device=self.pe_embed.device)
            * -(math.log(self.max_tokens_local_per_cell) / dim_embed),
        )
        self.pe_embed.data[:, 0::2] = torch.sin(position * div[: self.pe_embed[:, 0::2].shape[1]])
        self.pe_embed.data[:, 1::2] = torch.cos(position * div[: self.pe_embed[:, 1::2].shape[1]])

        dim_embed = cf.ae_global_dim_embed

        if self.rope_2D:
            # Precompute per-cell center coordinates (lat, lon in radians) for 2D RoPE.
            # Shape: (num_healpix_cells, ae_local_num_queries, 2)
            verts, _ = healpix_verts_rots(self.healpix_level, 0.5, 0.5)
            coords = r3tos2(verts.to(self.rope_coords.device)).to(self.rope_coords.dtype)
            # Per-cell coords for QueryAggregationEngine (no query expansion)
            self.rope_cell_coords.data.copy_(coords)
            coords = coords.unsqueeze(1).repeat(1, cf.ae_local_num_queries, 1)
            coords_flat = coords.flatten(0, 1).unsqueeze(0)
            offset = self.num_extra_tokens * cf.ae_local_num_queries
            self.rope_coords.data.fill_(0.0)
            self.rope_coords.data[:, offset : offset + coords_flat.shape[1], :].copy_(coords_flat)

        # pe_global: always initialized. RoPE handles relative position in Q/K, but pe_global
        # provides per-cell token identity which is critical for masked cells that have no
        # content from local assimilation. Without it, masked cells are identical and the
        # teacher representation (evaluated without dropout) collapses to low rank.
        self.pe_global.data.fill_(0.0)
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=self.pe_global.device) / dim_embed
        self.pe_global.data[..., 0::2] = 0.5 * torch.sin(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 0::2] += (
            torch.sin(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        self.pe_global.data[..., 1::2] = 0.5 * torch.cos(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 1::2] += (
            torch.cos(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )

        # healpix neighborhood structure

        hlc = self.healpix_level
        num_healpix_cells = self.num_healpix_cells
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(np.arange(num_healpix_cells), 2**hlc, order="nested").transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        # nbors *and* self
        self.hp_nbours.data[:, 0] = torch.arange(temp.shape[0], device=self.hp_nbours.device)
        self.hp_nbours.data[:, 1:] = torch.from_numpy(temp).to(self.hp_nbours.device)

        # 2-ring neighbourhood: union of 1-rings of all 1-ring cells, padded to _RING2_SIZE.
        # Uses hp_nbours (just populated above) as the 1-ring lookup.
        ring1_np = self.hp_nbours.data.cpu().numpy()  # (num_cells, 9): [self, n1..n8]
        ring2_padded = np.empty((num_healpix_cells, _RING2_SIZE), dtype=np.int32)
        for _i in range(num_healpix_cells):
            candidates = ring1_np[ring1_np[_i]].ravel()  # 9 cells × 9 each = 81 candidates
            unique_cells = np.unique(candidates)  # deduplicated; at most 25 for interior cells
            _n = len(unique_cells)
            ring2_padded[_i, :_n] = unique_cells
            if _n < _RING2_SIZE:
                ring2_padded[_i, _n:] = _i  # pad with self
        self.hp_nbours_2ring.data.copy_(
            torch.from_numpy(ring2_padded).to(self.hp_nbours_2ring.device)
        )

        # precompute for varlen attention
        self.q_cells_lens.data.fill_(1)
        self.q_cells_lens.data[0] = 0

        # ensure all params have grad set to False

        return


class Model(torch.nn.Module):
    """WeatherGenerator model architecture

    WeatherGenerator consists of the following components:

    embeds: embedding networks: Stream specific embedding networks.

    ae_local_blocks: Local assimilation engine: transformer based network to combine different input
        streams per healpix cell.

    ae_adapter: Assimilation engine adapter: Adapter to transform local assimilation engine
        information to the global assimilation engine.

    ae_aggregation_blocks: Query aggregation engine: after the learnable queries are created per
        non-masked healpix cell, this engine combines information from all non-masked cells by
        using dense attention layers.

    ae_global_blocks: Global assimilation engine: Transformer network alternating between local and
        global attention based upon global attention density rate.

    fe_blocks: Forecasting engine: Transformer network using the output of global attention to
        advance the latent representation in time.

    embed_target_coords: Embedding networks for coordinates: Initializes embedding networks tailored
        for metadata embedded target coordinates. The architecture is either a linear layer or a
        multi-layer perceptron, determined by the configuration of the embedding target coordinate
        networks.

    pred_adapter_kv: Prediction adapter: Adapter to transform the global assimilation/forecasting
        engine output to the prediction engine. Uses an MLP if `cf.pred_adapter_kv` is True,
        otherwise it uses an identity function.

    target_token_engines: Prediction engine: Transformer based prediction network that generates
        output corresponding to target coordinates.

    pred_heads: Prediction head: Final layers using target token engines output for mapping target
        coordinates to its physical space.
    """

    def __init__(
        self,
        cf: Config,
        sources_size,
        targets_num_channels,
        targets_coords_size,
        targets_channels=None,
    ):
        """
        Args:
            cf : Configuration with model parameters
            sources_size : List of number of channels for models
            targets_num_channels : List with size of each output sample for coordinates target
                embedding
            targets_coords_size : List with size of each input sample for coordinates target
                embedding
            targets_channels : List of list[str] of channel names per stream (for variable_groups)
        """
        super(Model, self).__init__()

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level

        self.cf = cf
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size
        self.targets_channels = targets_channels or [[] for _ in sources_size]
        # maps stream_name -> [(group_name, ch_indices, has_own_tte, ring_size), ...]
        # ring_size: 1 = default 1-ring KV (Decoder B default), 2 = 2-ring KV (Decoder B1)
        self.stream_groups: dict[str, list] = {}

        self.embed_target_coords = None
        self.encoder: EncoderModule | None = None
        self.forecast_engine: ForecastingEngine | IdentityEngine | None = None
        self.pred_heads = None
        self.q_cells: torch.Tensor | None = None
        self.streams: dict[str, typing.Any] = cf.streams
        self.stream_names: list[str] = list(cf.streams.keys())
        self.target_token_engines = None

        assert cf.get("forecast", {}).get("att_dense_rate", 1.0) == 1.0, (
            "Local attention not adapted for register tokens"
        )
        self.num_register_tokens = cf.num_register_tokens
        self.latent_heads = None
        self.latent_pre_norm = None
        # auxiliary tokens
        self.class_token_idxs = list(
            range(cf.num_register_tokens, cf.num_register_tokens + cf.num_class_tokens)
        )
        self.register_token_idxs = list(range(cf.num_register_tokens))
        self.aux_token_idxs = list(range(cf.num_register_tokens + cf.num_class_tokens))
        self.num_aux_tokens = cf.num_register_tokens + cf.num_class_tokens

    def _create_latent_pred_head(
        self, global_cfg, name, loss_cfg, use_class_token, use_patch_token
    ):
        if loss_cfg["head"].lower() == "mlp":
            return LatentPredictionHeadMLP(
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "transformer":
            return LatentPredictionHeadTransformer(
                global_cfg,
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "identity":
            return LatentPredictionHeadIdentity()
        else:
            assert False, f"Unknown latent prediction head type {loss_cfg['head']}"

    def create(self) -> "Model":
        """Create each individual module of the model"""
        cf = self.cf

        self.encoder = EncoderModule(
            cf, self.sources_size, self.targets_num_channels, self.targets_coords_size
        )

        mode_cfg = cf.training_config
        if cf.fe_num_blocks > 0:
            self.forecast_engine = ForecastingEngine(cf, mode_cfg, self.num_healpix_cells)
        else:
            self.forecast_engine = IdentityEngine()

        # embed coordinates yielding one query token for each target token
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleDict()
        self.target_token_engines = torch.nn.ModuleDict()
        self.pred_heads = torch.nn.ModuleDict()
        # Decoder A: per-stream group-identity query embeddings (one vector per group).
        # Populated when decoder_group_query: true; empty dict otherwise (no overhead).
        self.group_query_embeds = torch.nn.ModuleDict()

        # determine stream names once so downstream components use consistent keys
        loss_terms = [
            v.type for _, v in cf.training_config.losses.items() if v.get("enabled", True)
        ]
        if cf.validation_config.get("losses"):
            loss_terms += [
                v.type for _, v in cf.validation_config.losses.items() if v.get("enabled", True)
            ]

        if "LossPhysical" in loss_terms:
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                # skip for the moment to ensure target embedding and tte exist (ordering of
                # cf.streams is random)
                if si.get("pred_spatial_shared") is None:
                    # extract and setup relevant parameters
                    etc = si["embed_target_coords"]
                    tr = si["target_readout"]
                    num_layers = tr["num_layers"]
                    tr_mlp_hidden_factor = (
                        tr["mlp_hidden_factor"] if "mlp_hidden_factor" in tr else 2
                    )
                    tr_dim_head_proj = tr["dim_head_proj"] if "dim_head_proj" in tr else None
                    softcap = tr["softcap"] if "softcap" in tr else 0.0

                    dims_embed = [
                        si["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    if is_root():
                        logger.info("{} :: coord embed: :: {}".format(si["name"], dims_embed))

                    dim_coord_in = self.targets_coords_size[i_stream]

                    # embedding network for coordinates
                    if etc["net"] == "linear":
                        self.embed_target_coords[stream_name] = NamedLinear(
                            f"embed_target_coords_{stream_name}",
                            in_features=dim_coord_in,
                            out_features=dims_embed[0],
                            bias=False,
                        )
                    elif etc["net"] == "mlp":
                        self.embed_target_coords[stream_name] = MLP(
                            dim_coord_in,
                            dims_embed[0],
                            hidden_factor=8,
                            with_residual=False,
                            dropout_rate=dropout_rate,
                            norm_eps=self.cf.mlp_norm_eps,
                            name=f"embed_target_coords_{stream_name}",
                        )
                    else:
                        assert False

                    # Resolve variable groups upfront so TTE creation can be skipped if unneeded
                    if "variable_groups" in si and self.targets_channels[i_stream]:
                        groups = _resolve_variable_groups(
                            si["variable_groups"], self.targets_channels[i_stream]
                        )
                        # Shared TTE only needed when at least one group has no own target_readout
                        needs_shared_tte = any(
                            "target_readout" not in grp_cfg for _, _, grp_cfg in groups
                        )
                    else:
                        groups = None
                        needs_shared_tte = True

                    # Shared TTE — used by ungrouped path or by Option-A groups
                    if needs_shared_tte:
                        if cf.decoder_type == "Linear":
                            tte = BilinearDecoder(
                                stream_name,
                                dims_embed[0],
                                cf.ae_global_dim_embed,
                                self.targets_num_channels[i_stream],
                            )
                        else:
                            tte_cls = _pick_tte_cls(cf.decoder_type)
                            tte = tte_cls(
                                cf,
                                dims_embed,
                                dim_coord_in,
                                tr_dim_head_proj,
                                tr_mlp_hidden_factor,
                                softcap,
                                stream_config=si,
                            )
                        self.target_token_engines[stream_name] = tte

                    if groups is not None:
                        # Build per-group TTEs (Option B) and per-group pred heads
                        stream_group_meta = []
                        for group_name, ch_indices, group_cfg in groups:
                            has_own_tte = "target_readout" in group_cfg
                            if has_own_tte:
                                if cf.decoder_type == "Linear":
                                    raise ValueError(
                                        f"decoder_type='Linear' does not support per-group "
                                        f"target_readout (stream {stream_name!r}, "
                                        f"group {group_name!r})"
                                    )
                                grp_tr = group_cfg["target_readout"]
                                grp_dims = [dims_embed[0]] * (grp_tr["num_layers"] + 1)
                                grp_stream_cfg = {
                                    "name": f"{stream_name}/{group_name}",
                                    "target_readout": grp_tr,
                                }
                                grp_tte_cls = _pick_tte_cls(cf.decoder_type)
                                self.target_token_engines[f"{stream_name}/{group_name}"] = (
                                    grp_tte_cls(
                                        cf,
                                        grp_dims,
                                        dim_coord_in,
                                        tr_dim_head_proj,
                                        tr_mlp_hidden_factor,
                                        softcap,
                                        stream_config=grp_stream_cfg,
                                    )
                                )

                            n_ch = len(ch_indices)
                            ph_cfg = group_cfg.get("pred_head") or si.get("pred_head")
                            if ph_cfg is None:
                                raise ValueError(
                                    f"Group {group_name!r} in stream {stream_name!r} has no "
                                    "pred_head config, and the stream has no top-level "
                                    "pred_head to fall back to."
                                )
                            final_activation = ph_cfg.get("final_activation", "Identity")
                            self.pred_heads[f"{stream_name}/{group_name}"] = EnsPredictionHead(
                                dims_embed[-1],
                                n_ch,
                                ph_cfg["num_layers"],
                                ph_cfg["ens_size"],
                                norm_type=cf.norm_type,
                                final_activation=final_activation,
                                stream_name=f"{stream_name}/{group_name}",
                            )
                            ring_size = int(group_cfg.get("neighbourhood_ring", 1))
                            stream_group_meta.append(
                                (group_name, ch_indices, has_own_tte, ring_size)
                            )

                        self.stream_groups[stream_name] = stream_group_meta

                        # Decoder A: one learnable embedding per group gives the query
                        # a variable-identity signal so Q and T generate different queries.
                        if cf.get("decoder_group_query", False):
                            self.group_query_embeds[stream_name] = torch.nn.Embedding(
                                len(stream_group_meta), dims_embed[0]
                            )
                    else:
                        final_activation = si["pred_head"].get("final_activation", "Identity")
                        if is_root():
                            logger.debug(
                                f"{final_activation} activation of pred head of {si['name']} stream"
                            )
                        self.pred_heads[stream_name] = EnsPredictionHead(
                            dims_embed[-1],
                            self.targets_num_channels[i_stream],
                            si["pred_head"]["num_layers"],
                            si["pred_head"]["ens_size"],
                            norm_type=cf.norm_type,
                            final_activation=final_activation,
                            stream_name=stream_name,
                        )

            # iterate again to setup shared spatial pred heads if specified in config
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                pred_spatial_shared = si.get("pred_spatial_shared")
                if pred_spatial_shared is not None:
                    if pred_spatial_shared not in self.streams.keys():
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += " but no stream with that name found."
                        raise ValueError(msg)
                    if pred_spatial_shared == stream_name:
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += "but cannot share with itself."
                        raise ValueError(msg)
                    logger.debug(
                        f"{stream_name} shares spatial prediction head with {pred_spatial_shared}."
                    )

                    self.embed_target_coords[stream_name] = self.embed_target_coords[
                        pred_spatial_shared
                    ]
                    self.target_token_engines[stream_name] = self.target_token_engines[
                        pred_spatial_shared
                    ]

                    assert pred_spatial_shared in self.streams.keys()
                    si_other = self.streams[pred_spatial_shared]
                    dims_embed = [
                        si_other["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

        # Latent heads for losses
        self.latent_heads = nn.ModuleDict()
        self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)

        ssl_losses_cfgs = [
            v
            for _, v in cf.training_config.losses.items()
            if v.type == "LossLatentSSLStudentTeacher" and v.get("enabled", True)
        ]

        # TODO: support multiple LossLatentSSLStudentTeacher terms
        assert len(ssl_losses_cfgs) <= 1, "To be implemented."
        for ssl_target_losses in ssl_losses_cfgs:
            self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)
            for loss, loss_conf in ssl_target_losses.loss_fcts.items():
                if loss == "iBOT":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=True,
                    )
                elif loss == "JEPA":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=False,
                        use_patch_token=True,
                    )
                elif loss == "DINO":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=False,
                    )

        return self

    def reset_parameters(self):
        def _reset_params(module):
            if isinstance(module, nn.Linear | nn.LayerNorm):
                module.reset_parameters()
            else:
                pass

        self.apply(_reset_params)

    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

        num_params_embed = [
            get_num_parameters(self.encoder.embed_engine.embeds[name])
            for name in self.streams.keys()
        ]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = get_num_parameters(self.encoder.ae_local_engine.ae_local_blocks)
        num_params_ae_global = get_num_parameters(self.encoder.ae_global_engine.ae_global_blocks)

        num_params_q_cells = (
            np.prod(self.encoder.q_cells.shape) if self.encoder.q_cells.requires_grad else 0
        )
        num_params_ae_adapter = get_num_parameters(self.encoder.ae_local_global_engine)

        num_params_ae_aggregation = get_num_parameters(
            self.encoder.ae_aggregation_engine.ae_aggregation_blocks
        )

        num_params_latent_heads = get_num_parameters(self.latent_heads)
        num_params_latent_heads += get_num_parameters(self.latent_pre_norm)

        num_params_fe = get_num_parameters(self.forecast_engine.fe_blocks)

        mdict = self.embed_target_coords
        num_params_embed_tcs = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]
        mdict = self.target_token_engines
        num_params_tte = []
        for name in self.stream_names:
            if not mdict:
                num_params_tte.append(0)
                continue
            total = get_num_parameters(mdict[name]) if name in mdict else 0
            if name in self.stream_groups:
                total += sum(
                    get_num_parameters(mdict[f"{name}/{grp}"])
                    for grp, _, has_own, *_ in self.stream_groups[name]
                    if has_own and f"{name}/{grp}" in mdict
                )
            num_params_tte.append(total)
        mdict = self.pred_heads
        num_params_preds = []
        for name in self.stream_names:
            if not mdict:
                num_params_preds.append(0)
            elif name in mdict:
                num_params_preds.append(get_num_parameters(mdict[name]))
            elif name in self.stream_groups:
                num_params_preds.append(
                    sum(
                        get_num_parameters(mdict[f"{name}/{grp}"])
                        for grp, *_ in self.stream_groups[name]
                        if f"{name}/{grp}" in mdict
                    )
                )
            else:
                num_params_preds.append(0)

        print("-----------------")
        print(f"Total number of trainable parameters: {num_params_total:,}")
        print("Number of parameters:")
        print("  Embedding networks:")
        [
            print("    {} : {:,}".format(si["name"], np))
            for si, np in zip(self.streams.values(), num_params_embed, strict=False)
        ]
        print(f" Local assimilation engine: {num_params_ae_local:,}")
        print(f" Local-global adapter: {num_params_ae_adapter:,}")
        print(f" Learnable queries: {num_params_q_cells:,}")
        print(f" Query Aggregation engine: {num_params_ae_aggregation:,}")
        print(f" Global assimilation engine: {num_params_ae_global:,}")
        print(f" Latent prediction heads and pre-norm: {num_params_latent_heads:,}")
        print(f" Forecast engine: {num_params_fe:,}")
        print(" coordinate embedding, prediction networks and prediction heads:")
        zps = zip(
            self.streams.keys(),
            num_params_embed_tcs,
            num_params_tte,
            num_params_preds,
            strict=False,
        )
        for stream_name, np0, np1, np2 in zps:
            print(f"   {stream_name} : {np0:,} / {np1:,} / {np2:,}")
        print("-----------------")

    def tokens_to_latent_state(self, tokens_post_norm, tokens) -> LatentState:
        """
        Extract separate parts from global latent space representation and store in LatentState
        """
        toks_pn = tokens_post_norm
        return LatentState(
            register_tokens=toks_pn[:, self.register_token_idxs] if toks_pn is not None else None,
            class_token=toks_pn[:, self.class_token_idxs] if tokens_post_norm is not None else None,
            patch_tokens=toks_pn[:, self.num_aux_tokens :] if toks_pn is not None else None,
            z_pre_norm=tokens,
        )

    def forward(self, model_params: ModelParams, batch: ModelBatch) -> ModelOutput:
        """Forward pass of the model

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            batch
        Returns:
            A list containing all prediction results
        """

        output = ModelOutput(batch.get_output_len())

        tokens, posteriors = self.encoder(model_params, batch)
        output.add_latent_prediction(0, "posteriors", posteriors)

        # recover batch dimension and separate input_steps
        shape = (len(batch), batch.get_num_source_steps(), *tokens.shape[1:])
        # collapse along input step dimension
        tokens = tokens.reshape(shape).sum(axis=1)

        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in batch.get_output_idxs():
            without_grad = p_fwd and self.training and step != max(batch.get_output_idxs())
            if without_grad:
                # Pushforward mode: advance tokens without grad; no decoding with torch.no_grad():
                tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
                continue

            tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
            # decoder predictions
            output = self.predict_decoders(model_params, step, tokens, batch, output)
            # latent predictions (raw and with SSL heads)
            output = self.predict_latent(model_params, step, tokens, batch, output)

        return output

    def predict_latent(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute latent predictions
        """

        # safe latent prediction
        tokens_post_norm = self.latent_pre_norm(tokens) if step == 0 else None
        latent_state = self.tokens_to_latent_state(tokens_post_norm, tokens)
        output.add_latent_prediction(step, "latent_state", latent_state)

        # latent predictions for SSL training
        for name, head in self.latent_heads.items():
            output.add_latent_prediction(step, name, head(latent_state))

        return output

    def predict_decoders(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute decoder-based predictions

        Predict outputs at the specific target coordinates based on the input weather state and
        pre-training task and projects the latent space representation back to physical space.

        Args:
            model_params : Query and embedding parameters
            fstep : Number of forecast steps
            tokens : Tokens from global assimilation engine
            streams_data : Used to initialize target coordinates tokens and index information
                List of StreamData len(streams_data) == batch_size_per_gpu
            target_coords_idxs : Indices of target coordinates
        Returns:
            Prediction output tokens in physical representation for each target_coords.
        """
        # Empty dicts evaluate to False in python
        if not self.pred_heads:
            return output

        # remove register  and class tokens
        tokens = tokens[:, self.num_aux_tokens :]

        # get 1-ring neighborhood for prediction
        batch_size = len(batch)
        s = [batch_size, self.num_healpix_cells, self.cf.ae_local_num_queries, tokens.shape[-1]]
        idxs = model_params.hp_nbours.unsqueeze(0).repeat((batch_size, 1, 1)).flatten(0, 1)
        tokens_nbors = tokens.reshape(s).flatten(0, 1)[idxs.flatten()].flatten(0, 1)
        # TODO: precompute in model_params?
        tokens_nbors_lens = torch.full(
            (s[0] * s[1] + 1,), fill_value=9, dtype=torch.int32, device=tokens_nbors.device
        )
        tokens_nbors_lens[0] = 0

        # pair with tokens from assimilation engine to obtain target tokens
        for stream_name in self.streams.keys():
            # extract target coords for current stream and fstep and convert to one tensor
            t_coords = [
                batch.samples[i_b].streams_data[stream_name].target_coords[step]
                for i_b in range(batch_size)
            ]
            t_coords_lens = [len(t) for t in t_coords]
            t_coords = torch.cat(t_coords)

            if len(t_coords) == 0:
                continue

            # embed token coords
            tc_embed = self.embed_target_coords[stream_name]
            tc_tokens = checkpoint(tc_embed, t_coords, use_reentrant=False)

            # skip when coordinate embeddings yields nan (i.e. the coord embedding network diverged)
            if torch.isnan(tc_tokens).any():
                logger.warning(
                    (
                        f"Skipping prediction for {stream_name} because",
                        f" of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens.",
                    )
                )
                pred = torch.tensor([], device=tc_tokens.device)

            # skip empty lengths
            elif tc_tokens.shape[0] == 0:
                pred = torch.tensor([], device=tc_tokens.device)

            else:
                # lens for varlen attention
                tcls = torch.cat(
                    [
                        sample.streams_data[stream_name].target_coords_lens[step]
                        for sample in batch.samples
                    ]
                )
                tcs_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device=tcls.device), tcls])

                if self.cf.decoder_type == "Linear":
                    pred = self.target_token_engines[stream_name](
                        tc_tokens,
                        tokens.reshape(-1, s[-1]),  # collapse the batch and token dimensions
                        tcs_lens,
                    ).unsqueeze(0)  # add ensemble dim: shape is then [1, preds_per_coord, channels]
                else:
                    if stream_name in self.stream_groups:
                        # Coord embedding is shared; each group uses its own TTE (Option B) or the
                        # shared TTE (Option A). The shared TTE output is cached across groups.
                        tc_tokens_init = tc_tokens
                        i_stream = self.stream_names.index(stream_name)
                        n_ch_total = self.targets_num_channels[i_stream]
                        n_toks = tc_tokens_init.shape[0]
                        first_grp_name = self.stream_groups[stream_name][0][0]
                        ens_size = len(
                            self.pred_heads[f"{stream_name}/{first_grp_name}"].pred_heads
                        )
                        # pred is allocated lazily from the first grp_pred so its dtype
                        # matches the pred_head output (float32) rather than the
                        # mixed-precision token dtype (bfloat16).
                        pred = None
                        shared_tc_tokens = None
                        tokens_nbors_2ring = None  # lazy-computed for Decoder B1
                        tokens_nbors_2ring_lens = None
                        use_grp_query = stream_name in self.group_query_embeds

                        for i_grp, (group_name, ch_indices, has_own_tte, ring_size) in enumerate(
                            self.stream_groups[stream_name]
                        ):
                            # ── Decoder A: group-identity query ──────────────────────────
                            # Add a learnable group embedding to the coordinate query so that
                            # each variable group generates a distinct query direction and
                            # the cross-attention can route to group-relevant latent features.
                            if use_grp_query:
                                grp_idx = torch.tensor([i_grp], device=tc_tokens_init.device)
                                grp_q = self.group_query_embeds[stream_name](grp_idx)
                                tc_tokens_query = tc_tokens_init + grp_q  # [N, dim] + [1, dim]
                            else:
                                tc_tokens_query = tc_tokens_init

                            # ── Decoder B1: 2-ring KV selection ──────────────────────────
                            # Groups with neighbourhood_ring: 2 in their stream config get
                            # a wider spatial context (25 cells vs 9).  Only supported for
                            # Option-B groups (has_own_tte=True); Option-A groups warn and
                            # fall back to 1-ring (can't break the shared-TTE cache).
                            if ring_size == 2 and has_own_tte:
                                if tokens_nbors_2ring is None:
                                    idxs_2ring = (
                                        model_params.hp_nbours_2ring.unsqueeze(0)
                                        .repeat((batch_size, 1, 1))
                                        .flatten(0, 1)
                                    )
                                    tokens_nbors_2ring = (
                                        tokens.reshape(s)
                                        .flatten(0, 1)[idxs_2ring.flatten()]
                                        .flatten(0, 1)
                                    )
                                    tokens_nbors_2ring_lens = torch.full(
                                        (s[0] * s[1] + 1,),
                                        fill_value=_RING2_SIZE,
                                        dtype=torch.int32,
                                        device=tokens_nbors_2ring.device,
                                    )
                                    tokens_nbors_2ring_lens[0] = 0
                                kv_tokens = tokens_nbors_2ring
                                kv_lens = tokens_nbors_2ring_lens
                            else:
                                if ring_size == 2 and not has_own_tte:
                                    logger.warning(
                                        f"Group {group_name!r} in stream {stream_name!r}: "
                                        "neighbourhood_ring=2 requires Option-B (target_readout "
                                        "per group). Falling back to 1-ring for this group."
                                    )
                                kv_tokens = tokens_nbors
                                kv_lens = tokens_nbors_lens

                            # ── TTE dispatch ──────────────────────────────────────────────
                            if has_own_tte:
                                tc_tokens_grp = self.target_token_engines[
                                    f"{stream_name}/{group_name}"
                                ](
                                    latent=kv_tokens,
                                    output=tc_tokens_query,
                                    latent_lens=kv_lens,
                                    output_lens=tcs_lens,
                                    coordinates=t_coords,
                                )
                            else:
                                # Option A: shared TTE.  Cache is valid only when group query
                                # is disabled (same query for all groups) and ring_size=1.
                                if not use_grp_query and ring_size == 1:
                                    if shared_tc_tokens is None:
                                        shared_tc_tokens = self.target_token_engines[stream_name](
                                            latent=kv_tokens,
                                            output=tc_tokens_query,
                                            latent_lens=kv_lens,
                                            output_lens=tcs_lens,
                                            coordinates=t_coords,
                                        )
                                    tc_tokens_grp = shared_tc_tokens
                                else:
                                    # Group query active: each group needs its own TTE run.
                                    tc_tokens_grp = self.target_token_engines[stream_name](
                                        latent=kv_tokens,
                                        output=tc_tokens_query,
                                        latent_lens=kv_lens,
                                        output_lens=tcs_lens,
                                        coordinates=t_coords,
                                    )
                            grp_pred = self.pred_heads[f"{stream_name}/{group_name}"](tc_tokens_grp)
                            if pred is None:
                                pred = torch.zeros(
                                    ens_size,
                                    n_toks,
                                    n_ch_total,
                                    device=grp_pred.device,
                                    dtype=grp_pred.dtype,
                                )
                            ch_idx = torch.tensor(ch_indices, device=pred.device, dtype=torch.long)
                            pred[:, :, ch_idx] = grp_pred.to(pred.dtype)
                    else:
                        tc_tokens = self.target_token_engines[stream_name](
                            latent=tokens_nbors,
                            output=tc_tokens,
                            latent_lens=tokens_nbors_lens,
                            output_lens=tcs_lens,
                            coordinates=t_coords,
                        )
                        pred = self.pred_heads[stream_name](tc_tokens)

            # recover batch dimension (ragged, so as list)
            pred = torch.split(pred, t_coords_lens, dim=1)
            output.add_physical_prediction(step, stream_name, pred)

        return output
