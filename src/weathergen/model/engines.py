# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import math

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.blocks import CrossAttentionBlock, OriginalPredictionBlock, SelfAttentionBlock
from weathergen.model.embeddings import (
    StreamEmbedLinear,
    StreamEmbedTransformer,
)
from weathergen.model.layers import MLP
from weathergen.model.norms import RMSNorm
from weathergen.model.utils import ActivationFactory
from weathergen.utils.utils import get_dtype


class EmbeddingEngine(torch.nn.Module):
    name: "EmbeddingEngine"

    def __init__(self, cf: Config, sources_size) -> None:
        """
        Initialize the EmbeddingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param sources_size: List of source sizes for each stream.
        """
        super(EmbeddingEngine, self).__init__()
        self.cf = cf
        self.dtype = get_dtype(self.cf.mixed_precision_dtype)
        self.sources_size = sources_size  # KCT:iss130, what is this?
        self.embeds = torch.nn.ModuleDict()
        self.streams = cf.streams

        for i, (stream_name, si) in enumerate(self.streams.items()):
            if si.get("diagnostic", False) or self.sources_size[i] == 0:
                self.embeds[stream_name] = torch.nn.Identity()
                continue

            if si["embed"]["net"] == "transformer":
                self.embeds[stream_name] = StreamEmbedTransformer(
                    num_tokens=si["embed"]["num_tokens"],
                    token_size=si["token_size"],
                    num_channels=self.sources_size[i],
                    dim_embed=si["embed"]["dim_embed"],
                    dim_out=self.cf.ae_local_dim_embed,
                    num_blocks=si["embed"]["num_blocks"],
                    num_heads=si["embed"]["num_heads"],
                    dropout_rate=self.cf.embed_dropout_rate,
                    norm_type=self.cf.norm_type,
                    unembed_mode=self.cf.embed_unembed_mode,
                    stream_name=stream_name,
                )
            elif si["embed"]["net"] == "linear":
                self.embeds[stream_name] = StreamEmbedLinear(
                    self.sources_size[i] * si["token_size"],
                    self.cf.ae_local_dim_embed,
                    stream_name=stream_name,
                )
            else:
                raise ValueError("Unsupported embedding network type")

    def forward(self, batch, pe_embed):
        num_steps_input = batch.get_num_source_steps()

        num_tokens = torch.sum(batch.tokens_lens, 2).flatten().sum().item()
        tokens_all = torch.empty(
            (num_tokens, self.cf.ae_local_dim_embed), dtype=self.dtype, device=batch.get_device()
        )

        # iterate over all streams
        x_embeds = []
        for stream_name in self.streams.keys():
            # collect all source tokens from all input_steps and all samples in the batch
            sdata = []
            for istep in range(num_steps_input):
                for sample in batch.get_samples():
                    sdata += [sample.streams_data[stream_name].source_tokens_cells[istep]]

            if all(s is None for s in sdata):
                continue

            sdata = torch.cat(sdata).to(tokens_all.dtype)
            # skip empty stream
            if sdata.numel() == 0:
                continue

            # embedding from physical space to per patch latent representation
            x_embeds += [self.embeds[stream_name](sdata).flatten(0, 1)]

        # switch from stream to cell-based ordering and apply per cell positional encoding

        # if the assert is hit, max_number_tokens_local_per_cell in config needs to be increased
        max_tokens = self.cf.get("ae_local_max_tokens_per_cell", 64)
        assert batch.tokens_lens.flatten(0, 2).sum(0).max() <= max_tokens, (
            "max number of tokens per cell for positional encoding exceeded."
        )
        " Increase ae_local_max_tokens_per_cell in config."

        if batch.tokens_lens.shape[2] == 1:
            # trivial with one stream
            tokens_all = torch.cat(x_embeds)

        else:
            scatter_idxs = self.get_scatter_idxs_vectorized(batch)
            scatter_idxs = scatter_idxs.unsqueeze(1).repeat((1, self.cf.ae_local_dim_embed))

            # actual scatter operation and apply per cell positional encoding
            tokens_all.scatter_(0, scatter_idxs, torch.cat(x_embeds))

        pe_idxs = self.get_pe_idxs_vectorized(batch)
        tokens_all = tokens_all + pe_embed[pe_idxs]

        return tokens_all

    def get_pe_idxs_vectorized(self, batch):
        """
        Compute per cell indices into positional encoding
        """

        tok_counts = batch.tokens_lens.permute([2, 0, 1, 3]).sum(0).flatten()
        rows = torch.arange(tok_counts.max(), device=tok_counts.device).unsqueeze(0)
        rows = rows.expand(tok_counts.shape[0], -1)
        pe_idxs = rows[rows < tok_counts.unsqueeze(1)]

        return pe_idxs

    def get_scatter_idxs(self, batch):
        """
        Compute reordering index so that tokens from different streams but same cell are
        continguous

        Simple version (reference implementation)
        """

        dev = batch.get_device()
        # batch.tokens_lens : (num_steps_input, num_samples, num_streams, num_cells)
        # flatten leasds to streams x tokens per cell (across all cells for input steps and samples)
        tok_counts = batch.tokens_lens.permute([2, 0, 1, 3]).flatten(1, -1)

        scatter_idxs = []
        for i in range(len(tok_counts)):
            for j in range(tok_counts.shape[1]):
                if tok_counts[i, j] == 0:
                    continue
                # offset from preceding cells
                offset = tok_counts[:, :j].flatten().sum()
                # offset from preceding streams in cells
                offset += tok_counts[:i, j].sum()
                # scatter idxs is offset and idxs for all tokens in cell for current stream
                scatter_idxs += [offset[i, j] + torch.arange(tok_counts[i, j], device=dev)]

        scatter_idxs = torch.cat(scatter_idxs).to(torch.int64)

        return scatter_idxs

    def get_scatter_idxs_vectorized(self, batch):
        """
        Compute reordering index so that tokens from different streams but same cell are
        continguous

        Vectorized version
        """

        dev = batch.get_device()
        # batch.tokens_lens : (num_steps_input, num_samples, num_streams, num_cells)
        # flatten leasds to streams x tokens per cell (across all cells for input steps and samples)
        tok_counts = batch.tokens_lens.permute([2, 0, 1, 3]).flatten(1, -1)

        # partial sums for per cell offsets
        pad = torch.zeros((1, tok_counts.shape[1]), dtype=torch.int64, device=dev)
        offset = torch.cat([pad, tok_counts.cumsum(0)])[:-1]
        offset[:, 1:] += tok_counts.sum(0).cumsum(0)[:-1]

        ranges = torch.arange(tok_counts.max(), device=dev).repeat((tok_counts.numel(), 1))
        idxs = (offset.flatten() + ranges.transpose(1, 0)).transpose(1, 0)
        # select idxs[i][:ranges[i]] for each i; vectorized version
        col_indices = torch.arange(idxs.shape[1], device=dev).unsqueeze(0)
        valid_mask = col_indices < tok_counts.flatten().unsqueeze(1)
        scatter_idxs = idxs[valid_mask].to(torch.int64)

        return scatter_idxs


class LocalAssimilationEngine(torch.nn.Module):
    name: "LocalAssimilationEngine"

    def __init__(self, cf: Config) -> None:
        """
        Initialize the LocalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        """
        super(LocalAssimilationEngine, self).__init__()
        self.cf = cf
        self.ae_local_blocks = torch.nn.ModuleList()

        for _ in range(self.cf.ae_local_num_blocks):
            self.ae_local_blocks.append(
                MultiSelfAttentionHeadVarlen(
                    self.cf.ae_local_dim_embed,
                    num_heads=self.cf.ae_local_num_heads,
                    dropout_rate=self.cf.ae_local_dropout_rate,
                    with_qk_lnorm=self.cf.ae_local_with_qk_lnorm,
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )
            self.ae_local_blocks.append(
                MLP(
                    self.cf.ae_local_dim_embed,
                    self.cf.ae_local_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_local_dropout_rate,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, tokens_c, cell_lens_c, use_reentrant):
        for block in self.ae_local_blocks:
            tokens_c = block(tokens_c, cell_lens_c)
        return tokens_c


class Local2GlobalAssimilationEngine(torch.nn.Module):
    name: "Local2GlobalAssimilationEngine"

    def __init__(self, cf: Config) -> None:
        """
        Initialize the Local2GlobalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        """
        super(Local2GlobalAssimilationEngine, self).__init__()
        self.cf = cf
        self.ae_adapter = torch.nn.ModuleList()

        self.ae_adapter.append(
            MultiCrossAttentionHeadVarlenSlicedQ(
                self.cf.ae_global_dim_embed,
                self.cf.ae_local_dim_embed,
                num_slices_q=self.cf.ae_local_num_queries,
                dim_head_proj=self.cf.ae_adapter_embed,
                num_heads=self.cf.ae_adapter_num_heads,
                with_residual=self.cf.ae_adapter_with_residual,
                with_qk_lnorm=self.cf.ae_adapter_with_qk_lnorm,
                dropout_rate=self.cf.ae_adapter_dropout_rate,
                with_flash=self.cf.with_flash_attention,
                norm_type=self.cf.norm_type,
                qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                norm_eps=self.cf.norm_eps,
                attention_dtype=get_dtype(self.cf.attention_dtype),
            )
        )

        ae_adapter_num_blocks = cf.get("ae_adapter_num_blocks", 2)
        for _ in range(ae_adapter_num_blocks - 1):
            self.ae_adapter.append(
                MLP(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_adapter_dropout_rate,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )
            self.ae_adapter.append(
                MultiCrossAttentionHeadVarlenSlicedQ(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_local_dim_embed,
                    num_slices_q=self.cf.ae_local_num_queries,
                    dim_head_proj=self.cf.ae_adapter_embed,
                    num_heads=self.cf.ae_adapter_num_heads,
                    with_residual=self.cf.ae_adapter_with_residual,
                    with_qk_lnorm=self.cf.ae_adapter_with_qk_lnorm,
                    dropout_rate=self.cf.ae_adapter_dropout_rate,
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )

    def forward(self, tokens_c, tokens_global_c, q_cells_lens_c, cell_lens_c):
        for block in self.ae_adapter:
            tokens_global_c = block(
                tokens_global_c,
                tokens_c,
                q_cells_lens_c,
                cell_lens_c,
            )
        return tokens_global_c


class Local2GlobalSumEngine(torch.nn.Module):
    """Alternative to Local2GlobalAssimilationEngine.

    Instead of cross-attention (Q=learnable query, KV=local tokens), this engine
    sums local tokens per cell and projects to global dim. Masked cells are filled
    externally by the encoder using the learnable query + pe_global (unchanged).

    Forward signature matches Local2GlobalAssimilationEngine; tokens_global_c and
    q_cells_lens_c are unused (masked-cell filling happens in the encoder).
    """

    name: "Local2GlobalSumEngine"

    def __init__(self, cf: Config) -> None:
        super(Local2GlobalSumEngine, self).__init__()
        self.cf = cf
        self.proj = torch.nn.Linear(cf.ae_local_dim_embed, cf.ae_global_dim_embed, bias=False)
        ae_adapter_num_blocks = cf.get("ae_adapter_num_blocks", 2)
        self.mlp_blocks = torch.nn.ModuleList()
        for _ in range(ae_adapter_num_blocks - 1):
            self.mlp_blocks.append(
                MLP(
                    cf.ae_global_dim_embed,
                    cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=cf.ae_adapter_dropout_rate,
                    norm_type=cf.norm_type,
                    norm_eps=cf.mlp_norm_eps,
                )
            )

    def forward(self, tokens_c, tokens_global_c, q_cells_lens_c, cell_lens_c):
        # tokens_c:        (total_local_tokens, local_dim)
        # tokens_global_c: (num_unmasked_cells, num_queries, global_dim) — unused
        # cell_lens_c:     (num_unmasked_cells + 1,) with 0 at index 0
        num_cells = cell_lens_c.shape[0] - 1
        cell_counts = cell_lens_c[1:]

        # scatter-sum local tokens into per-cell summaries
        cell_idx = torch.repeat_interleave(
            torch.arange(num_cells, device=tokens_c.device, dtype=torch.long), cell_counts
        )
        cell_sums = torch.zeros(
            num_cells, tokens_c.shape[-1], device=tokens_c.device, dtype=tokens_c.dtype
        )
        cell_sums.scatter_add_(0, cell_idx.unsqueeze(1).expand_as(tokens_c), tokens_c)

        # project to global dim and match (num_cells, num_queries, global_dim)
        num_queries = tokens_global_c.shape[1]
        out = self.proj(cell_sums).unsqueeze(1).expand(-1, num_queries, -1)

        for blk in self.mlp_blocks:
            out = blk(out)

        return out


class LatentUpsamplingEngine(torch.nn.Module):
    """Decode-time latent capacity expansion (per stream).

    Maps each HEALPix cell's gathered 1-ring neighbourhood latent -- the KV that
    ``Model.predict_decoders`` already builds: the cell plus its 8 neighbours at the source
    HEALPix level -- into ``num_sub_latents`` (K) learned sub-latents via cross-attention. The
    decoder then attends each target point to these K sub-latents instead of the 9 neighbour
    latents, raising the per-cell decode capacity. The K learned queries carry their own
    (learned) within-cell positional seed, so the K outputs can specialise to different
    sub-regions of the ~150 km cell; the per-cell *content* comes from the cell-specific KV.

    Deterministic and decode-only: the encoder / global / forecast backbone is untouched and
    stays at the source level, so the expansion is cheap and added only where high resolution
    is needed. This is the working, decode-side analogue of ``ae_local_num_queries`` (which is
    unfinished/broken in the encoder; see docs/raina_knowledge_base.md sections 12-13).

    When ``target_lens`` is given, the expansion runs only over *active* cells (those with at
    least one target point this step). Inactive cells get zero-length latent, exactly matching
    their zero-length target set in the downstream varlen decoder attention, so the decoder
    output is identical to expanding every cell — but for a regional stream (e.g. CERRA over
    Europe) this cuts the upsampler's work from batch x 12,288 cells to the few hundred that
    matter.
    """

    name: "LatentUpsamplingEngine"

    def __init__(self, cf: Config, num_sub_latents: int) -> None:
        super().__init__()
        self.cf = cf
        self.num_sub_latents = num_sub_latents
        dim = cf.ae_global_dim_embed

        # K learned sub-latent queries, shared across cells (small init, as for encoder
        # q_cells): the residual base is near-zero so the cross-attention over each cell's KV
        # does the work and the K slots differentiate via their learned positional seed.
        self.q_sub = torch.nn.Parameter(torch.randn(num_sub_latents, dim) / dim)

        num_blocks = cf.get("decode_upsample_num_blocks", 1)
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            if i > 0:
                self.blocks.append(
                    MLP(
                        dim,
                        dim,
                        with_residual=True,
                        dropout_rate=cf.ae_adapter_dropout_rate,
                        norm_type=cf.norm_type,
                        norm_eps=cf.mlp_norm_eps,
                    )
                )
            self.blocks.append(
                MultiCrossAttentionHeadVarlen(
                    dim_embed_q=dim,
                    dim_embed_kv=dim,
                    num_heads=cf.ae_adapter_num_heads,
                    dim_head_proj=cf.ae_adapter_embed,
                    with_residual=True,
                    with_qk_lnorm=cf.ae_adapter_with_qk_lnorm,
                    dropout_rate=cf.ae_adapter_dropout_rate,
                    with_flash=cf.with_flash_attention,
                    norm_type=cf.norm_type,
                    qk_norm_type=cf.get("qk_norm_type", cf.norm_type),
                    norm_eps=cf.norm_eps,
                    attention_dtype=get_dtype(cf.attention_dtype),
                )
            )

    def forward(self, latent, latent_lens, target_lens=None):
        """Expand per-cell neighbourhood latent into K sub-latents.

        Args:
            latent: ``(sum(latent_lens), dim)`` KV tokens, grouped per cell by ``latent_lens``
                (the ``tokens_nbors`` built in ``predict_decoders``, 9 neighbours per cell —
                the same fixed count for every cell).
            latent_lens: ``(num_cells + 1,)`` int32, ``latent_lens[0] == 0`` and the rest the
                per-cell KV counts (9).
            target_lens: optional ``(num_cells + 1,)`` int32 per-cell target counts (the
                decoder's ``tcs_lens``). When given, only cells with ``target_lens > 0`` are
                expanded; the rest get zero-length latent (identical decode, far cheaper).

        Returns:
            sub_latent: ``(num_active * K, dim)`` -- the K sub-latents per active cell,
                flattened to match the layout the decoder expects for ``latent``.
            sub_lens: ``(num_cells + 1,)`` int32, ``[0] == 0``, ``K`` at active cells and
                ``0`` elsewhere (all ``K`` when ``target_lens`` is None).
        """
        num_cells = latent_lens.shape[0] - 1
        k = self.num_sub_latents
        dim = latent.shape[-1]

        if target_lens is not None:
            idx_active = (target_lens[1:] > 0).nonzero(as_tuple=True)[0]
            num_active = idx_active.shape[0]
            # per-cell KV count is uniform (9-neighbour gather), so a view-gather suffices
            kv = latent.reshape(num_cells, -1, dim)[idx_active].flatten(0, 1)
            kv_lens = torch.full(
                (num_active + 1,),
                latent.shape[0] // num_cells,
                dtype=torch.int32,
                device=latent.device,
            )
            kv_lens[0] = 0
        else:
            idx_active = None
            num_active = num_cells
            kv, kv_lens = latent, latent_lens

        sub = self.q_sub.unsqueeze(0).expand(num_active, k, dim).reshape(num_active * k, dim)
        sub_q_lens = torch.full((num_active + 1,), k, dtype=torch.int32, device=latent.device)
        sub_q_lens[0] = 0

        for block in self.blocks:
            if isinstance(block, MLP):
                sub = checkpoint(block, sub, use_reentrant=False)
            else:
                sub = checkpoint(block, sub, kv, sub_q_lens, kv_lens, use_reentrant=False)

        if idx_active is None:
            sub_lens = sub_q_lens
        else:
            sub_lens = torch.zeros((num_cells + 1,), dtype=torch.int32, device=latent.device)
            sub_lens[1:][idx_active] = k

        return sub, sub_lens


class QueryAggregationEngine(torch.nn.Module):
    name: "QueryAggregationEngine"

    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """
        Initialize the QueryAggregationEngine with the configuration.

        This engine is used for aggregating information from all query tokens coming
        from healpix cells, that are not masked.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        super(QueryAggregationEngine, self).__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells

        self.ae_aggregation_blocks = torch.nn.ModuleList()

        global_rate = int(1 / self.cf.ae_aggregation_att_dense_rate)
        for i in range(self.cf.ae_aggregation_num_blocks):
            ## Alternate between local and global attention
            #  as controlled by cf.ae_dense_local_att_dense_rate
            # Last block is always global attention
            if i % global_rate == 0 or i + 1 == self.cf.ae_aggregation_num_blocks:
                self.ae_aggregation_blocks.append(
                    MultiSelfAttentionHeadVarlen(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_aggregation_num_heads,
                        dropout_rate=self.cf.ae_aggregation_dropout_rate,
                        with_qk_lnorm=self.cf.ae_aggregation_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                        with_2d_rope=self.cf.get("rope_2D", False),
                    )
                )
            else:
                assert False, "Incompatible with batchsize > 1 here"
                self.ae_aggregation_blocks.append(
                    MultiSelfAttentionHeadLocal(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_aggregation_num_heads,
                        qkv_len=self.num_healpix_cells * self.cf.ae_local_num_queries,
                        block_factor=self.cf.ae_aggregation_block_factor,
                        dropout_rate=self.cf.ae_aggregation_dropout_rate,
                        with_qk_lnorm=self.cf.ae_aggregation_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                    )
                )
            # MLP block
            self.ae_aggregation_blocks.append(
                MLP(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_aggregation_dropout_rate,
                    hidden_factor=self.cf.ae_aggregation_mlp_hidden_factor,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, tokens, batch_lens, use_reentrant, coords=None):
        for block in self.ae_aggregation_blocks:
            aux_info = None
            if isinstance(block, MultiSelfAttentionHeadVarlen):
                tokens = block(tokens, x_lens=batch_lens, coords=coords)
            else:
                tokens = block(tokens, coords, aux_info)
        return tokens


class GlobalAssimilationEngine(torch.nn.Module):
    name: "GlobalAssimilationEngine"

    def __init__(self, cf: Config, num_healpix_cells: int) -> None:
        """
        Initialize the GlobalAssimilationEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        super(GlobalAssimilationEngine, self).__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells

        self.ae_global_blocks = torch.nn.ModuleList()

        global_rate = int(1 / self.cf.ae_global_att_dense_rate)
        for i in range(self.cf.ae_global_num_blocks):
            ## Alternate between local and global attention
            #  as controlled by cf.ae_global_att_dense_rate
            # Last block is always global attention
            if i % global_rate == 0 or i + 1 == self.cf.ae_global_num_blocks:
                self.ae_global_blocks.append(
                    MultiSelfAttentionHead(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_global_num_heads,
                        dropout_rate=self.cf.ae_global_dropout_rate,
                        with_qk_lnorm=self.cf.ae_global_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                        with_2d_rope=self.cf.get("rope_2D", False),
                    )
                )
            else:
                self.ae_global_blocks.append(
                    MultiSelfAttentionHeadLocal(
                        self.cf.ae_global_dim_embed,
                        num_heads=self.cf.ae_global_num_heads,
                        qkv_len=self.num_healpix_cells * self.cf.ae_local_num_queries,
                        block_factor=self.cf.ae_global_block_factor,
                        dropout_rate=self.cf.ae_global_dropout_rate,
                        with_qk_lnorm=self.cf.ae_global_with_qk_lnorm,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                        with_2d_rope=self.cf.get("rope_2D", False),
                    )
                )
            # MLP block
            self.ae_global_blocks.append(
                MLP(
                    self.cf.ae_global_dim_embed,
                    self.cf.ae_global_dim_embed,
                    with_residual=True,
                    dropout_rate=self.cf.ae_global_dropout_rate,
                    hidden_factor=self.cf.ae_global_mlp_hidden_factor,
                    norm_type=self.cf.norm_type,
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )
        if self.cf.get("ae_global_trailing_layer_norm", False):
            self.ae_global_blocks.append(
                torch.nn.LayerNorm(self.cf.ae_global_dim_embed, elementwise_affine=False)
            )

    def forward(self, tokens, coords=None):
        aux_info = None
        for block in self.ae_global_blocks:
            tokens = checkpoint(block, tokens, coords, aux_info, use_reentrant=False)
        return tokens


class IdentityEngine(torch.nn.Module):
    """Identity engine that passes tokens through unchanged."""

    def __init__(self):
        super().__init__()
        self.fe_blocks = torch.nn.ModuleList()

    def forward(self, tokens, *args, **kwargs):
        return tokens


class ForecastingEngine(torch.nn.Module):
    name: "ForecastingEngine"

    def __init__(self, cf: Config, mode_cfg, num_healpix_cells: int, dim_aux: int = None) -> None:
        """
        Initialize the ForecastingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param num_healpix_cells: Number of healpix cells used for local queries.
        """
        super(ForecastingEngine, self).__init__()
        self.cf = cf
        self.num_healpix_cells = num_healpix_cells
        self.fe_blocks = torch.nn.ModuleList()

        global_rate = int(1 / self.cf.forecast_att_dense_rate)
        if mode_cfg.get("forecast", {}).get("policy") is not None:
            for i in range(self.cf.fe_num_blocks):
                # Alternate between global and local attention
                if (i % global_rate == 0) or i + 1 == self.cf.fe_num_blocks:
                    self.fe_blocks.append(
                        MultiSelfAttentionHead(
                            self.cf.ae_global_dim_embed,
                            num_heads=self.cf.fe_num_heads,
                            dropout_rate=self.cf.fe_dropout_rate,
                            with_qk_lnorm=self.cf.fe_with_qk_lnorm,
                            with_flash=self.cf.with_flash_attention,
                            norm_type=self.cf.norm_type,
                            qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                            dim_aux=dim_aux,
                            norm_eps=self.cf.norm_eps,
                            attention_dtype=get_dtype(self.cf.attention_dtype),
                            with_2d_rope=self.cf.get("rope_2D", False),
                        )
                    )
                else:
                    self.fe_blocks.append(
                        MultiSelfAttentionHeadLocal(
                            self.cf.ae_global_dim_embed,
                            num_heads=self.cf.fe_num_heads,
                            qkv_len=self.num_healpix_cells * self.cf.ae_local_num_queries,
                            block_factor=self.cf.ae_global_block_factor,
                            dropout_rate=self.cf.fe_dropout_rate,
                            with_qk_lnorm=self.cf.fe_with_qk_lnorm,
                            with_flash=self.cf.with_flash_attention,
                            norm_type=self.cf.norm_type,
                            qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                            dim_aux=dim_aux,
                            norm_eps=self.cf.norm_eps,
                            attention_dtype=get_dtype(self.cf.attention_dtype),
                            with_2d_rope=self.cf.get("rope_2D", False),
                        )
                    )
                # Add MLP block
                self.fe_blocks.append(
                    MLP(
                        self.cf.ae_global_dim_embed,
                        self.cf.ae_global_dim_embed,
                        with_residual=True,
                        dropout_rate=self.cf.fe_dropout_rate,
                        norm_type=self.cf.norm_type,
                        dim_aux=dim_aux,
                        norm_eps=self.cf.mlp_norm_eps,
                    )
                )
                # Optionally, add LayerNorm after i-th layer
                if i in self.cf.get("fe_layer_norm_after_blocks", []):
                    self.fe_blocks.append(
                        torch.nn.LayerNorm(self.cf.ae_global_dim_embed, elementwise_affine=False)
                    )

        def init_weights_final(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, mean=0, std=0.001)

        for block in self.fe_blocks:
            block.apply(init_weights_final)

    def forward(self, tokens, fstep, coords=None):
        if self.training:
            # Impute noise to the latent state
            noise_std = self.cf.get("fe_impute_latent_noise_std", 0.0)
            if noise_std > 0.0:
                tokens = tokens + torch.randn_like(tokens) * torch.norm(tokens) * noise_std

        aux_info = None
        for _b_idx, block in enumerate(self.fe_blocks):
            if isinstance(block, torch.nn.modules.normalization.LayerNorm):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = checkpoint(block, tokens, coords, aux_info, use_reentrant=False)
        return tokens


class EnsPredictionHead(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        dim_out,
        ens_num_layers,
        ens_size,
        stream_name: str,
        norm_type="LayerNorm",
        hidden_factor=2,
        final_activation: None | str = None,
    ):
        """Constructor"""

        super(EnsPredictionHead, self).__init__()

        self.name = f"EnsPredictionHead_{stream_name}"

        dim_internal = dim_embed * hidden_factor
        # norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm
        enl = ens_num_layers

        self.pred_heads = torch.nn.ModuleList()
        for i in range(ens_size):
            self.pred_heads.append(torch.nn.ModuleList())

            # self.pred_heads[-1].append( norm( dim_embed))
            self.pred_heads[-1].append(
                torch.nn.Linear(dim_embed, dim_out if enl == 1 else dim_internal)
            )

            for i in range(ens_num_layers - 1):
                self.pred_heads[-1].append(torch.nn.GELU())
                self.pred_heads[-1].append(
                    torch.nn.Linear(dim_internal, dim_out if enl - 2 == i else dim_internal)
                )

            # Add optional final non-linear activation
            if final_activation is not None and enl >= 1:
                fal = ActivationFactory.get(final_activation)
                self.pred_heads[-1].append(fal)

    #########################################
    def forward(self, toks):
        preds = []
        for pred_head in self.pred_heads:
            cpred = toks
            for block in pred_head:
                cpred = block(cpred)
            preds.append(cpred)
        preds = torch.stack(preds, 0)

        return preds


class EnsPredictionHeadFourier(torch.nn.Module):
    """
    Fourier-feature pred head with one or more frequency bands (Tancik et al. NeurIPS 2020).

    For each scale σ in ``freq_scales``, builds an independent Fourier basis
    ``W_σ ~ N(0, σ²·I)`` and a dedicated MLP sub-head operating on the decoder
    output augmented with that band's sinusoidal features. Sub-head outputs are
    summed:

        pred = Σ_b  sub_head_b( [toks ; sin(2π·coords·W_b^T), cos(...)] )

    A single-element ``freq_scales`` (default ``[10.0]``) is the canonical
    single-band Random Fourier Feature head; multiple scales let the model route
    different channels to different frequency bands (e.g. q_850 → high-freq band,
    z_500 → low-freq band). This combats spectral bias on high-activity channels.

    Parameters
    ----------
    freq_scales : list[float]
        One scale (σ) per band. ``len(freq_scales)`` = number of bands/sub-heads.
        Default ``[10.0]`` (single band). Larger σ injects higher frequencies.
    num_freqs_per_band : int
        Fourier basis size per band (default 64), applied uniformly to all bands.
    learnable_freqs : bool
        If True the projection matrix trains; default False (canonical fixed RFF).

    Notes
    -----
    The projection matrix is a frozen ``Parameter`` (not a buffer): FSDP shards
    parameters into DTensors, while plain buffers stay as Tensors after the FSDP
    wrap and break state-dict gathering at checkpoint time.
    """

    def __init__(
        self,
        dim_embed,
        dim_out,
        dim_coord_in,
        ens_num_layers,
        ens_size,
        stream_name: str,
        num_freqs_per_band: int = 64,
        freq_scales: list[float] | tuple[float, ...] = (10.0,),
        learnable_freqs: bool = False,
        norm_type: str = "LayerNorm",
        hidden_factor: int = 2,
        final_activation: None | str = None,
    ):
        super().__init__()
        self.name = f"EnsPredictionHeadFourier_{stream_name}"

        self.num_bands = len(freq_scales)
        self.num_freqs_per_band = num_freqs_per_band

        # Fourier bases for all bands stacked: shape [num_bands, num_freqs_per_band, dim_coord_in].
        # Each band slice is drawn from N(0, σ_b²·I) so bands sample different frequency scales.
        scales = torch.tensor(list(freq_scales), dtype=torch.float32).view(-1, 1, 1)
        ws = torch.randn(self.num_bands, num_freqs_per_band, dim_coord_in) * scales
        self.Ws = torch.nn.Parameter(ws, requires_grad=learnable_freqs)

        # One sub-head per (ensemble member × band). Each takes [toks ; ff_b].
        dim_input = dim_embed + 2 * num_freqs_per_band
        dim_internal = dim_input * hidden_factor
        enl = ens_num_layers

        self.band_heads = torch.nn.ModuleList()
        for _ in range(ens_size):
            per_ens = torch.nn.ModuleList()
            for _b in range(self.num_bands):
                head = torch.nn.ModuleList()
                head.append(torch.nn.Linear(dim_input, dim_out if enl == 1 else dim_internal))
                for i in range(ens_num_layers - 1):
                    head.append(torch.nn.GELU())
                    head.append(
                        torch.nn.Linear(dim_internal, dim_out if enl - 2 == i else dim_internal)
                    )
                if final_activation is not None and enl >= 1:
                    head.append(ActivationFactory.get(final_activation))
                per_ens.append(head)
            self.band_heads.append(per_ens)

    def forward(self, toks, coords):
        preds = []
        for per_ens in self.band_heads:
            cpred = None
            for b, head in enumerate(per_ens):
                phases = 2.0 * torch.pi * (coords @ self.Ws[b].T)
                ff = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
                x = torch.cat([toks, ff], dim=-1)
                for block in head:
                    x = block(x)
                cpred = x if cpred is None else cpred + x
            preds.append(cpred)
        return torch.stack(preds, 0)


class TargetPredictionEngineClassic(nn.Module):
    def __init__(
        self,
        cf,
        dims_embed,
        dim_coord_in,
        tr_dim_head_proj,
        tr_mlp_hidden_factor,
        softcap,
        stream_config: dict,
    ):
        """
        Initialize the TargetPredictionEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param dims_embed: List of embedding dimensions for each layer.
        :param dim_coord_in: Input dimension for coordinates.
        :param tr_dim_head_proj: Dimension for head projection.
        :param tr_mlp_hidden_factor: Hidden factor for the MLP layers.
        :param softcap: Softcap value for the attention layers.
        """
        super(TargetPredictionEngineClassic, self).__init__()
        self.name = f"TargetPredictionEngine_{stream_config['name']}"

        self.cf = cf
        self.dims_embed = dims_embed
        self.dim_coord_in = dim_coord_in
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor
        self.softcap = softcap
        self.tte = torch.nn.ModuleList()

        for i in range(len(self.dims_embed) - 1):
            # Multi-Cross Attention Head
            self.tte.append(
                MultiCrossAttentionHeadVarlen(
                    dim_embed_q=self.dims_embed[i],
                    dim_embed_kv=self.cf.ae_global_dim_embed,
                    num_heads=stream_config["target_readout"]["num_heads"],
                    dim_head_proj=self.tr_dim_head_proj,
                    with_residual=True,
                    with_qk_lnorm=True,
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                    softcap=self.softcap,
                    dim_aux=self.dim_coord_in,
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )

            # Optional Self-Attention Head
            if self.cf.pred_self_attention:
                self.tte.append(
                    MultiSelfAttentionHeadVarlen(
                        dim_embed=self.dims_embed[i],
                        num_heads=stream_config["target_readout"]["num_heads"],
                        dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                        with_qk_lnorm=True,
                        with_flash=self.cf.with_flash_attention,
                        norm_type=self.cf.norm_type,
                        qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
                        dim_aux=self.dim_coord_in,
                        norm_eps=self.cf.norm_eps,
                        attention_dtype=get_dtype(self.cf.attention_dtype),
                    )
                )

            # MLP Block
            self.tte.append(
                MLP(
                    self.dims_embed[i],
                    self.dims_embed[i + 1],
                    with_residual=True,
                    hidden_factor=self.tr_mlp_hidden_factor,
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    norm_type=self.cf.norm_type,
                    dim_aux=(self.dim_coord_in if self.cf.pred_mlp_adaln else None),
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, latent, output, latent_lens, output_lens, coordinates):
        tc_tokens = output
        tcs_lens = output_lens
        tokens_stream = latent
        tokens_lens = latent_lens
        tcs_aux = coordinates

        for ib, block in enumerate(self.tte):
            if self.cf.pred_self_attention and ib % 3 == 1:
                tc_tokens = checkpoint(block, tc_tokens, tcs_lens, tcs_aux, use_reentrant=False)
            else:
                tc_tokens = checkpoint(
                    block,
                    tc_tokens,
                    tokens_stream,
                    tcs_lens,
                    tokens_lens,
                    tcs_aux,
                    use_reentrant=False,
                )
        return tc_tokens


class TargetPredictionEngine(nn.Module):
    def __init__(
        self,
        cf,
        dims_embed,
        dim_coord_in,
        tr_dim_head_proj,
        tr_mlp_hidden_factor,
        softcap,
        stream_name: str,
    ):
        """
        Initialize the TargetPredictionEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param dims_embed: List of embedding dimensions for each layer.
        :param dim_coord_in: Input dimension for coordinates.
        :param tr_dim_head_proj: Dimension for head projection.
        :param tr_mlp_hidden_factor: Hidden factor for the MLP layers.
        :param softcap: Softcap value for the attention layers.

        the decoder_type decides the how the conditioning is done

        PerceiverIO: is a simple CrossAttention layer with no MLP or Adaptive LayerNorm
        AdaLayerNormConditioning: only conditions via the Adaptive LayerNorm
        CrossAttentionConditioning: conditions via the CrossAttention layer but also uses an MLP
        CrossAttentionAdaNormConditioning: conditions via the CrossAttention layer and
            Adaptive LayerNorm
        PerceiverIOCoordConditioning: The conditioning is the coordinates and is a modified Adaptive
            LayerNorm that does not scale after the layer is applied
        """
        super(TargetPredictionEngine, self).__init__()
        self.name = f"TargetPredictionEngine_{stream_name}"

        self.cf = cf
        self.dims_embed = dims_embed
        self.dim_coord_in = dim_coord_in
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor
        self.softcap = softcap

        # For backwards compatibility

        self.cf = OmegaConf.merge(
            OmegaConf.create({"decoder_type": "PerceiverIOCoordConditioning"}), self.cf
        )

        attention_kwargs = {
            "with_qk_lnorm": True,
            "dropout_rate": 0.1,  # Assuming dropout_rate is 0.1
            "with_flash": self.cf.with_flash_attention,
            "norm_type": self.cf.norm_type,
            "qk_norm_type": self.cf.qk_norm_type,
            "softcap": self.softcap,
            "dim_aux": self.dim_coord_in,
            "norm_eps": self.cf.norm_eps,
            "attention_dtype": get_dtype(self.cf.attention_dtype),
        }
        self.tte = nn.ModuleList()
        self.output_in_norm = nn.LayerNorm(self.dims_embed[0])
        self.latent_in_norm = nn.LayerNorm(self.cf.ae_global_dim_embed)
        self.final_norm = nn.Identity()  # nn.RMSNorm(self.dims_embed[-1])
        self.dropout = nn.Dropout(0.2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 9, self.cf.ae_global_dim_embed))
        dim_aux = self.cf.ae_global_dim_embed

        target_readout_num_heads = next(self.cf.streams.values())["target_readout"]["num_heads"]
        for ith, dim in enumerate(self.dims_embed[:-1]):
            if self.cf.decoder_type == "PerceiverIO":
                # a single cross attention layer as per https://arxiv.org/pdf/2107.14795
                self.tte.append(
                    CrossAttentionBlock(
                        dim_q=dim,
                        dim_kv=dim_aux,
                        dim_aux=dim_aux,
                        num_heads=target_readout_num_heads,
                        with_self_attn=False,
                        with_adanorm=False,
                        with_mlp=False,
                        attention_kwargs=attention_kwargs,
                    )
                )
            elif self.cf.decoder_type == "AdaLayerNormConditioning":
                self.tte.append(
                    SelfAttentionBlock(
                        dim=dim,
                        dim_aux=dim_aux,
                        num_heads=target_readout_num_heads,
                        attention_kwargs=attention_kwargs,
                        with_adanorm=True,
                        dropout_rate=0.1,
                    )
                )
            elif self.cf.decoder_type == "CrossAttentionConditioning":
                self.tte.append(
                    CrossAttentionBlock(
                        dim_q=dim,
                        dim_kv=self.cf.ae_global_dim_embed,
                        dim_aux=dim_aux,
                        num_heads=target_readout_num_heads,
                        with_self_attn=True,
                        with_adanorm=False,
                        with_mlp=True,
                        dropout_rate=0.1,
                        attention_kwargs=attention_kwargs,
                    )
                )
            elif self.cf.decoder_type == "CrossAttentionAdaNormConditioning":
                self.tte.append(
                    CrossAttentionBlock(
                        dim_q=dim,
                        dim_kv=dim_aux,
                        dim_aux=dim_aux,
                        num_heads=target_readout_num_heads,
                        with_self_attn=True,
                        with_adanorm=True,
                        with_mlp=True,
                        dropout_rate=0.1,
                        attention_kwargs=attention_kwargs,
                    )
                )
            elif self.cf.decoder_type == "PerceiverIOCoordConditioning":
                self.tte.append(
                    OriginalPredictionBlock(
                        config=self.cf,
                        dim_in=dim,
                        dim_out=self.dims_embed[ith + 1],
                        dim_kv=dim_aux,
                        dim_aux=self.dim_coord_in,
                        num_heads=target_readout_num_heads,
                        attention_kwargs=attention_kwargs,
                        tr_dim_head_proj=tr_dim_head_proj,
                        tr_mlp_hidden_factor=tr_mlp_hidden_factor,
                        mlp_norm_eps=self.cf.mlp_norm_eps,
                    )
                )
            else:
                raise NotImplementedError(
                    f"{self.cf.decoder_type} is not implemented for prediction heads"
                )

    def forward(self, latent, output, latent_lens, output_lens, coordinates):
        latent = (
            self.dropout(self.latent_in_norm(latent + self.pos_embed))
            if self.cf.decoder_type != "PerceiverIOCoordConditioning"
            else latent
        )
        for layer in self.tte:
            if isinstance(layer, OriginalPredictionBlock):
                output = checkpoint(
                    layer,
                    latent=latent.flatten(0, 1),
                    output=output,
                    coords=coordinates,
                    latent_lens=latent_lens,
                    output_lens=output_lens,
                    use_reentrant=False,
                )
            elif isinstance(layer, CrossAttentionBlock):
                output = checkpoint(
                    layer,
                    x=output,
                    x_kv=latent.flatten(0, 1),
                    x_lens=output_lens,
                    aux=latent[:, 0],
                    x_kv_lens=latent_lens,
                    use_reentrant=False,
                )
            else:
                output = checkpoint(
                    layer,
                    x=output,
                    x_lens=output_lens,
                    aux=latent[:, 0],
                    use_reentrant=False,
                )
        output = (
            self.final_norm(output)
            if self.cf.decoder_type != "PerceiverIOCoordConditioning"
            else output
        )
        return output


class TargetPredictionEngineMLPMultiStage(nn.Module):
    """
    Multi-stage MLP decoder: K cross-attention lookups interleaved with MLP blocks.

    Splits the MLP stack into K equal stages, each preceded by its own cross-attention lookup
    into the same latent 1-ring neighbourhood.  The second (and later) lookups allow the
    decoder to re-attend to the latent after MLP processing, testing whether iterative
    spatial refinement helps over a single lookup.

    Architecture (num_cross_attn=2, num_layers=4 example):
        CrossAttn_0 → MLP_0 → MLP_1 → CrossAttn_1 → MLP_2 → MLP_3

    Forward signature is identical to TargetPredictionEngineClassic:
        forward(latent, output, latent_lens, output_lens, coordinates)

    Config consumed (from cf and stream_config):
        - cf.ae_global_dim_embed       : KV embedding dimension
        - cf.with_flash_attention      : flash attention toggle
        - cf.norm_type                 : LayerNorm or RMSNorm
        - cf.norm_eps, cf.mlp_norm_eps : norm epsilons
        - cf.pred_mlp_adaln            : enable coord conditioning in MLPs
        - cf.attention_dtype           : attention dtype (bf16 recommended)
        - stream_config["target_readout"]["num_layers"]    : total MLP blocks; must be divisible
                                                             by num_cross_attn
        - stream_config["target_readout"]["num_heads"]     : attention heads
        - stream_config["target_readout"]["num_cross_attn"]: number of stages (default: 2)
    Optional in cf:
        - cf.qk_norm_type              : qk-lnorm type override (defaults to norm_type)
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
    ):
        super(TargetPredictionEngineMLPMultiStage, self).__init__()
        self.name = f"TargetPredictionEngineMLPMultiStage_{stream_config['name']}"

        self.cf = cf
        self.dims_embed = dims_embed
        self.dim_coord_in = dim_coord_in
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor
        self.softcap = softcap

        num_cross_attn = stream_config["target_readout"].get("num_cross_attn", 2)
        num_mlp_blocks = len(self.dims_embed) - 1
        assert num_mlp_blocks % num_cross_attn == 0, (
            f"num_layers ({num_mlp_blocks}) must be divisible by num_cross_attn ({num_cross_attn})"
        )

        cross_attn_kwargs = dict(
            dim_embed_q=self.dims_embed[0],
            dim_embed_kv=self.cf.ae_global_dim_embed,
            num_heads=stream_config["target_readout"]["num_heads"],
            dim_head_proj=self.tr_dim_head_proj,
            with_residual=True,
            with_qk_lnorm=True,
            dropout_rate=0.1,
            with_flash=self.cf.with_flash_attention,
            norm_type=self.cf.norm_type,
            qk_norm_type=self.cf.get("qk_norm_type", self.cf.norm_type),
            softcap=self.softcap,
            dim_aux=self.dim_coord_in,
            norm_eps=self.cf.norm_eps,
            attention_dtype=get_dtype(self.cf.attention_dtype),
        )
        self.cross_attns = torch.nn.ModuleList(
            [MultiCrossAttentionHeadVarlen(**cross_attn_kwargs) for _ in range(num_cross_attn)]
        )

        self.mlp_blocks = torch.nn.ModuleList()
        for i in range(num_mlp_blocks):
            self.mlp_blocks.append(
                MLP(
                    self.dims_embed[i],
                    self.dims_embed[i + 1],
                    with_residual=True,
                    hidden_factor=self.tr_mlp_hidden_factor,
                    dropout_rate=0.1,
                    norm_type=self.cf.norm_type,
                    dim_aux=(self.dim_coord_in if self.cf.pred_mlp_adaln else None),
                    norm_eps=self.cf.mlp_norm_eps,
                )
            )

    def forward(self, latent, output, latent_lens, output_lens, coordinates):
        """
        Args:
            latent       : [total_kv_tokens, ae_global_dim_embed] flattened 1-ring KV tokens.
            output       : [total_query_tokens, dims_embed[0]] query embeddings (target coords).
            latent_lens  : [num_groups + 1] per-cell KV lens (typically 9 for 1-ring).
            output_lens  : [num_groups + 1] per-cell query lens.
            coordinates  : [total_query_tokens, dim_coord_in] raw coordinates for AdaLN.

        Returns:
            [total_query_tokens, dims_embed[-1]] decoded per-query embeddings ready for pred_head.
        """
        x = output
        mlps_per_stage = len(self.mlp_blocks) // len(self.cross_attns)
        for stage_idx, ca in enumerate(self.cross_attns):
            x = checkpoint(
                ca, x, latent, output_lens, latent_lens, coordinates, use_reentrant=False
            )
            start = stage_idx * mlps_per_stage
            for mlp in self.mlp_blocks[start : start + mlps_per_stage]:
                x = checkpoint(mlp, x, coordinates, use_reentrant=False)
        return x


class CoarseContextBranch(nn.Module):
    """Gated cross-attention from the target queries into the pooled coarse-scale latent.

    Kept as its own module rather than as loose attributes of ``MultiScaleContextDecoder`` for a
    concrete warm-start reason: when a checkpoint lacks parameters, ``load_model_state`` groups
    the missing keys, takes the highest-level module covering them, and calls ``to_empty()`` +
    ``reset_parameters()`` on it. With the new weights sitting directly on the decoder, that root
    would be the *whole decoder* -- and the re-init would discard the inherited
    ``TargetPredictionEngineClassic`` weights the fine-tune exists to reuse. Confining them here
    makes this branch the root, so only the new weights are initialised.
    """

    def __init__(self, cf, dims_embed, dim_coord_in, tr_dim_head_proj, softcap, stream_config):
        super().__init__()

        self.num_levels = len(cf.get("decode_context_levels") or [])
        assert self.num_levels > 0, (
            "decoder_type: MultiScaleContext requires a non-empty decode_context_levels"
        )
        self.tokens_per_cell = 9 * self.num_levels

        dim_kv = cf.ae_global_dim_embed
        # (level, compass direction) embedding for the coarse keys; small init so the slots are
        # distinguishable from the first step. gamma still gates the whole branch to zero.
        self.slot_embed = torch.nn.Parameter(torch.empty(self.tokens_per_cell, dim_kv))

        num_blocks = cf.get("decode_context_num_blocks", 1)
        self.ctx_attns = torch.nn.ModuleList(
            [
                MultiCrossAttentionHeadVarlen(
                    dim_embed_q=dims_embed[0],
                    dim_embed_kv=dim_kv,
                    num_heads=stream_config["target_readout"]["num_heads"],
                    dim_head_proj=tr_dim_head_proj,
                    # the delta is gated externally by gamma, so no internal residual
                    with_residual=False,
                    with_qk_lnorm=True,
                    dropout_rate=0.1,
                    with_flash=cf.with_flash_attention,
                    norm_type=cf.norm_type,
                    qk_norm_type=cf.get("qk_norm_type", cf.norm_type),
                    softcap=softcap,
                    dim_aux=dim_coord_in,
                    norm_eps=cf.norm_eps,
                    attention_dtype=get_dtype(cf.attention_dtype),
                )
                for _ in range(num_blocks)
            ]
        )
        self.gamma = torch.nn.Parameter(torch.empty(num_blocks))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise every weight under this branch.

        Called directly by the warm-start path after ``to_empty()``, so it must cover *all*
        parameters here, not just the two owned outright.
        """
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, RMSNorm):
                # RMSNorm holds a bare weight and defines no reset_parameters of its own
                torch.nn.init.ones_(module.weight)
            elif hasattr(module, "reset_parameters"):
                module.reset_parameters()
        torch.nn.init.normal_(self.slot_embed, std=1.0 / self.slot_embed.shape[-1])
        # zero init => the decoder starts bit-identical to TargetPredictionEngineClassic
        torch.nn.init.zeros_(self.gamma)

    def forward(self, output, ctx_latent, ctx_lens, output_lens, coordinates):
        """Returns the query tokens with the gated coarse-context delta added."""
        if ctx_latent is None or ctx_latent.shape[0] == 0:
            return output

        num_groups = ctx_latent.shape[0] // self.tokens_per_cell
        ctx = ctx_latent + self.slot_embed.repeat(num_groups, 1)

        q = output
        for i_block, ctx_attn in enumerate(self.ctx_attns):
            delta = checkpoint(
                ctx_attn, q, ctx, output_lens, ctx_lens, coordinates, use_reentrant=False
            )
            q = q + self.gamma[i_block] * delta
        return q


class MultiScaleContextDecoder(TargetPredictionEngineClassic):
    """
    Coordinate-conditioned readout with an added coarse-scale context branch.

    The production decoder lets every target point attend to the 9 latent vectors of its own
    HEALPix cell's 1-ring -- roughly 450 km at L5, at a single scale, with keys that carry no
    positional code. Where a sharp atmospheric gradient belongs (a front, a convergence line, a
    shear zone) is set by the deformation and thermal-gradient field over 500-1500 km, so the
    decoder is being asked to place fine structure while blind to the synoptic setting that
    determines it. This class widens the receptive field across HEALPix scales.

    Architecture::

        q = output + sum_i  gamma_i * ctx_attn_i(output, coarse_kv)     # new, gated
        x = TargetPredictionEngineClassic.forward(latent, q, ...)       # inherited, unchanged

    It **subclasses** ``TargetPredictionEngineClassic`` rather than wrapping it so the inherited
    block stack keeps its state-dict paths (``tte.*``). A checkpoint trained with the classic
    decoder therefore loads into this one with no re-keying: the only missing keys are
    ``context.*``.

    ``coarse_kv`` is built by ``utils.pool_latent_levels`` / ``utils.gather_context_latent``, which
    ``Model.predict_decoders`` calls per step: for each configured level in
    ``cf.decode_context_levels`` (e.g. ``[4, 3]``), the 1-ring of the target cell's ancestor at
    that level, mean-pooled over its descendants -- 9 tokens per level, ~2^(hl-l) x the cell
    width. Levels are concatenated per cell, so ``coarse_kv`` carries ``9 * num_levels`` tokens
    per active cell.

    Two properties are deliberate:

    - **Exact continuation.** ``gamma`` is zero-initialised and the context branch is a pure
      additive delta on the query, so at step 0 the module is bit-identical to
      ``TargetPredictionEngineClassic``. A checkpoint trained with the classic decoder can be
      fine-tuned into this one without a warm-up transient. (Merging the coarse tokens into the
      *same* KV set would not have this property: they would absorb softmax mass from the fine
      tokens even with their values zeroed.)
    - **Position on the keys.** ``slot_embed`` is a learned ``[9 * num_levels, dim_kv]`` table
      added to the coarse KV. Because ``hp.neighbours`` returns the 1-ring in a fixed order, slot
      ``j`` of a level is a consistent compass direction, so one table encodes both *which scale*
      and *which direction* a key sits in -- the relative geometry the cross-attention otherwise
      cannot see. This needs no change to ``MultiCrossAttentionHeadVarlen``, unlike RoPE (which
      has no hook there, and whose ``apply_rotary_pos_emb`` assumes q and k are index-aligned).

    Config consumed:
        - ``cf.decode_context_levels``     : coarse HEALPix levels, e.g. ``[4, 3]``. Required.
        - ``cf.decode_context_num_blocks`` : context cross-attention blocks (default 1).
        - everything ``TargetPredictionEngineClassic`` consumes.
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
    ):
        super(MultiScaleContextDecoder, self).__init__(
            cf,
            dims_embed,
            dim_coord_in,
            tr_dim_head_proj,
            tr_mlp_hidden_factor,
            softcap,
            stream_config=stream_config,
        )
        self.name = f"MultiScaleContextDecoder_{stream_config['name']}"

        self.context = CoarseContextBranch(
            cf, dims_embed, dim_coord_in, tr_dim_head_proj, softcap, stream_config
        )

    @property
    def tokens_per_cell(self) -> int:
        return self.context.tokens_per_cell

    def forward(
        self,
        latent,
        output,
        latent_lens,
        output_lens,
        coordinates,
        ctx_latent=None,
        ctx_lens=None,
    ):
        """
        Args beyond the TargetPredictionEngineClassic contract:
            ctx_latent : ``[num_active * 9 * num_levels, ae_global_dim_embed]`` pooled coarse KV,
                grouped per cell, levels concatenated. ``None`` skips the context branch.
            ctx_lens   : ``[num_groups + 1]`` int32, ``9 * num_levels`` at cells with targets this
                step and ``0`` elsewhere, ``[0] == 0``.
        """
        q = self.context(output, ctx_latent, ctx_lens, output_lens, coordinates)
        return super().forward(latent, q, latent_lens, output_lens, coordinates)


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


class FlowMatchingPointDecoder(TargetPredictionEngineClassic):
    """
    Conditional flow-matching readout: emits jointly sampled fields instead of a regression.

    Every other decoder on this branch factorises ``p(y | latent)`` over target points, so it can
    only ever produce correct one-point statistics. Pinball quantile heads make that concrete: the
    tau=0.97 map is "the 97th percentile everywhere at once", which is not a realization of
    anything. This decoder learns a velocity field that transports noise to data and integrates it
    at inference, so each forward pass is one coherent draw.

    **Why it is not vulnerable to the failure that sank the quantile arm.** A noise-conditioned
    regressor scored by a proper loss keeps a degenerate optimum available -- ignore the noise,
    emit the conditional mean. Flow matching removes that option: the noise *is* the state being
    transported, so at small ``t`` the network's input is essentially pure noise and being a
    function of it is unavoidable.

    **Where the spatial coherence comes from.** Not from correlated base noise -- the base is iid
    Gaussian per point. It comes from the inherited block stack, which already self-attends over
    the target points grouped by cell (``pred_self_attention``, varlen over ``tcs_lens``). That
    makes the model ``p(y_cell | latent)`` jointly over the ~750 points of a cell rather than
    point-by-point, which is exactly what pointwise quantiles structurally cannot do. Set
    ``pred_self_attention: True`` or this decoder degenerates into an expensive per-point sampler.

    Training (conditional flow matching, Lipman et al. 2023). With ``y1`` the target and
    ``y0 ~ N(0, I)``, on the linear path ``y_t = (1-t) y0 + t y1`` the target velocity is
    ``u = y1 - y0``. The module returns ``y0 + v_theta`` rather than ``v_theta``, so the ordinary
    ``mse`` term of ``LossPhysical`` computes ``||y1 - (y0 + v)||^2 = ||u - v||^2`` -- the flow
    matching objective exactly, with the existing channel/point weighting and NaN masking, and
    with no new loss function or plumbing.

    Evaluation. Integrates ``dy/dt = v_theta`` from ``t=0`` to ``1`` with ``flow_num_steps`` Euler
    steps, drawing ``pred_head.ens_size`` independent samples. Validation MSE is therefore a
    *sample* MSE and is expected to sit above a regression arm's by construction -- read the
    structure function instead, and specifically member-SF against mean-SF: a coherent generator
    has member-SF close to the target while mean-SF collapses.

    Config consumed:
        - ``cf.flow_num_steps`` : Euler steps at inference (default 24).
        - ``cf.flow_dim_time``  : width of the sinusoidal time embedding (default 32).
        - ``pred_head.ens_size``: number of samples drawn at evaluation.
        - everything ``TargetPredictionEngineClassic`` consumes.
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
        # so the inherited stack must be built for the widened aux vector
        super(FlowMatchingPointDecoder, self).__init__(
            cf,
            dims_embed,
            dim_coord_in + dim_time,
            tr_dim_head_proj,
            tr_mlp_hidden_factor,
            softcap,
            stream_config=stream_config,
        )
        self.name = f"FlowMatchingPointDecoder_{stream_config['name']}"

        self.dim_time = dim_time
        self.num_channels = num_channels
        self.num_steps = int(cf.get("flow_num_steps", 24))

        # current ODE state -> query token, and decoded token -> velocity
        self.embed_state = torch.nn.Linear(num_channels, dims_embed[0])
        # Default init on purpose. Zero-initialising an output head is the usual trick for a
        # *gated residual* branch, but here the head is the only path from the block stack to
        # the loss: with zero weights the gradient w.r.t. the decoded tokens is exactly zero, so
        # the whole stack and embed_state would receive no gradient at all on the first step.
        self.vel_head = torch.nn.Linear(dims_embed[-1], num_channels)

    def velocity(self, y_t, t, latent, output, latent_lens, output_lens, coordinates):
        """One evaluation of ``v_theta(y_t, t, token, coords)`` -> ``[N, num_channels]``."""
        q = output + self.embed_state(y_t.to(output.dtype))
        aux = torch.cat([coordinates, flow_time_embedding(t, self.dim_time)], dim=-1)
        tokens = super().forward(latent, q, latent_lens, output_lens, aux)
        return self.vel_head(tokens)

    def sample(self, latent, output, latent_lens, output_lens, coordinates, ens_size):
        """Integrate the probability flow ODE; returns ``[ens_size, N, num_channels]``."""
        n = output.shape[0]
        dt = 1.0 / self.num_steps
        preds = []
        # never needed with gradients: an unrolled 24-step ODE would hold 24x the activations
        with torch.no_grad():
            for _ in range(ens_size):
                y = torch.randn((n, self.num_channels), device=output.device, dtype=torch.float32)
                for i_step in range(self.num_steps):
                    t = torch.full((n, 1), i_step * dt, device=output.device, dtype=torch.float32)
                    v = self.velocity(y, t, latent, output, latent_lens, output_lens, coordinates)
                    y = y + dt * v.float()
                preds.append(y)
        return torch.stack(preds, 0)

    def forward(
        self,
        latent,
        output,
        latent_lens,
        output_lens,
        coordinates,
        target=None,
        ens_size=1,
    ):
        """
        Args beyond the TargetPredictionEngineClassic contract:
            target   : ``[N, num_channels]`` ground truth, required in training mode and ignored
                in eval. Must be in the decoder's point order (it is: ``LossPhysical`` pairs
                prediction and target by a plain reshape, with no permutation).
            ens_size : samples to draw in eval mode.

        Returns:
            ``[1, N, num_channels]`` in training (``y0 + v``, so plain mse is the flow-matching
            loss) or ``[ens_size, N, num_channels]`` in eval (integrated samples).
        """
        if not self.training:
            return self.sample(latent, output, latent_lens, output_lens, coordinates, ens_size)

        assert target is not None, (
            "FlowMatchingPointDecoder needs the target during training to build the "
            "probability path; predict_decoders must pass it."
        )
        y1 = target.float()
        y0 = torch.randn_like(y1)
        # masked / spoofed targets are NaN. Substituting y0 gives them zero velocity instead of
        # poisoning y_t; the loss masks these points anyway, so they contribute no gradient.
        y1 = torch.where(torch.isfinite(y1), y1, y0)

        t = torch.rand((y1.shape[0], 1), device=y1.device, dtype=y1.dtype)
        y_t = (1.0 - t) * y0 + t * y1

        v = self.velocity(y_t, t, latent, output, latent_lens, output_lens, coordinates)
        return (y0 + v.float()).unsqueeze(0)


@dataclasses.dataclass
class LatentState:
    """
    A dataclass to encapsulate the latent state aka the intput to latent heads.
    """

    class_token: torch.Tensor
    register_tokens: torch.Tensor
    patch_tokens: torch.Tensor
    z_pre_norm: torch.Tensor


class LatentPredictionHeadTransformer(nn.Module):
    def __init__(
        self,
        cf: Config,
        name: str,
        in_dim: int,
        loss_conf,
        use_class_token: bool,
        use_patch_token: bool,
    ):
        super().__init__()

        self.name = name

        out_dim, num_blocks, num_heads, with_qk_lnorm, intermediate_dim, dropout_rate = (
            loss_conf["out_dim"],
            loss_conf["num_blocks"],
            loss_conf["num_heads"],
            loss_conf["with_qk_lnorm"],
            loss_conf["intermediate_dim"],
            loss_conf["dropout_rate"],
        )

        self.global_cf = cf
        self.use_class_token = use_class_token
        self.use_patch_token = use_patch_token

        self.blocks = nn.ModuleList()

        # first map to intermediate_dim to introduce a bottleneck
        self.blocks.append(nn.Linear(in_dim, intermediate_dim, bias=False))

        for _ in range(num_blocks):
            self.blocks.append(
                MultiSelfAttentionHead(
                    intermediate_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    with_qk_lnorm=with_qk_lnorm,
                    with_flash=self.global_cf.with_flash_attention,
                    norm_type=self.global_cf.norm_type,
                    qk_norm_type=self.global_cf.qk_norm_type,
                    # dim_aux=dim_aux,
                    norm_eps=self.global_cf.norm_eps,
                    attention_dtype=get_dtype(self.global_cf.attention_dtype),
                )
            )
            # Add MLP block
            self.blocks.append(
                MLP(
                    intermediate_dim,
                    intermediate_dim,
                    hidden_factor=4,
                    with_residual=True,
                    dropout_rate=dropout_rate,
                    norm_type=self.global_cf.norm_type,
                    # dim_aux=dim_aux,
                    norm_eps=self.global_cf.mlp_norm_eps,
                )
            )

        # finally map from intermediate_dim to the out_dim
        self.blocks.append(nn.Linear(intermediate_dim, out_dim, bias=False))

    def forward(self, x: LatentState):
        # we concatenate the patch and class tokens to process them together
        # We concatenate in the token dimension [Batch, Tokens, Dim]
        patch_class_tokens = []
        if self.use_class_token:
            patch_class_tokens.append(x.class_token)
        if self.use_patch_token:
            patch_class_tokens.append(x.patch_tokens)
        patch_class_tokens = torch.cat(patch_class_tokens, dim=1)

        for _b_idx, block in enumerate(self.blocks):
            if isinstance(block, torch.nn.modules.normalization.LayerNorm):
                patch_class_tokens = block(patch_class_tokens)
            else:
                patch_class_tokens = checkpoint(block, patch_class_tokens, use_reentrant=False)
        return patch_class_tokens


class LatentPredictionHeadIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        return

    def forward(self, x: LatentState):
        return x.patch_tokens


class LatentPredictionHeadMLP(nn.Module):
    def __init__(self, name, in_dim: int, loss_conf, use_class_token: bool, use_patch_token: bool):
        super().__init__()

        self.name = name

        out_dim, num_layers, hidden_factor = (
            loss_conf["out_dim"],
            loss_conf["num_layers"],
            loss_conf["hidden_factor"],
        )

        self.use_class_token = use_class_token
        self.use_patch_token = use_patch_token

        # Create an MLP block
        self.blocks = MLP(in_dim, out_dim, num_layers, hidden_factor)

    def forward(self, x: LatentState):
        outputs = []
        if self.use_class_token:
            outputs.append(self.blocks(x.class_token))
        if self.use_patch_token:
            outputs.append(self.blocks(x.patch_tokens))

        return torch.cat(outputs, dim=1)


class EfficientBilinear(torch.nn.Module):
    def __init__(self, in_dim_lhs, in_dim_rhs, out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out, in_dim_lhs, in_dim_rhs))
        self.bias = nn.Parameter(torch.zeros(out)) if bias else 0.0
        self.total_in = in_dim_lhs * in_dim_rhs

    def forward(self, x_lhs, x_rhs):
        return torch.einsum("bi,oij,bj->bo", x_lhs, self.weight, x_rhs) + self.bias

    def reset_parameters(self):
        if isinstance(self.weight, nn.Parameter):
            bound = math.sqrt(2.0 / self.total_in)
            nn.init.uniform_(self.weight, -bound, bound)
        if isinstance(self.bias, nn.Parameter):
            nn.init.zeros_(self.bias)


class BilinearDecoder(nn.Module):
    def __init__(self, stream_name, coord_dim, latent_dim, out_dim):
        super().__init__()

        self.name = f"BilinearDecoder_{stream_name}"
        self.latent_dim = latent_dim
        self.bilin = EfficientBilinear(coord_dim, latent_dim, out_dim)

    def forward(self, coords_md, latent_nd, tcs_lens_n1):
        """
        Using Noam Shazeer notation
        N = Number of latent tokens*batch_size (N1 means N+1)
        M = Number of coordinates to decode
        D = Hidden dimension
        """
        latent_md = torch.repeat_interleave(latent_nd, tcs_lens_n1[1:], 0)
        return self.bilin(coords_md, latent_md)
