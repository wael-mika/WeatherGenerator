# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.batch import SampleMetaData
from weathergen.datasets.masking import Masker
from weathergen.datasets.tokenizer import Tokenizer
from weathergen.datasets.tokenizer_utils import (
    encode_times_source,
    encode_times_target,
    tokenize_apply_mask_source,
    tokenize_apply_mask_target,
    tokenize_space,
    tokenize_spacetime,
)


def readerdata_to_torch(rdata: IOReaderData) -> IOReaderData:
    """
    Convert data, coords, and geoinfos to torch tensor
    """
    if type(rdata.coords) is not torch.Tensor:
        rdata.coords = torch.tensor(rdata.coords)
    if type(rdata.geoinfos) is not torch.Tensor:
        rdata.geoinfos = torch.tensor(rdata.geoinfos)
    if type(rdata.data) is not torch.Tensor:
        rdata.data = torch.tensor(rdata.data)

    return rdata


class TokenizerMasking(Tokenizer):
    def __init__(self, healpix_level: int, masker: Masker):
        super().__init__(healpix_level)
        self.masker = masker
        self.rng = None
        self.token_size = None

    def reset_rng(self, rng) -> None:
        """
        Reset rng after mini_epoch to ensure proper randomization
        """
        self.masker.reset_rng(rng)
        self.rng = rng

    def get_tokens_windows(self, stream_info, data, pad_tokens):
        """
        Tokenize data (to amortize over the different views that are generated)

        """

        tok_spacetime = stream_info.get("tokenize_spacetime", False)
        tok = tokenize_spacetime if tok_spacetime else tokenize_space
        hl = self.healpix_level
        token_size = stream_info["token_size"]

        tokens = []
        for rdata in data:
            # skip empty data
            if rdata.is_empty():
                tokens += [(None, None)]
                continue
            # tokenize data
            idxs_cells, idxs_cells_lens = tok(
                readerdata_to_torch(rdata), token_size, hl, pad_tokens
            )
            tokens += [(idxs_cells, idxs_cells_lens)]

        return tokens

    def build_samples_for_stream(
        self,
        training_mode: str,
        num_cells: int,
        stream_info: dict,
        num_channels: int | None = None,
    ) -> tuple[np.typing.NDArray, list[np.typing.NDArray], list[SampleMetaData]]:
        """
        Create masks for samples
        """
        return self.masker.build_samples_for_stream(
            training_mode, num_cells, stream_info, num_channels=num_channels
        )

    def cell_to_token_mask(
        self,
        idxs_cells,
        idxs_cells_lens,
        mask,
        channel_drop_mask=None,
        group_spatial_masks=None,
    ):
        """Convert a cell-level spatial mask to a token-level mask.

        Args:
            mask: (num_cells,) bool — spatial keep mask.
            channel_drop_mask: optional (num_channels,) bool — True = keep channel.
                When set, returned as mask_channels for the source tokenizer so that
                dropped channels are zeroed out in the token data.
            group_spatial_masks: optional dict {group_name: (num_cells,) bool} — per-group
                spatial masks.  When set, a 2-D (num_visible_tokens × num_channels) mask is
                computed so that each token only carries channels from groups that cover its cell.
                (Requires num_channels_per_group mapping; handled in Feature 1 extension.)
        """

        mask_tokens, mask_channels = None, None
        num_tokens = torch.tensor([len(t) for t in idxs_cells_lens]).sum().item()

        # If there are no tokens, return empty lists.
        if num_tokens == 0:
            return (mask_tokens, mask_channels)

        # TODO, TODO, TODO: use np.repeat
        # https://stackoverflow.com/questions/26038778/repeat-each-values-of-an-array-different-times
        # build token level mask: for each cell replicate the keep flag across its tokens
        token_level_flags: list[np.typing.NDArray] = []
        for km, lens_cell in zip(mask, idxs_cells_lens, strict=True):
            num_tokens_cell = len(lens_cell)
            if num_tokens_cell == 0:
                continue
            token_level_flags.append(
                np.ones(num_tokens_cell, dtype=bool)
                if km
                else np.zeros(num_tokens_cell, dtype=bool)
            )
        if token_level_flags:
            mask_tokens = np.concatenate(token_level_flags)
        else:
            mask_tokens = np.array([], dtype=bool)

        # Channel-level masks (independent of spatial masking).
        # group_spatial_masks produces a 2D token×channel mask (Feature 1); the
        # simpler 1D channel_drop_mask is used when only Feature 3 is active.
        if group_spatial_masks is not None:
            # 2D case: built below once we know which cells are visible (mask_tokens).
            # Resolved in get_source / get_target_values with channel name info.
            mask_channels = None  # set by caller after resolving group→channel mapping
        elif channel_drop_mask is not None:
            # 1D case: same channel mask applied to every visible token.
            mask_channels = torch.from_numpy(np.asarray(channel_drop_mask))

        return (mask_tokens, mask_channels)

    def _build_group_channel_mask_2d(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        idxs_cells,
        idxs_cells_lens,
        mask_tokens: np.typing.NDArray,
        group_spatial_masks: dict,
    ) -> torch.Tensor | None:
        """Build a 2-D (num_visible_tokens, num_channels) channel mask from per-group cell masks.

        For each visible token (cell), channel c is True (keep) if the group that owns c has
        that cell in its spatial mask.  Channels not assigned to any group are always kept.

        Args:
            stream_info: stream config containing ``variable_groups``.
            rdata: reader data providing ``source_channels`` (list of channel names).
            idxs_cells: list of per-cell token index lists (from tokenize_space/spacetime).
            idxs_cells_lens: per-cell token-size lists.
            mask_tokens: (num_all_tokens,) bool — which tokens are visible.
            group_spatial_masks: {group_name: (num_cells,) bool tensor}.

        Returns:
            2-D bool tensor of shape (num_visible_tokens, num_channels), or None if
            variable_groups is not configured on the stream.
        """
        import re

        vgroups = stream_info.get("variable_groups")
        channel_names = getattr(rdata, "source_channels", None)
        if vgroups is None or channel_names is None or not group_spatial_masks:
            return None

        num_channels = len(channel_names)

        # Map channel index → group name (unassigned channels stay in no group → always kept)
        channel_to_group: dict[int, str] = {}
        assigned: set[int] = set()
        for gname, gcfg in vgroups.items():
            if gname == "_default":
                continue
            patterns = [re.compile(p) for p in gcfg.get("variables", [])]
            for i, ch in enumerate(channel_names):
                if any(pat.fullmatch(ch) for pat in patterns):
                    channel_to_group[i] = gname
                    assigned.add(i)
        # Assign unmatched channels to _default group if it exists
        if "_default" in vgroups:
            for i in range(num_channels):
                if i not in assigned:
                    channel_to_group[i] = "_default"

        # Build per-cell visibility: cell_idx → set of groups covering this cell
        # idxs_cells is a list over cells; each element is a list of token index tensors
        group_spatial_np: dict[str, np.typing.NDArray] = {}
        for gname, gmask in group_spatial_masks.items():
            arr = gmask.numpy() if isinstance(gmask, torch.Tensor) else np.asarray(gmask)
            group_spatial_np[gname] = arr

        # Enumerate visible tokens (those with mask_tokens[t] == True)
        # and determine which cell each visible token belongs to.
        num_tokens_per_cell = [len(lens) for lens in idxs_cells_lens]
        token_to_cell: list[int] = []
        for cell_idx, n in enumerate(num_tokens_per_cell):
            token_to_cell.extend([cell_idx] * n)

        visible_token_cells = [
            token_to_cell[t] for t, keep in enumerate(mask_tokens) if keep
        ]

        num_visible = len(visible_token_cells)
        if num_visible == 0:
            return None

        # Build 2-D mask: (num_visible_tokens, num_channels)
        mask_2d = torch.ones(num_visible, num_channels, dtype=torch.bool)
        for tok_i, cell_idx in enumerate(visible_token_cells):
            for ch_i, gname in channel_to_group.items():
                if gname in group_spatial_np:
                    mask_2d[tok_i, ch_i] = bool(group_spatial_np[gname][cell_idx])

        return mask_2d

    def get_source(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        idxs_cells_data,
        time_win: tuple,
        cell_mask: torch.Tensor,
        channel_drop_mask=None,
        group_spatial_masks=None,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = idxs_cells_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells,
            idxs_cells_lens,
            cell_mask,
            channel_drop_mask=channel_drop_mask,
            group_spatial_masks=group_spatial_masks,
        )

        # If group_spatial_masks present, build 2-D channel mask now that we
        # have both token-level cell indices and channel names from rdata.
        if group_spatial_masks is not None and mask_tokens is not None:
            mask_channels = self._build_group_channel_mask_2d(
                stream_info,
                rdata,
                idxs_cells,
                idxs_cells_lens,
                mask_tokens,
                group_spatial_masks,
            )

        source_tokens_cells, source_tokens_lens = tokenize_apply_mask_source(
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            stream_info["stream_id"],
            rdata,
            time_win,
            self.hpy_verts_rots_source[-1],
            encode_times_source,
        )

        return (source_tokens_cells, source_tokens_lens)

    def get_target_coords(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        # TODO: split up
        _, _, _, coords_local, coords_per_cell = tokenize_apply_mask_target(
            stream_info["stream_id"],
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
        )

        return (coords_local, coords_per_cell)

    def get_target_values(
        self,
        stream_info: dict,
        rdata: IOReaderData,
        token_data,
        time_win: tuple,
        cell_mask,
    ):
        # create tokenization index
        (idxs_cells, idxs_cells_lens) = token_data

        (mask_tokens, mask_channels) = self.cell_to_token_mask(
            idxs_cells, idxs_cells_lens, cell_mask
        )

        data, datetimes, coords, _, _ = tokenize_apply_mask_target(
            stream_info["stream_id"],
            self.hl_target,
            idxs_cells,
            idxs_cells_lens,
            mask_tokens,
            mask_channels,
            rdata,
            time_win,
            self.hpy_verts_rots_target,
            self.hpy_verts_local_target,
            self.hpy_nctrs_target,
            encode_times_target,
        )

        idxs_ord_inv = None
        if data.numel() > 0:
            # flatten per-token indices into one flat list
            idxs_flat = torch.cat([idxs for idxs_cell in idxs_cells for idxs in idxs_cell])
            # compute indices for inversion
            _, idxs_ord_inv = torch.sort(idxs_flat)

        return (data, datetimes, coords, idxs_ord_inv)
