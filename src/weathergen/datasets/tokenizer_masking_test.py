# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import astropy_healpix as hp
import numpy as np
import pytest
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.masking import Masker
from weathergen.datasets.tokenizer_masking import TokenizerMasking

HEALPIX_LEVEL = 1
NUM_CELLS = 12 * 4**HEALPIX_LEVEL

CHANNEL_NAMES = ["ch_a1", "ch_a2", "ch_b1", "ch_x"]

STREAM_INFO = {
    "name": "TESTSTREAM",
    "stream_id": 0,
    "token_size": 8,
    "variable_groups": {
        "grp_a": {"variables": ["ch_a.*"]},
        "grp_b": {"variables": ["ch_b1"]},
    },
}


def _make_rdata(with_channels: bool = True) -> IOReaderData:
    """One data point at every healpix cell center, all data values 1.0."""
    lons, lats = hp.healpix_to_lonlat(
        np.arange(NUM_CELLS), 2**HEALPIX_LEVEL, dx=0.5, dy=0.5, order="nested"
    )
    coords = np.stack([lats.deg, lons.deg], axis=-1).astype(np.float32)
    geoinfos = np.zeros((NUM_CELLS, 1), dtype=np.float32)
    data = np.ones((NUM_CELLS, len(CHANNEL_NAMES)), dtype=np.float32)
    datetimes = np.full(NUM_CELLS, np.datetime64("2020-01-01T03:00"), dtype="datetime64[ns]")
    rdata = IOReaderData(coords, geoinfos, data, datetimes)
    if with_channels:
        rdata.source_channels = CHANNEL_NAMES
        rdata.target_channels = CHANNEL_NAMES
    return rdata


def _make_tokenizer() -> TokenizerMasking:
    masker = Masker(HEALPIX_LEVEL, "train")
    tokenizer = TokenizerMasking(HEALPIX_LEVEL, masker)
    tokenizer.reset_rng(np.random.default_rng(42))
    return tokenizer


TIME_WIN = (np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-01T06:00"))


def _get_source_tokens(tokenizer, rdata, group_spatial_masks, cell_mask):
    token_data = tokenizer.get_tokens_windows(STREAM_INFO, [rdata], True)[0]
    return tokenizer.get_source(
        STREAM_INFO,
        rdata,
        token_data,
        TIME_WIN,
        cell_mask,
        group_spatial_masks=group_spatial_masks,
    )


def test_group_channel_mask_applied_in_get_source():
    """Per-group spatial masks must zero the channels of groups not covering a cell."""
    tokenizer = _make_tokenizer()
    rdata = _make_rdata()

    grp_a_mask = torch.zeros(NUM_CELLS, dtype=torch.bool)
    grp_a_mask[: NUM_CELLS // 2] = True  # grp_a sees the first half of the cells
    grp_b_mask = ~grp_a_mask  # grp_b sees the second half
    group_spatial_masks = {"grp_a": grp_a_mask, "grp_b": grp_b_mask}
    cell_mask = grp_a_mask | grp_b_mask  # union, as built by the masker

    tokens_cells, tokens_per_cell = _get_source_tokens(
        tokenizer, rdata, group_spatial_masks, cell_mask
    )

    assert (tokens_per_cell == 1).all(), "expected one token per cell"
    # data columns are the last len(CHANNEL_NAMES) columns of each token row
    num_ch = len(CHANNEL_NAMES)
    for i_cell, token in enumerate(tokens_cells):
        # first row of the token is the real data point (rest is padding)
        data_cols = token[0, -num_ch:]
        in_a = bool(grp_a_mask[i_cell])
        # ch_a1, ch_a2 owned by grp_a; ch_b1 by grp_b; ch_x unassigned -> always kept
        expected = torch.tensor([1.0 * in_a, 1.0 * in_a, 1.0 * (not in_a), 1.0], dtype=token.dtype)
        assert torch.equal(data_cols, expected), f"cell {i_cell}: {data_cols} != {expected}"


def test_channel_drop_composes_with_group_masks():
    """channel_drop_rate must keep working when per-group masks are active: a dropped
    channel is zeroed everywhere, including cells its group covers."""
    tokenizer = _make_tokenizer()
    rdata = _make_rdata()

    group_spatial_masks = {
        "grp_a": torch.ones(NUM_CELLS, dtype=torch.bool),
        "grp_b": torch.ones(NUM_CELLS, dtype=torch.bool),
    }
    cell_mask = torch.ones(NUM_CELLS, dtype=torch.bool)
    # drop ch_a2 (grp_a) and ch_x (ungrouped)
    channel_drop_mask = np.array([True, False, True, False])

    token_data = tokenizer.get_tokens_windows(STREAM_INFO, [rdata], True)[0]
    tokens_cells, _ = tokenizer.get_source(
        STREAM_INFO,
        rdata,
        token_data,
        TIME_WIN,
        cell_mask,
        channel_drop_mask=channel_drop_mask,
        group_spatial_masks=group_spatial_masks,
    )

    num_ch = len(CHANNEL_NAMES)
    for i_cell, token in enumerate(tokens_cells):
        data_cols = token[0, -num_ch:]
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=token.dtype)
        assert torch.equal(data_cols, expected), f"cell {i_cell}: {data_cols} != {expected}"


def test_group_masking_without_channel_names_raises():
    """Missing channel names must fail loudly, not silently skip group channel masking."""
    tokenizer = _make_tokenizer()
    rdata = _make_rdata(with_channels=False)

    group_spatial_masks = {"grp_a": torch.ones(NUM_CELLS, dtype=torch.bool)}
    cell_mask = torch.ones(NUM_CELLS, dtype=torch.bool)

    with pytest.raises(ValueError, match="channel names"):
        _get_source_tokens(tokenizer, rdata, group_spatial_masks, cell_mask)


def test_channel_group_id_default_group():
    """Unmatched channels fall to _default when present, else stay always-kept (-1)."""
    tokenizer = _make_tokenizer()

    stream_info = {
        "name": "S_DEFAULT",
        "variable_groups": {
            "grp_a": {"variables": ["ch_a.*"]},
            "_default": {},
        },
    }
    group_order, gid = tokenizer._get_channel_group_id(stream_info, CHANNEL_NAMES)
    assert group_order == ["grp_a", "_default"]
    assert gid.tolist() == [0, 0, 1, 1]  # ch_b1/ch_x fall to _default

    group_order, gid = tokenizer._get_channel_group_id(STREAM_INFO, CHANNEL_NAMES)
    assert gid.tolist() == [0, 0, 1, -1]  # no _default: ch_x always kept
