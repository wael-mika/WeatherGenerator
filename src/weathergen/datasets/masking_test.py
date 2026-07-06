# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
from omegaconf import OmegaConf

from weathergen.datasets.masking import Masker
from weathergen.train.utils import TRAIN

HEALPIX_LEVEL = 1
NUM_CELLS = 12 * 4**HEALPIX_LEVEL

MODE_CFG = OmegaConf.create(
    {
        "training_mode": "masking",
        "model_input": {
            "input_physical": {
                "masking_strategy": "random",
                "masking_strategy_config": {"rate": 0.5},
            }
        },
        "losses": {"masking": {"loss_fcts": {"mse": {"weight": 1.0}}}},
    }
)


def _make_masker(streams: dict) -> Masker:
    masker = Masker(HEALPIX_LEVEL, TRAIN, OmegaConf.create(streams), MODE_CFG)
    masker.reset_rng(np.random.default_rng(7))
    return masker


GROUPED_STREAM = {
    "name": "S_GROUPED",
    "train_source_channels": ["ch_a1", "ch_a2", "ch_b1"],
    "train_target_channels": ["ch_a1", "ch_b1"],
    "variable_groups": {
        "grp_a": {"variables": ["ch_a.*"], "masking_rate": 0.5},
        "grp_b": {"variables": ["ch_b1"], "masking_rate": 0.5},
    },
}


def test_mode_a_complement_target():
    """Plain (ungrouped) MAE: target must be the exact complement of the source."""
    stream_info = {
        "name": "S_PLAIN",
        "train_source_channels": ["ch_a1"],
        "train_target_channels": ["ch_a1"],
    }
    masker = _make_masker({"S_PLAIN": stream_info})

    target_masks, source_masks, mapping = masker.build_samples_for_stream(
        "masking", NUM_CELLS, stream_info
    )

    assert len(source_masks) == 1 and len(target_masks) == 1
    src, tgt = source_masks.get_mask(0), target_masks.get_mask(0)
    assert (src ^ tgt).all(), "target must be the complement of the source"


def test_mode_a_grouped_union_and_complements():
    """Grouped MAE: source = union of group masks; target = union of group complements."""
    stream_info = dict(GROUPED_STREAM)
    masker = _make_masker({"S_GROUPED": stream_info})

    target_masks, source_masks, _ = masker.build_samples_for_stream(
        "masking", NUM_CELLS, stream_info
    )

    src_groups = source_masks.get_group_spatial_masks(0)
    tgt_groups = target_masks.get_group_spatial_masks(0)
    assert src_groups is not None and tgt_groups is not None
    assert set(src_groups) == {"grp_a", "grp_b"} and set(tgt_groups) == {"grp_a", "grp_b"}

    for g in src_groups:
        assert (src_groups[g] ^ tgt_groups[g]).all(), f"group {g}: target != complement"

    src_union = source_masks.get_mask(0)
    tgt_union = target_masks.get_mask(0)
    assert (src_union == (src_groups["grp_a"] | src_groups["grp_b"])).all()
    assert (tgt_union == (tgt_groups["grp_a"] | tgt_groups["grp_b"])).all()


def test_forcing_stream_target_stays_empty():
    """Forcing streams are never decoded: the Mode-A complement override must not
    resurrect their all-False target mask."""
    for extra in ({}, {"variable_groups": GROUPED_STREAM["variable_groups"]}):
        stream_info = {
            "name": "S_FORCING",
            "forcing": True,
            "train_source_channels": ["ch_a1"],
            "train_target_channels": ["ch_a1"],
            **extra,
        }
        masker = _make_masker({"S_FORCING": stream_info})

        target_masks, source_masks, _ = masker.build_samples_for_stream(
            "masking", NUM_CELLS, stream_info
        )

        tgt = target_masks.get_mask(0)
        assert tgt.sum() == 0, "forcing stream must keep an all-False target mask"
        assert target_masks.get_group_spatial_masks(0) is None
        # source-side masking must still be honoured for forcing streams
        assert source_masks.get_mask(0).sum() > 0


def test_repeated_calls_fresh_masks_stable_mapping():
    """The correspondence cache must not freeze anything sample-dependent: repeated calls
    give fresh random masks but an identical source->target mapping, and targets stay
    exact complements every time."""
    stream_info = {
        "name": "S_REPEAT",
        "train_source_channels": ["c1"],
        "train_target_channels": ["c1"],
    }
    masker = _make_masker({"S_REPEAT": stream_info})

    seen_masks = []
    for _ in range(5):
        target_masks, source_masks, mapping = masker.build_samples_for_stream(
            "masking", NUM_CELLS, stream_info
        )
        assert mapping.tolist() == [0]
        src, tgt = source_masks.get_mask(0), target_masks.get_mask(0)
        assert (src ^ tgt).all(), "complement must hold on every call"
        seen_masks.append(tuple(src.tolist()))

    assert len(set(seen_masks)) > 1, "source masks must differ across samples (RNG not frozen)"


def test_mixed_strategy_redraws_per_sample():
    """Skipping the per-sample deepcopy must not leak the resolved 'mixed' sub-strategy
    back into the effective config: each call must re-draw the sub-strategy."""
    mode_cfg = OmegaConf.create(
        {
            "training_mode": "masking",
            "model_input": {
                "input_physical": {
                    "masking_strategy": "mixed",
                    "masking_strategy_config": {
                        "rate": 0.5,
                        "strategies": ["random", "healpix"],
                        "hl_mask_levels": [0],
                    },
                }
            },
            "losses": {"masking": {"loss_fcts": {"mse": {"weight": 1.0}}}},
        }
    )
    stream_info = {
        "name": "S_MIXED",
        "train_source_channels": ["c1"],
        "train_target_channels": ["c1"],
    }
    masker = Masker(HEALPIX_LEVEL, TRAIN, OmegaConf.create({"S_MIXED": stream_info}), mode_cfg)
    masker.reset_rng(np.random.default_rng(3))

    chosen = set()
    for _ in range(20):
        target_masks, source_masks, _ = masker.build_samples_for_stream(
            "masking", NUM_CELLS, stream_info
        )
        strategy = source_masks.metadata[0].params["masking_strategy"]
        assert strategy in ("random", "healpix"), "mixed must be resolved to a sub-strategy"
        chosen.add(strategy)
        # complement correspondence must hold for the resolved strategy too
        assert (source_masks.get_mask(0) ^ target_masks.get_mask(0)).all()

    assert chosen == {"random", "healpix"}, (
        f"sub-strategy frozen to {chosen}: resolved 'mixed' leaked into the shared config"
    )

    # the effective config itself must still say 'mixed' (no mutation leaked)
    eff = masker._effective_masking_cfgs["S_MIXED"]["model_input"]["input_physical"]
    assert eff["masking_strategy"] == "mixed"


def test_group_masking_apply_base_bounds_source_and_target():
    """With group_masking_apply_base, every group's source and target masks partition one
    shared stream-level base mask: encoder and decoder work stays bounded by the base rate
    instead of the union of independent group masks growing towards full coverage."""
    stream_info = dict(GROUPED_STREAM)
    stream_info["name"] = "S_BASE"
    stream_info["group_masking_apply_base"] = True
    masker = _make_masker({"S_BASE": stream_info})

    for _ in range(5):
        target_masks, source_masks, _ = masker.build_samples_for_stream(
            "masking", NUM_CELLS, stream_info
        )
        src_groups = source_masks.get_group_spatial_masks(0)
        tgt_groups = target_masks.get_group_spatial_masks(0)

        bases = []
        for g in src_groups:
            src_g, tgt_g = src_groups[g], tgt_groups[g]
            assert not (src_g & tgt_g).any(), f"group {g}: source/target overlap"
            bases.append(src_g | tgt_g)
        # all groups partition the SAME base mask
        for b in bases[1:]:
            assert (b == bases[0]).all(), "groups must share one base mask"
        base = bases[0]
        # base is a proper subset (rate 0.5 in MODE_CFG, so not the full sphere)
        assert 0 < base.sum() < NUM_CELLS
        # stream-level source/target unions stay inside the base
        assert not (source_masks.get_mask(0) & ~base).any()
        assert not (target_masks.get_mask(0) & ~base).any()


def test_channel_drop_keeps_at_least_one_channel():
    """Even at drop rate ~1.0 the channel-drop mask must keep one channel."""
    stream_info = {
        "name": "S_DROP",
        "train_source_channels": ["c1", "c2"],
        "train_target_channels": ["c1", "c2"],
        "channel_drop_rate": 1.0,
    }
    masker = _make_masker({"S_DROP": stream_info})

    _, source_masks, _ = masker.build_samples_for_stream(
        "masking", NUM_CELLS, stream_info, num_channels=2
    )

    drop_mask = source_masks.get_channel_drop_mask(0)
    assert drop_mask is not None
    assert drop_mask.sum() == 1, "exactly one channel must survive an all-drop draw"


def test_diagnostic_stream_with_target_override():
    """A stream whose masking_override defines target_input (era5_out pattern) must get an
    explicit target (Mode C) honouring the override, with an all-False source (diagnostic)."""
    stream_info = {
        "name": "S_DIAG",
        "diagnostic": True,
        "train_source_channels": ["ch_a1"],
        "train_target_channels": ["ch_a1"],
        "masking_override": {"target_input": {"masking_strategy_config": {"rate": 1.0}}},
    }
    masker = _make_masker({"S_DIAG": stream_info})

    target_masks, source_masks, _ = masker.build_samples_for_stream(
        "masking", NUM_CELLS, stream_info
    )

    assert target_masks.get_mask(0).all(), "target override rate 1.0 must keep all cells"
    assert source_masks.get_mask(0).sum() == 0, "diagnostic stream source must be all-False"
