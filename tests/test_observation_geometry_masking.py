from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from weathergen.common import config as common_config
from weathergen.datasets.masking import Masker
from weathergen.train.utils import TRAIN


def _selected_parent_indices(mask_tensor, hl_data: int, hl_mask: int) -> np.ndarray:
    mask = mask_tensor.numpy()
    num_parents = 12 * (4**hl_mask)
    num_children_per_parent = 4 ** (hl_data - hl_mask)
    parent_view = mask.reshape(num_parents, num_children_per_parent)
    assert np.all(parent_view == parent_view[:, :1])
    return np.flatnonzero(parent_view[:, 0])


def _build_mask(masker: Masker, masking_strategy_config: dict):
    return masker._get_mask(  # noqa: SLF001 - testing strategy behavior directly
        num_cells=masker.healpix_num_cells,
        strategy="observation_healpix",
        masking_strategy_config=masking_strategy_config,
        target_relationship_mask=("independent", None),
        target_metadata=None,
    )


def test_observation_healpix_selects_requested_parent_count_and_records_metadata():
    masker = Masker(healpix_level=5, stage=TRAIN)
    masker.reset_rng(np.random.default_rng(0))

    cfg = {
        "rate": 0.2,
        "hl_mask": 3,
        "num_swaths": 2,
        "swath_tilt_degrees": 20.0,
        "swath_tilt_degrees_random": True,
        "scanline_spacing_degrees": 8.0,
        "scanline_fill_fraction": 0.5,
        "scanline_retain_rate": 0.9,
        "latitudinal_sampling_mode": "polar_dense",
        "latitudinal_sampling_strength": 0.35,
    }

    mask, params = _build_mask(masker, cfg)
    selected_parents = _selected_parent_indices(mask, hl_data=5, hl_mask=3)

    assert selected_parents.size == round(cfg["rate"] * (12 * (4**cfg["hl_mask"])))
    assert params["num_swaths"] == 2
    assert len(params["swath_center_longitudes"]) == 2
    assert len(params["swath_tilt_degrees"]) == 2
    assert params["scanline_spacing_degrees"] == cfg["scanline_spacing_degrees"]
    assert params["latitudinal_sampling_mode"] == cfg["latitudinal_sampling_mode"]


def test_observation_healpix_scanline_mask_follows_latitude_bands():
    masker = Masker(healpix_level=5, stage=TRAIN)
    masker.reset_rng(np.random.default_rng(1))

    cfg = {
        "rate": 0.2,
        "hl_mask": 3,
        "num_swaths": 1,
        "swath_center_longitudes": [0.0],
        "swath_tilt_degrees": 0.0,
        "swath_tilt_degrees_random": False,
        "scanline_spacing_degrees": 15.0,
        "scanline_fill_fraction": 0.25,
        "scanline_retain_rate": 1.0,
        "scanline_phase_degrees": 0.0,
        "latitudinal_sampling_mode": "uniform",
        "latitudinal_sampling_strength": 0.0,
    }

    mask, params = _build_mask(masker, cfg)
    selected_parents = _selected_parent_indices(mask, hl_data=5, hl_mask=3)
    _, lats_deg = masker._get_healpix_lonlat_degrees(3)  # noqa: SLF001 - cached geometry helper
    selected_lats = lats_deg[selected_parents]

    spacing = params["scanline_spacing_degrees"]
    fill_width = spacing * params["scanline_fill_fraction"]
    phase = params["scanline_phase_degrees"]
    assert np.all(np.mod(selected_lats + 90.0 + phase, spacing) < fill_width + 1e-9)


def test_observation_healpix_latitudinal_sampling_changes_selected_latitudes():
    polar_masker = Masker(healpix_level=5, stage=TRAIN)
    polar_masker.reset_rng(np.random.default_rng(7))
    equatorial_masker = Masker(healpix_level=5, stage=TRAIN)
    equatorial_masker.reset_rng(np.random.default_rng(7))

    base_cfg = {
        "rate": 0.2,
        "hl_mask": 3,
        "num_swaths": 1,
        "swath_center_longitudes": [0.0],
        "swath_tilt_degrees": 0.0,
        "swath_tilt_degrees_random": False,
        "latitudinal_sampling_strength": 1.0,
    }

    polar_mask, _ = _build_mask(
        polar_masker,
        {**base_cfg, "latitudinal_sampling_mode": "polar_dense"},
    )
    equatorial_mask, _ = _build_mask(
        equatorial_masker,
        {**base_cfg, "latitudinal_sampling_mode": "equatorial_dense"},
    )

    polar_selected = _selected_parent_indices(polar_mask, hl_data=5, hl_mask=3)
    equatorial_selected = _selected_parent_indices(equatorial_mask, hl_data=5, hl_mask=3)
    _, lats_deg = polar_masker._get_healpix_lonlat_degrees(3)  # noqa: SLF001

    assert np.abs(lats_deg[polar_selected]).mean() > np.abs(lats_deg[equatorial_selected]).mean()


def test_observation_geometry_config_smoke_loads():
    cfg_path = Path(
        "config/week3/config_jepa_frozen_2drope_qkrms_student_observation_geometry_1.yml"
    )
    cfg = OmegaConf.load(cfg_path)
    cfg = common_config._load_streams_in_config(cfg)

    assert len(cfg.streams) > 0
    assert cfg.training_config.model_input.observation_easy.masking_strategy == "observation_healpix"
    assert cfg.training_config.target_input.full_teacher_target.masking_strategy == "healpix"
    assert cfg.training_config.model_input.observation_easy.masking_strategy_config.num_swaths == 2
