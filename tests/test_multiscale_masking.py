from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.common import config as common_config
from weathergen.datasets.masking import Masker
from weathergen.train.loss_modules.loss_module_ssl import jepa_loss
from weathergen.train.utils import TRAIN, get_target_idxs_from_cfg


def _build_multiscale_stage_cfg():
    return OmegaConf.create(
        {
            "losses": {
                "student-teacher": {
                    "type": "LossLatentSSLStudentTeacher",
                    "loss_fcts": {
                        "JEPA": {
                            "weight": 4,
                            "loss_extra_args": {},
                            "target_source_correspondence": {0: {0: "subset", 1: "subset"}},
                        }
                    },
                }
            },
            "model_input": {
                "random_easy": {
                    "masking_strategy": "random",
                    "num_samples": 1,
                    "num_steps_input": 1,
                    "masking_strategy_config": {
                        "diffusion_rn": True,
                        "rate": 0.2,
                        "rate_sampling": False,
                    },
                },
                "healpix_easy": {
                    "masking_strategy": "healpix",
                    "num_samples": 1,
                    "num_steps_input": 1,
                    "masking_strategy_config": {"rate": 0.2, "hl_mask": 1, "rate_sampling": False},
                },
            },
            "target_input": {
                "full_teacher_target": {
                    "masking_strategy": "healpix",
                    "num_samples": 1,
                    "masking_strategy_config": {"rate": 1.0, "hl_mask": 0, "rate_sampling": False},
                }
            },
        }
    )


def _build_stream_cfg():
    return OmegaConf.create(
        {
            "name": "ERA5",
            "forcing": False,
            "diagnostic": False,
            "train_source_channels": ["2t"],
            "train_target_channels": ["2t"],
            "val_source_channels": ["2t"],
            "val_target_channels": ["2t"],
        }
    )


def test_jepa_loss_broadcasts_single_teacher_to_multiple_students():
    student_patches = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]],
        ],
        dtype=torch.float32,
    )
    student_masks = torch.tensor(
        [[[True, False, True]], [[False, True, True]]],
        dtype=torch.bool,
    )
    teacher_patches = torch.tensor(
        [[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]],
        dtype=torch.float32,
    )
    teacher_masks = torch.tensor([[[True, True, True]]], dtype=torch.bool)

    loss_broadcast = jepa_loss(student_patches, student_masks, teacher_patches, teacher_masks)
    loss_expanded = jepa_loss(
        student_patches,
        student_masks,
        teacher_patches.expand(student_patches.shape[0], -1, -1),
        teacher_masks.expand(student_patches.shape[0], -1, -1),
    )

    assert loss_broadcast.ndim == 0
    assert torch.isfinite(loss_broadcast)
    assert torch.allclose(loss_broadcast, loss_expanded)


def test_multiscale_correspondence_maps_two_students_to_one_teacher():
    stage_cfg = _build_multiscale_stage_cfg()
    stream_cfg = _build_stream_cfg()
    masker = Masker(healpix_level=5, stage=TRAIN)
    masker.reset_rng(np.random.default_rng(0))

    num_cells = 12 * (4**5)
    target_masks, source_masks, source_to_target = masker.build_samples_for_stream(
        training_mode="student_teacher",
        num_cells=num_cells,
        stage_cfg=stage_cfg,
        stream_cfg=stream_cfg,
    )

    assert len(target_masks) == 1
    assert len(source_masks) == 2
    assert source_to_target.tolist() == [0, 0]
    assert [meta.global_params["relationship"] for meta in source_masks.metadata] == [
        "subset",
        "subset",
    ]


def test_multiscale_config_smoke_loads_and_matches_expected_view_counts():
    cfg_path = Path(
        "config/week3/config_jepa_frozen_2drope_qkrms_student_multiscale_random_healpix_1.yml"
    )
    cfg = OmegaConf.load(cfg_path)
    cfg = common_config._load_streams_in_config(cfg)
    merged = common_config.merge_configs(OmegaConf.load("config/default_config.yml"), cfg)

    assert len(cfg.streams) > 0
    assert set(cfg.training_config.model_input.keys()) == {"random_easy", "healpix_easy"}
    assert sum(
        source_cfg.get("num_samples", 1) for _, source_cfg in cfg.training_config.model_input.items()
    ) == 2
    assert set(cfg.training_config.target_input.keys()) == {"full_teacher_target"}
    assert get_target_idxs_from_cfg(merged.training_config, "student-teacher") == [0]
    assert (
        merged.training_config.losses["student-teacher"].loss_fcts.JEPA.target_source_correspondence
        == {0: {0: "subset", 1: "subset"}}
    )
