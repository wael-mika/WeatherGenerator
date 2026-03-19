import math

import torch
from omegaconf import OmegaConf

from weathergen.train.lr_scheduler import LearningRateScheduler


def test_small_cosine_warmup_falls_back_safely():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-7)
    lr_cfg = OmegaConf.create(
        {
            "num_steps_warmup": 512,
            "num_steps_cooldown": 512,
            "lr_start": 5e-7,
            "lr_max": 5e-5,
            "lr_final_decay": 1e-6,
            "lr_final": 0.0,
            "policy_warmup": "cosine",
            "policy_decay": "constant",
            "policy_cooldown": "linear",
            "parallel_scaling_policy": "sqrt",
        }
    )

    scheduler = LearningRateScheduler(
        optimizer,
        batch_size=2,
        world_size=4,
        istep=0,
        lr_steps=10,
        lr_cfg=lr_cfg,
    )

    assert scheduler.n_steps_warmup == 1
    assert scheduler.effective_policy_warmup == "linear"

    learning_rates = [scheduler.get_lr()]
    for _ in range(12):
        learning_rates.append(scheduler.step())

    assert all(math.isfinite(lr) for lr in learning_rates)
