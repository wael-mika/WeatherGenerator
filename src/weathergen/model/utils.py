# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_num_parameters(block):
    nps = filter(lambda p: p.requires_grad, block.parameters())
    return sum([torch.prod(torch.tensor(p.size())) for p in nps])


def freeze_weights(block):
    if hasattr(block, "name"):
        logger.info(f"Freeze block {block.name}")
    for p in block.parameters():
        p.requires_grad = False


def set_to_eval(block):
    if hasattr(block, "name"):
        logger.info(f"Set block {block.name} to eval mode")
    block.eval()


def apply_fct_to_blocks(model, blocks, fct):
    """
    Apply a function to specific blocks of a model.
    Args:
        model : model instance with attribute named_modules
        blocks : regex pattern to match block names
        fct : function to apply to matching blocks
    """

    for name, module in model.named_modules():
        name = module.name if hasattr(module, "name") else name
        # avoid the whole model element which has name ''
        if (re.fullmatch(blocks, name) is not None) and (name != ""):
            fct(module)


def _resolve_variable_groups(variable_groups_cfg, channel_names):
    """Map variable-group config to (group_name, sorted_channel_indices, group_cfg) tuples.

    _default group collects all unmatched channels.
    Raises ValueError on overlapping patterns or missing _default when channels are unmatched.
    """
    assigned = set()
    groups = []
    default_cfg = None

    for group_name, group_cfg in variable_groups_cfg.items():
        if group_name == "_default":
            default_cfg = group_cfg
            continue
        patterns = [re.compile(p) for p in group_cfg.get("variables", [])]
        indices = [
            i for i, ch in enumerate(channel_names) if any(pat.fullmatch(ch) for pat in patterns)
        ]
        overlap = set(indices) & assigned
        if overlap:
            raise ValueError(f"Channels at indices {overlap} matched by multiple variable groups")
        assigned.update(indices)
        groups.append((group_name, sorted(indices), group_cfg))

    default_indices = [i for i in range(len(channel_names)) if i not in assigned]
    if default_indices and default_cfg is None:
        raise ValueError(
            f"{len(default_indices)} channel(s) unmatched by any variable group "
            "but no _default group is defined"
        )
    if default_cfg is not None:
        groups.append(("_default", default_indices, default_cfg))

    return groups


class ActivationFactory:
    _registry = {
        "identity": nn.Identity,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "prelu": nn.PReLU,
        "softplus": nn.Softplus,
        "linear": nn.Linear,
        "logsoftmax": nn.LogSoftmax,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }

    @classmethod
    def get(cls, name: str, **kwargs):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unsupported activation type: '{name}'")
        fn = cls._registry[name]
        return fn(**kwargs) if callable(fn) else fn
