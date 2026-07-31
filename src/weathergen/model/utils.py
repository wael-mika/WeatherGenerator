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


def neighbour_gather_idxs(
    hp_nbours: torch.Tensor, batch_size: int, num_healpix_cells: int
) -> torch.Tensor:
    """Row indices gathering each cell's HEALPix 1-ring out of a (batch, cell)-flattened tensor.

    ``hp_nbours`` is ``[num_healpix_cells, 9]`` holding cell indices in
    ``[0, num_healpix_cells)`` -- it is shared by every sample. The tensor it indexes is
    flattened over ``(batch, cell)`` and therefore has ``batch_size * num_healpix_cells`` rows,
    so each sample's ring must be shifted into its own block. Omitting that offset makes every
    batch element read sample 0's latent, which is silent: the shapes still line up.

    Args:
        hp_nbours: ``[num_healpix_cells, 9]`` 1-ring, self first.
        batch_size: samples in the batch.
        num_healpix_cells: cells per sample.

    Returns:
        ``[batch_size * num_healpix_cells, 9]`` indices into the flattened tensor.
    """
    idxs = hp_nbours.unsqueeze(0).repeat((batch_size, 1, 1)).long()
    cell_offsets = (torch.arange(batch_size, device=idxs.device) * num_healpix_cells).view(-1, 1, 1)
    return (idxs + cell_offsets).flatten(0, 1)
