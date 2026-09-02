# ruff: noqa: B006

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging

import torch
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import distribute_tensor

from weathergen.common.config import Config, get_path_model, merge_configs
from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelParams
from weathergen.model.utils import apply_fct_to_blocks, freeze_weights
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import register_nvtx_hooks
from weathergen.utils.utils import get_dtype

logger = logging.getLogger(__name__)


# same as in config: student_teacher, forecasting, masking
type TrainingMode = str


def init_model_and_shard(
    cf,
    dataset,
    run_id_contd,
    mini_epoch_contd,
    training_mode,
    device,
    with_ddp,
    with_fsdp,
    overrides={},
):
    model_creation_device = "meta" if with_ddp and with_fsdp else "cuda"
    with torch.device(model_creation_device):
        model = get_model(cf, training_mode, dataset, overrides)

    if cf.get("profiling", {}).get("nvtx_annotate", False):
        logger.info("Registering NVTX hooks for model.")
        register_nvtx_hooks(model)

    # freeze request model part
    apply_fct_to_blocks(model, cf.freeze_modules, freeze_weights)

    # TODO: this should be handled in the encoder to be close where q_cells is defined
    if "q_cells" in cf.freeze_modules:
        model.encoder.q_cells.requires_grad = False

    if with_ddp and not with_fsdp:
        # `ddp_static_graph` is off by default, so existing runs are unchanged.
        #
        # Turn it on when DDP raises "Parameter ... has been marked as ready twice". That happens
        # when one module is legitimately used more than once per iteration -- e.g. a stream's
        # decoder runs at every output step, and `_get_output_length` is `offset + num_steps`, so
        # a forecast config with offset 1 and num_steps 1 decodes twice. torch tolerated this
        # until ~2.9 and now errors; the flow-matching pretrain l4m31jbd trained fine on an older
        # torch with a byte-identical config.
        #
        # static_graph is the documented fix for both listed causes (reuse across checkpointed
        # sub-graphs, and unused parameters), and it subsumes find_unused_parameters as long as
        # the set of unused parameters is the SAME every iteration. That holds here: the flow
        # decoder owns its velocity head and leaves `pred_heads` permanently unused. It would NOT
        # hold for a model that varies which parameters participate from step to step.
        static_graph = bool(cf.get("ddp_static_graph", False))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=True,
            find_unused_parameters=(
                False if static_graph else cf.get("ddp_find_unused_parameters", True)
            ),
            gradient_as_bucket_view=True,
            bucket_cap_mb=512,
            static_graph=static_graph,
        )

    elif with_ddp and with_fsdp:
        # with DDP *and() FSDP
        fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=get_dtype(cf.mixed_precision_dtype),
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }
        modules_to_shard = (
            MLP,
            MultiSelfAttentionHeadLocal,
            MultiSelfAttentionHead,
            MultiCrossAttentionHeadVarlen,
            MultiCrossAttentionHeadVarlenSlicedQ,
            MultiSelfAttentionHeadVarlen,
        )

        for module in model.encoder.ae_local_engine.ae_local_blocks.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.encoder.ae_local_global_engine.ae_adapter.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.encoder.ae_global_engine.ae_global_blocks.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        for module in model.forecast_engine.fe_blocks.modules():
            if isinstance(module, modules_to_shard):
                # reshard_after_forward=False keeps FE parameters unsharded
                # during the multi-step rollout loop.
                # Needed for pushforward trick.
                fully_shard(module, reshard_after_forward=False, **fsdp_kwargs)

        for module in model.latent_heads.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        full_precision_fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }

        for module in model.target_token_engines.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **full_precision_fsdp_kwargs)

    if with_ddp and with_fsdp:
        fully_shard(model)
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            assert tensor.device == torch.device("meta")

        # For reasons we do not yet fully understand, when using train continue in some
        # instances, FSDP2 does not register the forward_channels and forward_columns
        # functions in the embedding engine as forward functions. Thus, yielding a crash
        # because the input tensors are not converted to DTensors. This seems to primarily
        # occur during validation.
        for embed in model.encoder.embed_engine.embeds.values():
            torch.distributed.fsdp.register_fsdp_forward_method(embed, "forward")

    # complete initalization and load model if inference/continuing a run
    if run_id_contd is not None:
        if is_root():
            logger.info(f"Continuing run with id={run_id_contd} at mini_epoch {mini_epoch_contd}.")
        model = load_model(cf, model, device, run_id_contd, mini_epoch_contd)
    elif cf.get("load_chkpt", {}).get("run_id", None):
        run_id = cf.load_chkpt.run_id
        mini_epoch = cf.load_chkpt.get("mini_epoch", -1)
        if is_root():
            logger.info(f"Loading checkpoint from id={run_id} at mini_epoch {mini_epoch}.")
        model = load_model(cf, model, device, run_id, mini_epoch)
    else:
        if with_ddp and with_fsdp:
            model.to_empty(device="cuda")
            if with_fsdp:
                model.reset_parameters()

    # model params
    model_params = ModelParams(cf).create(cf)
    model_params.reset_parameters(cf)
    model_params = model_params.to(f"cuda:{cf.local_rank}")

    return model, model_params


def load_model(cf, model, device, run_id: str, mini_epoch=-1):
    """Loads model state from checkpoint and checks for missing and unused keys.
    Args:
        run_id : model_id of the trained model
        mini_epoch : The mini_epoch to load. Default (-1) is the latest mini_epoch
    """

    path_run = get_path_model(run_id=run_id)
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    filename = f"{run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(
        path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
    )

    is_model_sharded = cf.with_ddp and cf.with_fsdp
    if is_model_sharded:
        meta_sharded_sd = model.state_dict()
        maybe_sharded_sd = {}
        for param_name, full_tensor in params.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if sharded_meta_param is None:
                logger.warning(f"Parameter {param_name} from checkpoint not found in model.")
                continue
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            # maybe_sharded_sd[param_name.replace("module.", "")] = nn.Parameter(sharded_tensor)
            maybe_sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
        # choose `assign=True` for sharded model since we cannot call `copy_` on meta tensor
        mkeys, ukeys = model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)

        # new network parts (e.g. for fine-tuning)
        if mkeys:
            # Get the unique parent modules for the missing parameters
            new_modules_to_init = {key.rsplit(".", 1)[0] for key in mkeys}

            # Find the highest-level "root" new modules to avoid redundant initializations
            root_new_modules = set()
            for path in sorted(list(new_modules_to_init)):
                if not any(path.startswith(root + ".") for root in root_new_modules):
                    root_new_modules.add(path)

            # Get all modules for quick lookup and initialize the new ones
            all_modules = dict(model.named_modules())
            for path in root_new_modules:
                if is_root():
                    logger.info(f"Initializing new module not found in checkpoint: {path}")
                module_to_init = all_modules[path]
                module_to_init.to_empty(device="cuda")
                module_to_init.reset_parameters()

    else:
        # fix mismatch between state_dict keys that can occur between interactive/non-interactive
        model_has_prefix_module = list(model.state_dict().keys())[0].split(".")[0] == "module"
        params_has_prefix_module = list(params.keys())[0].split(".")[0] == "module"
        if model_has_prefix_module and not params_has_prefix_module:
            # add "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp["module." + k] = params[k]
            params = params_temp
        elif not model_has_prefix_module and params_has_prefix_module:
            # remove "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp[k.replace("module.", "")] = params[k]
            params = params_temp
        # load checkpoint
        mkeys, ukeys = model.load_state_dict(params, strict=False)
        model = model.to(device)

    # warn about difference in checkpoint and model
    if len(mkeys) == 0 and len(ukeys) == 0:
        logger.info(f"Checkpoint {filename} loaded successfully with all weights matching.")
    if len(mkeys) > 0:
        logger.warning(f"Missing keys when loading model: {mkeys}")
    if len(ukeys) > 0:
        logger.warning(f"Unused keys when loading model: {ukeys}")
        # Unused keys are weights the PARENT trained and this code has nowhere to put. Unlike
        # missing keys -- which are reinitialised on purpose, and are how a new decoder branch
        # gets its weights -- this silently produces a DIFFERENT, worse model than the checkpoint
        # claims to continue, and training proceeds without complaint.
        #
        # This is not hypothetical: ypj5be4l carries a per-stream `geo_bias` MLP that does not
        # exist on this branch, so eight tensors were dropped and every continuation ran a base
        # 48% worse on MSE than the parent's published numbers -- for 28 mini-epochs, with the
        # evidence sitting in the log unread.
        if not bool(cf.get("allow_unused_checkpoint_keys", False)):
            raise RuntimeError(
                f"{len(ukeys)} checkpoint tensors have no home in this model: {ukeys[:8]}"
                f"{' ...' if len(ukeys) > 8 else ''}. The parent trained weights this code cannot "
                "use, so continuing would silently run a different model. Either restore the "
                "missing module, pick a compatible parent, or set "
                "`allow_unused_checkpoint_keys: True` to accept the degraded base deliberately."
            )

    return model


def get_model(cf: Config, training_mode: TrainingMode, dataset, overrides):
    """
    Create model

    cf :
    training_mode :
    dataset :
    """

    # TODO: how to avoid the dependence on dataset
    sources_size = dataset.get_sources_size()
    targets_num_channels = dataset.get_targets_num_channels()
    targets_coords_size = dataset.get_targets_coords_size()

    cf_with_overrides = merge_configs(cf, overrides)
    return Model(
        cf_with_overrides, sources_size, targets_num_channels, targets_coords_size
    ).create()
