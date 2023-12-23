"""container for FSDP modules."""

from __future__ import annotations

from contextlib import contextmanager
from typing import OrderedDict

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.optimizer import Optimizer

import storch
from storch.distributed.factory._base import CheckpointInterfaceBase, ParallelFactoryBase
from storch.utils.version import is_torch_version_geq


@contextmanager
def state_dict_type(
    module: FSDP, state_dict_type: StateDictType, state_dict_config, optim_state_dict_config=None
) -> None:
    """Set the state dict type using the api prefered by different pytorch versions.

    FSDP is in beta, and there might be further changes in the API.

    Args:
    ----
        module (FSDP): The wrapped FSDP module to set the state dict type for.
        state_dict_type (StateDictType): either one of StateDictType enum.
        state_dict_config (): The state dict config corresponding to the state_dict_type.
        optim_state_dict_config (): The optimizer's state dict config corresponding to the state_dict_type.
            This argument is required for PyTorch>=2.0.0. Deafult: None.
    """
    if is_torch_version_geq('2.0.0'):
        assert optim_state_dict_config is not None
        with FSDP.state_dict_type(module, state_dict_type, state_dict_config, optim_state_dict_config):
            yield
    else:
        with FSDP.state_dict_type(module, state_dict_type, state_dict_config):
            yield


class FSDPModuleCheckpointInterface(CheckpointInterfaceBase):
    """Interface for checkpointing models wrapped with FSDP.

    This class will be constructed automatically by `distributed.DistributedHelper.prepare_for_checkpointing`.

    FSDP models shards the parameters and requires special API to proprely get the state_dict of the
    original model. This interface provides methods to easily get/set the state_dict, automatically
    configuring state_dict_type.

    Currently this class only support `StateDictType.FULL_STATE_DICT` and currently there are no plans to
    add support for other types.

    Args:
    ----
        to_checkpoint (nn.Module): The module to wrap witch this interface.
        state (dict): a dict containing all objects needed for setting state dict type when get/set state_dicts.
    """

    def __init__(self, module: nn.Module, state: dict) -> None:  # noqa: D107
        super().__init__(module)
        self.state = state

    def state_dict(self) -> OrderedDict:
        """Properly sets the state_dict type and return the state_dict of the original model.

        Returns
        -------
            OrderedDict: the state dict of the original model
        """
        with state_dict_type(self.to_checkpoint, self.state.type, self.state.config, self.state.optim_config):
            state_dict = self.to_checkpoint.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Properly sets the state dict type and loads the state dict to the parallelized model.

        Args:
        ----
            state_dict (OrderedDict): the state dict of the non-parallelized model.
        """
        with state_dict_type(self.to_checkpoint, self.state.type, self.state.config, self.state.optim_config):
            self.to_checkpoint.load_state_dict(state_dict)


class FSDPOptimizerCheckpointInterface(CheckpointInterfaceBase):
    """Interface for optimizers used to train FSDP wrapped models.

    As same as the wrapped model, the optimizer also requires a special API to get state_dict properly.

    Args:
    ----
        optimizer (Optimizer): the optimizer.
        module (FSDP): the correspoing module.
        state (dict):  a dict containing all objects needed for setting state dict type when get/set state_dicts.
    """

    def __init__(self, optimizer: Optimizer, module: nn.Module, state: dict) -> None:  # noqa: D107
        super().__init__(optimizer)
        self.module = module
        self.state = state

    def state_dict(self) -> OrderedDict:
        """Properly sets the state_dict type and return the state_dict of the optimizer.

        Returns
        -------
            OrderedDict: the state dict of the original model
        """
        if is_torch_version_geq('2.0.0'):
            with state_dict_type(self.module, self.state.type, self.state.config, self.state.optim_config):
                state_dict = FSDP.optim_state_dict(self.module, self.to_checkpoint)
        else:
            state_dict = FSDP.full_optim_state_dict(self.module, self.to_checkpoint)
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Properly sets the state dict type and loads the state dict to the optimizer.

        Args:
        ----
            state_dict (OrderedDict): the state dict of the optimizer.
        """
        if is_torch_version_geq('2.0.0'):
            with state_dict_type(self.module, self.state.type, self.state.config, self.state.optim_config):
                flattened_osd = FSDP.optim_state_dict_to_load(state_dict, self.module, self.to_checkpoint)
                self.to_checkpoint.load_state_dict(flattened_osd)
        else:
            flattened_osd = FSDP.shard_full_optim_state_dict(state_dict, self.module)
            self.to_checkpoint.load_state_dict(flattened_osd)


class FullyShardedDataParallelFactory(ParallelFactoryBase):
    """FSDP factory.

    Factory class that provides methods to wrap the input model with the pytorch native FSDP class and
    creates interfaces for checkpointing. This class is only used inside `distributed.DistributedHelper`,
    and users should not be seeing this class from code.
    """

    def wrap_module(self, module: nn.Module, mixed_precision: bool | str, **kwargs) -> FSDP:
        """Wrap the input `module` using pytorch native FSDP class.

        Args:
        ----
            module (nn.Module): module to wrap.
            mixed_precision (bool | str): use mixed precision?
            **kwargs: other keyword arguments passed to FSDP class. used when `torch.compile` is used together
                with FSDP.

        Returns:
        -------
            FSDP: the wrapped module.
        """
        if mixed_precision:
            dtype = torch.float16
            mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        else:
            mixed_precision = None

        wrapped_module = FSDP(module, mixed_precision=mixed_precision, **kwargs)
        self._wrapped_module = wrapped_module

        return wrapped_module

    def create_checkpoint_interface(
        self, optim, offload_to_cpu: bool, **kwargs
    ) -> tuple[FSDPModuleCheckpointInterface, FSDPOptimizerCheckpointInterface]:
        """Create interfaces for checkpointing.

        Args:
        ----
            optim (Optimizer): The corresponding optimizer for training the wrapped module.
            offload_to_cpu (bool): offload state_dict to CPU when state_dict is called. Default: True.
            **kwargs: currently not used.

        Returns:
        -------
            tuple[FSDPModuleCheckpointInterface, FSDPOptimizerCheckpointInterface]: Interface for checkpointing.
        """
        assert self.is_wrapped, 'Call "wrap_module()" first.'

        state = storch.EasyDict()
        state.type = StateDictType.FULL_STATE_DICT
        state.config = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=offload_to_cpu)
        if is_torch_version_geq('2.0.0'):
            from torch.distributed.fsdp.api import FullOptimStateDictConfig

            state.optim_config = FullOptimStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=offload_to_cpu)
        else:
            state.optim_config = None

        mci = FSDPModuleCheckpointInterface(self._wrapped_module, state)
        oci = optim if optim is None else FSDPOptimizerCheckpointInterface(optim, self._wrapped_module, state)

        return mci, oci
