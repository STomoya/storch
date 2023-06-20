"""container for FSDP modules"""

from __future__ import annotations

from contextlib import contextmanager
from typing import OrderedDict

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType
from torch.optim.optimizer import Optimizer

import storch
from storch.distributed.factory._base import (CheckpointInterfaceBase,
                                              ParallelFactoryBase)
from storch.utils.version import is_torch_version_geq


@contextmanager
def state_dict_type(module, state_dict_type, state_dict_config, optim_state_dict_config=None):
    if is_torch_version_geq('2.0.0'):
        assert optim_state_dict_config is not None
        with FSDP.state_dict_type(
            module, state_dict_type, state_dict_config, optim_state_dict_config
        ):
            yield
    else:
        with FSDP.state_dict_type(
            module, state_dict_type, state_dict_config
        ):
            yield


class FSDPModuleCheckpointInterface(CheckpointInterfaceBase):
    def __init__(self, module: nn.Module, state: dict) -> None:
        super().__init__(module)
        self.state = state

    def state_dict(self):
        with state_dict_type(
            self.to_checkpoint, self.state.type, self.state.config, self.state.optim_config
        ):
            state_dict = self.to_checkpoint.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict):
        with state_dict_type(
            self.to_checkpoint, self.state.type, self.state.config, self.state.optim_config
        ):
            self.to_checkpoint.load_state_dict(state_dict)


class FSDPOptimizerCheckpointInterface(CheckpointInterfaceBase):
    def __init__(self, optimizer: Optimizer, module: nn.Module, state: dict) -> None:
        super().__init__(optimizer)
        self.module = module
        self.state = state

    def state_dict(self):
        if is_torch_version_geq('2.0.0'):
            with state_dict_type(
                self.module, self.state.type, self.state.config, self.state.optim_config
            ):
                state_dict = FSDP.optim_state_dict(self.module, self.to_checkpoint)
        else:
            state_dict = FSDP.full_optim_state_dict(self.module, self.to_checkpoint)
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict):
        if is_torch_version_geq('2.0.0'):
            with state_dict_type(
                self.module, self.state.type, self.state.config, self.state.optim_config
            ):
                flattened_osd = FSDP.optim_state_dict_to_load(state_dict, self.module, self.to_checkpoint)
                self.to_checkpoint.load_state_dict(flattened_osd)
        else:
            flattened_osd = FSDP.shard_full_optim_state_dict(state_dict, self.module)
            self.to_checkpoint.load_state_dict(flattened_osd)


class FullyShardedDataParallelFactory(ParallelFactoryBase):

    def wrap_module(self, module: nn.Module, mixed_precision: bool|str, **kwargs):
        if mixed_precision:
            dtype = torch.float16
            mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        else: mixed_precision = None

        wrapped_module = FSDP(module, mixed_precision=mixed_precision, **kwargs)
        self._wrapped_module = wrapped_module

        return wrapped_module

    def create_checkpoint_interface(self, optim, offload_to_cpu: bool, **kwargs):
        assert self.is_wrapped, f'Call "wrap_module()" first.'

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
