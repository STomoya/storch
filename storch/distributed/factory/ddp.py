"""container for DDP modules"""

from collections import OrderedDict

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from storch.distributed.factory._base import (CheckpointInterfaceBase,
                                              ParallelFactoryBase)


class DDPModuleCheckpointInterface(CheckpointInterfaceBase):

    def state_dict(self):
        return self.to_checkpoint.module.state_dict()

    def load_state_dict(self, state_dict: OrderedDict):
        self.to_checkpoint.module.load_state_dict(state_dict)


class DistributedDataParallelFactory(ParallelFactoryBase):

    def wrap_module(self, module: nn.Module, device_ids: list=None, **kwargs):
        wrapped_module = DDP(module, device_ids=device_ids)
        self._wrapped_module = wrapped_module
        return wrapped_module

    def create_checkpoint_interface(self, optim, **kwargs):
        assert self.is_wrapped, f'Call "wrap_module()" first.'
        module_interface = DDPModuleCheckpointInterface(self._wrapped_module)
        return module_interface, optim
