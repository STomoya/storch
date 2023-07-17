"""container for DDP modules"""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer

from storch.distributed.factory._base import (CheckpointInterfaceBase,
                                              ParallelFactoryBase)


class DDPModuleCheckpointInterface(CheckpointInterfaceBase):
    """Interface for checkpointing models wrapped with DDP. This class will be constructed automatically
    by `distributed.DistributedHelper.prepare_for_checkpointing`.

    DDP models holds the original model at `.module` attribute. Getting and setting the state_dict can
    be achieved by accessing this attr.
    If this is not done, all of the parameters' key will contain `module.` prefix, and we will have to
    erase this prefix when loading parameters to non-parallelized models. This interface provides methods
    to easily get/set the state_dict dealing with parameter names.

    Args:
        to_checkpoint (nn.Module): The module to wrap witch this interface.
    """

    def state_dict(self) -> OrderedDict:
        """This function returns the state_dict of the original module.

        Returns:
            OrderedDict: the state_dict
        """
        return self.to_checkpoint.module.state_dict()

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """This function properly sets the state_dict to the parallelized model.

        Args:
            state_dict (OrderedDict): The non-parallelized model's state_dict to load.
        """
        self.to_checkpoint.module.load_state_dict(state_dict)


class DistributedDataParallelFactory(ParallelFactoryBase):
    """Factory class that provides methods to wrap the input model with the pytorch native DDP class and
    creates interfaces for checkpointing. This class is only used inside `distributed.DistributedHelper`,
    and users should not be seeing this class from code.
    """
    def wrap_module(self, module: nn.Module, device_ids: list=None, **kwargs) -> DDP:
        """This funcion wraps the input `module` using pytorch native DDP class.

        Args:
            module (nn.Module): module to wrap.
            device_ids (list): device ids.

        Returns:
            DDP: the wrapped module.
        """
        wrapped_module = DDP(module, device_ids=device_ids)
        self._wrapped_module = wrapped_module
        return wrapped_module

    def create_checkpoint_interface(self, optim: Optimizer, **kwargs) -> tuple[DDPModuleCheckpointInterface, Optimizer]:
        """Create interfaces for checkpointing. DDP does not require code changes for optimizers so they,
        are returned as-is.

        Args:
            optim (Optimizer): The corresponding optimizer for training the wrapped module.

        Returns:
            tuple[DDPModuleCheckpointInterface, Optimizer]: Interface for checkpointing and the optimizer.
        """
        assert self.is_wrapped, f'Call "wrap_module()" first.'
        module_interface = DDPModuleCheckpointInterface(self._wrapped_module)
        return module_interface, optim
