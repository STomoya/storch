"""Base container classes for data parallel wrapped modules."""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
from torch.optim.optimizer import Optimizer


class CheckpointInterfaceBase:
    """Interface for get/set state_dict from a module.

    This class is for enabling checkpointing models without any code changes between
    different parallelization methods (DDP or FSDP).

    inherited classes should work as follows:
    >>> module = {FSDP,DDP}(original_module, ...)
    >>> ckpt_if = CheckpointInterface(module, ...)
    >>> state_dict = ckpt_if.state_dict()
    >>> if primary_process:
    ...     torch.save(state_dict, './state_dict')
    >>> state_dict = torch.load('./state_dict')
    >>> ckpt_if.load_state_dict(state_dict)
    """

    def __init__(self, to_checkpoint: nn.Module | Optimizer) -> None:
        self.to_checkpoint = to_checkpoint

    def state_dict(self):
        return self.to_checkpoint.state_dict()

    def load_state_dict(self, state_dict: OrderedDict):
        self.to_checkpoint.load_state_dict(state_dict)


class ParallelFactoryBase:
    """Factory for wrapped modules."""

    def __init__(self) -> None:
        self._wrapped_module = None

    @property
    def is_wrapped(self) -> None:
        return self._wrapped_module is not None

    def wrap_module(self, module: nn.Module, **kwargs):
        raise NotImplementedError

    def create_checkpoint_interface(self, optim, **kwargs):
        raise NotImplementedError
