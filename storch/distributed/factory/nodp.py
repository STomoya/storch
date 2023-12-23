"""container without wrapping. for single GPU."""

import torch.nn as nn

from storch.distributed.factory._base import ParallelFactoryBase


class NoParallelFactory(ParallelFactoryBase):
    """Dummy class for no distributed training."""

    def wrap_module(self, module: nn.Module, **kwargs):  # noqa: D102
        self._wrapped_module = module
        return module

    def create_checkpoint_interface(self, optim, **kwargs):  # noqa: D102
        assert self.is_wrapped, 'Call "wrap_module()" first.'
        return self._wrapped_module, optim
