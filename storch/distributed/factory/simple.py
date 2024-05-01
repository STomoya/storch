"""Simple state_dict API.

Some functions are modifications of implementations in the pytorch repo.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP


class ModelStateDict(Stateful):
    """ModelStateDict interface."""

    def __init__(self, model: nn.Module, options: StateDictOptions) -> None:
        """ModelStateDict.

        This class implements `.state_dict` and `.load_state_dict` functions for parallel models.
        By using this wrapper, users will be able to save / load model parameters with minimal code changes.
        Functions in `torch.distributed.checkpoint` are used as backends.

        Args:
        ----
            model (nn.Module): The model to get/set the parameters.
            options (StateDictOptions): StateDictOptions object.

        """
        self._model = model
        self._options = options

    def load_state_dict(self, model_state_dict: dict[str, Any]) -> None:
        """Load state_dict to model.

        Args:
        ----
            model_state_dict (dict[str, Any]): The state_dict to load.

        """
        # This line is a workaround for loading weights on FSDP models before any forward pass.
        # The state of FSDP is lazily finalized using `_lazy_init` at the first forward call. (From `torch>=2.0.0`)
        # Calling `.state_dict` also initializes the state, additionally, does not require an input, and implemented in
        # DDP, FSDP, and nn.Module.
        # Without this, the below exception is triggered in `_verify_state_dict`.
        #   `RuntimeError: The model has FSDP modules but no FSDP root module exists.`
        self._model.state_dict()

        set_model_state_dict(self._model, model_state_dict, options=self._options)

    def state_dict(self) -> dict[str, Any]:
        """Get state_dict from model.

        Returns
        -------
            dict[str, Any]: The state_dict.

        """
        return get_model_state_dict(self._model, options=self._options)


class OptimStateDict(Stateful):
    """OptimStateDict interface."""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, options: StateDictOptions) -> None:
        """OptimStateDict interface.

        This class implements `.state_dict` and `.load_state_dict` functions for parallel optimizers.
        By using this wrapper, users will be able to save / load model parameters with minimal code changes.
        Functions in `torch.distributed.checkpoint` are used as backends.

        Args:
        ----
            model (nn.Module): The model optimized by the optimizers.
            optimizer (optim.Optimizer): The optimizer,
            options (StateDictOptions): StateDictOptions object.

        """
        self.model = model
        self.optimizer = optimizer
        self.options = options

    def load_state_dict(self, optim_state_dict: dict[str, Any]) -> None:
        """Load state_dict to optimizer.

        Args:
        ----
            optim_state_dict (dict[str, Any]): The state_dict to load.

        """
        set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict=optim_state_dict, options=self.options)

    def state_dict(self) -> dict[str, Any]:
        """Get state_dict from optimizer.

        Returns
        -------
            dict[str, Any]: The state_dict.

        """
        return get_optimizer_state_dict(self.model, self.optimizer, options=self.options)


def wrap_module(
    module: nn.Module,
    strategy: str | None,
    mixed_presision: str | bool | None,
    device_ids: list | None = None,
    compile: bool = False,
    **kwargs,
) -> DDP | FSDP | nn.Module:
    """Wrap a module according to the `strategy`.

    If `torch.compile` is used on FSDP models, automatically adds `use_orig_params=True`.

    Args:
    ----
        module (nn.Module): The module to wrap.
        strategy (str | None): wrap strategy.
        mixed_presision (str | bool | None): mixed_precision setting.
        device_ids (list | None, optional): `device_ids` option for DDP. Default: None.
        compile (bool, optional): Apply `torch.compile` to model? Default: False.
        **kwargs: Additional kwargs for parallel wrappers.

    Returns:
    -------
        DDP | FSDP | nn.Module: The wrapped module.

    """
    if strategy in [None, 'none']:
        return module

    elif strategy == 'ddp':
        ParallelCls = DDP
        kwargs['device_ids'] = device_ids
    elif strategy == 'fsdp':
        ParallelCls = FSDP
        if mixed_presision:
            dtype = torch.float16
            if isinstance(mixed_presision, str) and mixed_presision in ['bf16', 'bfloat16']:
                dtype = torch.bfloat16
            kwargs['mixed_precision'] = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        if compile:
            kwargs['use_orig_params'] = True

    wrapped_module = ParallelCls(module, **kwargs)

    return wrapped_module


def create_checkpoint_interface(
    model: nn.Module,
    optim: optim.Optimizer,
    full_state_dict: bool = True,
    cpu_offload: bool = True,
    strict: bool = True,
) -> tuple[ModelStateDict, OptimStateDict]:
    """Create interfaces for checkpointing.

    This function creates a stateful object that get/set state_dict from model/optimizer wrapped with parallel
    wrappers without changing any codes between different APIs. Non-parallel models are also acceptable.
    `optim` argument can be a `None` if the model is not meant for training.

    Args:
    ----
        model (nn.Module): The model.
        optim (optim.Optimizer): The optimizer used to optimize the model.
        full_state_dict (bool, optional): Return full state_dict for FSDP models. Default: True.
        cpu_offload (bool, optional): Offload weights to CPU. Default: True.
        strict (bool, optional): Strict loading. Default: True.

    Returns:
    -------
        tuple[ModelStateDict, OptimStateDict]: Created interfaces.

    Examples:
    --------
        >>> model = Model(...)
        >>> model = FSDP(model) # or DDP(model) or as-is.
        >>> optim = torch.optim.Adam(model.parameters())
        >>> model_sd, optim_sd = create_checkpoint_interface(model, optim)
        >>> model_state_dict = model_sd.state_dict() # returns full state_dict
        >>> model_sd.load_state_dict(model_state_dict) # load full state_dict on FSDP model.

        >>> model_sd, optim_sd = create_checkpoint_interface(model, None) # None as optimizer is allowed.
        >>> optim_sd is None
        True

    """
    model_unwrapped = model._orig_mod if hasattr(model, 'dynamo_ctx') else model

    options = StateDictOptions(full_state_dict=full_state_dict, cpu_offload=cpu_offload, strict=strict)

    model_stateful = ModelStateDict(model_unwrapped, options)
    optim_stateful = OptimStateDict(model_unwrapped, optim, options) if optim is not None else None

    return model_stateful, optim_stateful
