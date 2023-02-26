
import inspect
import json
from collections import OrderedDict
from functools import wraps
from itertools import chain
from typing import Iterable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from storch._optimizer_step import get_optimizer_step


def get_parameters_and_buffers(module: nn.Module) -> Iterable:
    """get parameters and buffers from a module.

    Args:
        module (nn.Module): The module to collect the parameters and buffers from.

    Returns:
        Iterable: An iterable containing all parameters and buffers.
    """
    parameters = module.parameters()
    buffers = module.buffers()
    return chain(parameters, buffers)


def get_named_parameters_and_buffers(module: nn.Module) -> Iterable:
    """get named parameters and buffers from a module.

    Args:
        module (nn.Module): The module to collect the parameters and buffers from.

    Returns:
        Iterable: An iterable containing all named parameters and buffers.
    """
    named_parameters = module.named_parameters()
    named_buffers = module.named_buffers()
    return chain(named_parameters, named_buffers)


def get_device_from_module(module: nn.Module) -> torch.device:
    """get the device of a module.

    Args:
        module (nn.Module): The module to get the device from.

    Raises:
        UserWarning: No parameters nor buffers.

    Returns:
        torch.device: The device.
    """
    module_vars = get_parameters_and_buffers(module)
    try:
        first_var = next(module_vars)
    except StopIteration:
        raise UserWarning(f"'{module.__class__.__qualname__}' does not have any parameters nor buffers to determine the device.")
    return first_var.device


def get_dtype_from_module(module: nn.Module) -> torch.dtype:
    """get the dtype of a module.

    Args:
        module (nn.Module): The module to get the dtype from.

    Raises:
        UserWarning: No parameters nor buffers.

    Returns:
        torch.dtype: The dtype.
    """
    module_vars = get_parameters_and_buffers(module)
    try:
        first_var = next(module_vars)
    except StopIteration:
        raise UserWarning(f"'{module.__class__.__qualname__}' does not have any parameters nor buffers to determine the dtype.")
    return first_var.dtype


class ModelMixin(nn.Module):

    def __init_subclass__(cls) -> None:
        cls.__init__ = register_init_args(cls.__init__)

    @property
    def device(self):
        return get_device_from_module(self)

    @property
    def dtype(self):
        return get_dtype_from_module(self)


    def save_config(self, filename: str):
        with open(filename, 'w') as fout:
            json.dump(self._config_repr, fout, indent=2)


    @classmethod
    def from_config(cls, config_file: str, weight_file: str=None, map_location: torch.device=None):
        with open(config_file, 'r') as fin:
            kwargs = json.load(fin)
        model = cls(**kwargs)
        if weight_file is not None:
            state_dict = torch.load(weight_file, map_location=map_location)
            model.to(map_location)
            model.load_state_dict(state_dict)
        return model


    def save(self, filename: str, location: torch.device=None):
        state_dict = self.state_dict()

        new_state_dict = OrderedDict()
        if location is not None:
            for key, value in state_dict.items():
                new_state_dict[key] = value.to(location)
        else: new_state_dict = state_dict

        torch.save(new_state_dict, filename)


    def extra_repr(self):
        return ', '.join([f'{k}={v}' for k, v in self._config_repr.items()])


def register_init_args(init):

    @wraps(init)
    def inner(self, *args, **kwargs):
        init(self, *args, **kwargs)

        config = {}
        signiture = inspect.signature(init)
        parameters = {k: v.default for k, v in signiture.parameters.items() if k != 'self'}

        for arg, name in zip(args, parameters.keys()):
            config[name] = arg
        config.update({
            k: kwargs.get(k, default) for k, default in parameters.items() if k not in config
        })
        self._config_repr = config

    return inner


class Engine:
    """engine for training.

    Args:
        model (nn.Module): model to optimize.
        optimizer (Optimizer): optimizer.
        scaler (GradScaler, optional): gradient scaler. Default: None.
        scheduler (LRScheduler, optional): scheduler. calls step every STEP not EPOCH. Default: None.

        gradient_accumulation_steps (int, optional): gradient accumulation steps. Default: 1.
        num_iters_per_epoch (int, optional): number of iterations per epoch.
            required for gradient accumulation. Default: None.

        zero_grad (bool, optional): kwarg for optimizer step. Default: True.
        set_to_none (bool, optional): kwarg for optimizer step. Default: True.
        clip_grad_norm (bool, optional): kwarg for optimizer step. Default: False.
        max_norm (float, optional): kwarg for optimizer step. Default: 5.0.
        grad_nan_to_num (bool, optional): kwarg for optimizer step. Default: False.
        update_scaler (bool, optional): kwarg for optimizer step. Default: True.

    Examples:
        >>> model = get_model(...)
        >>> optimizer = get_optimizer(model.parameters(), ...)
        >>> engine = Engine(model, optimizer)
        >>> input = torch.randn(10, 10)
        >>> loss = criterion(engine(input), label)
        >>> engine.step(loss)
    """
    def __init__(self,
        model: nn.Module, optimizer: Optimizer,
        scaler: GradScaler=None, scheduler: _LRScheduler=None,
        *,
        # kwargs for optimizer step getter
        gradient_accumulation_steps: int=1,
        num_iters_per_epoch: int=None,
        # kwargs for optimizer step
        zero_grad: bool=True, set_to_none: bool=True,
        clip_grad_norm: bool=False, max_norm: float=5.0,
        grad_nan_to_num: bool=False, update_scaler: bool = True
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler

        self.optimizer_step = get_optimizer_step(
            trigger_gradient_scaling_via_gradscaler=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_iters_per_epoch=num_iters_per_epoch,
            no_sync_context = model.no_sync if hasattr(model, 'no_sync') else None
        )
        self.optimizer_step_kwargs = dict(
            zero_grad=zero_grad, set_to_none=set_to_none,
            clip_grad_norm=clip_grad_norm, max_norm=max_norm,
            grad_nan_to_num=grad_nan_to_num, update_scaler=update_scaler
        )


    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def step(self, loss, **overrides):
        kwargs = self.optimizer_step_kwargs.copy()
        if overrides != {}:
            assert all(key in self.optimizer_step_kwargs.keys() for key in overrides.keys())
            kwargs.update(overrides)
        self.optimizer_step(
            loss, self.optimizer, self.scaler, self.scheduler, self.model,
            **kwargs
        )
