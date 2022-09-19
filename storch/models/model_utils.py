
import inspect
import json
from collections import OrderedDict
from functools import wraps
from itertools import chain
from typing import Iterable

import torch
import torch.nn as nn


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

        torch.save(state_dict, filename)


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
