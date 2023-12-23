"""Model utils."""

from itertools import chain
from typing import Iterable

import torch
import torch.nn as nn


def get_parameters_and_buffers(module: nn.Module) -> Iterable:
    """Get parameters and buffers from a module.

    Args:
    ----
        module (nn.Module): The module to collect the parameters and buffers from.

    Returns:
    -------
        Iterable: An iterable containing all parameters and buffers.
    """
    parameters = module.parameters()
    buffers = module.buffers()
    return chain(parameters, buffers)


def get_named_parameters_and_buffers(module: nn.Module) -> Iterable:
    """Get named parameters and buffers from a module.

    Args:
    ----
        module (nn.Module): The module to collect the parameters and buffers from.

    Returns:
    -------
        Iterable: An iterable containing all named parameters and buffers.
    """
    named_parameters = module.named_parameters()
    named_buffers = module.named_buffers()
    return chain(named_parameters, named_buffers)


def get_device_from_module(module: nn.Module) -> torch.device:
    """Get the device of a module.

    Args:
    ----
        module (nn.Module): The module to get the device from.

    Raises:
    ------
        UserWarning: No parameters nor buffers.

    Returns:
    -------
        torch.device: The device.
    """
    module_vars = get_parameters_and_buffers(module)
    try:
        first_var = next(module_vars)
    except StopIteration as e:
        raise UserWarning(
            f"'{module.__class__.__qualname__}' does not have any parameters nor buffers to determine the device."
        ) from e
    return first_var.device


def get_dtype_from_module(module: nn.Module) -> torch.dtype:
    """Get the dtype of a module.

    Args:
    ----
        module (nn.Module): The module to get the dtype from.

    Raises:
    ------
        UserWarning: No parameters nor buffers.

    Returns:
    -------
        torch.dtype: The dtype.
    """
    module_vars = get_parameters_and_buffers(module)
    try:
        first_var = next(module_vars)
    except StopIteration as e:
        raise UserWarning(
            f"'{module.__class__.__qualname__}' does not have any parameters nor buffers to determine the dtype."
        ) from e
    return first_var.dtype
