"""functional API for distributed utilities"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from storch.distributed._state import DistributedState

__all__ = [
    'is_available',
    'is_torchrun',
    'is_primary',
    'wait_for_everyone',
    'get_backend',
    'get_world_size',
    'get_rank',
    'get_local_rank',
    'get_device',
    'gather',
    'reduce',
    'only_on_primary'
]


def is_available() -> bool:
    """is distributed package available?"""
    return dist.is_available()


def is_torchrun() -> bool:
    """is python started via torchrun command?"""
    return dist.is_torchelastic_launched()


def is_primary() -> bool:
    """is this the primary process."""
    return DistributedState().is_main_process


def wait_for_everyone() -> None:
    """torch.distributed.barrier with a recognizable name.
    Does nothing when distributed isn't used.
    """
    state = DistributedState()
    if state.is_distributed:
        dist.barrier()


def get_backend() -> None:
    """get backend name."""
    return DistributedState().backend


def get_world_size() -> int:
    """get world size."""
    return DistributedState().num_processes


def get_rank() -> int:
    """get rank"""
    return DistributedState().process_index


def get_local_rank() -> int:
    """get local rank"""
    return DistributedState().local_process_index


def get_device() -> torch.device:
    """get device"""
    return DistributedState().device


def gather(obj: Any, dst: int=None, into_tensor: bool=True) -> torch.Tensor | tuple[torch.Tensor] | tuple[Any]:
    """gather objects between devices.

    Can be a torch.Tensor or a picklable python object.

    By default tensors are gathered into a single tensor. To gather into a list of tensors,
    set `into_tensor=False`. Python objects are not affected by this argument and are always
    gathered into a list.

    By default the objects are gathered to all devices. You can specify the device to gather
    to by passing a valid process index to the `dst` argument (e.g., 0). If `dst` argument
    is specified, `None` will be returned to all other processes.

    If is not a distributed environment, this function will just return the input `obj`.

    Args:
        obj (Any): object to gather. Can be a Tensor or picklable python object.
        dst (int, optional): destination device. If not given gathers to all devices. Default: None.
        into_tensor (bool, optional): If True and obj is a Tensor gather into a Tensor instead of a list. Default: True.

    Returns:
        torch.Tensor | tuple[torch.Tensor] | tuple[Any]: gathered object.
    """
    state = DistributedState()
    if not state.is_distributed:
        return obj
    else:
        if torch.is_tensor(obj):
            output = [torch.empty_like(obj) for _ in range(state.num_processes)]
            if dst is None and into_tensor:
                output = torch.cat(output)
                dist.all_gather_into_tensor(output, obj)
                return output
            elif dst is None:
                dist.all_gather(output, obj)
                return output
            else:
                output = output if state.process_index == dst else None
                dist.gather(obj, output, dst)
                return output
        else:
            output = [None for _ in range(state.num_processes)]
            if dst is None:
                dist.all_gather_object(output, obj)
                return output
            else:
                output = output if state.process_index == dst else None
                dist.gather_object(obj, output, dst)
                return output


def reduce(tensor: torch.Tensor, dst: int=None, op: ReduceOp=ReduceOp.SUM) -> torch.Tensor:
    """reduce a tensor according to the given `ReduceOp` enum.

    In contrast to `gather`, this function does not support python objects. If reducing
    a python number, convert object to a Tensor beforehand.

    By default the objects are reduced to all devices. You can specify the device
    by passing a valid process index to the `dst` argument (e.g., 0). If `dst` argument
    is specified, `None` will be returned to all other processes.

    If is not a distributed environment, this function will just return the input `obj`.

    Args:
        tensor (torch.Tensor): Tensor to reduce.
        dst (int, optional): destination device. If not given reduced to all device. Default: None.
        op (ReduceOp, optional): reduce option. Default: ReduceOp.SUM.

    Returns:
        torch.Tensor: reduced tensor.
    """
    state = DistributedState()

    if not state.is_distributed:
        return tensor

    elif dst is None:
        dist.all_reduce(tensor, op)
        return tensor

    else:
        dist.reduce(tensor, dst, op)
        return tensor if state.process_index == dst else None


def only_on_primary(func: Callable) -> Callable:
    """decorator for executing function only on primary process.

    Examples:
        >>> @only_on_primary
        ... def print0(*args, **kwargs):
        ...     print(*args, **kwargs)

    Args:
        func (Callable): the function to wrap.

    Returns:
        Callable: wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_primary():
            return func(*args, **kwargs)
    return wrapper
