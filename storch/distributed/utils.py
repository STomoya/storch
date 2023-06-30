"""functional API for distributed utilities"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from storch.distributed._state import DistributedState


def is_available() -> bool:
    return dist.is_available()


def is_torchrun() -> bool:
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


def get_device() -> torch.device:
    """get device"""
    return DistributedState().device


def gather(obj: Any, dst: int=None, into_tensor: bool=True) -> torch.Tensor | tuple[torch.Tensor] | tuple[Any]:
    """gather.

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
    """reduce

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
