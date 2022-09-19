
from __future__ import annotations

from functools import wraps
from typing import Callable

from torch.profiler import (ProfilerActivity, profile, record_function,
                            schedule, tensorboard_trace_handler)


def get_tb_profile(
    tb_folder: str,
    activities: list[ProfilerActivity]=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule: schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
    record_shapes: bool=True, profile_memory: bool=False, with_stack: bool=True
) -> profile:
    """A profile object with tensorboard_trace_handler.

    Args:
        tb_folder (str): Folder for tensorboard trace output.
        activities (list[ProfilerActivity], optional): Activities to profile.
            Default: [ProfilerActivity.CPU, ProfilerActivity.CUDA].
        schedule (schedule, optional): Profiling schedule. Use when tracing long process.
            Default: schedule(wait=1, warmup=1, active=3, repeat=2).
        record_shapes (bool, optional): Record shapes. Default: True.
        profile_memory (bool, optional): Profile memory. Disabled for default to avoid 'ERROR: 139'
            (segmentation fault caused by untraced allocation of a memory block.) Default: False.
        with_stack (bool, optional): Profile TorchScript stack trace. Default: True.

    Returns:
        profile: profiler.
    """
    profile_obj = profile(
        activities=activities, schedule=schedule,
        on_trace_ready=tensorboard_trace_handler(f'{tb_folder}'),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack
    )
    return profile_obj


def profiled_function(func: Callable) -> Callable:
    """Decorator for profiling a function.

    Args:
        func (Callable): The function to profile.

    Returns:
        Callable: The wrapped function.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        with record_function(func.__qualname__):
            return func(*args, **kwargs)
    return inner
