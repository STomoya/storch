"""learning rate schedulers."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    'build_scheduler'
]


def _warmup(current_step: int, num_warmup_steps: int):
    """calc factor on warmup"""
    return current_step / max(1.0, num_warmup_steps)


"""Lambdas for LambdaLR"""

def get_constant_schedule(num_warmup_steps: int=None) -> Callable:
    """get function for constant schedule.

    Args:
        num_warmup_steps (int, optional): number of warmup steps.

    Returns:
        Callable: always returns 1.0
    """
    if num_warmup_steps is None:
        def lr_lambda(current_step: int):
            return 1.0
    else:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            return 1.0
    return lr_lambda


def get_multistep_schedule(milestones: list, num_warmup_steps: int=None, gamma=0.1) -> Callable:
    """function for multistep schedules

    Args:
        milestones (list): list of steps on where to decay.
        num_warmup_steps (int, optional): number of warmup steps.
        gamma (float, optional): factor to decay on each milestone. Defaults to 0.1.

    Returns:
        Callable: function for LambdaLR
    """
    milestones = np.asarray(milestones)
    if num_warmup_steps is None:
        def lr_lambda(current_step: int):
            bigger_count = sum(milestones < current_step)
            return gamma ** bigger_count
    else:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            bigger_count = sum(milestones < current_step)
            return gamma ** bigger_count

    return lr_lambda


def get_linear_schedule(num_training_steps: int, num_warmup_steps: int) -> Callable:
    """function for linear schedule.

    Args:
        num_training_steps (int): total number of training steps.
        num_warmup_steps (int, optional): number of warmup steps.

    Returns:
        Callable: function for LambdaLR
    """
    if num_warmup_steps is None:
        def lr_lambda(current_step: int):
            return max(0.0,
                (num_training_steps - current_step) / max(1.0, num_training_steps)
            )
    else:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            return max(0.0,
                (num_training_steps - current_step) / max(1.0, num_training_steps - num_warmup_steps)
            )
    return lr_lambda


def get_cosine_schedule(num_training_steps: int, num_warmup_steps: int=None, num_cycles: float=0.5) -> Callable:
    """function for consine schedule.

    Args:
        num_training_steps (int): total number of training steps.
        num_warmup_steps (int, optional): number of warmup steps.
        num_cycles (float, optional): The number of waves in the cosine schedule. Default: 0.5.

    Returns:
        Callable: function for LambdaLR
    """
    if num_warmup_steps is None:
        def lr_lambda(current_step: int):
            progress = current_step / max(1, num_training_steps)
            return max(0.0, 0.5 *  (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    else:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return _warmup(current_step, num_warmup_steps)
            progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 *  (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return lr_lambda


def build_scheduler(
    optimizer: optim.Optimizer, type: str, num_training_steps: int,
    *,
    num_iter_per_step: int=1, num_warmup_steps: int=None,
    milestones: list[int]=None, gamma: float=0.1, last_epoch: int=-1
) -> LambdaLR:
    """build scheduler

    Args:
        optimizer (Optimizer): the optimizer.
        type (str): name of the scheduler.
        num_training_steps (int): total number of training steps. assumes epochs.
            to use iterations use num_iters_per_step or pass in iterations as steps.
        num_iter_per_step (int, optional): number of iterations per step. Default: 1.
        num_warmup_steps (int, optional): number of warmup steps. If None, no warmup phase. Default: None.
        milestones (list[int], optional): milestones for multistep scheduler. Default: None.
        gamma (float, optional): gamma for multistep scheduler. Default: 0.1.
        last_epoch (int, optional): last epoch for resume training. Default: -1.

    Returns:
        LambdaLR: learning rate scheduler.
    """
    num_training_steps = num_training_steps * num_iter_per_step
    num_warmup_steps = num_warmup_steps * num_iter_per_step if num_warmup_steps is not None else None

    if type == 'constant':
        lr_lambda = get_constant_schedule(num_warmup_steps)
    elif type == 'linear':
        lr_lambda = get_linear_schedule(num_training_steps, num_warmup_steps)
    elif type == 'multistep':
        assert milestones is not None
        milestones = [milestone * num_iter_per_step for milestone in milestones]
        lr_lambda = get_multistep_schedule(milestones, num_warmup_steps, gamma)
    elif type == 'cosine':
        lr_lambda = get_cosine_schedule(num_training_steps, num_warmup_steps)
    else:
        raise Exception(f'build_scheduler: No such scheduler type "{type}".')

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
