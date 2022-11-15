"""learning rate schedulers."""

from __future__ import annotations

import math
from typing import Callable, List

import numpy as np
import torch.optim as optim
from stutil.exceptions import deprecated
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

__all__ = [
    'build_scheduler',
    'ConstantMultiplier',
    'LinearMultiplier',
    'CosineMultiplier',
    'ExponentialMultiplier',
    'MultiStepMultiplier',
    'Compose',
    'MultiplyLR',
    'with_warmup'
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


'''
Learning rate scheduler.
Code is taken from fvcore: https://github.com/facebookresearch/fvcore
and modified by STomoya: https://github.com/STomoya
'''


class Multiplier:
    WHERE_EPSILON = 1e-6

    def __call__(self, where: float) -> float:
        raise NotImplementedError()


@deprecated(favor_of='build_scheduler', recommendation='build_scheduler')
class ConstantMultiplier(Multiplier):
    def __init__(self,
        value: float
    ) -> None:
        self.value = value
    def __call__(self, where: float) -> float:
        return self.value


@deprecated(favor_of='build_scheduler', recommendation='build_scheduler')
class LinearMultiplier(Multiplier):
    def __init__(self,
        start: float, end: float
    ) -> None:
        self.start = start
        self.end = end

    def __call__(self, where: float) -> float:
        return self.end * where + self.start * (1 - where)


@deprecated(favor_of='build_scheduler', recommendation='build_scheduler')
class CosineMultiplier(Multiplier):
    def __init__(self,
        start: float, min_value: float, T_max: float=1.
    ) -> None:
        self.start = start
        self.min_value = min_value
        self.T_max = T_max

    def __call__(self, where: float) -> float:
        return self.min_value \
            + 0.5 * (self.start - self.min_value) \
            * (1 + math.cos(math.pi * where / self.T_max))


@deprecated(favor_of='build_scheduler', recommendation='build_scheduler')
class ExponentialMultiplier(Multiplier):
    def __init__(self,
        start: float, decay: float
    ) -> None:
        self.start = start
        self.decay = decay
    def __call__(self, where: float) -> float:
        return self.start * (self.decay ** where)


@deprecated(favor_of='build_scheduler', recommendation='build_scheduler')
class MultiStepMultiplier(Multiplier):
    def __init__(self,
        milestones: List[int], max_iters: int, gamma: float, initial_scale: float=1.
    ) -> None:
        self.milestones = milestones
        self.max_iters = max_iters
        self.gamma = gamma
        self.curret = initial_scale

    def __call__(self, where: float) -> float:
        epoch_num = int((where + self.WHERE_EPSILON) * self.max_iters)
        if epoch_num in self.milestones:
            self.curret *= self.gamma
        return self.curret


@deprecated(favor_of='build_scheduler', recommendation='build_scheduler')
class Compose(Multiplier):
    def __init__(self,
        multipliers: List[Multiplier], lengths: List[int], scaling: List[str]
    ) -> None:
        assert len(multipliers) == len(lengths)
        assert 0 <= (sum(lengths) - 1.) < 1e-3
        assert all([s in ['scaled', 'fixed'] for s in scaling])

        self.multipliers = multipliers
        self.lengths = lengths
        self.scaling = scaling

    def __call__(self, where: float) -> float:
        running_total = 0.
        for i, length in enumerate(self.lengths):
            running_total += length
            if where + self.WHERE_EPSILON <= running_total:
                break

        multiplier = self.multipliers[i]

        if self.scaling[i] == 'scaled':
            start = running_total - self.lengths[i]
            where = (where - start) / self.lengths[i]

        return multiplier(where)


@deprecated(favor_of='build_scheduler', recommendation='build_scheduler')
def with_warmup(
    multiplier: Multiplier, warmup_factor: float, warmup_length: float, warmup_method: str='linear'
) -> Compose:
    assert warmup_method in ['linear', 'constant']
    end = multiplier(warmup_length)
    start = warmup_factor * multiplier(0.)
    if warmup_method == 'linear':
        warmup = LinearMultiplier(start, end)
    elif warmup_method == 'constant':
        warmup = ConstantMultiplier(start)

    return Compose(
        [warmup, multiplier],
        [warmup_length, (1 - warmup_length)],
        ['scaled', 'fixed']
    )


@deprecated(favor_of='build_scheduler', recommendation='build_scheduler')
class MultiplyLR(_LRScheduler):

    def __init__(self,
        optimizer: optim.Optimizer,
        multiplier: Multiplier,
        max_iter: int,
        last_iter: int=-1
    ) -> None:
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self) -> list:
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> float:
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]
