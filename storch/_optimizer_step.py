"""functions for one optimization step."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import _LRScheduler


def grad_nan_to_num_(input: nn.Module|list[torch.Tensor], nan: float=0.0, posinf: float=1e5, neginf: float=1e-5) -> None:
    """et nan gardients to a number.

    This is an inplace operation.

    Args:
        input (nn.Module | list[torch.Tensor]): Module or list of tensors with .grad attribute.
        nan (float, optional): Value to replace nan. Default: 0.0.
        posinf (float, optional): Value to replace positive inf. Default: 1e5.
        neginf (float, optional): Value to replace negative inf. Default: 1e-5.
    """
    if isinstance(input, nn.Module):
        input = input.parameters()
    params = [param for param in input if param.grad is not None]
    for param in params:
        param.grad = torch.nan_to_num(param.grad, nan, posinf, neginf)


def _assert_require_module(maybe_module: None|nn.Module):
    assert maybe_module is not None, f'"module" is required for "clip_grad_norm" or "grad_nan_to_num" option.'


def get_optimizer_step(
    trigger_gradient_scaling_via_gradscaler=False,
    gradient_accumulation_steps: int=1,
    num_iters_per_epoch: int=None,
    no_sync_context: Callable=None,
    module: nn.Module=None,
    post_backward_hooks: list[Callable]=[],
    post_optim_step_hooks: list[Callable]=[]
):

    if module is not None and no_sync_context is None:
        no_sync_context = getattr(module, 'no_sync', None)

    func = OptimizerStep(
        gradient_accumulation_steps,
        num_iters_per_epoch,
        no_sync_context
    )

    if len(post_backward_hooks) > 0:
        assert all(callable(hook) for hook in post_backward_hooks), f'all hooks must be callable.'
        func.register_hooks('backward', *post_backward_hooks)
    if len(post_optim_step_hooks) > 0:
        assert all(callable(hook) for hook in post_optim_step_hooks), f'all hooks must be callable.'
        func.register_hooks('step', *post_optim_step_hooks)

    return func


class OptimizerStep:
    """A function which calls backward on loss and step on optimizer, supporting gradient accumulation.

    Args:
        gradient_accumulation_steps (int): number of gradient accumulation steps. Default: 1.
        num_iters_per_epoch (int | None): number of iterations per epoch. used to adjust the
            accumulation steps of the last few batches. Pass None for infinite. Default: None.
        no_syn_context (Callable): contextmanager which suppresses device synchronization on backward. Default: None.

    Examples:
        >>> optimizer_step = OptimizerStep(grad_steps, total_iters)
    """
    def __init__(self,
        gradient_accumulation_steps: int=1, num_iters_per_epoch: int|None=None, no_sync_context: Callable=None
    ) -> None:
        # set attrs for gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_iters_per_epoch = num_iters_per_epoch if isinstance(num_iters_per_epoch, int) else gradient_accumulation_steps

        quotient, remainder = divmod(self.num_iters_per_epoch, self.gradient_accumulation_steps)
        self.last_gradient_accumulation_steps = remainder
        self.last_accumulation_from = quotient * self.gradient_accumulation_steps

        self.accumulation_count = 0
        self.total_step_count = 0

        if gradient_accumulation_steps != 1 and no_sync_context is not None:
            self.no_sync = no_sync_context
        else:
            self.no_sync = nullcontext

        self._post_backward_hooks = []
        self._post_optim_step_hooks = []


    def _call_hooks(self, hooks):
        for hook in hooks:
            hook()


    def register_hooks(self, when: str, *hooks):
        """Register hooks to be executed inside the step function.

        Args:
            when (str): either 'backward' or 'step'
            *hooks: callables to be executed.
        """
        assert when in ['backward', 'step']
        for hook in hooks:
            is_valid = callable(hook)
            if is_valid and when == 'backward':
                self._post_backward_hooks.append(hook)
            elif is_valid and when == 'step':
                self._post_optim_step_hooks.append(hook)


    # functions for gradient accumulation

    def _reset_count(self):
        """reset count attrs"""
        self.accumulation_count = 0
        self.total_step_count = self.total_step_count % self.num_iters_per_epoch


    def _step(self):
        """one gradient accumulation step"""
        self.accumulation_count += 1
        self.total_step_count += 1

        grad_accum_steps = self.gradient_accumulation_steps
        if self.total_step_count > self.last_accumulation_from:
            grad_accum_steps = self.last_gradient_accumulation_steps

        if self.accumulation_count == grad_accum_steps:
            self._reset_count()
            return True, 1.0 / grad_accum_steps
        return False, 1.0 / grad_accum_steps


    def __call__(self,
        loss: torch.Tensor, optimizer: optim.Optimizer, scaler: GradScaler=None, scheduler: _LRScheduler=None, module: nn.Module=None,
        *,
        zero_grad: bool=True, set_to_none: bool=True, update_scaler: bool=True,
        clip_grad_norm: bool=False, max_norm: float=1.0, grad_nan_to_num: bool=False,
    ) -> None:
        """optimization step which supports gradient scaling for AMP.

        Args:
            loss (torch.Tensor): loss to backpropagate.
            optimizer (optim.Optimizer): optimizer.
            scaler (GradScaler, optional): Optional GradScaler object.
                If specified, uses it to scale the loss and call .step(). Default: None.
            scheduler (_LRScheduler, optional): learning rate scheduler. should only be specified when calling .step()
                on every batch. Default: None.
            module (nn.Module, optional): The module to optimize. Used for gradient clipping. Default: None.
            zero_grad (bool, optional): Call .zero_grad() on optimizer before calling .backward() on loss. Default: True.
            set_to_none (bool, optional): Set None to .grad instead of setting them to 0. Default: True.
            clip_grad_norm (bool, optional): Clip gradient norm. Default: False.
            max_norm (float, optional): Maximum norm used when clipping gradients. Default: 1.0.
            grad_nan_to_num (bool, optional): Replace nan gradient: a number. Default: False.
            update_scaler (bool, optional): Update the scaler if not None. Default: False.
        """
        if clip_grad_norm or grad_nan_to_num:
            _assert_require_module(module)

        should_call_step, loss_scale = self._step()
        loss = loss * loss_scale

        no_sync = nullcontext if should_call_step else self.no_sync

        with no_sync():
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        self._call_hooks(self._post_backward_hooks)

        if not should_call_step:
            return

        if scaler is None:
            self._general_step(
                optimizer=optimizer, module=module,
                clip_grad_norm=clip_grad_norm, max_norm=max_norm, grad_nan_to_num=grad_nan_to_num
            )
        else:
            self._simple_step(
                optimizer=optimizer, scaler=scaler, module=module,
                clip_grad_norm=clip_grad_norm, max_norm=max_norm, grad_nan_to_num=grad_nan_to_num,
                update_scaler=update_scaler
            )

        self._call_hooks(self._post_optim_step_hooks)

        if scheduler is not None:
            scheduler.step()

        if zero_grad:
            optimizer.zero_grad(set_to_none=set_to_none)


    def _general_step(self,
        optimizer: optim.Optimizer, module: nn.Module=None,
        *,
        clip_grad_norm: bool=False, max_norm: float=1.0, grad_nan_to_num: bool=False
    ) -> None:
        """optimizer.step without using GradScaler"""

        if clip_grad_norm or grad_nan_to_num:
            if grad_nan_to_num:
                grad_nan_to_num_(module.parameters())
            if clip_grad_norm:
                if isinstance(module, FSDP):
                    module.clip_grad_norm_(max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)

        optimizer.step()


    def _simple_step(self,
        optimizer: optim.Optimizer, scaler: GradScaler, module: nn.Module=None,
        *,
        clip_grad_norm: bool=False, max_norm: float=1.0, grad_nan_to_num: bool=False,
        update_scaler: bool=True
    ) -> None:
        """optimizer.step with GradScaler"""

        if clip_grad_norm or grad_nan_to_num:
            scaler.unscale_(optimizer)
            if grad_nan_to_num:
                grad_nan_to_num_(module.parameters())
            if clip_grad_norm:
                if isinstance(module, FSDP):
                    module.clip_grad_norm_(max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)

        scaler.step(optimizer)

        if update_scaler:
            scaler.update()
