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

    '''from: https://github.com/NVlabs/stylegan3/blob/1c6608208cb51b7773da32f40ee2232f684c3a21/training/training_loop.py#L283-L292'''
    # if len(params):
    #     flat = torch.cat([param.grad.flatten() for param in params])
    #     flat = torch.nan_to_num(flat, nan, posinf, neginf)
    #     grads = flat.split([param.numel() for param in params])
    #     for param, grad in zip(params, grads):
    #         param.grad = grad.reshape(param.shape)


def optimizer_step(
    loss: torch.Tensor, optimizer: optim.Optimizer, scaler: GradScaler=None, scheduler: _LRScheduler=None, module: nn.Module=None,
    *,
    zero_grad: bool=True, set_to_none: bool=True,
    clip_grad_norm: bool=False, max_norm: float=1.0, grad_nan_to_num: bool=False,
    update_scaler: bool=False
) -> None:
    """optimization step which supports gradient scaling for AMP.

    Args:
        loss (torch.Tensor): loss to backpropagate.
        optimizer (optim.Optimizer): optimizer.
        scaler (GradScaler, optional): Optional GradScaler object.
            If specified, uses it to scale the loss and call .step(). Default: None.
        scheduler (_LRScheduler, optional): learning rate scheduler. should only be specified when calling .step()
            on every batch. Default: None.
        zero_grad (bool, optional): Call .zero_grad() on optimizer before calling .backward() on loss. Default: True.
        set_to_none (bool, optional): Set None to .grad instead of setting them to 0. Default: True.
        clip_grad_norm (bool, optional): Clip gradient norm. Default: False.
        max_norm (float, optional): Maximum norm used when clipping gradients. Default: 1.0.
        grad_nan_to_num (bool, optional): Replace nan gradient: a number. Default: False.
        update_scaler (bool, optional): Update the scaler if not None. Default: False.
    """
    assert scaler is None or isinstance(scaler, GradScaler)
    if clip_grad_norm or grad_nan_to_num:
        assert module is not None, f'"clip_grad_norm" or "grad_nan_to_num" option requires module argument.'

    if zero_grad:
        optimizer.zero_grad(set_to_none=set_to_none)

    if scaler is not None:
        scaler.scale(loss).backward()

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
    else:
        loss.backward()

        if clip_grad_norm or grad_nan_to_num:
            if grad_nan_to_num:
                grad_nan_to_num_(module.parameters())
            if clip_grad_norm:
                if isinstance(module, FSDP):
                    module.clip_grad_norm_(max_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)

        optimizer.step()

    if scheduler is not None:
        scheduler.step()


class optimizer_step_with_gradient_accumulation:
    """"torchops.optimizer_step" supporting gradient accumulation.

    Args:
        gradient_accumulation_steps (int): number of gradient accumulation steps.
        num_iters_per_epoch (int): number of iterations per epoch. used to adjust the
            accumulation steps of the last few batches.

    Examples:
        >>> optimizer_step = optimizer_step_with_gradient_accumulation(grad_steps, total_iters)
        >>> # then "optimizer_step" can be used as same as "torchops.optimizer_step"
    """

    def __init__(self, gradient_accumulation_steps: int, num_iters_per_epoch: int, no_sync_context: Callable=None) -> None:
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_iters_per_epoch = num_iters_per_epoch

        # calc last accumulation steps and where it starts.
        # if gradient accumulation steps and total iterations are 4 and 13 respectively,
        # last accumulation steps and where it starts would be 1 and 12 respectively.
        quotient, remainder = divmod(num_iters_per_epoch, gradient_accumulation_steps)
        self.last_gradient_accumulation_steps = remainder
        self.last_accumulation_from = quotient * gradient_accumulation_steps

        # counters
        self.accumulation_count = 0
        self.total_step_count = 0

        self.no_sync = no_sync_context if no_sync_context is not None else nullcontext

    def _reset(self):
        """reset counters"""
        self.accumulation_count = 0
        self.total_step_count = self.total_step_count % self.num_iters_per_epoch


    def _step(self) -> tuple[bool, float]:
        """one step

        Returns:
            bool: whether to call .step() on optimizer or not.
            float: value to scale the loss. generally equal to gradient accumulation steps.
        """
        self.accumulation_count += 1
        self.total_step_count += 1

        gradient_accumulation_steps = self.gradient_accumulation_steps
        if self.total_step_count > self.last_accumulation_from:
            gradient_accumulation_steps = self.last_gradient_accumulation_steps

        if self.accumulation_count == gradient_accumulation_steps:
            self._reset()
            return True, 1.0 / gradient_accumulation_steps
        return False, 1.0 / gradient_accumulation_steps


    def __call__(self,
        loss: torch.Tensor, optimizer: optim.Optimizer, scaler: GradScaler=None, scheduler: _LRScheduler=None, module: nn.Module=None,
        *,
        zero_grad: bool=True, set_to_none: bool=True,
        clip_grad_norm: bool=False, max_norm: float=1.0, grad_nan_to_num: bool=False,
        update_scaler: bool=False
    ) -> None:
        """optimization step which supports gradient scaling for AMP.

        Args:
            loss (torch.Tensor): loss to backpropagate.
            optimizer (optim.Optimizer): optimizer.
            scaler (_type_, optional): Optional GradScaler object.
                If specified, uses it to scale the loss and call .step(). Default: None.
            scheduler (_LRScheduler, optional): learning rate scheduler. should only be specified when calling .step()
                on every batch. Default: None.
            zero_grad (bool, optional): Call .zero_grad() on optimizer before calling .backward() on loss. Default: True.
            set_to_none (bool, optional): Set None to .grad instead of setting them to 0. Default: True.
            clip_grad_norm (bool, optional): Clip gradient norm. Default: False.
            max_norm (float, optional): Maximum norm used when clipping gradients. Default: 1.0.
            grad_nan_to_num (bool, optional): Replace nan gradient: a number. Default: False.
            update_scaler (bool, optional): Update the scaler if not None. Default: False.
        """
        assert scaler is None or isinstance(scaler, GradScaler)
        if clip_grad_norm or grad_nan_to_num:
            assert module is not None, f'"clip_grad_norm" or "grad_nan_to_num" option requires module argument.'

        should_call_step, loss_scale = self._step()
        loss = loss * loss_scale

        no_sync = nullcontext if should_call_step else self.no_sync

        with no_sync():
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if not should_call_step:
            return

        if scaler is not None:
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

        else:
            if clip_grad_norm or grad_nan_to_num:
                if grad_nan_to_num:
                    grad_nan_to_num_(module.parameters())
                if clip_grad_norm:
                    if isinstance(module, FSDP):
                        module.clip_grad_norm_(max_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if zero_grad:
            optimizer.zero_grad(set_to_none=set_to_none)
