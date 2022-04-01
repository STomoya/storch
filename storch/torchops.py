
from __future__ import annotations

from collections.abc import Callable
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import numpy as np


@torch.no_grad()
def freeze(model: nn.Module):
    '''freeze the model'''
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


@torch.no_grad()
def unfreeze(model: nn.Module):
    '''unfreeze the model'''
    for param in model.parameters():
        param.requires_grad = True
    model.train()


@torch.no_grad()
def update_ema(
    model: torch.nn.Module, model_ema: torch.nn.Module,
    decay: float=0.999, copy_buffers: bool=False, force_cpu: bool=False
) -> None:
    '''Update exponential moving avg
    w'_new = w' * decay + w * (1 - decay)

    Arguments:
        model: torch.nn.Module
            Model, which actually is updated
        model_ema: torch.nn.Module
            Copy of the model, which is updated by exponential moving average.
        decay: float (default: 0.999)
            Decay for exponential modving average.
        copy_buffers: bool (default: False)
            If True, also copy buffers inside the model.
        force_cpu: bool (default: True)
            If True, process on cpu.
    '''
    if force_cpu:
        org_device = next(model.parameters()).device
        model.to('cpu')
        model_ema.to('cpu')

    model.eval()
    param_ema = dict(model_ema.named_parameters())
    param     = dict(model.named_parameters())
    for key in param_ema.keys():
        param_ema[key].data.mul_(decay).add_(param[key].data, alpha=(1 - decay))
    if copy_buffers:
        buffer_ema = dict(model_ema.named_buffers())
        buffer     = dict(model.named_buffers())
        for key in buffer_ema.keys():
            buffer_ema[key].data.copy_(buffer[key].data)
    model.train()

    if force_cpu:
        model.to(org_device)


def deterministic(seed: int=3407) -> tuple[Callable, torch.Generator]:
    '''reproducible training

    Arguments:
        seed: int (default: 3407)
            seed for random

    Returns:
        - seed_worker: Callable
            Function for DataLoader's worker_init_fn option.
        - generator: torch.Generator
            torch.generator for DataLoader's generator option.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    generator = torch.Generator()
    generator.manual_seed(0)

    return seed_worker, generator


def shuffle_batch(batch: torch.Tensor, return_permutation: bool=False):
    '''randomly shuffle a batched tensor and optionally return pthe permutation.

    Arguments:
        batch: torch.Tensor
            the batch to shuffle
        return_permutation: bool (default: False)
            If True, return permutation
    '''
    permutation = torch.randperm(batch.size(0))
    shuffled = batch[permutation]
    if return_permutation:
        return shuffled, permutation
    return shuffled


def optimizer_step(
    loss: torch.Tensor, optimizer: optim.Optimizer, scaler=None
) -> None:
    '''optimization step which supports gradient scaling for AMP.

    Arguments:
        loss: torch.Tensor
            loss to backpropagate.
        optimizer: torch.optim.Optimizer
            optimizer.
        scaler: GradScaler (default: None)
            Optional GradScaler object.
            If specified, uses it to scale the loss and call .step()
    '''
    assert scaler is None or isinstance(scaler, GradScaler)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
    else:
        loss.backward()
        optimizer.step()


def assert_shape(tensor: torch.Tensor, shape: torch.Size|tuple|list):
    '''assert shape of tensor

    Arguments:
        tensor: torch.Tensor
            tensor to check the shape of
        shape: torch.Size|tuple|list
            expected shape of tensor.
            pass -1 or None for arbitrary size.
    '''
    assert tensor.ndim == len(shape), f'Wrong number of dimensions: got {tensor.ndim} expected {len(shape)}'
    for i, (size, exp_size) in enumerate(zip(tensor.size(), shape)):
        if exp_size is None or exp_size == -1:
            continue
        assert size == exp_size, f'Wrong size for dimension {i}: got {size} expected {exp_size}'
