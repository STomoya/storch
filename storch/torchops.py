
from __future__ import annotations

import random
import warnings
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

import storch

__all__ = [
    'auto_get_device',
    'freeze',
    'unfreeze',
    'update_ema',
    'deterministic',
    'shuffle_batch',
    'optimizer_step',
    'assert_shape'
]


def auto_get_device(force_cpu: bool=False, no_gpu_msg_type='warn'):
    '''automatically return a torch.device object.
    'cuda' if torch.cuda.is_available() else 'cpu'

    Argumnets:
        force_cpu: bool (default: False)
            force device to be CPU
        no_gpu_msg_type: str (default: 'warn')
            How this function tells you, you do not have any GPUs. Ignored when force_cpu=True.
            - 'warn': warnings.warn
            - 'except': raise an exception
            - any other str: does nothing
    '''
    if torch.cuda.is_available() or not force_cpu:
        return torch.device('cuda')
    if force_cpu:
        return torch.device('cpu')

    no_gpu_msg = f'No GPU found on your environment.'
    if no_gpu_msg_type == 'warn':
        warnings.warn(no_gpu_msg + ' Falling back to CPU.')
    elif no_gpu_msg_type == 'except':
        raise Exception(no_gpu_msg)

    return torch.device('cpu')


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


def grad_nan_to_num(module: nn.Module, nan: float=0.0, posinf: float=1e5, neginf: float=1e-5):
    '''set nan gardients to a number.

    Arguments:
        module: nn.Module
            The module with parameters holding .grad attribute.
        nan: float (default: 0)
            Value to replace nan.
        posinf, neginf: float (default: 1e5, 1e-5)
            Value to replace positive/negative inf.
    '''
    params = [param for param in module.parameters() if param.grad is not None]
    if len(params):
        flat = torch.cat([param.grad.flatten() for param in params])
        flat = torch.nan_to_num(flat, nan, posinf, neginf)
        grads = flat.split([param.numel() for param in params])
        for param, grad in zip(params, grads):
            param.grad = grad.reshape(param.shape)


def optimizer_step(
    loss: torch.Tensor, optimizer: optim.Optimizer, scaler=None,
    zero_grad: bool=True, set_to_none: bool=True, update_scaler: bool=False
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
        zero_grad: bool (default: True)
            Call .zero_grad() on optimizer before calling .backward() on loss
        set_no_none: bool (default: True)
            Set None to .grad instead of setting them to 0.
        update_scaler: bool (default: False)
            Update the scaler if not None.
    '''
    assert scaler is None or isinstance(scaler, GradScaler)

    if zero_grad:
        optimizer.zero_grad(set_to_none)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        if update_scaler:
            scaler.update()
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


def print_module_summary(module: nn.Module, inputs: list|tuple, max_nesting: int=3, skip_redundant: bool=True, print_fn: Callable=print):
    '''Print module summary.
    Taken from: https://github.com/NVlabs/stylegan3/blob/583f2bdd139e014716fc279f23d362959bcc0f39/torch_utils/misc.py#L196-L264

    Arguments:
        module: nn.Module
            The module to summrize.
        inputs: list|tuple
            List of input tensors.
        max_nesting: int (default: 3)
        skip_redundant: bool (default: True)
        print_fn: Callable (default: print)
            Function for printing the summary.
    '''
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(storch.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print_fn()
    for row in rows:
        print_fn('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print_fn()
    return outputs
