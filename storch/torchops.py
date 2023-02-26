'''PyTorch operations.'''

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

import storch
from storch._optimizer_step import (
    get_optimizer_step, grad_nan_to_num_, optimizer_step,
    optimizer_step_with_gradient_accumulation, simple_optimizer_step,
    simple_optimizer_step_with_gradient_accumulation)
from storch.utils import version

__all__ = [
    'auto_get_device',
    'freeze',
    'unfreeze',
    'update_ema',
    'set_seeds',
    'deterministic',
    'shuffle_batch',
    'get_optimizer_step',
    'optimizer_step',
    'simple_optimizer_step',
    'optimizer_step_with_gradient_accumulation',
    'simple_optimizer_step_with_gradient_accumulation',
    'assert_shape',
    'print_module_summary',
    'grad_nan_to_num_',
    'inference_mode',
    'convert_outputs_to_fp32'
]


if version.is_torch_version_geq('1.9.0'):
    inference_mode: Callable = torch.inference_mode
else:
    inference_mode: Callable = torch.no_grad


def auto_get_device(force_cpu: bool=False, no_gpu_msg_type='warn') -> torch.device:
    """automatically return a torch.device object.
    'cuda' if torch.cuda.is_available() else 'cpu'

    Args:
        force_cpu (bool, optional): Force device to be CPU. Default: False.
        no_gpu_msg_type (str, optional): How this function tells you, you do not have any GPUs. Ignored when force_cpu=True.
            - 'warn': warnings.warn
            - 'except': raise an exception
            - any other str: does nothing.
            Default: 'warn'.

    Raises:
        Exception: Raised when no GPUs found.

    Returns:
        torch.device: Device.
    """
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
def freeze(model: nn.Module) -> None:
    """freeze the model.

    This is an inplace operation.

    Args:
        model (nn.Module): The module to freeze.
    """
    model.eval()
    model.requires_grad_(False)


@torch.no_grad()
def unfreeze(model: nn.Module) -> None:
    """unfreeze the model

    This is an inplace operation.

    Args:
        model (nn.Module): The module to unfreeze.
    """
    model.requires_grad_(True)
    model.train()


@torch.no_grad()
def update_ema(
    model: torch.nn.Module, model_ema: torch.nn.Module,
    decay: float=0.999, copy_buffers: bool=False, force_cpu: bool=False
) -> None:
    """Update exponential moving avg.
    w'_new = w' * decay + w * (1 - decay)

    Args:
        model (torch.nn.Module): Model, which actually is updated
        model_ema (torch.nn.Module): Copy of the model, which is updated by exponential moving average.
        decay (float, optional):  Decay for exponential modving average. Default: 0.999.
        copy_buffers (bool, optional): If True, also copy buffers inside the model. Default: False.
        force_cpu (bool, optional): If True, process on cpu. Expects the model_ema to be on CPU. Default: False.
    """
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


def set_seeds(
    seed: int=3407,
    use_deterministic_algorithms: bool=False, warn_only: bool=False,
    cudnn_benchmark: bool=False
) -> tuple[Callable, torch.Generator]:
    """Settings for reproducible training.

    Args:
        seed (int, optional): Random number generator seed. Default: 3407.
        use_deterministic_algorithms (bool, optional): use deterministic algorithms?
            True for reproducibility. Default: False.
        warn_only (bool, optional): Warn instead of an exception when using an module
            without a deterministic implementation. Default: False.
        cudnn_benchmark (bool, optional): cudnn benchmark. Default: False.

    Returns:
        Callable: Function for DataLoader's worker_init_fn option.
        torch.Generator: torch.Generator for DataLoader's generator option.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic_algorithms, warn_only=warn_only)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    generator = torch.Generator()
    generator.manual_seed(0)

    return seed_worker, generator


def deterministic(seed: int=3407) -> tuple[Callable, torch.Generator]:
    """Settings for reproducible training.

    Deprecated

    Args:
        seed (int, optional): Random number generator seed. Default: 3407.

    Returns:
        Callable: Function for DataLoader's worker_init_fn option.
        torch.Generator: torch.Generator for DataLoader's generator option.
    """
    import warnings
    warnings.warn(
        f'"deterministic" is deprecated in favor of "set_seeds" and will be erased in the future version.',
        DeprecationWarning
    )
    set_seeds(seed, use_deterministic_algorithms=True)


def shuffle_batch(batch: torch.Tensor, return_permutation: bool=False) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """randomly shuffle a batched tensor and optionally return pthe permutation.

    Args:
        batch (torch.Tensor): Batched tensor to shuffle.
        return_permutation (bool, optional): If True, return permutation. Default: False.

    Returns:
        torch.Tensor: The shuffled tensor.
        torch.Tensor: The permutation. Only when return_permutation==True.
    """
    permutation = torch.randperm(batch.size(0))
    shuffled = batch[permutation]
    if return_permutation:
        return shuffled, permutation
    return shuffled


def assert_shape(tensor: torch.Tensor, shape: torch.Size|tuple|list) -> None:
    """assert shape of tensor

    Args:
        tensor (torch.Tensor): tensor to check the shape of
        shape (torch.Size | tuple | list): expected shape of tensor. -1 or None for arbitrary size.
    """
    assert tensor.ndim == len(shape), f'Wrong number of dimensions: got {tensor.ndim} expected {len(shape)}'
    for i, (size, exp_size) in enumerate(zip(tensor.size(), shape)):
        if exp_size is None or exp_size == -1:
            continue
        assert size == exp_size, f'Wrong size for dimension {i}: got {size} expected {exp_size}'


def print_module_summary(module: nn.Module, inputs: list|tuple, max_nesting: int=3, skip_redundant: bool=True, print_fn: Callable=print):
    """Print module summary.
    Taken from: https://github.com/NVlabs/stylegan3/blob/583f2bdd139e014716fc279f23d362959bcc0f39/torch_utils/misc.py#L196-L264

    Args:
        module (nn.Module): The module to summarize.
        inputs (list | tuple): List of input tensors.
        max_nesting (int, optional): Max nested modules to print. Default: 3.
        skip_redundant (bool, optional): Filter out redundant entries. Default: True.
        print_fn (Callable, optional): Function for printing the summary. Default: print.
    """
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


def convert_outputs_to_fp32(func: Callable) -> Callable:
    def convert_to_fp32(tensor: torch.Tensor):
        return tensor.float()

    @wraps(func)
    def inner(*args, **kwargs):
        retvals = func(*args, **kwargs)
        retvals = storch.recursive_apply(convert_to_fp32, retvals, torch.is_tensor)
        return retvals

    return inner


def get_grad_scaler(enabled=True, is_fsdp=False, disable_with_none=False) -> GradScaler|None:
    """Get the proper gradient scaler.

    Args:
        enabled (bool, optional): Enable gradient scaling? Default: True.
        is_fsdp (bool, optional): is distributed mode FSDP? Default: False.
        disable_with_none (bool, optional): Disable grdient scaling by returning None. Default: False.

    Returns:
        GradScaler | None: gradient scaler class
    """
    scaler = GradScaler(enabled=enabled) if not is_fsdp else ShardedGradScaler(enabled=enabled)
    if not enabled and disable_with_none:
        scaler = None
    return scaler
