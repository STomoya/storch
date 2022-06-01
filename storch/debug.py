'''Import stuff for debug
Usage:
    from storch.debug import *
'''

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from torchvision.utils import save_image

from storch.imageops import dummy_tensor, save_images

__all__ = [
    'torch',
    'dummy_tensor',
    'save_image',
    'save_images',
    'test_model'
]

def test_model(
    models: nn.Module|Iterable[nn.Module],
    input_shapes: tuple|Iterable[tuple],
    batch_size: int=3, call_backward: bool=False,
    device: str|torch.device='cpu',
    input_sampler: Callable=torch.randn
) -> None:
    '''Test is the model works properly by forwarding dummy tensors (and optionally backward.)

    Arguments:
        models: nn.Module|list[nn.Module]
            nn.Module or a list of nn.Module to be tested.
        input_shapes: tuple|Iterable[tuple]
            tuple of iterable of tuples, each presenting the input shape of the model.
        batch_size: int (default: 3)
            Size of the batch dimension.
        device: str|torch.device (default: 'cpu')
            The device to perform the test on.
        input_sampler: Callable (default: torch.randn)
            A callable which is used to sample the dummy inputs.
    '''
    if isinstance(models, (list, tuple)):
        assert len(models) == len(input_shapes)
    else: models, input_shapes = [models], [input_shapes]

    models = [model.to(device) for model in models]

    for model, input_shape in zip(models, input_shapes):
        input = input_sampler(batch_size, *input_shape, device=device)
        output = model(input)

        name = model.__class__.__name__
        in_size  = input.size()
        out_size = output.size()
        mean     = output.mean()
        std      = output.std()
        print(f'[NAME] {name} [IN] size: {in_size}, [OUT] size: {out_size}, mean: {mean}, std: {std}')
        if call_backward:
            print(f'Calling .backward() on {name}(input).mean().')
            mean.backward()
