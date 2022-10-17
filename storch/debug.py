'''Import stuff for debug
Examples::
    >>> from storch.debug import *
'''

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from storch.imageops import dummy_tensor, save_images
from storch.torchops import print_module_summary

__all__ = [
    'np',
    'torch',
    'dummy_tensor',
    'save_image',
    'save_images',
    'test_model',
    'print_module_summary',
    'Image',
    'TF',
]

def test_model(
    models: nn.Module|Iterable[nn.Module],
    input_shapes: tuple|Iterable[tuple],
    batch_size: int=3, call_backward: bool=False,
    device: str|torch.device='cpu',
    input_sampler: Callable=torch.randn
) -> None:
    """Test if the model works properly by forwarding dummy tensors (and optionally backward.)

    Args:
        models (nn.Module | Iterable[nn.Module]): nn.Module or a list of nn.Module to be tested.
        input_shapes (tuple | Iterable[tuple]): tuple of iterable of tuples, each presenting the input shape of the model.
        batch_size (int, optional): Size of the batch dimension. Defaults to 3.
        call_backward (bool, optional): The device to perform the test on. Defaults to False.
        device (str | torch.device, optional): _description_. Defaults to 'cpu'.
        input_sampler (Callable, optional): A callable which is used to sample the dummy inputs. Defaults to torch.randn.
    """
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
