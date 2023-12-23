"""Import stuff for debug.

Examples::
    >>> from storch.debug import *
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from storch.imageops import dummy_tensor, save_images
from storch.torchops import print_module_summary

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


__all__ = [
    'np',
    'torch',
    'nn',
    'dummy_tensor',
    'save_image',
    'save_images',
    'test_model',
    'print_module_summary',
    'Image',
    'TF',
    'RandomDataset',
]


def test_model(
    models: nn.Module | Iterable[nn.Module],
    input_shapes: tuple | Iterable[tuple],
    batch_size: int = 3,
    call_backward: bool = False,
    device: str | torch.device = 'cpu',
    input_sampler: Callable = torch.randn,
) -> None:
    """Test if the model works properly by forwarding dummy tensors (and optionally backward.).

    Args:
    ----
        models (nn.Module | Iterable[nn.Module]): nn.Module or a list of nn.Module to be tested.
        input_shapes (tuple | Iterable[tuple]): tuple of iterable of tuples, each presenting
            the input shape of the model.
        batch_size (int, optional): Size of the batch dimension. Defaults to 3.
        call_backward (bool, optional): The device to perform the test on. Defaults to False.
        device (str | torch.device, optional): _description_. Defaults to 'cpu'.
        input_sampler (Callable, optional): A callable which is used to sample the dummy inputs.
            Defaults to torch.randn.
    """
    if isinstance(models, (list, tuple)):
        assert len(models) == len(input_shapes)
    else:
        models, input_shapes = [models], [input_shapes]

    models = [model.to(device) for model in models]

    for model, input_shape in zip(models, input_shapes, strict=False):
        input = input_sampler(batch_size, *input_shape, device=device)
        output = model(input)

        name = model.__class__.__name__
        in_size = input.size()
        out_size = output.size()
        mean = output.mean()
        std = output.std()
        print(f'[NAME] {name} [IN] size: {in_size}, [OUT] size: {out_size}, mean: {mean}, std: {std}')
        if call_backward:
            print(f'Calling .backward() on {name}(input).mean().')
            mean.backward()


class RandomDataset(Dataset):
    """Random data creation dataset."""

    def __init__(self, generate_data: Callable, num_samples: int = 32) -> None:  # noqa: D107
        super().__init__()
        self.generate_data = generate_data
        self.num_samples = num_samples

    def __len__(self):  # noqa: D105
        return self.num_samples

    def __getitem__(self, _):  # noqa: D105
        return self.generate_data()

    @classmethod
    def create(cls, generate_data: Callable | str, num_samples: int = 32, batch_size: int = 32, **kwargs):  # noqa: D102
        if isinstance(generate_data, str):
            generate_data = _build_data_generators(generate_data, **kwargs)
        dataset = cls(generate_data, num_samples)
        dataloader = DataLoader(dataset, batch_size)
        return dataloader


def _build_data_generators(type: str, image_size: int = 224, num_classes: int = 1000, sr_scale: int = 2):
    if type == 'image':

        def generate_data():
            return torch.randn(3, image_size, image_size)
    elif type == 'classification':

        def generate_data():
            return torch.randn(3, image_size, image_size), torch.randint(0, num_classes, (1,)).long()
    elif type == 'multilabel':

        def generate_data():
            return torch.randn(3, image_size, image_size), torch.where(
                torch.rand(num_classes) < 0.5,  # noqa: PLR2004
                torch.zeros(num_classes),
                torch.ones(num_classes),
            )
    elif type == 'i2i':

        def generate_data():
            return torch.randn(3, image_size, image_size), torch.randn(3, image_size, image_size)
    elif type == 'sr':

        def generate_data():
            return torch.randn(3, image_size // sr_scale, image_size // sr_scale), torch.randn(
                3, image_size, image_size
            )
    else:
        raise Exception(f'No random generator type "{type}"')
    return generate_data
