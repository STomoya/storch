
from __future__ import annotations
from typing import Callable

from dataclasses import dataclass, field

import torchvision.transforms as T

def make_simple_transform(
    image_size: tuple[int]|int, crop: str='center', hflip: bool=True,
    mean: tuple[float]|float=0.5, std: tuple[float]|float=0.5
) -> Callable:
    '''make a simple transform

    Arguments:
        image_size: tuple|int
            size of the image to be resized to. if a single int value, (image_size, image_size) will be used.
        crop: str (default: 'center')
            'center': CenterCrop
            'random': RandomResizedCrop
        hflip: bool (default: True)
            If True, add RandomHorizontalFlip
        mean, std: tuple|float (default: 0.5)
            mean and std used to normalize the image.
            mean=0.0, std=1.0 for no normalization.
    '''
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if crop == 'random':
        transform = [T.RandomResizeCrop(image_size)]
    elif crop == 'center':
        transform = [T.Resize(max(image_size)), T.CenterCrop(image_size)]

    if hflip:
        transform.append(T.RandomHorizontalFlip())

    transform.extend([
        T.ToTensor(),
        T.Normalize(mean, std)])

    return T.Compose(transform)


def build_transform(name: str, **params):
    if hasattr(T, name):
        return getattr(T, name)(**params)
    raise UserWarning(f'torchvision.transforms.{name} does not exist.')

def make_transform_from_config(configs: list[dict]):
    '''make transform from list of TransformConfigs'''
    transform = []
    for config in configs:
        transform.append(build_transform(**config))
    return T.Compose(transform)
