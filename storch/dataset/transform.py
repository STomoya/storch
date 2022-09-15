
from __future__ import annotations

from typing import Callable

import torchvision.transforms as T

import storch


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
    '''Build a transform inside torchvision.transforms by their class name

    Arguments:
        name: str
            The name of the transform. ex) ToTenor, Normalize
        **params: Any
            Keyword arguments of passed to the transform.
    '''
    if hasattr(T, name):
        return getattr(T, name)(**params)
    else:
        transform = storch.construct_class_by_name(class_name=name, **params)
        assert callable(transform), f'User defined tranform {name} is not callable'
        return transform


def make_transform_from_config(configs: list[dict]):
    '''make transform from list of TransformConfigs

    Usage:
        transforms = make_transform_from_config(
            [
                {'name': 'Resize', 'size': (224, 224)},
                {'name': 'ToTensor'},
                {'name': 'Normalize', 'mean': 0.5, 'std': 0.5}
            ]
        )

    Arguments:
        configs: list[dict]
            List of dicts containing a least 'name' key.
    '''
    transform = []
    for config in configs:
        if config.get('name') in ['RandomChoice', 'RandomOrder', 'RandomApply']:
            inner_transform = []
            for inner_config in config.get('transforms'):
                inner_transform.append(build_transform(**inner_config))
            params = dict(transforms=inner_transform)
            if config.get('p', None) is not None:
                params.update(dict(p=config.get('p')))
            transform.append(build_transform(config.get('name'), **params))
        else:
            transform.append(build_transform(**config))
    return T.Compose(transform)
