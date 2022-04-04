
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


@dataclass
class TransformConfig:
    '''Config for transforms

    Arguments:
        name: str
            name of the transform implemented in torchvision.transforms
        params: dict
            keyword arguments for the transform
    '''
    name: str
    params: dict=field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict):
        '''make config from a dict

        Required shape (yaml):
            - name: Resize
              size: 128
            - name: ToTensor
            - name: Normalize
              mean: 0.5
              std: 0.5

            # or

            - name: Resize
              params:
                size: 128
            - name: ToTensor
            - name: Normalize
              params:
                mean: 0.5
                std: 0.5
        '''
        name = config['name']
        if 'params' in config:
            params = config['params']
        else:
            params = {key: value for key, value in config.items() if key != 'name'}
        return cls(name, params)

    def exists(self):
        return hasattr(T, self.name)

def make_transform_from_config(configs: list[TransformConfig]):
    '''make transform from list of TransformConfigs'''
    transform = []
    for config in configs:
        if not config.exists():
            raise UserWarning(f'torchvision.transoforms.{config.name} does not exist.')
        tcls = getattr(T, config.name)
        transform.append(tcls(**config.params))
    return T.Compose(transform)
