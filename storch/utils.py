
from __future__ import annotations

import glob
import json
import math
import os
from collections.abc import Iterable
from typing import Any

__all__=[
    'calc_num_sampling',
    'check_folder',
    'dynamic_default',
    'EasyDict',
    'glob_inside',
    'prod',
    'save_command_args'
]


class EasyDict(dict):
    '''dict that can access keys like attributes.'''
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def calc_num_sampling(high_resl, low_resl):
    '''calculate number of sampling times when scale factor is 2'''
    return int(math.log2(high_resl)-math.log2(low_resl))


def dynamic_default(value: Any|None, default_value: Any):
    '''dynamic default value'''
    return value if value is not None else default_value


def prod(iter: Iterable):
    '''np.prod for python iterables'''
    result = 1
    for value in iter:
        result *= value
    return result


def save_command_args(args, filename: str='args.json'):
    '''save Namespace object to json file'''
    args_dict = vars(args)
    with open(filename, 'w') as fout:
        json.dump(args_dict, fout, indent=2)


def check_folder(folder: str, make: bool=False):
    '''check if a folder exists and create it if not.'''
    exists = os.path.exists(folder)
    if make and not exists:
        os.makedirs(folder)
    return exists


def glob_inside(folder: str, pattern: str='*', recursive: bool=True):
    '''glob for files/dirs that matches pattern.'''
    pattern = f'**/{pattern}' if recursive else pattern
    return glob.glob(os.path.join(folder, pattern), recursive=recursive)
