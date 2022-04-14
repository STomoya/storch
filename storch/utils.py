
from __future__ import annotations

import glob
import json
import math
import os
import re
from collections.abc import Iterable
from typing import Any

__all__=[
    'calc_num_sampling',
    'check_folder',
    'dynamic_default',
    'EasyDict',
    'glob_inside',
    'natural_sort',
    'prod',
    'recursive_apply',
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


# from: https://github.com/google/flax/blob/2387439a6f5c88627754905e6feadac4f33d9800/flax/training/checkpoints.py
UNSIGNED_FLOAT_RE = re.compile(r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
def natural_sort(iter):
    '''sort files by numbers'''
    def maybe_num(s):
        return float(s) if UNSIGNED_FLOAT_RE.match(s) else s
    def split_keys(s):
        return [maybe_num(c) for c in UNSIGNED_FLOAT_RE.split(s)]
    return sorted(iter, key=split_keys)


def recursive_apply(func, data, cond_fn):
    '''recursively apply func to data that satisfies cond_fn

    Arguments:
        func: Callable
            the function to apply
        data: Any
            data to be applied
        cond_fn: Callable
            a function that returns a bool, which decides whether to apply the func or not.
    '''
    if isinstance(data, (tuple, list)):
        return type(data)(recursive_apply(func, element, cond_fn) for element in data)
    elif isinstance(data, dict):
        return {key: recursive_apply(func, value, cond_fn) for key, value in data.items()}
    elif cond_fn(data):
        data = func(data)
    return data
