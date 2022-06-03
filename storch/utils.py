
from __future__ import annotations

import datetime
import functools
import glob
import importlib
import json
import math
import os
import re
import sys
import traceback
import types
from collections.abc import Iterable
from typing import Any, Tuple

__all__=[
    'calc_num_sampling',
    'check_folder',
    'construct_class_by_name',
    'dynamic_default',
    'EasyDict',
    'get_now_string',
    'glob_inside',
    'natural_sort',
    'prod',
    'recursive_apply',
    'save_command_args',
    'save_exec_status'
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


'''
Bellow is taken from: https://github.com/NVlabs/stylegan3/blob/583f2bdd139e014716fc279f23d362959bcc0f39/dnnlib/util.py#L233-L303
Creates an object using it's name and parameters.
Modified by: STomoya (https://github.com/STomoya)
'''

def _get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    '''Searches for the underlying module behind the name to some python object.
        Returns the module and the object name (original name with module part removed).
    '''

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            _get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            _get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def _get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    '''Traverses the object name and returns the last (rightmost) python object.
    '''
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def _get_obj_by_name(name: str) -> Any:
    '''Finds the python object with the given name.'''
    module, obj_name = _get_module_from_obj_name(name)
    return _get_obj_from_module(module, obj_name)


def _call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    '''Finds the python object with the given name and calls it as a function.
    '''
    assert func_name is not None
    func_obj = _get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args: Any, class_name: str = None, **kwargs: Any) -> Any:
    '''Finds the python class with the given name and constructs it with the given arguments.

    Arguments:
        *args: Any
            Positional arguments of the class
        class_name: str
            The name of the class. It should be the full name (like torch.optim.Adam).
        **kwargs: Any
            Keyword arguments of the class
    '''
    return _call_func_by_name(*args, func_name=class_name, **kwargs)


def get_now_string(format: str='%Y%m%d%H%M%S'):
    return datetime.datetime.now().strftime(format)


def save_exec_status(path: str='./execstatus.txt', mode: str='a'):
    '''Decorator that saves execution status to a file.
    Useful if you cannot access traceback messages like inside detached docker containers.

    basic code is from huggingface/knockknock,
    but save message to a file instead of sending e-mails etc.

    Usage:
        @storch.save_exec_status('./path/to/output.txt', 'a')
        def hello():
            print('hello')
        hello()
        # OR
        def hello():
            print('hello')
        storch.save_exec_status('./path/to/output.txt', 'a')(hello)()

    Arguments:
        path: str (default: './execstatus.txt')
            File to save the output to.
        mode: str (default: 'a')
            File open mode. 'w' will overwrite previous outputs.
    '''

    messgae_format = '' \
    + '**  MAIN CALL   **: {func_name}\n' \
    + '**  STATUS      **: {status}\n' \
    + '**  START TIME  **: {start_time}\n' \
    + '**  END TIME    **: {end_time}\n' \
    + '**  DURATION    **: {duration}'

    date_format = "%Y-%m-%d %H:%M:%S"

    def _save(message):
        with open(path, mode) as fp:
            fp.write(message)


    def decorator(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):

            start_time = datetime.datetime.now()
            func_name = func.__qualname__

            try:
                retval = func(*args, **kwargs)

                # successful run
                end_time = datetime.datetime.now()
                duration = end_time - start_time
                message = messgae_format.format(
                    func_name=func_name,
                    status='FINISHED 🐈',
                    start_time=start_time.strftime(date_format),
                    end_time=end_time.strftime(date_format),
                    duration=str(duration)
                )
                _save(message)

                return retval

            except Exception as ex:
                # error
                end_time = datetime.datetime.now()
                duration = end_time - start_time
                message = messgae_format.format(
                    func_name=func_name,
                    status='CRASHED 👿',
                    start_time=start_time.strftime(date_format),
                    end_time=end_time.strftime(date_format),
                    duration=str(duration)
                ) + '\n' \
                + f'**  ERROR       **: {ex}\n' \
                + f'**  TRACEBACK   **: \n{traceback.format_exc()}' # add traceback and error message
                _save(message)

                raise ex
        return inner

    return decorator
