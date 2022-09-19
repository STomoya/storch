
import atexit
import os
import warnings
from functools import wraps
from typing import Any, Callable

import torch

__all__ = [
    'autosave'
]

class autosave:
    """autosave registered objects when code exits.
    This class is only a container for the objects to be saved,
    with functions to register objects.

    NOTE: This implementation might be a memory and computation overhead and
          forces the user to take care of this, such as registering torch tensors on GPU
          or calling registering functions multiple times.

    Attributes:
        _container (dict): contains the objects to be saved.
    """

    _container = {}

    @classmethod
    def register_object(cls, name: str, obj: Any, overwrite: bool=False) -> None:
        """register an object to save

        Args:
            name (str): name to identify the object.
            obj (Any): the object to register.
            overwrite (bool, optional): overwrite the object if already exists. Default: False.

        Examples::
            >>> # anything that can be torch.save()-ed can be registered
            >>> # optionally overwrite the object using 'overwrite' argument.
            >>> autosave.register_object('name-of-object', torch.randn(10), overwrite=True)

            >>> # model.state_dict() returns a reference so there is no need to overwrite
            >>> # and only needs to be called once
            >>> autosave.register_object('model', model.state_dict())
        """
        if name not in cls._container:
            cls._container[name] = obj
        elif overwrite:
            cls._container[name] = obj

    @classmethod
    def register_function_output(cls, func: Callable) -> Callable:
        """register a function to save the outputs.

        Args:
            func (Callable): The function to save the outputs of

        Returns:
            Callable: The registered function.

        Examples::
            >>> # it can be used like a function
            >>> func = autosave.register_function_output(func)
            >>> # or a decorator
            >>> @autosave.register_function_output
            >>> def func():
            >>>     return
        """
        name = func.__qualname__
        @wraps(func)
        def inner(*args, **kwargs):
            retval = func(*args, **kwargs)
            cls.register_object(name, retval, overwrite=True)
            return retval
        return inner



def _save(folder='.'):
    """save registered objects.
    This function does not need to be called manually. It is automatically called at exit
    via atexit.

    Args:
        folder (str, optional): The directory to save the registered objects. Default: '.'.
    """
    if len(autosave._container.keys()):
        torch.save(autosave._container, os.path.join(folder, 'autosave.torch'))

# these should be always called when this module is imported
warnings.warn('"autosave" is only for unexpected exits from Python programms. Please save them munually by yourselves.')
atexit.register(_save)
