"""Checkpointing
"""

import glob
import os
import random
import warnings
from collections import deque

import numpy as np
import torch

import storch
from storch.distributed import utils as distutils
from storch.path import Path


class Checkpoint:
    """class for saving checkpoints.

    Easy saving/loading of intermediate training procedure. The main API of this class is three functions:

    - `register`: register objects which have `state_dict` and `load_state_dict` methods.

    - `save`: serialize the registered model.

    - `load`/`load_latest`: load (latest) checkpoint.

    `save` and `load_latest` does not have any required arguments.

    This class automatically saves/loads the random state of python builtin `random`, numpy, and PyTorch.

    Supports distributed training (serialization only on primary process).

    Args:
        folder (str): folder to save the checkpoints to.
        keep_last (int, optional): keep last n checkpoints. None to keep all. Default: 3.
        filename_format (str, optional): format for filename of the checkpoint. Default: 'checkpoint.{count}.torch'.

    Examples:
        >>> checkpoint = Checkpoint('./checkpoints')
        >>> # the objects must have state_dict and load_state_dict function.
        >>> checkpoint.register(model=torch.nn.Linear(1, 1))

        >>> # this will save the registered models.
        >>> checkpoint.save()
        >>> # you can also save others, including objects without state_dict and load_state_dict func.
        >>> checkpoint.save(constant=1)

        >>> # update models
        >>> train(...)

        >>> # this will load the models from a specific checkpoint to the registered objects.
        >>> # no need to call load_state_dict.
        >>> # it also returns additionally saved objects. (These must be over written by user code.)
        >>> checkpoint.load('./checkpoints/checkpoint.0.torch')
        {'constant': 1}

        >>> # this will load the state_dict from the latest checkpoint.
        >>> checkpoint.load_latest()
        {'constant': 1}
    """

    _container = storch.EasyDict()


    def __init__(self, folder: str, keep_last=3, filename_format: str='checkpoint.{count}.torch', disthelper=None) -> None:
        self.folder = Path(folder)
        self.filename_format = filename_format

        if keep_last is not None:
            keep_last += 1
        self.file_deque = deque(maxlen=keep_last)
        self.count = 0

        if disthelper is not None:
            warnings.warn(
                f'This class does not require DistributedHelper anymore, and the argument will be erased in future versions.',
                FutureWarning
            )


    def _container_as_state_dict(self) -> dict:
        """call state_dict on registered objects and return as dict.

        Returns:
            dict: the state_dict of registered objects.
        """
        state_dict = dict()
        for key, value in self._container.items():
            state_dict[key] = value.state_dict()
        return state_dict


    def state_dict(self):
        """state_dict

        Returns:
            dict: state_dict of this class.
        """
        return dict(
            storch_version=storch.__version__,
            file_deque=self.file_deque,
            count=self.count
        )


    def load_state_dict(self, state_dict: dict):
        """load from a saved state_dict
        """
        _storch_version = state_dict.pop('storch_version')
        if storch.__version__ != _storch_version and distutils.is_primary():
            warnings.warn(
                f'Checkpoint.load_state_dict: You are using a different version of storch ({_storch_version} -> {storch.__version__}). This might cause errors.',
                UserWarning)
        self.file_deque = state_dict.pop('file_deque')
        self.count = state_dict.pop('count')


    def save(self, **constants):
        """save the registered objects along with additional objects.

        Returns:
            str: path to where the checkpoint was saved
        """
        filename = self.folder / self.filename_format.format(count=self.count)
        self.file_deque.append(filename)
        self.count += 1

        state_dict = {}
        state_dict['state_dict'] = self._container_as_state_dict()

        if constants != {}:
            state_dict.update({'constants': constants})

        state_dict['random_state'] = dict(
            builtin=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()
        )

        state_dict.update({'__checkpoint_state': self.state_dict()})

        if distutils.is_primary():
            torch.save(state_dict, filename)

            if self.file_deque.maxlen is not None and len(self.file_deque) == self.file_deque.maxlen:
                to_erase = self.file_deque.popleft()
                os.remove(to_erase)

        return filename


    def load(self, path: str, map_location: torch.device=None):
        """load checkpoint from "path".

        Args:
            path (str): path to the saved checkpoint.
            map_location (torch.device, optional): location to load the checkpoint. Default: None.

        Raises:
            Exception: the checkpoint was not saved by this class

        Returns:
            dict: additional objects saved when save (if any).
        """
        distutils.wait_for_everyone()

        state_dict = torch.load(path, map_location=map_location)

        _self_state = state_dict.pop('__checkpoint_state', None)
        if _self_state is None:
            raise Exception(f'Checkpoint.load: This checkpoint seems to be not saved via storch.Checkpoint.')
        self.load_state_dict(_self_state)

        random_state = state_dict.pop('random_state')
        random.setstate(random_state['builtin'])
        np.random.set_state(random_state['numpy'])
        torch.set_rng_state(random_state['torch'])

        constants = state_dict.pop('constants', None)

        for key, value in state_dict.pop('state_dict').items():
            self._container[key].load_state_dict(value)

        return constants


    def load_latest(self, map_location: torch.device=None):
        """load checkpoint from latest file (if any).

        Args:
            map_location (torch.device, optional): location to load the checkpoint. Default: None.

        Returns:
            dict: additional objects saved when save (if any).
        """
        paths = storch.natural_sort(glob.glob(self.folder / self.filename_format.format(count='*')))
        if len(paths) > 0:
            return self.load(paths[-1], map_location)


    def register(self, **kwargs):
        """register objects to be saved when "save" is called. also used to load checkpoints via "load".
        The objects must have a "state_dict" and "load_state_dict" function.

        Args:
            **kwargs: objects to be registered as keyword arguments.
        """
        not_registered_keys = []
        for key, value in kwargs.items():
            if (hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict') and
                callable(value.state_dict) and callable(value.load_state_dict)):
                self._container[key] = value
            else:
                not_registered_keys.append(key)

        if len(not_registered_keys) and distutils.is_primary():
            warnings.warn(
                f'Checkpoint.register: The following object could not be registered: {not_registered_keys}',
                UserWarning
            )
