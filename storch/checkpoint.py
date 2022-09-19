
from __future__ import annotations

import copy
import glob

import torch

import storch
from storch.path import Path


class Checkpoint:
    """Checkpointing

    Args:
        root (str): Where to save the chaeckpoint files.
        prefix (str, optional): Prefix for the saved checkpoint file name. Defaults to 'checkpoint'.
        separator (str, optional): Separator for the saved checkpoint file name. Defaults to '_'.
        overwrite (bool, optional): Whether to overwrite the saved checkpoint everytime .save() is called. Defaults to False.
    """
    def __init__(self, root: str, prefix: str='checkpoint', separator: str='_', overwrite: bool=False) -> None:
        self._root = Path(root)
        self._root.mkdir()
        self._compactname = self._root / f'{prefix}.torch'

        self._prefix = prefix
        self._separator = separator
        self._overwrite = overwrite

        self._state_dict = None


    def save(self, state_dict: str, step: int|str, force_cpu=False) -> None:
        """Save checkpoint.

        Args:
            state_dict (str): The object to be saved. Any pickle-able object can be passed in.
            step (int | str): Current step. Ignored if overwrite=True.
            force_cpu (bool, optional): Force the state_dict to be on CPU before saving. Default: False.
        """
        filename = self._root / f'{self._prefix}{self._separator}{step}.torch'

        if force_cpu:
            def is_tensor(data): return isinstance(data, torch.Tensor)
            def to_cpu(tensor): return tensor.cpu()
            state_dict = storch.recursive_apply(to_cpu, state_dict, is_tensor)

        torch.save(state_dict, filename if not self._overwrite else self._compactname)


    def load(self, step: int|str, map_location: str='cpu') -> dict:
        """_summary_

        Args:
            step (int | str): The specific step to be loaded.
            map_location (str, optional): Device for the state_dict to be on when loaded.

        Returns:
            dict: The loaded state dict
        """
        filename = self._root / f'{self._prefix}{self._separator}{step}.torch'
        state_dict = torch.load(filename, map_location=map_location)
        return state_dict


    def load_latest(self, map_location: str='cpu') -> dict:
        """Load latest checkpoint

        Args:
            map_location (str, optional): Device for the state_dict to be on when loaded. Defaults to 'cpu'.

        Returns:
            dict: The loaded state dict
        """
        pattern = self._root / f'{self._prefix}*.torch'
        latest = storch.natural_sort(glob.glob(pattern))[-1]
        state_dict = torch.load(latest, map_location=map_location)
        return state_dict


    # for keeping best model state.
    def keep(self, state_dict: dict):
        """Keep a deepcopy of a state_dict

        Args:
            state_dict (dict): The state_dict to be kept.
        """
        self._state_dict = copy.deepcopy(state_dict)

    def get_kept(self) -> dict:
        """Return the deepcopy of the state_dict.

        Returns:
            dict: The state_dict that was kept via .keep()
        """
        return self._state_dict
