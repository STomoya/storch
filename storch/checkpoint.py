
import copy
import glob

import torch

import storch
from storch.path import Path


class Checkpoint:
    def __init__(self, root, prefix='checkpoint', separator='_', overwrite=False) -> None:
        self._root = Path(root)
        self._root.mkdir()
        self._compactname = self._root / f'{prefix}.torch'

        self._prefix = prefix
        self._separator = separator
        self._overwrite = overwrite

        self._state_dict = None


    def save(self, state_dict, step, force_cpu=False):
        filename = self._root / f'{self._prefix}{self._separator}{step}.torch'

        if force_cpu:
            def is_tensor(data): return isinstance(data, torch.Tensor)
            def to_cpu(tensor): return tensor.cpu()
            state_dict = storch.recursive_apply(to_cpu, state_dict, is_tensor)

        torch.save(state_dict, filename if not self._overwrite else self._compactname)


    def load(self, step, map_location='cpu'):
        filename = self._root / f'{self._prefix}{self._separator}{step}.torch'
        state_dict = torch.load(filename, map_location=map_location)
        return state_dict


    def load_latest(self, map_location='cpu'):
        pattern = self._root / f'{self._prefix}*.torch'
        latest = storch.natural_sort(glob.glob(pattern))[-1]
        state_dict = torch.load(latest, map_location=map_location)
        return state_dict


    # for keeping best model state.
    def keep(self, state_dict):
        self._state_dict = copy.deepcopy(state_dict)
    def get_kept(self):
        return self._state_dict
