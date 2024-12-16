"""Model utils.

classes and functions in this file is for user defined models, not for pre-defined models in
libraries like timm, etc.
"""

from __future__ import annotations

import inspect
import json
import os
import struct
from enum import Enum
from functools import wraps
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullStateDictConfig, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import safetensors
except ImportError:
    safetensors = None

from storch.distributed.utils import is_primary, wait_for_everyone

__all__ = ['ModelMixin', 'is_safetensors_available', 'load_model', 'save_model']


class ModelMixin(nn.Module):
    """model mixin."""

    def __init_subclass__(cls) -> None:
        """Initialize subclasses to save constructor args."""

        def register_init_args(init: Callable):
            @wraps(init)
            def inner(self, *args, **kwargs):
                init(self, *args, **kwargs)

                config = {}
                signiture = inspect.signature(init)
                parameters = {k: v.default for k, v in signiture.parameters.items() if k != 'self'}

                for arg, name in zip(args, parameters.keys(), strict=False):
                    config[name] = arg
                config.update({k: kwargs.get(k, default) for k, default in parameters.items() if k not in config})
                self.__model_config = config

            return inner

        cls.__init__ = register_init_args(cls.__init__)

    @property
    def model_config(self) -> dict:
        """Model config."""
        return self.__model_config

    @classmethod
    def from_saved(cls, fp, device='cpu'):
        """Create model from saved file."""
        model_config, state_dict = load_model(fp, device)
        model = cls(**model_config)
        model.load_state_dict(state_dict)
        return model

    def extra_repr(self):
        """Repr."""
        return ', '.join([f'{k}={v}' for k, v in self._config_repr.items()])


# serialization


class WeightExt(Enum):
    TORCH = '.torch'
    SAFETENSORS = '.safetensors'


def is_safetensors_available():
    """Is safetensors available."""
    return safetensors is not None


def parse_safetensors_metadata(file: str) -> dict:
    """Load metadata entry inside file header.

    Args:
        file (str): the file to read metadata from.

    """
    with open(file, 'rb') as fp:
        header_length = struct.unpack('<Q', fp.read(8))[0]
        header_string = struct.unpack(f'<{header_length}s', fp.read(header_length))[0]
    header = json.loads(header_string)
    metadata = header.get('__metadata__', {})
    return metadata


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap the model from torch.compile and DDP.

    FSDP is not unwrapped because it needs an special process to get state_dicts. See `get_resolved_state_dict`.

    Args:
        model (nn.Module): the model to unwrap.

    Returns:
        nn.Module: unwrapped model.

    """
    if hasattr(model, 'dynamo_ctx'):
        model = model._orig_mod
    if isinstance(model, DDP):
        model = model.module
    return model


def get_resolved_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return `state_dict` that can be loaded by unwrapped models.

    Supports FSDP and DDP models. `state_dict` for FSDP models are always offloaded to cpu and only gathered to the
    primary rank.

    Args:
        model (nn.Module): the model to collect the weights.

    Returns:
        (dict[str, torch.Tensor]): the `state_dict` of the model.

    """
    # FSDP state_dict.
    if isinstance(model, FSDP):
        offload_to_cpu = model.sharding_strategy != ShardingStrategy.NO_SHARD
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=offload_to_cpu),
        ):
            return model.state_dict()

    # DDP state_dict.
    # NOTE: the model is already uwrapped in `unwrap_model` if using DDP.

    # No parallel model.
    return model.state_dict()


def save_model(model: ModelMixin, fp: str, framework: str = 'safetensors'):
    """Save model.

    `fp` does not need file extension. `framework` does not equal to safetensors' framework keyword argument.

    The `model` can be one of a normal `nn.Module` or wrapped via `DistributedDataParallel` or
    `FullyShardedDataParallel` as long as the original model inherits the `ModelMixin` class. The arguments and keyword
    arguments used to construct the model is automatically saved.

    if `model` is wrapped via FSDP module, you must pass the wrapped model, not the un-wrapped one. Also, the weights
    will always be offloaded to CPU and gathered only on rank 0.

    If safetensors is not available, this function fallbacks to `torch.save` to pickle weights.

    Args:
        model (ModelMixin): the model to save the weights. it can be wrapped via DDP or FSDP. If the model
            is wrapped via FSDP, you must pass the wrapped model for gathering the full state dict properly.
        fp (str): the filename to save the weights. the path does not need an file extension.
        framework (str): one of 'pytorch', 'safetensors', or 'both'. 'pytorch' will use `torch.save`,
            'safetensors' will use `safetensors.torch.save_file` to serialize the weights. 'both' will save
            the weights with both ways. Default: 'safetensors'.

    """
    model = unwrap_model(model)
    model_config = getattr(model, 'model_config', None)
    if model_config is None:
        raise Exception('The `model` argument must be an object of a class inheriting ModelMixin.')

    state_dict = get_resolved_state_dict(model)

    # do binarization only once.
    if is_primary():
        framework = framework.lower()
        if is_safetensors_available() and framework in ['st', 'safetensor', 'safetensors', 'both']:
            from safetensors.torch import save_file

            # save model_config dict as string inside __metadata__ of safetensors file header.
            metadata = {'model_config': json.dumps(model_config)}
            save_file(state_dict, fp + WeightExt.SAFETENSORS.value, metadata=metadata)

        # fallback to pickle, if safetensors is not available.
        if not is_safetensors_available() or framework in ['pt', 'pytorch', 'torch', 'both']:
            torch_weight_dict = {'model_config': model_config, 'state_dict': state_dict}
            torch.save(torch_weight_dict, fp + WeightExt.TORCH.value)


def load_model(fp: str, device: str | torch.device = 'cpu') -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Load serialized file.

    The file must be saved via `save_model` function. This function returns a dict containing keyword arguments to
    instantiate the model and the saved `state_dict`. It is recommended to use `ModelMixin.from_saved` classmethod that
    does the instantiation and weigt loading.

    Args:
        fp (str): _description_
        device (str | torch.device, optional): _description_. Defaults to 'cpu'.

    Returns:
        (tuple[dict[str, Any], dict[str, torch.Tensor]]): _description_

    """
    # synchronize devices
    wait_for_everyone()

    ext = os.path.splitext(fp)[-1]

    if ext == WeightExt.TORCH.value:
        state_dict = torch.load(fp, map_location=device)
        model_config = state_dict.get('model_config', None)
        assert model_config is not None, 'The file must be saved using `save_model` function.'
        state_dict = state_dict.get('state_dict')

    elif ext == WeightExt.SAFETENSORS.value:
        if not is_safetensors_available():
            raise Exception('`safetensors` must be installed.')
        from safetensors.torch import load_file

        metadata = parse_safetensors_metadata(fp)
        model_config = metadata.get('model_config', None)
        assert model_config is not None, 'The file must be saved using `save_model` function.'
        model_config = json.loads(model_config)
        state_dict = load_file(fp, device=device)

    return model_config, state_dict
