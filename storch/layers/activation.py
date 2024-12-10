"""Activations."""

import torch.nn as nn


def get_activation(name: str, **kwargs) -> nn.Module:
    """Get activation layer by name.

    Supported functions:
      - `relu`: `torch.nn.ReLU`
      - `lrelu`: `torch.nn.LeakyReLU`
      - `prelu`: `torch.nn.PReLU`
      - `gelu`: `torch.nn.GELU`
      - `silu`: `torch.nn.SiLU`
      - `tanh`: `torch.nn.Tanh`
      - `sigmoid`: `torch.nn.Sigmoid`

    Args:
        name (str): Name of the activation function.
        **kwargs: keyword arguments for the activation functions.

    Raises:
        Exception: Unknown activation name.

    Returns:
        nn.Module: The activation function module.

    """
    if name == 'relu':
        return nn.ReLU(**kwargs)
    elif name == 'lrelu':
        return nn.LeakyReLU(**kwargs)
    elif name == 'prelu':
        return nn.PReLU(**kwargs)
    elif name == 'gelu':
        return nn.GELU(**kwargs)
    elif name in ['swish', 'silu']:
        return nn.SiLU(**kwargs)
    elif name == 'tanh':
        return nn.Tanh(**kwargs)
    elif name == 'sigmoid':
        return nn.Sigmoid(**kwargs)
    raise Exception(f'Activation: {name}')
