
import torch.nn as nn


def get_activation(name: str, **kwargs) -> nn.Module:
    """Get activation layer by name.

    Args:
        name (str): Name of the activation function.

    Raises:
        Exception: Unknown name.

    Returns:
        nn.Module: The activation function module.
    """
    if   name == 'relu':  return nn.ReLU(**kwargs)
    elif name == 'lrelu': return nn.LeakyReLU(**kwargs)
    elif name == 'prelu': return nn.PReLU(**kwargs)
    elif name == 'gelu':  return nn.GELU(**kwargs)
    elif name in ['swish', 'silu']: return nn.SiLU(**kwargs)
    elif name == 'tanh':  return nn.Tanh(**kwargs)
    elif name == 'sigmoid': return nn.Sigmoid(**kwargs)
    raise Exception(f'Activation: {name}')
