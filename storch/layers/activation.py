
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Get activation layer by name

    NOTE: - Always inplace=True.
          - negative_slope=0.2 for LeakyReLU

    Args:
        name (str): Name of the activation function.

    Raises:
        Exception: Unknown name.

    Returns:
        nn.Module: The activation function module.
    """
    if   name == 'relu':  return nn.ReLU(True)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, True)
    elif name == 'prelu': return nn.PReLU()
    elif name == 'gelu':  return nn.GELU()
    elif name in ['swish', 'silu']: return nn.SiLU(True)
    elif name == 'tanh':  return nn.Tanh()
    elif name == 'sigmoid': return nn.Sigmoid()
    raise Exception(f'Activation: {name}')
