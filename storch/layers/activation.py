
import torch.nn as nn


def get_activation(name):
    if   name == 'relu':  return nn.ReLU(True)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, True)
    elif name == 'prelu': return nn.PReLU()
    elif name == 'gelu':  return nn.GELU()
    elif name in ['swish', 'silu']: return nn.SiLU(True)
    elif name == 'tanh':  return nn.Tanh()
    elif name == 'sigmoid': return nn.Sigmoid()
    raise Exception(f'Activation: {name}')
