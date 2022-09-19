
import torch
import torch.nn as nn

import storch


def get_normalization2d(name: str, channels: int, affine: bool=None) -> nn.Module:
    """Get 2d normalization layers by name

    Args:
        name (str): Name of the normalization layer
        channels (int): Input tensor channel width
        affine (bool, optional): Whether to enable trainability. If None, uses the default behaviour. Default: None.

    Raises:
        Exception: Unknown normalization layer name.

    Returns:
        nn.Module: normalization layer module.
    """
    if   name == 'bn': return nn.BatchNorm2d(channels,    affine=storch.dynamic_default(affine, True))
    elif name == 'in': return nn.InstanceNorm2d(channels, affine=storch.dynamic_default(affine, False))
    elif name == 'ln': return nn.GroupNorm(1, channels,   affine=storch.dynamic_default(affine, True))
    elif name == 'gn': return nn.GroupNorm(16, channels,  affine=storch.dynamic_default(affine, True))
    raise Exception(f'Normalization: {name}')

def get_normalization1d(name: str, channels: int, affine: bool=None):
    """Get 1d normalization layers by name

    Args:
        name (str): Name of the normalization layer
        channels (int): Input tensor channel width
        affine (bool, optional): Whether to enable trainability. If None, uses the default behaviour. Default: None.

    Raises:
        Exception: Unknown normalization layer name.

    Returns:
        nn.Module: normalization layer module.
    """
    if   name == 'bn': return nn.BatchNorm1d(channels,    affine=storch.dynamic_default(affine, True))
    elif name == 'in': return nn.InstanceNorm1d(channels, affine=storch.dynamic_default(affine, False))
    elif name == 'ln': return nn.LayerNorm(channels,      affine=storch.dynamic_default(affine, True))
    elif name == 'gn': return get_normalization2d(name, channels, affine=storch.dynamic_default(affine, True))
    raise Exception(f'Normalization: {name}')


class AdaptiveNorm2d(nn.Module):
    """Adaptive Normalization Layer

    Args:
        norm_name (str): Name of the base normalization layer.
        style_dim (int): Dimension of the style vector
        channels (int): Input tensor channel width
        affine_layer (nn.Module, optional): nn.Module to transform style vector. Default: nn.Linear.
        biasfree (bool, optional): Bias free. Default: False.
    """
    def __init__(self,
        norm_name: str, style_dim: int, channels: int, affine_layer: nn.Module=nn.Linear, biasfree: bool=False
    ) -> None:
        super().__init__()
        self._norm_name = norm_name
        self._style_dim = style_dim
        self._channels  = channels
        self._biasfree  = biasfree

        self.norm = get_normalization2d(norm_name, channels, affine=False)

        if biasfree: scalebias = channels
        else:        scalebias = channels * 2
        self.affine = affine_layer(style_dim, scalebias, bias=False)
        self.affine_bias = nn.Parameter(torch.zeros(scalebias))
        self.affine_bias.data[:channels] = 1.

    def forward(self, x, style):
        scale_bias = self.affine(style) + self.affine_bias

        if self._biasfree:
            scale, bias = scale_bias[:, :, None, None], 0
        else:
            scale, bias = scale_bias[:, :, None, None].chunk(2, dim=1)

        norm = self.norm(x)
        return scale * norm + bias

    def extra_repr(self):
        return f'norm_name={self._norm_name}, style_dim={self._style_dim}, channels={self._channels}, biasfree={self._biasfree}'
