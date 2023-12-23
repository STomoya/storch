"""Normalization."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn


def get_normalization2d(name: str, channels: int, **kwargs) -> nn.Module:
    """Get 2d normalization layers by name.

    Args:
    ----
        name (str): Name of the normalization layer
        channels (int): Input tensor channel width
        **kwargs: extra args.

    Raises:
    ------
        Exception: Unknown normalization layer name.

    Returns:
    -------
        nn.Module: normalization layer module.
    """
    if name == 'bn':
        return nn.BatchNorm2d(channels, **kwargs)
    elif name == 'in':
        return nn.InstanceNorm2d(channels, **kwargs)
    elif name == 'ln':
        return LayerNorm2d(channels, **kwargs)
    elif name == 'gn':
        if 'num_groups' not in kwargs:
            raise Exception(f'Normalization: "{name}" requires "num_groups" argument.')
        return nn.GroupNorm(num_channels=channels, **kwargs)
    raise Exception(f'Normalization: {name}')


def get_normalization1d(name: str, channels: int, **kwargs) -> nn.Module:
    """Get 1d normalization layers by name.

    Args:
    ----
        name (str): Name of the normalization layer
        channels (int): Input tensor channel width
        **kwargs: extra args.

    Raises:
    ------
        Exception: Unknown normalization layer name.

    Returns:
    -------
        nn.Module: normalization layer module.
    """
    if name == 'bn':
        return nn.BatchNorm1d(channels, **kwargs)
    elif name == 'in':
        return nn.InstanceNorm1d(channels, **kwargs)
    elif name == 'ln':
        return nn.LayerNorm(channels, **kwargs)
    elif name == 'gn':
        if 'num_groups' not in kwargs:
            raise Exception(f'Normalization: "{name}" requires "num_groups" argument.')
        return nn.GroupNorm(num_channels=channels, **kwargs)
    raise Exception(f'Normalization: {name}')


class LayerNorm2d(nn.Module):
    """LayerNorm2d."""

    __constants__: ClassVar[list[str, str, str]] = ['channels', 'eps', 'elementwise_affine']
    channels: int
    eps: float
    elementwise_affine: bool

    def __init__(self, channels: int, eps: float = 1e-5, elementwise_affine: bool = True, device=None, dtype=None):  # noqa: D107
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            factory_kwargs = dict(device=device, dtype=dtype)
            self.weight = nn.Parameter(torch.empty(channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(channels, **factory_kwargs))
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)

        self.reset_parameters()

    def reset_parameters(self):  # noqa: D102
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):  # noqa: D102
        input_dtype = x.dtype
        x = x.float()
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = x.to(dtype=input_dtype)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class AdaptiveNorm2d(nn.Module):
    """Adaptive Normalization Layer.

    Args:
    ----
        norm_name (str): Name of the base normalization layer.
        style_dim (int): Dimension of the style vector
        channels (int): Input tensor channel width
        affine_layer (nn.Module, optional): nn.Module to transform style vector. Default: nn.Linear.
        biasfree (bool, optional): Bias free. Default: False.
    """

    def __init__(  # noqa: D107
        self, norm_name: str, style_dim: int, channels: int, affine_layer: nn.Module = nn.Linear, biasfree: bool = False
    ) -> None:
        super().__init__()
        self._norm_name = norm_name
        self._style_dim = style_dim
        self._channels = channels
        self._biasfree = biasfree

        self.norm = get_normalization2d(norm_name, channels, affine=False)

        if biasfree:
            scalebias = channels
        else:
            scalebias = channels * 2
        self.affine = affine_layer(style_dim, scalebias, bias=False)
        self.affine_bias = nn.Parameter(torch.zeros(scalebias))
        self.affine_bias.data[:channels] = 1.0

    def forward(self, x, style):  # noqa: D102
        scale_bias = self.affine(style) + self.affine_bias

        if self._biasfree:
            scale, bias = scale_bias[:, :, None, None], 0
        else:
            scale, bias = scale_bias[:, :, None, None].chunk(2, dim=1)

        norm = self.norm(x)
        return scale * norm + bias

    def extra_repr(self):  # noqa: D102
        return (
            f'norm_name={self._norm_name}, '
            f'style_dim={self._style_dim}, '
            f'channels={self._channels}, '
            f'biasfree={self._biasfree}'
        )
