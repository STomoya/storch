"""PatchGAN D."""

import torch.nn as nn

from storch import layers
from storch.models import ModelMixin


class PatchDiscriminator(ModelMixin):
    """PatchGAN D."""

    def __init__(
        self, in_channels: int, num_layers: int = 3, channels: int = 64, norm_name: str = 'bn', act_name: str = 'lrelu'
    ) -> None:
        """PatchGAN D.

        Args:
        ----
            in_channels (int): Channel width of the input image.
            num_layers (int, optional): Number of strided convolution layers. Default: 3.
            channels (int, optional): Base channel width. Default: 64.
            norm_name (str, optional): Normalization layer name. Default: 'bn'.
            act_name (str, optional): Activation function name. Default: 'lrelu'.

        """
        super().__init__()
        bias = norm_name != 'bn'
        modules = [nn.Conv2d(in_channels, channels, 4, 2, 1, bias=bias), layers.get_activation(act_name)]
        for _ in range(num_layers):
            modules.extend(
                [
                    nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=bias),
                    layers.get_normalization2d(norm_name, channels * 2),
                    layers.get_activation(act_name),
                ]
            )
            channels *= 2
        modules.extend(
            [
                nn.Conv2d(channels, channels * 2, 4, 1, 1, bias=bias),
                layers.get_normalization2d(norm_name, channels * 2),
                layers.get_activation(act_name),
                nn.Conv2d(channels * 2, 1, 4, 1, 1, bias=bias),
            ]
        )
        self.net = nn.Sequential(*modules)

    def forward(self, x):  # noqa: D102
        return self.net(x)
