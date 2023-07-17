
import torch.nn as nn

from storch.layers import get_activation


class SpatialSqueezeAndChannelExcitation(nn.Module):
    """Spatial squeeze and excitation block.
    This is equivalent to the original squeeze and excitation block, but with some options.

    NOTE: - if with default values => original SE block.
        - if pool_size=4, act_name='swish' => SE block used in FastGAN.

    Args:
        channels (int): Channel width of the input feature vector.
        reduction (int, optional): Reduction factor. Default: 4.
        pool_size (int, optional): Adaptive average pooling size. Default: 1.
        act_name (str, optional): Activation function name. Default: 'relu'.
    """
    def __init__(self,
        channels: int, reduction: int=4, pool_size: int=1, act_name: str='relu'
    ) -> None:
        super().__init__()
        self.global_content = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(channels, channels//reduction, pool_size, bias=False),
            get_activation(act_name),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            get_activation('sigmoid')
        )

    def forward(self, x):
        x = x * self.global_content(x)
        return x


class ChannelSqueezeAndSpatialExcitation(nn.Module):
    """Channel squeeze and excitation block.
    Attention.

    Args:
        channels (int): Channel width of the input feature vector.
    """
    def __init__(self,
        channels: int
    ) -> None:
        super().__init__()
        self.global_content = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            get_activation('sigmoid')
        )

    def forward(self, x):
        x = x * self.global_content(x)
        return x


class SpatialChannelSqueezeAndExcitation(nn.Module):
    """Spatial channel squeeze and excitation block.

    Args:
        channels (int): Channel width of the input feature vector.
        reduction (int, optional): Reduction factor. Default: 4.
        pool_size (int, optional): Adaptive average pooling size. Default: 1.
        act_name (str, optional): Activation function name. Default: 'relu'.
    """
    def __init__(self,
        channels: int, reduction: int=4, pool_size: int=1, act_name: str='relu'
    ) -> None:
        super().__init__()
        self.spatial_se = SpatialSqueezeAndChannelExcitation(channels, reduction, pool_size, act_name)
        self.channel_se = ChannelSqueezeAndSpatialExcitation(channels)

    def forward(self, x):
        s = self.spatial_se(x)
        c = self.channel_se(x)
        x = s + c
        return x


# alias
SE = cSE = SpatialSqueezeAndChannelExcitation
sSE = ChannelSqueezeAndSpatialExcitation
scSE = SpatialChannelSqueezeAndExcitation
