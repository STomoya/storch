
import torch.nn as nn

from storch.layers import get_activation


class SpatialSqueezeAndExcitation(nn.Module):
    '''Spatial squeeze and excitation block.
    This is equivalent to the original squeeze and excitation block, but with some options.

    NOTE: - if with default values => original SE block.
          - if pool_size=4, act_name='swish' => SE block used in FastGAN.

    Arguments:
        channels : int
            Channel width of the input feature vector.
        reduction: int (default: 4)
            Reduction factor.
        pool_size: int (default: 4)
            Adaptive average pooling size.
        act_name: str (default: 'relu')
            Activation function name.
    '''
    def __init__(self,
        channels, reduction=4, pool_size=1, act_name='relu'
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


class ChannelSqueezeAndExcitation(nn.Module):
    '''Channel squeeze and excitation block.
    Attention.

    Arguments:
        channels: int
            Channel width of the input feature vector.
    '''
    def __init__(self,
        channels
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
    '''Spatial channel squeeze and excitation block.

    Arguments:
        channels : int
            Channel width of the input feature vector.
        reduction: int (default: 4)
            Reduction factor.
        pool_size: int (default: 4)
            Adaptive average pooling size.
        act_name: str (default: 'relu')
            Activation function name.
    '''
    def __init__(self,
        channels, reduction=4, pool_size=1, act_name='relu'
    ) -> None:
        super().__init__()
        self.spatial_se = SpatialSqueezeAndExcitation(channels, reduction, pool_size, act_name)
        self.channel_se = ChannelSqueezeAndExcitation(channels)

    def forward(self, x):
        s = self.spatial_se(x)
        c = self.channel_se(x)
        x = s + c
        return x


# alias
sSE = SpatialSqueezeAndExcitation
cSE = ChannelSqueezeAndExcitation
scSE = SpatialChannelSqueezeAndExcitation
