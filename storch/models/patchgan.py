
import torch.nn as nn

from storch import layers


class PatchDiscriminator(nn.Sequential):
    '''PatchGAN D'''
    def __init__(self,
        in_channels: int, num_layers: int=3, channels: int=64, norm_name: str='bn', act_name: str='lrelu'
    ) -> None:
        bias = norm_name != 'bn'
        modules = [
            nn.Conv2d(in_channels, channels, 4, 2, 1, bias=bias),
            layers.get_activation(act_name)]
        for _ in range(num_layers):
            modules.extend([
                nn.Conv2d(channels, channels*2, 4, 2, 1, bias=bias),
                layers.get_normalization2d(norm_name, channels*2),
                layers.get_activation(act_name)])
            channels *= 2
        modules.extend([
            nn.Conv2d(channels, channels*2, 4, 1, 1, bias=bias),
            layers.get_normalization2d(norm_name, channels*2),
            layers.get_activation(act_name),
            nn.Conv2d(channels*2, 1, 4, 1, 1, bias=bias)])
        super().__init__(*modules)
