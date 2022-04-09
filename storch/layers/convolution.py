
import torch.nn as nn


class DepthWiseConv2d(nn.Conv2d):
    def __init__(self,
        channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'
    ) -> None:
        super().__init__(
            channels, channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=channels, bias=bias, padding_mode=padding_mode)

class PointWiseConv2d(nn.Conv2d):
    def __init__(self,
        in_channels, out_channels, bias=True
    ) -> None:
        super().__init__(in_channels, out_channels, 1, bias=bias)

class DepthSepConv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, bias=True, padding_mode='zeros'
    ) -> None:
        super().__init__()
        self.dwconv = DepthWiseConv2d(in_channels, kernel_size, stride, padding, dilation, False, padding_mode)
        self.pwconv = PointWiseConv2d(in_channels, out_channels, bias)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x
