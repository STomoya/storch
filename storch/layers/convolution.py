
import torch.nn as nn


class DepthWiseConv2d(nn.Conv2d):
    '''Depth-wise convolution 2d is a convolution layer with groups=channel width.

    NOTE: in_channels = out_channels = groups
          and this is set using 'channels' argument.

    Arguments:
        channels: int
            Channel width of the input/output feature tensor.
        kernel_size: int
            Kernel size of the convolution weights.
        stride: int (default: 1)
            Stride.
        padding: int (default: 0)
            Padding.
        dilation: int (default: 1)
            Dilation.
        bias: bool (default: True)
            Whether to use bias or not.
        padding_mode: str (default: 'zeros')
            Padding mode.
    '''
    def __init__(self,
        channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'
    ) -> None:
        super().__init__(
            channels, channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=channels, bias=bias, padding_mode=padding_mode)

class PointWiseConv2d(nn.Conv2d):
    '''Point-wise convolution 2d is a convolution layer with 1x1 kernel.

    Arguments:
        in_channels: int
            Channel width of the input feature tensor.
        out_channels: int
            Channel width of the resulting tensor.
        bias: bool (default: True)
            Whether to use bias or not.
    '''
    def __init__(self,
        in_channels, out_channels, bias=True
    ) -> None:
        super().__init__(in_channels, out_channels, 1, bias=bias)

class DepthSepConv2d(nn.Module):
    '''Depth-wise separable convolution 2d splits the convolution to depth-wise convolution and
    point-wise convolution.

    Arguments:
        in_channels: int
            Channel width of the input feature tensor.
        out_channels: int
            Channel width of the resulting tensor.
        kernel_size: int
            Kernel size of the convolution weights.
        stride: int (default: 1)
            Stride.
        padding: int (default: 0)
            Padding.
        dilation: int (default: 1)
            Dilation.
        bias: bool (default: True)
            Whether to use bias or not.
        padding_mode: str (default: 'zeros')
            Padding mode.
    '''
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
