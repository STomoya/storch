
import torch.nn as nn


class DepthWiseConv2d(nn.Conv2d):
    """Depth-wise convolution 2d is a convolution layer with groups=channel width.

    Args:
        channels (int): Channel width of the input/output feature tensor.
        kernel_size (int): Kernel size of the convolution weights.
        stride (int, optional): Stride. Default: 1.
        padding (int, optional): Padding. Default: 0.
        dilation (int, optional): Dilation. Default: 1.
        bias (bool, optional): Whether to use bias or not. Default: True.
        padding_mode (str, optional): Padding mode. Default: 'zeros'.
    """
    def __init__(self,
        channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1,
        bias: bool=True, padding_mode: str='zeros'
    ) -> None:
        super().__init__(
            channels, channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=channels, bias=bias, padding_mode=padding_mode)

class PointWiseConv2d(nn.Conv2d):
    """Point-wise convolution 2d is a convolution layer with 1x1 kernel.

    Args:
        in_channels (int): Channel width of the input feature tensor.
        out_channels (int): Channel width of the resulting tensor.
        bias (bool, optional): Whether to use bias or not. Default: True.
    """
    def __init__(self,
        in_channels: int, out_channels: int, bias: bool=True
    ) -> None:
        super().__init__(in_channels, out_channels, 1, bias=bias)

class DepthSepConv2d(nn.Module):
    """Depth-wise separable convolution 2d splits the convolution to depth-wise convolution and
    point-wise convolution.

    Args:
        in_channels (int): Channel width of the input feature tensor.
        out_channels (int): Channel width of the resulting tensor.
        kernel_size (int): Kernel size of the convolution weights.
        stride (int, optional): Stride. Default: 1.
        padding (int, optional): Padding. Default: 0.
        dilation (int, optional): Dilation. Default: 1.
        bias (bool, optional): Whether to use bias or not. Default: True.
        padding_mode (str, optional): Padding mode. Default: 'zeros'.
    """
    def __init__(self,
        in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0,
        dilation: int=1, bias: bool=True, padding_mode: str='zeros'
    ) -> None:
        super().__init__()
        self.dwconv = DepthWiseConv2d(in_channels, kernel_size, stride, padding, dilation, False, padding_mode)
        self.pwconv = PointWiseConv2d(in_channels, out_channels, bias)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x
