"""Interpolation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_blur_kernel(filter_size: int) -> torch.Tensor:
    """Integer aproximation of gaussian kernel.

    Args:
        filter_size (int): The size of the filter.

    Returns:
        torch.Tensor: The resampling filter.

    """

    def _pascal_triangle():
        """Return binomial filter from size."""

        def c(n, k):
            if k <= 0 or n <= k:
                return 1
            else:
                return c(n - 1, k - 1) + c(n - 1, k)

        return [c(filter_size - 1, j) for j in range(filter_size)]

    filter = torch.tensor(_pascal_triangle(), dtype=torch.float32)
    filter2d = torch.outer(filter, filter)
    filter2d /= filter2d.sum()
    return filter2d.unsqueeze(0).unsqueeze(0)


class Blur(nn.Module):
    """Blur layer used in StyleGANs.

    Args:
        filter_size (int): Size of the low pass filter. Default: 4

    """

    def __init__(self, filter_size: int = 4) -> None:  # noqa: D107
        super().__init__()
        self._filter_size = filter_size
        self.register_buffer('kernel', _make_blur_kernel(filter_size))

        if filter_size % 2 == 0:
            pad1, pad2 = (filter_size - 1) // 2, filter_size // 2
        else:
            pad1 = pad2 = filter_size // 2
        self.padding = (pad1, pad2, pad1, pad2)

    def forward(self, x):  # noqa: D102
        C = x.size(1)
        x = F.pad(x, self.padding)
        x = F.conv2d(x, self.kernel.expand(C, -1, -1, -1), groups=C)
        return x

    def extra_repr(self):  # noqa: D102
        return f'filter_size={self._filter_size}'


class BlurUpsample(nn.Sequential):
    """Upsample then blur.

    Args:
        filter_size (int, optional): Size of the low pass filter. Default: 4.
        scale_factor (int, optional): Scale factor for upsampling. Default: 2.
        mode (str, optional): Upsampling mode. Default: 'bilinear'.
        align_corners (bool, optional): Align corners. Default: True.

    """

    def __init__(  # noqa: D107
        self, filter_size: int = 4, scale_factor: int = 2, mode: str = 'bilinear', align_corners: bool = True
    ) -> None:
        super().__init__(
            nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners), Blur(filter_size)
        )


class BlurDownsample(nn.Sequential):
    """Blur then downsample.

    Args:
        filter_size (int, optional): Size of the low pass filter. Default: 4.
        scale (int, optional):  Scale for downsampling. Default: 2.

    """

    def __init__(self, filter_size: int = 4, scale: int = 2) -> None:  # noqa: D107
        super().__init__(Blur(filter_size), nn.AvgPool2d(scale))


class AABilinearInterp(nn.Module):
    """Bilinear interpolation with antialias option enabled.

    Args:
        size (int | None, optional): The output size. Default: None.
        scale_factor (int | None, optional): Scale factor.. Default: None.

    """

    def __init__(self, size: int | None = None, scale_factor: int | None = None) -> None:  # noqa: D107
        super().__init__()
        assert size is not None or scale_factor is not None
        self._size = size
        self._scale_factor = scale_factor

    def forward(self, x):  # noqa: D102
        return F.interpolate(
            x, size=self._size, scale_factor=self._scale_factor, mode='bilinear', align_corners=True, antialias=True
        )

    def extra_repr(self):  # noqa: D102
        return f'size={self._size}, scale_factor={self._scale_factor}'
