"""Mini-batch stadard deviation."""

import torch


class MinibatchStdDev(torch.nn.Module):
    """Mini-batch standard deviation.

    Args:
        group_size (int): Size of the group to calculate the statistics.
        num_channels (int, optional): Number of channels to be appended. Defaults to 1.

    """

    def __init__(self, group_size: int, num_channels: int = 1) -> None:  # noqa: D107
        super().__init__()
        self._group_size = group_size
        self._num_channels = num_channels

    def forward(self, x):  # noqa: D102
        N, C, H, W = x.shape
        G = self._group_size if N % self._group_size == 0 else N
        F = self._num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4])
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        x = torch.cat([x, y], dim=1)

        return x

    def extra_repr(self):  # noqa: D102
        return f'group_size={self._group_size}, num_channels={self._num_channels}'
