
import torch

class MinibatchStdDev(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self._group_size = group_size
        self._num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.group_size if N % self.group_size == 0 else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2,3,4])
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        x = torch.cat([x, y], dim=1)

        return x

    def extra_repr(self):
        return f'group_size={self._group_size}, num_channels={self._num_channels}'
