"""Multiscale D."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class MultiScale(nn.Module):
    """Multi Scale.

    Only supports models with same structure.
    """

    def __init__(
        self,
        build_model: Callable,
        num_scales: int = 2,
        downsample: Callable | None = None,
        collate_fn: Callable | None = None,
    ) -> None:
        """Multi scale D.

        Args:
            build_model (Callable): Func that builds the model. Requires to have no arguments.
                Use functools.partial, etc.
            num_scales (int, optional): Number of scales. Default: 2.
            downsample (Callable | None, optional): Callable to downsample the input tensor.
                If not specified, nn.AvgPool2d(2) will be used. Default: None.
            collate_fn (Callable | None, optional): Func to collate the outputs. Default: None.

        """
        super().__init__()
        self._num_scales = num_scales
        self.downsample = downsample if downsample is not None else nn.AvgPool2d(2)

        self.models = nn.ModuleList()
        for _ in range(num_scales):
            self.models.append(build_model())

        self.collate = collate_fn if collate_fn is not None else self.default_collate

    def forward(self, x):  # noqa: D102
        out = []
        for i in range(self._num_scales):
            out.append(self.models[i](x))
            if i != self._num_scales - 1:
                x = self.downsample(x)
        return self.collate(out)

    def default_collate(self, out: list):  # noqa: D102
        if isinstance(out[0], torch.Tensor):
            return torch.cat([x.flatten() for x in out])
        elif isinstance(out[0], (tuple, list)):
            collated = []
            for i in range(len(out[0])):
                collated.append(self.collate([x[i] for x in out]))
            return tuple(collated)
