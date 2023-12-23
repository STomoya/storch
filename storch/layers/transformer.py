"""Transformer block."""

import torch
import torch.nn as nn


class MetaformerBlock(nn.Module):
    """Pass in nn.Module objects to construct a transformer block.

    Configuration:
        - pre-normalization.
        - trainable layer scaling with initial value of 1e-2

    Args:
    ----
        token_mixer (nn.Module): nn.Module which implements an attention.
        feedforward (nn.Module): nn.Module which implements a feed forward network.
        norm_layer1 (nn.Module): Normalization layer applied before the attention.
        norm_layer2 (nn.Module): Normalization layer applied before the feed forward network.
    """

    def __init__(  # noqa: D107
        self, token_mixer: nn.Module, feedforward: nn.Module, norm_layer1: nn.Module, norm_layer2: nn.Module
    ) -> None:
        super().__init__()
        should_be_module = (token_mixer, feedforward, norm_layer1, norm_layer2)
        assert all(isinstance(module, nn.Module) for module in should_be_module)
        self.norm1 = norm_layer1
        self.token_mixer = token_mixer
        self.norm2 = norm_layer2
        self.feedforward = feedforward

        self.scale_mixer = nn.Parameter(torch.ones([]) * 1e-2)
        self.scale_ff = nn.Parameter(torch.ones([]) * 1e-2)

    def forward(self, x):  # noqa: D102
        x = x + self.scale_mixer * self.token_mixer(self.norm1(x))
        x = x + self.scale_ff * self.feedforward(self.norm2(x))
        return x
