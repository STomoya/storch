
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    '''Pass in nn.Module objects to construct a transformer block'''
    def __init__(self,
        attention, feedforward, norm_layer1, norm_layer2
    ) -> None:
        super().__init__()
        should_be_module = (attention, feedforward, norm_layer1, norm_layer2)
        assert all([isinstance(module, nn.Module) for module in should_be_module])
        self.norm1       = norm_layer1
        self.attention   = attention
        self.norm2       = norm_layer2
        self.feedforward = feedforward

        self.scale_attn = nn.Parameter(torch.ones([]) * 1e-2)
        self.scale_ff   = nn.Parameter(torch.ones([]) * 1e-2)

    def forward(self, x):
        x = x + self.scale_attn * self.attention(self.norm1(x))
        x = x + self.scale_ff   * self.feedforward(self.norm2(x))
        return x
