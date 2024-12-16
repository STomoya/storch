"""ViT attn map."""

from __future__ import annotations

from functools import partial
from types import MethodType
from typing import Any, Callable

import torch
import torch.nn as nn
from timm.models import VisionTransformer

from storch.imageops import tensor2heatmap


def _collect_attn_maps(attn_layers: list[nn.Module]):
    attn_maps = []
    for layer in attn_layers:
        attn_maps.append(layer._attn_map)
    return attn_maps


def _recover_original_model(attn_layers: list[nn.Module]):
    for layer in attn_layers:
        orig_forward = layer._original_forward
        layer.forward = orig_forward
        del layer._original_forward
        if hasattr(layer, '_attn_map'):
            del layer._attn_map


def replace_vit_attn_forward(
    vit_model: VisionTransformer,
) -> tuple[Callable[[], list[torch.Tensor]], Callable[[], None]]:
    """Replace forward of Attention layers in ViT.

    Args:
        vit_model (VisionTransformer): This ViT model.

    Returns:
        (tuple[Callable[[], list[torch.Tensor]], Callable[[], None]]): Func to collect attention maps from Attention
            layers and func to convert back to the original model.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # torch SDPA is always disabled to collect attention maps.

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self._attn_map = attn
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    attn_layers = []
    for block in vit_model.blocks.children():
        block.attn._original_forward = block.attn.forward
        block.attn.forward = MethodType(forward, block.attn)
        attn_layers.append(block.attn)

    collect_attn_maps = partial(_collect_attn_maps, attn_layers)
    recover_original_model = partial(_recover_original_model, attn_layers)

    return collect_attn_maps, recover_original_model


def create_vit_heat_maps(
    attn_maps: list[torch.Tensor],
    num_prefix_tokens: int,
    image_tokens_size: tuple[int, int],
    image_size: tuple[int, int] | None = None,
    token_index: int | None = None,
    include_avg: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert attention maps to heat maps.

    Args:
        attn_maps (list[torch.Tensor]): The collected attn maps.
        num_prefix_tokens (int): Number of prefix tokens. VisionTransformer.num_prefix_tokens.
        image_tokens_size (tuple[int, int]): Shape of image tokens in 2D.
        image_size (tuple[int, int], optional): input image size. Heat maps will be resized to this size. Default: None.
        token_index (int, optional): The index of the token to create the heat map on. Default: None.
        include_avg (bool, optional): include averaged image. Default: True.

    Returns:
        (dict[str, torch.Tensor]): dict of heat maps.

    """
    token_index = token_index or 0
    heat_maps = {}
    for index, attn_map in enumerate(attn_maps, 1):
        # [B,H,L,L]
        # Mean over heads
        attn_map = attn_map.mean(dim=1)  # noqa: PLW2901
        # target token
        target_token = attn_map[:, token_index, num_prefix_tokens:]
        target_token = target_token.reshape(-1, 1, *image_tokens_size)
        # as heat map
        heat_map = tensor2heatmap(
            target_token, image_size if image_size is not None else image_tokens_size, normalize=False
        )

        heat_maps[index] = heat_map.squeeze().clone().cpu()

    if include_avg:
        heat_maps['avg'] = torch.stack(list(heat_maps.values())).mean(dim=0)
    return heat_maps


def create_attn_heat_maps(
    model: VisionTransformer,
    input: torch.Tensor | tuple[Any, ...],
    target_token_index: int = 0,
    return_avg: bool = False,
    resize: bool = True,
) -> dict[str, torch.Tensor]:
    """Create heat maps of softmax attention.

    Currently only objects of VisionTransformer class of timm is supportted.

    Args:
        model (VisionTransformer): The ViT model.
        input (torch.Tensor | tuple[Any, ...]): input Tensor or tuple of tensors. The first element must be the image.
        target_token_index (int, optional): The index of the token to create the heat map on. Default: 0 (CLS token).
        return_avg (bool, optional): include averaged image. Default: False.
        resize (bool, optional): resize the heat map. Default: True.

    Returns:
        (dict[str, torch.Tensor]): dict of heat maps.

    """
    assert isinstance(model, VisionTransformer), 'This function only supports timm.models.VisionTransformer class.'

    if isinstance(input, torch.Tensor):
        input = (input,)

    image_size = input[0].size()[-2:]
    image_tokens_size = model.patch_embed.dynamic_feat_size(image_size)
    num_prefix_tokens = model.num_prefix_tokens

    collect_attn_maps, recover_original_model = replace_vit_attn_forward(model)

    model(*input)
    attn_maps = collect_attn_maps()
    recover_original_model()

    heat_maps = create_vit_heat_maps(
        attn_maps=attn_maps,
        num_prefix_tokens=num_prefix_tokens,
        image_tokens_size=image_tokens_size,
        image_size=image_size if resize else None,
        token_index=target_token_index,
        include_avg=return_avg,
    )

    return heat_maps
