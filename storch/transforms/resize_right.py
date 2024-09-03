"""Resize right."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from storch.transforms.ResizeRight import interp_methods, resize_right

INTERP_METHODS = {
    'cubic': interp_methods.cubic,
    'box': interp_methods.box,
    'linear': interp_methods.linear,
    'lanczos2': interp_methods.lanczos2,
    'lanczos3': interp_methods.lanczos3,
}


def resize(
    input: np.ndarray | torch.Tensor,
    scale_factors: int | None = None,
    out_shape: tuple[int] | None = None,
    interp_method: str | Callable = interp_methods.cubic,
    support_sz: int | None = None,
    antialiasing: bool = True,
    by_convs: bool = False,
    scale_tolerance: float | None = None,
    max_numerator: int = 10,
    pad_mode: str = 'constant',
) -> np.ndarray | torch.Tensor:
    """Wrap storch.transforms.ResizeRight.resize_right.resize, supporting string for interp_method arg.

    Args:
        input (np.ndarray | torch.Tensor): input image to resize.
        scale_factors (int, optional): scaler factors. Defaults to None.
        out_shape (tuple[int], optional): output shape. Defaults to None.
        interp_method (str | Callable, optional): interpolation method. One of
            {cubic,box,linear,lanczos2,lanczos3}. Defaults to interp_methods.cubic.
        support_sz (int, optional): support size. Defaults to None.
        antialiasing (bool, optional): antialiasing. Defaults to True.
        by_convs (bool, optional): use convolution. Defaults to False.
        scale_tolerance (float, optional): scale tolerance. Defaults to None.
        max_numerator (int, optional): max numerator. Defaults to 10.
        pad_mode (str, optional): padding mode. Defaults to 'constant'.

    Returns:
        (np.ndarray | torch.Tensor): resized image

    """
    if isinstance(interp_method, str):
        interp_method = INTERP_METHODS[interp_method]
    return resize_right.resize(
        input,
        scale_factors,
        out_shape,
        interp_method,
        support_sz,
        antialiasing,
        by_convs,
        scale_tolerance,
        max_numerator,
        pad_mode,
    )
