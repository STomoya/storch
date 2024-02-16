"""Cutout."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def random_box(size: tuple, lambda_: float) -> tuple[tuple[int], float]:
    """Make a random box within size.

    Args:
    ----
        size (tuple): size of the image.
        lambda_ (float): lambda sampled from beta.

    Returns:
    -------
        tuple[int]: xyxy
        float: adjusted lambda

    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1.0 - lambda_)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return (x1, y1, x2, y2)


def cutout(
    images: torch.Tensor, alpha: float = 1.0, p: float = 0.5, filler: Callable = torch.zeros, sample_wise: bool = True
) -> torch.Tensor:
    """Cutout augmentation.

    Args:
    ----
        images (torch.Tensor): Tensor of images to apply Cutout to.
        alpha (float, optional): Parameter for sampling random numbers from the Beta distribution. Default: 1.0.
        p (float, optional): Probability to apply Cutout to images.
            If sample_wise is true, determined on every sample. Default: 0.5.
        filler (Callable, optional): A function which ganerates a tensor to fill the cropped out box.
            Requires the function to follow the argument format of tensor making functions in PyTorch.
            Default: torch.zeros.
        sample_wise (bool, optional): Make a mask for each samples in the batch. Default: True.

    Returns:
    -------
        torch.Tensor: The mixed image.

    """
    B, C, W, H = images.size()

    mask = torch.ones(B, 1, W, H, device=images.device)
    if sample_wise:
        for i in range(B):
            if np.random.random(1) < p:
                lambda_ = np.random.beta(alpha, alpha)
                xyxy = random_box((W, H), lambda_)
                mask[i, :, xyxy[0] : xyxy[2], xyxy[1] : xyxy[3]] = 0.0
    elif np.random.random(1) < p:
        lambda_ = np.random.beta(alpha, alpha)
        xyxy = random_box((W, H), lambda_)
        mask[:, :, xyxy[0] : xyxy[2], xyxy[1] : xyxy[3]] = 0.0

    images = images * mask + filler(B, C, W, H, device=images.device, dtype=images.dtype) * (1 - mask)
    return images
