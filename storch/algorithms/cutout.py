
from typing import Callable

import numpy as np
import torch


def random_box(size: tuple, lambda_: float):
    '''Make a random box within size'''
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lambda_)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return (x1, y1, x2, y2)


def cutout(
    images: torch.Tensor, alpha: float=1.0, p: float=0.5, filler: Callable=torch.zeros, sample_wise: bool=True
) -> torch.Tensor:
    '''Cutout augmentation

    Arguments:
        images: torch.Tensor
            Tensor of images to apply Cutout to.
        alpha: float (default: 1.0)
            Parameter for sampling random numbers from the Beta distribution.
        p: float (default: 0.5)
            Probability to apply Cutout to images.
            If sample_wise is true, determined on every sample.
        filler: Callable (default: torch.zeros)
            A function which ganerates a tensor to fill the cropped out box.
            Requires the function to follow the argument format of tensor making functions in PyTorch.
        sample_wise: bool (default: True)
            Make a mask for each samples in the batch.
    '''
    B, C, W, H = images.size()

    mask = torch.ones(B, 1, W, H, device=images.device)
    if sample_wise:
        for i in range(B):
            if np.random.random(1) < p:
                lambda_ = np.random.beta(alpha, alpha)
                xyxy = random_box((W, H), lambda_)
                mask[i, :, xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]] = 0.0
    elif np.random.random(1) < p:
        lambda_ = np.random.beta(alpha, alpha)
        xyxy = random_box((W, H), lambda_)
        mask[:, :, xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]] = 0.0

    images = images * mask + filler(B, C, W, H, device=images.device, dtype=images.dtype) * (1 - mask)
    return images
