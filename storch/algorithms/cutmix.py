
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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

    adjusted_lambda = 1 - ((x2 - x1) * (y2 - y1) / (H * W))

    return (x1, y1, x2, y2), adjusted_lambda


def cutmix(
    images: torch.Tensor, targets: torch.Tensor, alpha: float=0.2, p: float=1.0, sample_wise: bool=True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''CutMix augmentation

    Arguments:
        images: torch.Tensor
            Tensor of images to apply CutMix to.
        targets: torch.Tensor
            Tensor of targets.
        alpha: float (default: 1.0)
            Parameter for sampling random numbers from the Beta distribution.
        p: float (default: 0.5)
            Probability to apply CutMix to images.
            If sample_wise is true, determined on every sample.
        sample_wise: bool (default: True)
            Make a mask for each samples in the batch.

    Returns:
        mixed: torch.Tensor
            The mixed image.
        target: torch.Tensor
            The target in the order of the shuffled images.
        lambdas: torch.Tensor
            The lambda used to make the mask.
    '''
    B, _, W, H = images.size()

    mask = torch.ones(B, 1, W, H, device=images.device)
    lambdas = torch.ones(B, device=images.device)

    if sample_wise:
        # make mask for each sample in the batch
        for i in range(B):
            if np.random.random(1) < p:
                lambda_ = np.random.beta(alpha, alpha)
                xyxy, adjusted_lambda = random_box((W, H), lambda_)
                mask[i, :, xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]] = 0.0
                lambdas[i] = adjusted_lambda

    elif np.random.random(1) < p:
        # share a mask for all samples in the batch
        lambda_ = np.random.beta(alpha, alpha)
        xyxy, adjusted_lambda = random_box((W, H), lambda_)
        mask[:, :, xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]] = 0.0
        lambdas = lambdas * adjusted_lambda


    permutation = torch.randperm(B, device=images.device)
    mixed = images * mask + images[permutation] * (1 - mask)

    return mixed, targets[permutation], lambdas


def mixed_cross_entropy_loss(logits, targets_a, targets_b, lambdas):
    '''Cross entropy loss for mixed images

    Arguments:
        logits: torch.Tensor
            Output logits of the model.
        targets_a: torch.Tensor
            targets of the original batch.
        targets_b: torch.Tensor
            targets returned by cutmix()
        lambdas: torch.Tensor
            lambdas returned by cutmix()

    Returns:
        loss: torch.Tensor
            The calculated loss.
    '''
    ce_loss_a = F.cross_entropy(logits, targets_a, reduction='none') * lambdas
    ce_loss_b = F.cross_entropy(logits, targets_b, reduction='none') * (1 - lambdas)
    loss = (ce_loss_a + ce_loss_b).mean()
    return loss
