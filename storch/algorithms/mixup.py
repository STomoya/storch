"""Mixup."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mixup(
    images: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2, p: float = 1.0, sample_wise: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixup augmentation.

    Args:
    ----
        images (torch.Tensor): Tensor of images to apply Mixup to.
        targets (torch.Tensor): Tensor of targets.
        alpha (float, optional): Parameter for sampling random numbers from the Beta distribution. Default: 0.2.
        p (float, optional): Probability to apply Mixup to images. If sample_wise is true, determined on every sample.
            Default: 1.0.
        sample_wise (bool, optional): Make a mask for each samples in the batch. Default: True.

    Returns:
    -------
        torch.Tensor: The mixed image.
        torch.Tensor: The target in the order of the shuffled images.
        torch.Tensor: The lambda used to make the mask.
    """
    B = images.size(0)

    lambdas = torch.ones(B, 1, 1, 1, device=images.device)
    if sample_wise:
        for i in range(B):
            if np.random.random(1) < p:
                lambda_ = np.random.beta(alpha, alpha)
                lambdas[i] = lambda_
    elif np.random.random(1) < p:
        lambda_ = np.random.beta(alpha, alpha)
        lambdas = lambdas * lambda_

    permutation = torch.randperm(B, device=images.device)
    mixed = images * lambdas + images[permutation] * (1 - lambdas)

    return mixed, targets[permutation], lambdas.view(-1)


def mixed_cross_entropy_loss(
    logits: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lambdas: torch.Tensor
) -> torch.Tensor:
    """Cross entropy loss for mixed images.

    Args:
    ----
        logits (torch.Tensor): Output logits of the model.
        targets_a (torch.Tensor): targets of the original batch.
        targets_b (torch.Tensor): targets returned by mixup()
        lambdas (torch.Tensor): lambdas returned by mixup()

    Returns:
    -------
        torch.Tensor: the loss
    """
    ce_loss_a = F.cross_entropy(logits, targets_a, reduction='none') * lambdas
    ce_loss_b = F.cross_entropy(logits, targets_b, reduction='none') * (1 - lambdas)
    loss = (ce_loss_a + ce_loss_b).mean()
    return loss
