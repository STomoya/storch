
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from storch.helpers import to_2tuple
from storch.imageops import gaussian_2d
from storch.metrics.utils import reduce_dimension


def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float=1.0, reduction: str='elementwise_mean') -> torch.Tensor:
    """Peak signal-to-noise ratio (PSNR)

    Args:
        input (torch.Tensor): predicted image.
        target (torch.Tensor): original image.
        max_val (float, optional): maximum value of the images. either 1 or 255. Default: 1.0.
        reduction (str, optional): how to reduce dimension. Default: 'elementwise_mean'

    Returns:
        torch.Tensor: calculated PSNR score
    """
    mse_error = (input.double() - target.double()).square().mean(dim=(1, 2, 3), keepdim=True)
    psnr = 10.0 * torch.log10(max_val ** 2 / mse_error)
    return reduce_dimension(psnr, reduction).type(input.dtype)


def ssim(input: torch.Tensor, target: torch.Tensor, max_val: float=1.0,
    kernel_size: int|Sequence[int]=(11, 11), sigma: float|Sequence[float]=(1.5, 1.5),
    k1: float=0.01, k2: float=0.03, reduction: str='elementwise_mean'
) -> torch.Tensor:
    """Structual similarity index measure (SSIM)

    Args:
        input (torch.Tensor): predicted image.
        target (torch.Tensor): original image.
        max_val (float, optional): maximum value of the images.. Default: 1.0.
        kernel_size (int | Sequence[int], optional): kernel size for the gaussian filter. Default: (11, 11).
        sigma (float | Sequence[float], optional): sigma for the gaussian filter. Default: (1.5, 1.5).
        k1 (float, optional): SSIM hyperparameter. Default: 0.01.
        k2 (float, optional): SSIM hyperparameter. Default: 0.03.
        reduction (str, optional): how to reduce dimension. Default: 'elementwise_mean'

    Returns:
        torch.Tensor: _description_
    """
    B, C, device = input.size(0), input.size(1), input.device
    kernel_size, sigma = to_2tuple(kernel_size), to_2tuple(sigma)
    c1, c2 = (k1 * max_val)**2, (k2 * max_val)**2
    padh, padw = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2

    input = F.pad(input, [padw, padw, padh, padh], mode='reflect')
    target = F.pad(target, [padw, padw, padh, padh], mode='reflect')

    gaussian_filter = gaussian_2d(kernel_size, sigma)[None, None, ...].expand(C, 1, -1, -1).to(device)
    filtered = F.conv2d(torch.cat([input, target, input*input, target*target, input*target]), gaussian_filter, groups=C)
    filtered = filtered.split(B, dim=0)

    mu_input_sq = filtered[0].square()
    mu_target_sq = filtered[1].square()
    mu_input_target = filtered[0] * filtered[1]

    sigma_input_sq = filtered[2] - mu_input_sq
    sigma_target_sq = filtered[3] - mu_target_sq
    sigma_input_target = filtered[4] - mu_input_target

    numer = (2 * mu_input_target + c1) * (2 * sigma_input_target + c2)
    denom = (mu_input_sq + mu_target_sq + c1) * (sigma_input_sq + sigma_target_sq + c2)
    ssim = numer / denom
    return reduce_dimension(ssim, reduction)
