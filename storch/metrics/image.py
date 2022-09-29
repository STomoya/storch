
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from storch.helpers import to_2tuple
from storch.imageops import gaussian_2d


def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float=1.0):
    psnr_base_e = 2 * torch.log(torch.tensor(max_val)) - torch.log(F.mse_loss(input, target))
    psnr_vals = psnr_base_e * (10 / torch.log(torch.tensor(10.0)))
    return psnr_vals


def ssim(input: torch.Tensor, target: torch.Tensor, max_val: float=1.0, kernel_size: int|Sequence[int]=(11, 11), sigma: float|Sequence[float]=(1.5, 1.5), k1: float=0.01, k2: float=0.03):
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
    ssim = (numer / denom).mean(dim=[1, 2, 3])
    return ssim
