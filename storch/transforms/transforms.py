
import cv2
import numpy as np
import torch.nn as nn

from storch.transforms.converters import pil_to_numpy
from storch.transforms.degradations import (all_kernels, random_gaussian_noise,
                                            random_jpg_compression,
                                            random_mixed_kernels,
                                            random_poisson_noise)
from storch.transforms.resize_right import resize


class PILToNumpy(nn.Module):
    def __init__(self, like_to_tensor: bool=True) -> None:
        super().__init__()
        self.like_to_tensor = like_to_tensor
    def forward(self, image):
        return pil_to_numpy(image, self.like_to_tensor)


class RandomMixedGaussianBlur(nn.Module):
    def __init__(self,
        kernel_list: tuple=all_kernels, kernel_probs: tuple=[1/6 for _ in range(6)], kernel_size:int=21, sigma_range: tuple=[0.6, 5],
        rotation_range: tuple=[-np.pi, np.pi], betag_range: tuple=[0.5, 8], betap_range: tuple=[0.5, 8], noise_range: tuple=None
    ) -> None:
        super().__init__()
        self.kernel_list = kernel_list
        self.kernel_probs = kernel_probs
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.rotation_range = rotation_range
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.noise_range = noise_range

    def forward(self, image):
        kernel = random_mixed_kernels(
            self.kernel_list, self.kernel_probs, self.kernel_size, self.sigma_range, self.sigma_range,
            self.rotation_range, self.betag_range, self.betap_range, self.noise_range
        )

        image = cv2.filter2d(image, -1, kernel)

        return image


class RandomGaussianNoise(nn.Module):
    def __init__(self, sigma_range: tuple=[0.0, 10.0], gray_prob: float=0.0) -> None:
        super().__init__()
        self.sigma_range = sigma_range
        self.gray_prob = gray_prob
    def forward(self, image):
        image = random_gaussian_noise(image, self.sigma_range, self.gray_prob)
        return image


class RandomPoissonNoise(nn.Module):
    def __init__(self, scale_range: tuple=[0.0, 1.0], gray_prob: float=0.0) -> None:
        super().__init__()
        self.scale_range = scale_range
        self.gray_prob = gray_prob

    def forward(self, image):
        image = random_poisson_noise(image, self.scale_range, self.gray_prob)
        return image


class RandomJPEGCompression(nn.Module):
    def __init__(self, quality_range: tuple=[90, 100]) -> None:
        super().__init__()
        self.quality_range = quality_range

    def forward(self, image):
        image = random_jpg_compression(image, self.quality_range)
        return image


class ResizeRight(nn.Module):
    def __init__(self, size, interp_method='cubic', antialias=True) -> None:
        super().__init__()
        self.size = size
        self.interp_method = interp_method
        self.antialias = antialias

    def forward(self, x):
        x = resize(x, out_shape=self.size, interp_method=self.interp_method, antialiasing=self.antialias)
        return x
