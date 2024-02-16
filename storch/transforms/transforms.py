"""Tranforms."""

import cv2
import numpy as np
import torch.nn as nn

from storch.transforms.converters import pil_to_numpy
from storch.transforms.degradations import (
    all_kernels,
    random_gaussian_noise,
    random_jpg_compression,
    random_mixed_kernels,
    random_poisson_noise,
)
from storch.transforms.resize_right import resize


class PILToNumpy(nn.Module):
    """PIL to numpy."""

    def __init__(self, like_to_tensor: bool = True) -> None:
        """PIL to numpy.

        Args:
        ----
            like_to_tensor (bool, optional): Default: True.

        """
        super().__init__()
        self.like_to_tensor = like_to_tensor

    def forward(self, image):  # noqa: D102
        return pil_to_numpy(image, self.like_to_tensor)


class RandomMixedGaussianBlur(nn.Module):
    """Random mixed Gaussian blur."""

    def __init__(
        self,
        kernel_list: tuple = all_kernels,
        kernel_probs: tuple = [1 / 6 for _ in range(6)],
        kernel_size: int = 21,
        sigma_range: tuple = [0.6, 5],
        rotation_range: tuple = [-np.pi, np.pi],
        betag_range: tuple = [0.5, 8],
        betap_range: tuple = [0.5, 8],
        noise_range: tuple | None = None,
    ) -> None:
        """Gaussian blur transform.

        Args:
        ----
            kernel_list (tuple, optional): Default: all_kernels.
            kernel_probs (tuple, optional): Default: [1 / 6 for _ in range(6)].
            kernel_size (int, optional): Default: 21.
            sigma_range (tuple, optional): Default: [0.6, 5].
            rotation_range (tuple, optional): Default: [-np.pi, np.pi].
            betag_range (tuple, optional): Default: [0.5, 8].
            betap_range (tuple, optional): Default: [0.5, 8].
            noise_range (tuple | None, optional): Default: None.

        """
        super().__init__()
        self.kernel_list = kernel_list
        self.kernel_probs = kernel_probs
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.rotation_range = rotation_range
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.noise_range = noise_range

    def forward(self, image):  # noqa: D102
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_probs,
            self.kernel_size,
            self.sigma_range,
            self.sigma_range,
            self.rotation_range,
            self.betag_range,
            self.betap_range,
            self.noise_range,
        )

        image = cv2.filter2d(image, -1, kernel)

        return image


class RandomGaussianNoise(nn.Module):
    """Random Gaussian noise."""

    def __init__(self, sigma_range: tuple = [0.0, 10.0], gray_prob: float = 0.0) -> None:
        """Gaussian noise transform.

        Args:
        ----
            sigma_range (tuple, optional): Default: [0.0, 10.0].
            gray_prob (float, optional): Default: 0.0.

        """
        super().__init__()
        self.sigma_range = sigma_range
        self.gray_prob = gray_prob

    def forward(self, image):  # noqa: D102
        image = random_gaussian_noise(image, self.sigma_range, self.gray_prob)
        return image


class RandomPoissonNoise(nn.Module):
    """Random Poisson noise."""

    def __init__(self, scale_range: tuple = [0.0, 1.0], gray_prob: float = 0.0) -> None:
        """Poisson noise transform.

        Args:
        ----
            scale_range (tuple, optional): Default: [0.0, 1.0].
            gray_prob (float, optional): Default: 0.0.

        """
        super().__init__()
        self.scale_range = scale_range
        self.gray_prob = gray_prob

    def forward(self, image):  # noqa: D102
        image = random_poisson_noise(image, self.scale_range, self.gray_prob)
        return image


class RandomJPEGCompression(nn.Module):
    """Random JPEG compression."""

    def __init__(self, quality_range: tuple = [90, 100]) -> None:
        """JPEG compression.

        Args:
        ----
            quality_range (tuple, optional): Default: [90, 100].

        """
        super().__init__()
        self.quality_range = quality_range

    def forward(self, image):  # noqa: D102
        image = random_jpg_compression(image, self.quality_range)
        return image


class ResizeRight(nn.Module):
    """Resize right."""

    def __init__(self, size, interp_method='cubic', antialias=True) -> None:
        """Resize iright.

        Args:
        ----
            size (_type_): size to resize.
            interp_method (str, optional): Default: 'cubic'.
            antialias (bool, optional): Default: True.

        """
        super().__init__()
        self.size = size
        self.interp_method = interp_method
        self.antialias = antialias

    def forward(self, x):  # noqa: D102
        x = resize(x, out_shape=self.size, interp_method=self.interp_method, antialiasing=self.antialias)
        return x
