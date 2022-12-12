"""transform used in RealESRGAN.
from:
    - blur kernels: https://github.com/XPixelGroup/BasicSR/blob/b0ee3c8414bd39da34f0216cd6bfd8110b85da60/basicsr/data/realesrgan_dataset.py#L109-L182
    - transforms: https://github.com/XPixelGroup/BasicSR/blob/b0ee3c8414bd39da34f0216cd6bfd8110b85da60/basicsr/models/realesrnet_model.py#L82-L169
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch.nn as nn
from skimage.transform import rescale, resize

from storch.transforms.degradations import (all_kernels,
                                            circular_lowpass_kernel,
                                            random_gaussian_noise,
                                            random_jpg_compression,
                                            random_mixed_kernels,
                                            random_poisson_noise)


@dataclass
class RealESRTransformConfig:
    sinc_prob: float = 0.1
    kernel_list: tuple = all_kernels
    kernel_probs: tuple = (0.45, 0.25, 0.12, 0.03, 0.12, 0.03)
    blur_kernel_size: int = 21
    blur_sigma: tuple = (0.2, 3)
    betag_range: tuple = (0.5, 4)
    betap_range: tuple = (1, 2)

    resize_probs: tuple = (0.2, 0.7, 0.1)
    resize_range: tuple = (0.15, 1.5)
    gaussian_noise_prob: float = 0.5
    gray_noise_prob: float = 0.4
    noise_range: tuple = (1, 30)
    poisson_scale_range: tuple = (0.05, 3)
    jpeg_range: tuple = (30, 95)

    second_blur_prob: float = 0.8
    sinc_prob2: float = 0.1
    kernel_list2: tuple = all_kernels
    kernel_probs2: tuple = (0.45, 0.25, 0.12, 0.03, 0.12, 0.03)
    blur_kernel_size2: int = 21
    blur_sigma2: tuple = (0.2, 1.5)
    betag_range2: tuple = (0.5, 4)
    betap_range2: tuple = (1, 2)

    resize_probs2: tuple = (0.3, 0.4, 0.3)
    resize_range2: tuple = (0.3, 1.2)
    gaussian_noise_prob2: float = 0.5
    gray_noise_prob2: float = 0.4
    noise_range2: tuple = (1, 25)
    poisson_scale_range2: tuple = (0.05, 2.5)
    jpeg_range2: tuple = (30, 95)

    final_sinc_prob: float = 0.8


class RealESRTransform(nn.Module):
    def __init__(self,
        scale: float,
        **kwargs
    ) -> None:
        """Degredation process presented in RealESRGAN.

        Args:
            scale (float): scale
            config (RealESRTransformConfig, optional): config. see RealESRTransformConfig for details.
                Default: RealESRTransformConfig().
        """
        super().__init__()

        config = RealESRTransformConfig()

        self.scale = scale
        self.kernel_range = [2*v+1 for v in range(3, 11)]
        if kwargs is not {}:
            config = replace(config, **kwargs)
        self.config = config

        self.pulse_kernel = np.zeros((21, 21), dtype=np.float32)
        self.pulse_kernel[10, 10] = 1.0

    def forward(self, image):

        org_height, org_width = image.shape[:2]

        # first order degredation
        kernel_size = np.random.choice(self.kernel_range)
        if np.random.rand() < self.config.sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.config.kernel_list, self.config.kernel_probs, self.config.blur_kernel_size,
                self.config.blur_sigma, self.config.blur_sigma, [-np.pi, np.pi],
                self.config.betag_range, self.config.betap_range, None
            )
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        image = cv2.filter2D(image, -1, kernel)

        resize_type = np.random.choice(['up', 'down', 'keep'], p=self.config.resize_probs)
        if resize_type == 'up':
            scale = np.random.uniform(1, self.config.resize_range[1])
        elif resize_type == 'down':
            scale = np.random.uniform(self.config.resize_range[0], 1)
        else:
            scale = 1
        image = rescale(image, scale, channel_axis=2, anti_aliasing=True)

        if np.random.rand() < self.config.gaussian_noise_prob:
            image = random_gaussian_noise(image, self.config.noise_range, self.config.gray_noise_prob)
        else:
            image = random_poisson_noise(image, self.config.poisson_scale_range, self.config.gray_noise_prob)
        image = random_jpg_compression(image, self.config.jpeg_range)

        # second order degredation
        if np.random.rand() < self.config.second_blur_prob:
            kernel_size = np.random.choice(self.kernel_range)
            if np.random.rand() < self.config.sinc_prob2:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.config.kernel_list2, self.config.kernel_probs2, self.config.blur_kernel_size2,
                    self.config.blur_sigma2, self.config.blur_sigma2, [-np.pi, np.pi],
                    self.config.betag_range2, self.config.betap_range2, None
                )
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            image = cv2.filter2D(image, -1, kernel)

        resize_type = np.random.choice(['up', 'down', 'keep'], p=self.config.resize_probs2)
        if resize_type == 'up':
            scale = np.random.uniform(1, self.config.resize_range2[1])
        elif resize_type == 'down':
            scale = np.random.uniform(self.config.resize_range2[0], 1)
        else:
            scale = 1
        new_size = (int(org_height / self.scale * scale), int(org_width / self.scale * scale))
        image = resize(image, new_size, anti_aliasing=True)

        if np.random.rand() < self.config.gaussian_noise_prob2:
            image = random_gaussian_noise(image, self.config.noise_range2, self.config.gray_noise_prob2)
        else:
            image = random_poisson_noise(image, self.config.poisson_scale_range2, self.config.gray_noise_prob2)

        final_size = (int(org_height / self.scale), int(org_width / self.scale))
        if np.random.rand() < self.config.final_sinc_prob:
            kernel_size = np.random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        else:
            sinc_kernel = self.pulse_kernel

        if np.random.rand() < 0.5:
            image = resize(image, final_size, anti_aliasing=True)
            image = cv2.filter2D(image, -1, sinc_kernel)
            image = random_jpg_compression(image, self.config.jpeg_range2)
        else:
            image = random_jpg_compression(image, self.config.jpeg_range2)
            image = resize(image, final_size, anti_aliasing=True)
            image = cv2.filter2D(image, -1, sinc_kernel)

        image = (image * 255.0).round().clip(0.0, 255.0) / 255.0

        return image
