
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datapoints
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F

from storch.transforms.degradations import (all_kernels, random_gaussian_noise,
                                            random_jpg_compression,
                                            random_mixed_kernels,
                                            random_poisson_noise)
from storch.transforms.resize_right import resize


def to_image_dp(image: Image.Image, dtype: torch.dtype=torch.float) -> datapoints.Image:
    image = datapoints.Image(image)
    image = F.convert_dtype_image_tensor(image, dtype=dtype)
    return image


def to_mask_dp(mask: Image.Image) -> datapoints.Mask:
    return datapoints.Mask(mask)


class ToNumpy(Transform):
    _transformed_types = (torch.Tensor, Image.Image)
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if torch.is_tensor(inpt):
            inpt = inpt.numpy()
            if inpt.ndim == 3:
                inpt = inpt.transpose(1, 2, 0)
        else:
            inpt = np.array(inpt)

        return inpt


class ToTenor(Transform):
    """ToTensor with same functionality as v1 ToTensor class."""
    _transformed_types = (datapoints.Image, Image.Image)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, Image.Image):
            inpt = F.to_image_tensor(inpt)
        return F.convert_dtype_image_tensor(inpt)


class RandomMixedGaussianBlur(Transform):
    _transformed_types = (np.ndarray, )
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

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
        kernel = random_mixed_kernels(
            self.kernel_list, self.kernel_probs, self.kernel_size, self.sigma_range, self.sigma_range,
            self.rotation_range, self.betag_range, self.betap_range, self.noise_range
        )

        image = cv2.filter2d(image, -1, kernel)

        return image


class RandomGaussianNoise(Transform):
    _transformed_types = (np.ndarray, )
    def __init__(self, sigma_range: tuple=[0.0, 10.0], gray_prob: float=0.0) -> None:
        super().__init__()
        self.sigma_range = sigma_range
        self.gray_prob = gray_prob

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
        image = random_gaussian_noise(image, self.sigma_range, self.gray_prob)
        return image


class RandomPoissonNoise(Transform):
    _transformed_types = (np.ndarray, )
    def __init__(self, scale_range: tuple=[0.0, 1.0], gray_prob: float=0.0) -> None:
        super().__init__()
        self.scale_range = scale_range
        self.gray_prob = gray_prob

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
        image = random_poisson_noise(image, self.scale_range, self.gray_prob)
        return image


class RandomJPEGCompression(Transform):
    _transformed_types = (np.ndarray, )
    def __init__(self, quality_range: tuple=[90, 100]) -> None:
        super().__init__()
        self.quality_range = quality_range

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
        image = random_jpg_compression(image, self.quality_range)
        return image


class ResizeRight(Transform):
    _transformed_types = (np.ndarray, )
    def __init__(self, size, interp_method='cubic', antialias=True) -> None:
        super().__init__()
        self.size = size
        self.interp_method = interp_method
        self.antialias = antialias

    def _transform(self, x: Any, params: Dict[str, Any]) -> Any:
        x = resize(x, out_shape=self.size, interp_method=self.interp_method, antialiasing=self.antialias)
        return x
