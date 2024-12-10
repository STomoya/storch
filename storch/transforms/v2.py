"""v2 transforms."""

from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image

from storch.utils.version import is_v2_transforms_available

if not is_v2_transforms_available():
    raise Exception('v2 transforms is not available. Use torchvision>=0.16.0.')

from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F

from storch.transforms.degradations import (
    all_kernels,
    random_gaussian_noise,
    random_jpg_compression,
    random_mixed_kernels,
    random_poisson_noise,
)
from storch.transforms.resize_right import resize


def to_image(image: Image.Image, dtype: torch.dtype = torch.float) -> tv_tensors.Image:
    """Convert to image tensor.

    Args:
        image (Image.Image): input image.
        dtype (torch.dtype, optional): dtype. Default: torch.float.

    Returns:
        tv_tensors.Image: converted image.

    """
    image = F.to_image(image)  # PIL -> torch.Tensor (dtype: uint8)
    image = F.to_dtype(image, dtype=dtype)
    return image


def to_mask(mask: Image.Image) -> tv_tensors.Mask:
    """Convert to mask tensor.

    Args:
        mask (Image.Image): mask as image.

    Returns:
        tv_tensors.Mask: mask.

    """
    return tv_tensors.Mask(mask)


class ToNumpy(Transform):
    """Convert input to numpy array."""

    _transformed_types = (torch.Tensor, Image.Image)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if torch.is_tensor(inpt):
            inpt = inpt.numpy()
            if inpt.ndim == 3:  # noqa: PLR2004
                inpt = inpt.transpose(1, 2, 0)
        else:
            inpt = np.array(inpt)

        return inpt


class ToTensor(Transform):
    """ToTensor with same functionality as v1 ToTensor class."""

    _transformed_types = (tv_tensors.Image, Image.Image)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, Image.Image):
            inpt = F.to_image_tensor(inpt)
        return F.convert_dtype_image_tensor(inpt)


class RandomMixedGaussianBlur(Transform):
    """Random mixed Gaussian blur."""

    _transformed_types = (np.ndarray,)

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

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
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


class RandomGaussianNoise(Transform):
    """Random Gaussian noise."""

    _transformed_types = (np.ndarray,)

    def __init__(self, sigma_range: tuple = [0.0, 10.0], gray_prob: float = 0.0) -> None:
        """Gaussian noise transform.

        Args:
            sigma_range (tuple, optional): Default: [0.0, 10.0].
            gray_prob (float, optional): Default: 0.0.

        """
        super().__init__()
        self.sigma_range = sigma_range
        self.gray_prob = gray_prob

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
        image = random_gaussian_noise(image, self.sigma_range, self.gray_prob)
        return image


class RandomPoissonNoise(Transform):
    """Random Poisson noise."""

    _transformed_types = (np.ndarray,)

    def __init__(self, scale_range: tuple = [0.0, 1.0], gray_prob: float = 0.0) -> None:
        """Poisson noise transform.

        Args:
            scale_range (tuple, optional): Default: [0.0, 1.0].
            gray_prob (float, optional): Default: 0.0.

        """
        super().__init__()
        self.scale_range = scale_range
        self.gray_prob = gray_prob

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
        image = random_poisson_noise(image, self.scale_range, self.gray_prob)
        return image


class RandomJPEGCompression(Transform):
    """Random JPEG compression."""

    _transformed_types = (np.ndarray,)

    def __init__(self, quality_range: tuple = [90, 100]) -> None:
        """JPEG compression.

        Args:
            quality_range (tuple, optional): Default: [90, 100].

        """
        super().__init__()
        self.quality_range = quality_range

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
        image = random_jpg_compression(image, self.quality_range)
        return image


class ResizeRight(Transform):
    """Resize right."""

    _transformed_types = (np.ndarray,)

    def __init__(self, size, interp_method='cubic', antialias=True) -> None:
        """Resize iright.

        Args:
            size (_type_): size to resize.
            interp_method (str, optional): Default: 'cubic'.
            antialias (bool, optional): Default: True.

        """
        super().__init__()
        self.size = size
        self.interp_method = interp_method
        self.antialias = antialias

    def _transform(self, x: Any, params: Dict[str, Any]) -> Any:
        x = resize(x, out_shape=self.size, interp_method=self.interp_method, antialiasing=self.antialias)
        return x
