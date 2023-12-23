"""Convert data."""

import numpy as np
from PIL import Image


def pil_to_numpy(image: Image.Image, like_to_tensor: bool = True) -> np.ndarray:
    """_summary_.

    Args:
    ----
        image (Image.Image): PIL image obj.
        like_to_tensor (bool, optional): Scale to [0,1]. Default: True.

    Returns:
    -------
        np.ndarray: Converted image.
    """
    assert isinstance(image, Image.Image)
    image = np.asarray(image)
    if like_to_tensor:
        image = image.astype(np.float32) / 255.0
    return image
