
import numpy as np
from PIL import Image


def pil_to_numpy(image, like_to_tensor: bool=True):
    assert isinstance(image, Image.Image)
    image = np.asarray(image)
    if like_to_tensor:
        image = image.astype(np.float32) / 255.0
    return image
