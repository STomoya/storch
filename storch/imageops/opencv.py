"""OpenCV image ops."""

from __future__ import annotations

from functools import wraps

import cv2
import numpy as np
from PIL import Image
from skimage import segmentation
from skimage.color import label2rgb

from storch.imageops.utils import random_box

__all__ = [
    'color_hints',
    'color_palette',
    'cv2_load_image',
    'mosaic',
    'mosaic_area',
    'slic',
    'sobel',
    'xdog',
]


def pil_io(func):
    """Handle PIL image input/output for functions expecting np.ndarray input."""

    @wraps(func)
    def inner(image, *args, **kwargs):
        input_is_pil = False
        if isinstance(image, Image.Image):
            input_is_pil = True
            image = np.array(image)
        image = func(image, *args, **kwargs)
        if input_is_pil:
            image = Image.fromarray(image)
            image = image.convert('RGB')
        return image

    return inner


def cv2_load_image(path: str, rgb: bool = True, dtype: np.dtype = np.uint8) -> np.ndarray:
    """Load image using opencv.

    Args:
    ----
        path (str): The path of to the image.
        rgb (bool, optional): convert color to RGB format. Default: True.
        dtype (np.dtype, optional): data type of the loaded image. Default: np.uint8.

    Returns:
    -------
        np.ndarray: loaded image as numpy array

    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype != dtype:
        image = image.astype(dtype)
    return image


@pil_io
def xdog(
    image: np.ndarray, sigma: float = 1, k: float = 1.2, p: float = 200, eps: float = 0, phi: float = 2
) -> np.ndarray:
    """Edge detection via XDoG.

    Args:
    ----
        image (np.ndarray): image to detect edges on.
        sigma (float, optional): Default: 1.
        k (float, optional): Default: 1.2.
        p (float, optional): Default: 200.
        eps (float, optional): Default: 0.
        phi (float, optional): Default: 2.

    Returns:
    -------
        np.ndarray: line image.

    """
    if image.max() > 1:
        image = image / 255

    def sharp(image):
        g1 = cv2.GaussianBlur(image, (0, 0), sigma)
        g2 = cv2.GaussianBlur(image, (0, 0), sigma * k)
        return (1 + p) * g1 - p * g2

    def soft_threshold(s):
        T = 1 + np.tanh(phi * (s - eps))
        T[s > eps] = 1
        return T

    S = sharp(image)
    SI = np.multiply(image, S)
    T = soft_threshold(SI) * 255

    return T.astype(np.uint8)


@pil_io
def sobel(image: np.ndarray) -> np.ndarray:
    """Edge detection via Sobel.

    Args:
    ----
        image (np.ndarray): image to detect edges on.

    Returns:
    -------
        np.ndarray: line image.

    """
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.Sobel(image, cv2.CV_8U, 1, 1, ksize=5)
    image = 255 - image  # invert to fit XDoG implementation defaults.
    return image


@pil_io
def slic(image: np.ndarray, num_segments=200, compactness=10) -> np.ndarray:
    """super-pixel segmentation via SLIC.

    Args:
    ----
        image (np.ndarray): image to apply super-pixel segmentation.
        num_segments (int, optional): number of segments. Default: 200.
        compactness (int, optional): compactness. Default: 10.

    Returns:
    -------
        np.ndarray: segmented image.

    """
    segments = segmentation.slic(image, n_segments=num_segments, compactness=compactness, start_label=1)
    image = label2rgb(segments, image, kind='avg', bg_label=0)
    return image.astype(np.uint8)


@pil_io
def color_hints(image: np.ndarray, num_dots: int = 25, dot_size=3, superpixeled_color: bool = False) -> np.ndarray:
    """Generate random atari (color hints).

    TODO: implement line ataris

    Args:
    ----
        image (np.ndarray): image to create color hints from.
        num_dots (int, optional): number of dots. Default: 25.
        dot_size (int, optional): the size of each dots. Default: 3.
        superpixeled_color (bool, optional): use superpixeled image. Default: False.

    Returns:
    -------
        np.ndarray: color hint image.

    """
    h, w = image.shape[:2]
    hint = np.zeros((h, w, 3), dtype=np.uint8)
    if superpixeled_color:
        color_map = slic(image)
    else:
        color_map = image

    for _ in range(num_dots):
        x = np.random.randint(w)
        y = np.random.randint(h)
        color = color_map[y, x, :].tolist()
        hint = cv2.circle(hint, (x, y), dot_size, color, -1)

    return hint


@pil_io
def color_palette(image: np.ndarray, num_colors: int = 5, palette_size: int | tuple = 32) -> np.ndarray:
    """Create color palette.

    Args:
    ----
        image (np.ndarray): image to create color palette from.
        num_colors (int, optional): number of color in the palette. Default: 5.
        palette_size (int | tuple, optional): the size of each color panel. Default: 32.

    Returns:
    -------
        np.ndarray: color palette

    """
    if isinstance(palette_size, tuple):
        x, y = palette_size
    else:
        x = y = palette_size
    output = np.zeros((y, x * num_colors, 3), dtype=np.uint8)
    segment = slic(image, num_colors + 5)

    def palette(image):
        arr = np.ascontiguousarray(image)
        arr = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))
        palette, index = np.unique(arr.ravel(), return_inverse=True)
        palette = palette.view(image.dtype).reshape(-1, image.shape[-1])
        count = np.bincount(index)
        order = np.argsort(count)
        return palette[order[::-1]]

    colors = palette(segment)
    for i, color in enumerate(colors):
        output[:, x * i : x * (i + 1), :] = color

    return output


@pil_io
def mosaic(image: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """Create a mosaiced image.

    Args:
    ----
        image (np.ndarray): The image.
        ratio (float, optional): downsample image to.

    Returns:
    -------
        np.ndarray: image.

    """
    small = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(small, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return image


@pil_io
def mosaic_area(
    image: np.ndarray,
    box: tuple[int] | None = None,
    ratio: float = 0.1,
    margin: int = 0,
    min_size: float = 0.1,
    max_size: int = 0.8,
) -> np.ndarray:
    """Create a mozaic region given a bbox.

    If box is not given a random bbox is created.

    Args:
    ----
        image (np.ndarray): image.
        box (tuple[int] | None, optional): box to apply mozaic. Default: None.
        ratio (float, optional): ratio. Default: 0.1.
        margin (int, optional): Margin for random bbox. Default: 0.
        min_size (float, optional): minimum size of the random bbox. Default: 0.1.
        max_size (int, optional): maximum size of the random bbox. Default: 0.8.

    Returns:
    -------
        np.ndarray: image.

    """
    if box is None:
        box = random_box(image.shape[:2], min_size, max_size, margin)
    assert len(box) == 4  # noqa: PLR2004
    image[box[0] : box[2], box[1] : box[3]] = mosaic(image[box[0] : box[2], box[1] : box[3]], ratio)
    return image
