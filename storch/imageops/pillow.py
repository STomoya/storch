"""PIL image ops."""

from __future__ import annotations

import urllib.request
from io import BytesIO

import torch
from PIL import Image, ImageFile

__all__ = ['download', 'gif_from_files', 'pil_load_image', 'pil_configuration']


def pil_configuration(
    max_image_pixels: int = Image.MAX_IMAGE_PIXELS, load_truncated_images: bool = ImageFile.LOAD_TRUNCATED_IMAGES
) -> None:
    """Configure PIL globals.

    Args:
    ----
        max_image_pixels (int, optional): maximum image pixels. Default: Image.MAX_IMAGE_PIXELS.
        load_truncated_images (bool, optional): load truncated images? Default: False

    """
    Image.MAX_IMAGE_PIXELS = max_image_pixels
    ImageFile.LOAD_TRUNCATED_IMAGES = load_truncated_images


def pil_load_image(path: str, color_mode='RGB') -> Image.Image:
    """Load image using PIL.

    Args:
    ----
        path (str): The path of to the image.
        color_mode (str, optional): color mode. Default: 'RGB'.

    Returns:
    -------
        Image.Image: The loaded image.

    """
    return Image.open(path).convert(color_mode)


@torch.no_grad()
def gif_from_files(
    image_paths: list[str], filename: str = 'out.gif', optimize: bool = False, duration: int = 500, loop: int = 0
) -> None:
    """Make GIF from filelist.

    Args:
    ----
        image_paths (list[str]): List of images to contain in the gif.
        filename (str, optional): filename of the save gif file. Default: 'out.gif'.
        optimize (bool, optional): optimize. Default: False.
        duration (int, optional): duration of each image. Default: 500.
        loop (int, optional): loop mode. Default: infinite loop. Default: 0.

    """
    images = [Image.open(str(path)) for path in image_paths]
    images[0].save(filename, save_all=True, append_images=images[1:], optimize=optimize, duration=duration, loop=loop)


def download(url: str, filename: str | None = None) -> Image.Image:
    """Download image from url and optionally save.

    Args:
    ----
        url (str): URL.
        filename (str, optional): filename. Default: None.

    Returns:
    -------
        Image.Image: The loaded image.

    """
    b = BytesIO(urllib.request.urlopen(url).read())
    image = Image.open(b)
    if filename is not None:
        image.save(filename)
    return image
