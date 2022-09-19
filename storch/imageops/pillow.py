
from __future__ import annotations

import urllib.request
from io import BytesIO

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__=[
    'download',
    'gif_from_files',
    'pil_load_image'
]


def pil_load_image(path: str, color_mode='RGB') -> Image.Image:
    """Load image using PIL

    Args:
        path (str): The path of to the image.
        color_mode (str, optional): color mode. Default: 'RGB'.

    Returns:
        Image.Image: The loaded image.
    """
    return Image.open(path).convert(color_mode)


@torch.no_grad()
def gif_from_files(
    image_paths: list[str], filename: str='out.gif',
    optimize: bool=False, duration: int=500, loop: int=0
) -> None:
    """make GIF from filelist

    Args:
        image_paths (list[str]): List of images to contain in the gif.
        filename (str, optional): filename of the save gif file. Default: 'out.gif'.
        optimize (bool, optional): optimize. Default: False.
        duration (int, optional): duration of each image. Default: 500.
        loop (int, optional): loop mode. Default: infinite loop. Default: 0.
    """
    images = [Image.open(str(path)) for path in image_paths]
    images[0].save(filename,
        save_all=True, append_images=images[1:],
        optimize=optimize, duration=duration, loop=loop)


def download(url: str, filename: str=None) -> Image.Image:
    """download image from url and optionally save.

    Args:
        url (str): URL.
        filename (str, optional): filename. Default: None.

    Returns:
        Image.Image: The loaded image.
    """
    b = BytesIO(urllib.request.urlopen(url).read())
    image = Image.open(b)
    if filename is not None:
        image.save(filename)
    return image
