
from __future__ import annotations

import urllib.request
from io import BytesIO

import torch
from PIL import Image

__all__=[
    'gif_from_files',
    'download'
]

@torch.no_grad()
def gif_from_files(
    image_paths: list[str], filename: str='out.gif',
    optimize: bool=False, duration: int=500, loop: int=0
) -> None:
    '''make GIF from filelist'''
    images = [Image.open(str(path)) for path in image_paths]
    images[0].save(filename,
        save_all=True, append_images=images[1:],
        optimize=optimize, duration=duration, loop=loop)


def download(url: str, filename: str=None) -> Image.Image:
    '''download image from url

    Arguments:
        url: str
            URL of the image to download
        filename: str (default: None)
            If not None, the image will be saved with the given filename.
    '''
    b = BytesIO(urllib.request.urlopen(url).read())
    image = Image.open(b)
    if filename is not None:
        image.save(filename)
    return image
