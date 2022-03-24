
from __future__ import annotations

import torch
import torchvision.transforms.functional as TF

from storch.imageops.opencv import *
from storch.imageops.pillow import *
from storch.imageops.torch  import *


INUGAMI_URL = 'https://stomoya.sakura.ne.jp/images/inugami-512pix.jpg'
'''
The image ↑ is a screenshot of the official MMD model of Korone Inugami from Hololive.
MMD model: https://3d.nicovideo.jp/works/td63650
credits: Ⓒ 2016 COVER Corp.
'''

def dummy_tensor(image_size: int|tuple=256, batched: bool=True) -> torch.Tensor:
    '''dummy tensor (image)

    Argument:
        image_size: int|tuple (default: 256)
            Size of the dummy tensor.
        batched: bool (default: True)
            Whether to add the batch dimension to the tensor or not.
    '''
    image = download(INUGAMI_URL)
    image = TF.resize(image, image_size)
    image = to_tensor(image)
    if batched:
        image = image.unsqueeze(0)
    return image
