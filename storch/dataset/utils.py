"""Dataset utils."""

import os
from functools import partial
from typing import Any

import torch
from torch.utils.data import default_collate

import storch

# from https://github.com/pytorch/vision/blob/6512146e447b69cc9fb379eb05e447a17d7f6d1c/torchvision/datasets/folder.py#L242
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}


def is_image_file(path: str) -> bool:
    """Return whether if the path is a PIL.Image.Image openable file.

    Args:
    ----
        path (str): the path to an image.

    Returns:
    -------
        bool: file?
    """
    ext = set(os.path.splitext(os.path.basename(path))[-1].lower())
    return ext.issubset(IMG_EXTENSIONS)


def get_loader_kwargs() -> storch.EasyDict:
    """Loader keyword arguments.

    Returns
    -------
        storch.EasyDict: default loader keyword arguments as dict.
    """
    loader_kwargs = storch.EasyDict()
    loader_kwargs.batch_size = None
    loader_kwargs.shuffle = True
    loader_kwargs.drop_last = True
    loader_kwargs.num_workers = os.cpu_count()
    loader_kwargs.pin_memory = torch.cuda.is_available()
    loader_kwargs.worker_init_fn = None
    loader_kwargs.generator = None
    return loader_kwargs


def to(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Send tensor to device.

    Args:
    ----
        data (torch.Tensor): Data to be sent.
        device (torch.device): The device to send to.

    Returns:
    -------
        torch.Tensor: tensor on device.
    """
    return data.to(device)


def device_placement_collate_fn(batch: Any, device: torch.device) -> Any:
    """Collate fn with device placement.

    Args:
    ----
        batch (Any): batch from the dataset.
        device (torch.device): device to send to.

    Returns:
    -------
        Any: The collated batch.
    """
    batch = default_collate(batch)
    batch = storch.recursive_apply(partial(to, device=device), batch, torch.is_tensor)
    return batch
