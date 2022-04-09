
import os
from functools import partial

import torch
from torch.utils.data import default_collate

import storch

# from https://github.com/pytorch/vision/blob/6512146e447b69cc9fb379eb05e447a17d7f6d1c/torchvision/datasets/folder.py#L242
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

def is_image_file(path):
    ext = set([os.path.splitext(os.path.basename(path))[-1].lower()])
    return ext.issubset(IMG_EXTENSIONS)

def get_loader_kwargs():
    loader_kwargs = storch.EasyDict()
    loader_kwargs.batch_size     = None
    loader_kwargs.shuffle        = True
    loader_kwargs.drop_last      = True
    loader_kwargs.num_workers    = os.cpu_count()
    loader_kwargs.pin_memory     = torch.cuda.is_available()
    loader_kwargs.worker_init_fn = None
    loader_kwargs.generator      = None
    return loader_kwargs


def recursive_apply(func, data):
    if isinstance(data, (tuple, list)):
        return type(data)(recursive_apply(func, element) for element in data)
    elif isinstance(data, dict):
        return {key: recursive_apply(func, value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        data = func(data)
    return data

def to(data, device):
    return data.to(device)

def device_placement_collate_fn(batch, device):
    batch = default_collate(batch)
    batch = recursive_apply(partial(to, device=device), batch)
    return batch
