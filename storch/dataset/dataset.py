
from __future__ import annotations

import glob
import os
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from storch.dataset.utils import get_loader_kwargs, is_image_file


class DatasetBase(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = get_loader_kwargs()

    def setup_loader(self, batch_size: int, **kwargs):
        self.kwargs.batch_size = batch_size
        for key, value in kwargs.items():
            assert key in self.kwargs, f'Unknown keyword argument {key}.'
            self.kwargs[key] = value
        return self

    def toloader(self):
        return DataLoader(self, **self.kwargs)


class ImageFolder(DatasetBase):
    '''ImageFolder, but w/o class'''
    def __init__(self,
        data_root, transform, num_images=None, filter_fn=None
    ) -> None:
        super().__init__()
        images = glob.glob(os.path.join(data_root, '**', '*'), recursive=True)
        if isinstance(filter_fn, Callable):
            images = [file for file in images if is_image_file(file) and filter_fn(file)]
        if num_images is not None and len(images) > num_images:
            images = images[:num_images]
        self.images    = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image).convert('RGB')
        image = self.transform(image)
        return image
