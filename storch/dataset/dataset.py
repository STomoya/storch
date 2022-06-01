
from __future__ import annotations

import glob
import os
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from storch.dataset.utils import get_loader_kwargs, is_image_file


class DatasetBase(Dataset):
    '''Base class for datasets.
    Provides functions to easily transform the dataset to a DataLoader object.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = get_loader_kwargs()

    def setup_loader(self, batch_size: int, **kwargs):
        '''Setup keyword arguments for DataLoader

        Arguments:
            batch_size: int
                Batch size
            **kwargs: Any
                Other keyword arguments to be passed to the DataLoader class
        '''
        self.kwargs.batch_size = batch_size
        for key, value in kwargs.items():
            assert key in self.kwargs, f'Unknown keyword argument {key}.'
            self.kwargs[key] = value
        return self

    def toloader(self):
        '''Transform dataset to DataLoader object'''
        return DataLoader(self, **self.kwargs)


class ImageFolder(DatasetBase):
    '''ImageFolder, but w/o class labels

    Arguments:
        data_root: str
            Root directory of images. Images are searched recursively inside this folder.
        transform: Callable
            A callable that transforms the image.
        num_images: int|None (default: None)
            If given, the dataset will be reduced to have at most num_images samples.
        filter_fn: Callable|None (default: None)
            A callable that inputs a path and returns a bool to filter the files.
    '''
    def __init__(self,
        data_root: str, transform: Callable, num_images: int|None =None, filter_fn: Callable|None=None
    ) -> None:
        super().__init__()
        images = glob.glob(os.path.join(data_root, '**', '*'), recursive=True)
        images = [file for file in images if is_image_file(file)]
        if isinstance(filter_fn, Callable):
            images = [file for file in images if filter_fn(file)]
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
