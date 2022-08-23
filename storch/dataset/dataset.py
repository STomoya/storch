
from __future__ import annotations

import glob
import os
import random
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from storch import natural_sort
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


def _collect_image_paths(root, filter_fn):
    images = glob.glob(os.path.join(root, '**', '*'), recursive=True)
    images = [file for file in images if is_image_file(file)]
    if isinstance(filter_fn, Callable):
        images = [file for file in images if is_image_file(file)]
    return natural_sort(images)


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
        images = _collect_image_paths(data_root, filter_fn)
        if num_images is not None and len(images) > num_images:
            random.shuffle(images)
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


class ImageFolders(DatasetBase):
    '''ImageFolder, but w/o class labels and w/ multiple folder support.

    Arguments:
        data_roots: list[str]
            Root directory of images. Images are searched recursively inside this folder.
        transforms: list[Callable]
            A callable that transforms the image.
        num_images: int|None (default: None)
            If given, the dataset will be reduced to have at most num_images samples.
        filter_fn: Callable|None (default: None)
            A callable that inputs a path and returns a bool to filter the files.
    '''
    def __init__(self,
        data_roots: list[str], transforms: Callable|list[Callable], num_images: int|None=None, filter_fn: Callable|None=None
    ) -> None:
        super().__init__()
        self.images = {index: _collect_image_paths(data_root, filter_fn) for index, data_root in enumerate(data_roots)}
        if num_images is not None and num_images < len(self):
            # cut paths from first list only.
            random.shuffle(self.images[0])
            self.images[0] = self.images[0][:num_images]
        if isinstance(transforms, list):
            assert len(self.images) == len(transforms)
            self.transforms = transforms
        else:
            self.transforms = [transforms for _ in range(len(self.images))]

    def __len__(self):
        return min([len(value) for value in self.images.values()])

    def __getitem__(self, index):
        image_paths = [paths[index] for paths in self.images.values()]
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        images = [self.transforms[index](image) for index, image in enumerate(images)]
        return tuple(images)

    def shuffle(self, index: int|None=None):
        if index is not None:
            random.shuffle(self.images[index])
        else:
            for index, value in self.images.items():
                if index != 0:
                    random.shuffle(value)


def _extract_image_paths(file: str, filter_fn: Callable):
    with open(file, 'r') as fin:
        lines = fin.read().strip().split('\n')
    images = [path for path in lines if is_image_file(path) and os.path.exists(path)]
    if isinstance(filter_fn, Callable):
        images = [file for file in images if filter_fn(file)]
    return images


class ImagePathFile(DatasetBase):
    '''Dataset expecting a file with paths to images.

    Format:
        [path_to_images.txt]
            ./path/to/image/000.jpg
            ./path/to/image/001.jpg
            ./path/to/image/002.jpg
            ...

    Arguments:
        path: str
            Path to a file with path to images.
        transform: Callable
            A callable that transforms the image.
        num_images: int|None (default: None)
            If given, the dataset will be reduced to have at most num_images samples.
        filter_fn: Callable|None (default: None)
            A callable that inputs a path and returns a bool to filter the files.
    '''
    def __init__(self,
        path: str, transform: Callable, num_images: int|None=None, filter_fn: Callable|None=None
    ) -> None:
        super().__init__()
        images = _extract_image_paths(path)
        if num_images is not None and len(images) > num_images:
            images = images[:num_images]
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image).convert('RGB')
        image = self.transform(image)
        return image


class ImagePathFiles(DatasetBase):
    '''Dataset expecting a file with paths to images w/ multiple files support.

    Format:
        [path_to_images.txt]
            ./path/to/image/000.jpg
            ./path/to/image/001.jpg
            ./path/to/image/002.jpg
            ...

    Arguments:
        paths: list[str]
            Path to a file with path to images.
        transforms: list[Callable]
            A callable that transforms the image.
        num_images: int|None (default: None)
            If given, the dataset will be reduced to have at most num_images samples.
        filter_fn: Callable|None (default: None)
            A callable that inputs a path and returns a bool to filter the files.
    '''
    def __init__(self,
        paths: str, transforms: Callable, num_images: int|None=None, filter_fn: Callable|None=None
    ) -> None:
        super().__init__()
        self.images = {index: _extract_image_paths(path, filter_fn) for index, path in enumerate(paths)}
        if num_images is not None and num_images < len(self):
            random.shuffle(self.images[0])
            self.images[0] = self.images[0][:num_images]
        if isinstance(transforms, list):
            assert len(self.images) == len(transforms)
            self.transforms = transforms
        else:
            self.transforms = [transforms for _ in range(len(self.images))]

    def __len__(self):
        return min([len(value) for value in self.images.values()])

    def __getitem__(self, index):
        image_paths = [paths[index] for paths in self.images.values()]
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        images = [self.transforms[index](image) for index, image in enumerate(images)]
        return tuple(images)

    def shuffle(self, index: int|None=None):
        if index is not None:
            random.shuffle(self.images[index])
        else:
            for index, value in self.images.items():
                if index != 0:
                    random.shuffle(value)
