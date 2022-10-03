
from __future__ import annotations

import random
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from storch.dataset import make_transform_from_config
from storch.dataset.dataset import _collect_image_paths


def build_dataset(
    root_dir: str, num_images: int, synthesized_size: int|tuple|list, synthetic: bool,
    batch_size: int=64, num_workers: int=4,
    feature_extractor_input_size: int|tuple|list=(299, 299), filter_fn: Callable=None
) -> DataLoader:
    """build dataset for metrics

    Args:
        root_dir (str): root folder to images
        num_images (int): number of images
        synthesized_size (int | tuple | list): synthesized image size.
        synthetic (bool): is the dataset fake?
        batch_size (int, optional): batch size. Default: 64.
        num_workers (int, optional): number of workers for dataloader. Default: 4.
        feature_extractor_input_size (int | tuple | list, optional): input size of feature extractor. Defaults to (299, 299).
        filter_fn (Callable, optional): callable to filter files. Defaults to None.

    Returns:
        DataLoader: dataset
    """
    dataset = CleanResizeDataset(root_dir, num_images, synthesized_size, feature_extractor_input_size, synthetic, filter_fn)
    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
    return dataloader


class CleanResizeDataset(Dataset):
    def __init__(self,
        root_dir, num_images, syn_size, image_size=(299, 299), synthetic=False, filter_fn: Callable=None
    ) -> None:
        super().__init__()
        self.image_paths = _collect_image_paths(root_dir, filter_fn)
        assert len(self.image_paths) >= num_images, 'number of images must be smaller than the total image numbers.'
        self.num_images = num_images
        random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[:num_images]

        self.transform = make_transform_from_config([
            dict(name='ToTensor'), dict(name='Normalize', mean=0.5, std=0.5)
        ])

        self.syn_size = syn_size if isinstance(syn_size, (tuple, list)) else (syn_size, syn_size)
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        # if the generated images are too small, downsample reals then upsample.
        self.downsample_before_resize = not synthetic and min(self.syn_size) < min(self.image_size)


    def __len__(self):
        return self.num_images


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.downsample_before_resize:
            image = self.clean_resize(image, self.syn_size)
        image = self.clean_resize(image, self.image_size)
        image = self.transform(image)
        return image


    def clean_resize(self, image: Image.Image, size, mode=Image.BICUBIC) -> Image.Image:
        """resize with anti-aliasing.

        Args:
            image (Image.Image): Image.Image object.
            size (tuple | list): size in (height, width) order.
            mode (_type_, optional): Interpolation mode. Default: Image.BICUBIC.

        Returns:
            Image.Image: resized image.
        """
        image_splits = image.split()
        new_image = []
        for image_split in image_splits:
            image_split = image_split.resize(size[::-1], resample=mode)
            new_image.append(np.asarray(image_split).clip(0, 255).reshape(*size, 1))
        image = Image.fromarray(np.concatenate(new_image, axis=2))
        return image
