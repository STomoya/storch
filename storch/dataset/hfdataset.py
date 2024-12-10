"""`datasets` dataset."""

from __future__ import annotations

import glob
import os
from typing import Callable

import numpy as np

try:
    from datasets import Dataset, Image
except ImportError:
    Dataset = None

from storch.dataset.utils import is_image_file
from storch.torchops import local_seed_numpy


def is_datasets_available():
    """Is HuggingFace datasets available."""
    return Dataset is not None


def imagepaths(
    paths: list[str],
    transforms: Callable,
    num_images: int | None = None,
    filter_fn: Callable | None = None,
    seed: int | None = None,
):
    """Create `dataset.Dataset` object.

    The arguments are made equivalent to `dataset.ImageFolder` class.

    Args:
        paths (list[str]): list of paths to image files.
        transforms (Callable): A callable that transforms the image.
        num_images (int | None, optional): If given, the dataset will be reduced to have at most num_images samples.
            Default: None.
        filter_fn (Callable | None, optional): A callable that inputs a path and returns a bool to filter the files.
            Default: None.
        seed (int | None): seed for random sampling when reducing data according to `num_images`. Default: None.

    Returns:
        Dataset: The created dataset.

    """
    assert is_datasets_available(), 'This function requires the `datasets` module.'

    dataset = Dataset.from_dict({'image': list(filter(is_image_file, paths))})
    dataset = dataset.sort('image')  # always sort the data.

    if callable(filter_fn):
        dataset = dataset.filter(filter_fn)

    total_images = len(dataset)
    if num_images is not None and num_images < total_images:
        # Reduce dataset size using random permutation.
        with local_seed_numpy(seed=seed, enabled=seed is not None):
            permutation = np.random.permutation(total_images)[:num_images]
        # Sort indices to keep the dataset order.
        permutation = np.sort(permutation)
        dataset = dataset.select(permutation)

    dataset = dataset.cast_column('image', Image())

    def transform_sample(samples):
        samples['image'] = [transforms(image) for image in samples['image']]
        return samples

    dataset = dataset.with_transform(transform_sample)

    return dataset


def imagefolder(
    data_root: str,
    transforms: Callable,
    num_images: int | None = None,
    filter_fn: Callable | None = None,
    seed: int | None = None,
):
    """Create `dataset.Dataset` object.

    The arguments are made equivalent to `dataset.ImageFolder` class.

    Args:
        data_root (str): Root directory of images. Images are searched recursively inside this folder.
        transforms (Callable): A callable that transforms the image.
        num_images (int | None, optional): If given, the dataset will be reduced to have at most num_images samples.
            Default: None.
        filter_fn (Callable | None, optional): A callable that inputs a path and returns a bool to filter the files.
            Default: None.
        seed (int | None): seed for random sampling when reducing data according to `num_images`. Default: None.

    Returns:
        Dataset: The created dataset.

    """
    return imagepaths(
        glob.glob(os.path.join(data_root, '**', '*'), recursive=True),
        transforms=transforms,
        num_images=num_images,
        filter_fn=filter_fn,
        seed=seed,
    )


def imagecsv(
    csv: str | list[str],
    transforms: Callable,
    num_images: int | None = None,
    filter_fn: Callable | None = None,
    seed: int | None = None,
):
    """Create `dataset.Dataset` object.

    The arguments are made equivalent to `dataset.ImageFolder` class.

    Args:
        csv (str | list[str]): csv file containing the path to image files in each row in the first column. A csv file,
            multiple csv files in a list, and directory to csv files are supported.
        transforms (Callable): A callable that transforms the image.
        num_images (int | None, optional): If given, the dataset will be reduced to have at most num_images samples.
            Default: None.
        filter_fn (Callable | None, optional): A callable that inputs a path and returns a bool to filter the files.
            Default: None.
        seed (int | None): seed for random sampling when reducing data according to `num_images`. Default: None.

    Returns:
        Dataset: The created dataset.

    """

    def load_csv_first_column(path):
        with open(path, 'r') as fp:
            lines = fp.read().strip().splitlines()
        image_paths = [line.split(',') for line in lines]
        return image_paths

    if isinstance(csv, str) and os.path.exists(csv):
        if os.path.isfile(csv):
            csv = [csv]
        else:
            csv = glob.glob(os.path.join(csv, '*.csv'))

    image_paths = []
    for csv_file in csv:
        image_paths.extend(load_csv_first_column(csv_file))

    return imagepaths(image_paths, transforms=transforms, num_images=num_images, filter_fn=filter_fn, seed=seed)
