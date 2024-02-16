"""Dataset."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from storch.dataset import make_transform_from_config
from storch.dataset.dataset import _collect_image_paths


def build_dataset(
    root_dir: str,
    synthesized_size: int | tuple | list,
    synthetic: bool,
    batch_size: int = 64,
    num_workers: int = 4,
    num_images: int | None = None,
    feature_extractor_input_size: int | tuple | list = (299, 299),
    filter_fn: Callable | None = None,
) -> DataLoader:
    """Build dataset for metrics.

    Args:
    ----
        root_dir (str): root folder to images
        synthesized_size (int | tuple | list): synthesized image size.
        synthetic (bool): is the dataset fake?
        batch_size (int, optional): batch size. Default: 64.
        num_workers (int, optional): number of workers for dataloader. Default: 4.
        num_images (int, optional): number of images. Default: None.
        feature_extractor_input_size (int | tuple | list, optional): input size of feature extractor.
            Default: (299, 299).
        filter_fn (Callable, optional): callable to filter files. Default: None.

    Returns:
    -------
        DataLoader: dataset

    """
    dataset = CleanResizeDataset(
        root_dir, synthesized_size, feature_extractor_input_size, synthetic, num_images, filter_fn
    )
    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
    return dataloader


class CleanResizeDataset(Dataset):
    """Clean resize dataset."""

    def __init__(
        self,
        root_dir: str,
        syn_size: int,
        image_size: tuple[int, int] = (299, 299),
        synthetic: bool = False,
        num_images: int | None = None,
        filter_fn: Callable | None = None,
    ) -> None:
        """CleanResizeDataset.

        Resizes the images using antialiased interpolation methods.

        Args:
        ----
            root_dir (str): Dir to images.
            syn_size (int): Synthesized size.
            image_size (tuple, optional): Image size. Default: (299, 299).
            synthetic (bool, optional): Is fake set. Default: False.
            num_images (int, optional): Number of images. Default: None.
            filter_fn (Callable | None, optional): Callable to filter image paths. Default: None.

        """
        super().__init__()
        self.image_paths = _collect_image_paths(root_dir, filter_fn)
        self.num_images = len(self.image_paths)
        random.shuffle(self.image_paths)
        if num_images is not None:
            assert len(self.image_paths) >= num_images, 'number of images must be smaller than the total image numbers.'
            self.num_images = num_images
            self.image_paths = self.image_paths[:num_images]

        self.transform = make_transform_from_config([dict(name='ToTensor'), dict(name='Normalize', mean=0.5, std=0.5)])

        self.syn_size = syn_size if isinstance(syn_size, (tuple, list)) else (syn_size, syn_size)
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        # if the generated images are too small, downsample reals then upsample.
        self.maybe_downsample_before_resize = not synthetic and min(self.syn_size) < min(self.image_size)

    def __len__(self):  # noqa: D105
        return self.num_images

    def __getitem__(self, index):  # noqa: D105
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        # pre-resizing: (d, u: downsample, upsample to output)
        #   - real = fake = output:  no ops.
        #   - real >= fake > output: d(real), d(fake). This is ok.
        #   - output > fake >= real: u(real), u(fake). This is ok.
        #   - real > output > fake:  d(real), u(fake). We want to deal with this situation.
        #   - fake > output > real:  we want to believe this never happens...
        # => if (real > output > fake) then u(d'(real)), u(fake), where d' is downsample to fake.
        if self.maybe_downsample_before_resize and min(image.size) > min(self.image_size):
            image = self.clean_resize(image, self.syn_size)
        image = self.clean_resize(image, self.image_size)
        image = self.transform(image)
        return image

    def clean_resize(self, image: Image.Image, size, mode=Image.BICUBIC) -> Image.Image:
        """Resize with anti-aliasing.

        Args:
        ----
            image (Image.Image): Image.Image object.
            size (tuple | list): size in (height, width) order.
            mode (_type_, optional): Interpolation mode. Default: Image.BICUBIC.

        Returns:
        -------
            Image.Image: resized image.

        """
        image_splits = image.split()
        new_image = []
        for image_split in image_splits:
            image_split = image_split.resize(size[::-1], resample=mode)  # noqa: PLW2901
            new_image.append(np.asarray(image_split).clip(0, 255).reshape(*size, 1))
        image = Image.fromarray(np.concatenate(new_image, axis=2))
        return image
