from __future__ import annotations

from typing import Hashable

import numpy as np


class FeatureCache:
    _cache: dict[tuple[str, str, int], np.ndarray] = {}


    @classmethod
    def get(cls, folder: str, model: str, num_images: int) -> np.ndarray | None:
        """get cached features.

        Args:
            folder (str): folder to images
            model (str): name of the model used to extract the features.
            num_images (int): number of images.

        Returns:
            np.ndarray | None: cached feature. None when not registered.
        """
        return cls._cache.get(cls.make_key(folder, model, num_images), None)


    @classmethod
    def set(cls, folder: str, model: str, num_images: int, features: np.ndarray, force: bool=False) -> None:
        """set features to cache

        Args:
            folder (str): folder to images
            model (str): name of the model used to extract the features.
            num_images (int): number of images.
            features (np.ndarray): the extracted features
            force (bool, optional): force to register features exven if already exists. Defaults to False.
        """
        key = cls.make_key(folder, model, num_images)
        if force or key not in cls._cache:
            cls._cache[key] = features


    @staticmethod
    def make_key(folder: str, model: str, num_images: int, /) -> tuple[str, str, int]:
        """create dictionary key for the cache.

        Args:
            folder (str): folder to images
            model (str): name of the model used to extract the features.
            num_images (int): number of images.

        Returns:
            tuple[str, str, int]: key.
        """
        key = tuple(folder, model, num_images)
        assert all(isinstance(element, Hashable) for element in key)
        return key
