
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sklearn.metrics
import torch
from scipy import linalg
from tqdm import tqdm

import storch
from storch.metrics.utils.dataset import build_dataset
from storch.metrics.utils.inceptionv3 import InceptionV3, InceptionV3JIT
from storch.metrics.utils.resnet import ResNetIN, ResNetSwAVIN
from storch.torchops import freeze

__all__=[
    'calc_metrics',
    'MetricFlags'
]


"""register feature extractors"""

def _add_extractor(registry, name, builder, input_size):
    registry.__all__.append(name)
    registry[name] = storch.EasyDict()
    registry[name].builder = builder
    registry[name].input_size = input_size if isinstance(input_size, (tuple, list)) else (input_size, input_size)

_EXTRACTORS = storch.EasyDict()
_EXTRACTORS.__all__ = []
_add_extractor(_EXTRACTORS, 'inception', InceptionV3, (299, 299))
_add_extractor(_EXTRACTORS, 'jit_inception', InceptionV3JIT, (299, 299))
_add_extractor(_EXTRACTORS, 'resnet_in', ResNetIN, (224, 224))
_add_extractor(_EXTRACTORS, 'resnet_swav_in', ResNetSwAVIN, (224, 224))


"""functions"""

@torch.no_grad()
def get_features(dataset, model: torch.nn.Module, device: torch.device, progress: bool=False) -> np.ndarray:
    """extractor features from all images inside the dataset.

    Args:
        dataset (DataLoader): The dataset.
        model (torch.nn.Module): feature extractor model.
        device (torch.device): device
        progress (boo, optional): show progress bar. Default: False

    Returns:
        np.ndarray: the extracted features.
    """
    features = []
    for image in tqdm(dataset, disable=not progress):
        image = image.to(device)
        output = model(image)
        features.append(output.cpu().numpy())
    features = np.concatenate(features)
    return features


"""FID"""

def feature_statistics(features: np.ndarray) -> tuple[np.ndarray]:
    """compute mean and covariance of the given feature array.

    Args:
        features (np.ndarray): the extracted features.

    Returns:
        np.ndarray: mean
        np.ndarray: covariance
    """
    mean = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mean, sigma


def frechet_distance(feature1: np.ndarray, feature2: np.ndarray, eps: float=1e-6) -> float:
    """calculate Frechet distance between the given two feature arrays.

    Args:
        feature1 (np.ndarray): feature array. shape: [N, feat_dim]
        feature2 (np.ndarray): feature array. shape: [N, feat_dim]
        eps (float, optional): eps. Defaults to 1e-6.

    Raises:
        ValueError: imaginary value found.

    Returns:
        float: fid score
    """
    mu1, sigma1 = feature_statistics(feature1)
    mu2, sigma2 = feature_statistics(feature2)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates'
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


"""KID"""

def kernel_distance(feature1: np.ndarray, feature2: np.ndarray, num_subsets=100, max_subset_size=1000) -> float:
    """calculate kernel distance between the given two feature arrays.

    Args:
        feature1 (np.ndarray): feature array. shape: [N, feat_dim]
        feature2 (np.ndarray): feature array. shape: [N, feat_dim]
        num_subsets (int, optional): Defaults to 100.
        max_subset_size (int, optional): Defaults to 1000.

    Returns:
        float: kid score
    """
    n = feature1.shape[1]
    m = min(min(feature1.shape[0], feature2.shape[0]), max_subset_size)
    t = 0
    for _ in range(num_subsets):
        x = feature2[np.random.choice(feature2.shape[0], m, replace=False)]
        y = feature1[np.random.choice(feature1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


"""Precision Recall Density Coverage"""

def pairwise_distance(feature1: np.ndarray, feature2: np.ndarray=None) -> np.ndarray:
    """calculate pairwise distances.

    Args:
        feature1 (np.ndarray): feature array.
        feature2 (np.ndarray, optional): feature array. Defaults to None.

    Returns:
        np.ndarray: distances.
    """
    if feature2 is None:
        feature2 = feature1
    distance = sklearn.metrics.pairwise_distances(
        feature1, feature2, metric='euclidean', n_jobs=8
    )
    return distance


def nearest_neighbour_distances(features: np.ndarray, nearest_k: int) -> np.ndarray:
    """calculate nearest neighbour distances.

    Args:
        features (np.ndarray): feature array.
        nearest_k (int): nearest k.

    Returns:
        np.ndarray: Distances to kth nearest neighbours.
    """
    distances = pairwise_distance(features)
    indices = np.argpartition(distances, nearest_k + 1, axis=-1)[..., :nearest_k + 1]
    k_smallests = np.take_along_axis(distances, indices, axis=-1)
    radii = k_smallests.max(axis=-1)
    return radii


def precision(real_nearest_neighbour_distances: np.ndarray, distance_real_fake: np.ndarray) -> float:
    """precision

    Args:
        real_nearest_neighbour_distances (np.ndarray): nearest neighbour distances of real features
        distance_real_fake (np.ndarray): pairwise distances of real and fake features

    Returns:
        float: precision score
    """
    return (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()


def recall(fake_nearest_neighbour_distances: np.ndarray, distance_real_fake: np.ndarray) -> float:
    """recall

    Args:
        fake_nearest_neighbour_distances (np.ndarray): nearest neighbour distances of fake features
        distance_real_fake (np.ndarray): pairwise distances of real and fake features

    Returns:
        float: recall score
    """
    return (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()


def density(real_nearest_neighbour_distances: np.ndarray, distance_real_fake: np.ndarray, nearest_k: int) -> float:
    """density

    Args:
        real_nearest_neighbour_distances (np.ndarray): nearest neighbour distances of real features
        distance_real_fake (np.ndarray): pairwise distances of real and fake features
        nearest_k (int): nearest k.

    Returns:
        float: density
    """
    return (1. / float(nearest_k)) * (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0).mean()


def coverage(real_nearest_neighbour_distances: np.ndarray, distance_real_fake: np.ndarray) -> float:
    """coverage

    Args:
        real_nearest_neighbour_distances (np.ndarray): nearest neighbour distances of real features
        distance_real_fake (np.ndarray): pairwise distances of real and fake features

    Returns:
        float: coverage score
    """
    return (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()


@dataclass
class MetricFlags:
    fid: bool = True
    kid: bool = False
    precision: bool = False
    recall : bool = False
    density: bool = False
    coverage: bool = False

    @classmethod
    def disabled(cls):
        """all flags disabled."""
        return cls(fid=False)

    def need_nn_dists(self):
        """do we need to calculate nearest neighbour distances?"""
        return any([self.precision, self.recall, self.density, self.coverage])

    def any(self):
        """are there any enabled flag?"""
        return any([self.fid, self.kid, self.precision, self.recall, self.density, self.coverage])


def calc_metrics(real_root, fake_root, num_images, synthesized_size, device,
    metric_flags: MetricFlags=MetricFlags(), model_name='jit_inception',
    batch_size=64, num_workers=4, verbose: bool=False
):
    results = storch.EasyDict()

    # fast exit.
    # we can disable evaluation by disabling all flags.
    # we return an empty dict even when no metrics are calculated.
    if not metric_flags.any():
        return results

    global _EXTRACTORS
    assert model_name in _EXTRACTORS.__all__, f'model_name must be one of {_EXTRACTORS.__all__}'
    model_config = _EXTRACTORS[model_name]

    model = model_config.builder()
    freeze(model)
    model.to(device)

    real_dataset = build_dataset(real_root, num_images, synthesized_size, synthetic=False, batch_size=batch_size,
        num_workers=num_workers, feature_extractor_input_size=model_config.input_size)
    real_features = get_features(real_dataset, model, device, verbose)
    del real_dataset # manual delete to confirm reduce memory.

    fake_dataset = build_dataset(fake_root, num_images, synthesized_size, synthetic=True, batch_size=batch_size,
        num_workers=num_workers, feature_extractor_input_size=model_config.input_size)
    fake_features = get_features(fake_dataset, model, device, verbose)
    del fake_dataset, model # manual delete to confirm reduce memory.


    if metric_flags.fid:
        fid_score = frechet_distance(real_features, fake_features)
        results.fid = fid_score
    if metric_flags.kid:
        kid_score = kernel_distance(real_features, fake_features)
        results.kid = kid_score


    if metric_flags.need_nn_dists():
        nearest_k = 5
        real_nn_dists = nearest_neighbour_distances(real_features, nearest_k)
        fake_nn_dists = nearest_neighbour_distances(fake_features, nearest_k)
        real_fake_dists = pairwise_distance(real_features, fake_features)

        if metric_flags.precision:
            precision_score = precision(real_nn_dists, real_fake_dists)
            results.precision = precision_score
        if metric_flags.recall:
            recall_score = recall(fake_nn_dists, real_fake_dists)
            results.recall = recall_score
        if metric_flags.density:
            density_score = density(real_nn_dists, real_fake_dists, nearest_k)
            results.density = density_score
        if metric_flags.coverage:
            coverage_score = coverage(real_nn_dists, real_fake_dists)
            results.coverage = coverage_score

    return results
