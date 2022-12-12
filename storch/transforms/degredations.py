"""Degredations
from: https://github.com/XPixelGroup/BasicSR/blob/b0ee3c8414bd39da34f0216cd6bfd8110b85da60/basicsr/data/degradations.py
modified by: STomoya (https://github.com/STomoya)

What
    - refactoring for readability.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import special

# from scipy.stats import multivariate_normal


# gaussian blur

def sigma_matrix2(sig_x: float, sig_y: float, theta: float) -> np.ndarray:
    """Calculate the rotated sigma matrix (two dimensional matrix).

    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.

    Returns:
        ndarray: Rotated sigma matrix.
    """
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def mesh_grid(kernel_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the mesh grid, centering at zero.

    Args:
        kernel_size (int):

    Returns:
        ndarray: with the shape (kernel_size, kernel_size, 2)
        ndarray: with the shape (kernel_size, kernel_size)
        ndarray: with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Calculate PDF of the bivariate Gaussian distribution.

    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.

    Returns:
        ndarrray: un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


# def cdf2(d_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
#     """Calculate the CDF of the standard bivariate Gaussian distribution.
#         Used in skewed Gaussian distribution.

#     Args:
#         d_matrix (ndarray): skew matrix.
#         grid (ndarray): generated by :func:`mesh_grid`,
#             with the shape (K, K, 2), K is the kernel size.

#     Returns:
#         ndarray:
#     """
#     rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
#     grid = np.dot(grid, d_matrix)
#     cdf = rv.cdf(grid)
#     return cdf


def bivariate_Gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, grid: np.ndarray=None, isotropic: bool=True) -> np.ndarray:
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool): Default: True

    Returns:
        ndarray: normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_generalized_Gaussian(kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, grid: np.ndarray=None, isotropic: bool=True) -> np.ndarray:
    """Generate a bivariate generalized Gaussian kernel.
    ``Paper: Parameter Estimation For Multivariate Generalized Gaussian Distributions``
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        ndarray:
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau(kernel_size: int, sig_x: float, sig_y: float, theta: float, beta: float, grid: np.ndarray=None, isotropic: bool=True) -> np.ndarray:
    """Generate a plateau-like anisotropic kernel.
    1 / (1+x^(beta))
    Reference: https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        ndarray:
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_Gaussian(kernel_size: int,
    sigma_x_range: tuple[float], sigma_y_range: tuple[float],
    rotation_range: tuple[float], noise_range: tuple[float]=None, isotropic: bool=True
) -> np.ndarray:
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.
    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None
        isotropic (bool): Default: True

    Returns:
        ndarray:
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_generalized_Gaussian(kernel_size: int,
    sigma_x_range: tuple[float], sigma_y_range: tuple[float], rotation_range: tuple[float],
    beta_range: tuple[float], noise_range: tuple[float]=None, isotropic: bool=True
) -> np.ndarray:
    """Randomly generate bivariate generalized Gaussian kernels.
    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
        isotropic (bool, optional): Default: True

    Returns:
        ndarray:
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # assume beta_range[0] < 1 < beta_range[1]
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_plateau(kernel_size: int,
    sigma_x_range: tuple[float], sigma_y_range: tuple[float], rotation_range: tuple[float],
    beta_range: tuple[float], noise_range: tuple[float]=None, isotropic: bool=True
):
    """Randomly generate bivariate plateau kernels.
    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi/2, math.pi/2]
        beta_range (tuple): [1, 4]
        noise_range (tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
        isotropic (bool, optional): Default: True

    Returns:
        ndarray:
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # TODO: this may be not proper
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)
    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)

    return kernel


all_kernels = ('iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso')

def random_mixed_kernels(kernel_list: tuple[str], kernel_prob: tuple[float], kernel_size: int=21,
    sigma_x_range: tuple[float]=(0.6, 5), sigma_y_range: tuple[float]=(0.6, 5),
    rotation_range: tuple[float]=(-np.pi, np.pi),
    betag_range: tuple[float]=(0.5, 8), betap_range: tuple[float]=(0.5, 8), noise_range: tuple[float]=None
) -> np.ndarray:
    """Randomly generate mixed kernels.

    Args:
        kernel_list (tuple): a list name of kernel types,
            support ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        kernel_prob (tuple): corresponding kernel probability for each kernel type
        kernel_size (int):
        sigma_x_range (tuple): Default: [0.6, 5]
        sigma_y_range (tuple): Default: [0.6, 5]
        rotation range (tuple): Default: [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None

    Returns:
        ndarray:
    """
    kernel_type = np.random.choice(kernel_list, p=kernel_prob)
    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=False)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range,
            rotation_range, betag_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range,
            rotation_range, betag_range, noise_range=noise_range, isotropic=False)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=True)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=False)
    else:
        raise Exception(f'random_mixed_kernels: Unknown kernel type {kernel_type}')

    return kernel


np.seterr(divide='ignore', invalid='ignore')


def circular_lowpass_kernel(cutoff: float, kernel_size: int, pad_to: int=0) -> np.ndarray:
    """2D sinc filter
    Reference: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero. Default: 0

    Returns:
        ndarray:
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


# gaussian noise

def _clip_rounds(image: np.ndarray, clip: bool=True, rounds: bool=False) -> np.ndarray:
    """clip and round input array

    Args:
        image (np.ndarray):
        clip (bool, optional): Defaults to True.
        rounds (bool, optional): Defaults to False.

    Returns:
        ndarray:
    """
    if clip and rounds:
        image = np.clip((image * 255.0).round(), 0, 255) / 255.
    elif clip:
        image = np.clip(image, 0, 1)
    elif rounds:
        image = (image * 255.0).round() / 255.
    return image


def generate_gaussian_noise(size: tuple[float], sigma: float=10.0, gray_noise: bool=False) -> np.ndarray:
    """generate gaussian noise.

    Args:
        size (tuple[float]):
        sigma (float, optional): Defaults: 10.0.
        gray_noise (bool, optional): Defaults: False.

    Returns:
        ndarray:
    """
    if gray_noise:
        noise = np.random.randn(*size[:2])
        noise = noise[..., np.newaxis].repeat(3, axis=2)
    else:
        noise = np.random.randn(*size)

    noise = (noise * sigma / 255.0).astype(np.float32)
    return noise


def add_gaussian_noise(image: np.ndarray, sigma: float=10.0, gray_noise: bool=False, clip: bool=True, rounds: bool=False) -> np.ndarray:
    """add gaussian noise to image

    Args:
        image (np.ndarray): image with shape [H,W,C] in range [0,1]
        sigma (float, optional): Default: 10.0.
        gray_noise (bool, optional): Default: False.
        clip (bool, optional): clip output. Default: True.
        rounds (bool, optional): round output before clipping. Default: False.

    Returns:
        ndarray
    """
    noise = generate_gaussian_noise(image.shape, sigma, gray_noise)
    image = image + noise
    image = _clip_rounds(image, clip, rounds)
    return image


def random_gaussian_noise(image: np.ndarray, sigma_range: tuple[float]=(0.0, 10.0), gray_prob: float=0.0, clip: bool=True, rounds: bool=False) -> np.ndarray:
    """add gaussian noise with random sigma to image

    Args:
        image (np.ndarray): image with shape [H,W,C] in range [0,1]
        sigma_range (tuple[float], optional): range of sigma. Default: (0.0, 10.0).
        gray_prob (float, optional): probability to use gray noise. Default: 0.0.
        clip (bool, optional): clip output. Default: True.
        rounds (bool, optional): round output before clipping. Default: False.

    Returns:
        ndarray:
    """
    sigma = np.random.uniform(*sigma_range)
    gray_noise = np.random.rand() < gray_prob
    image = add_gaussian_noise(image, sigma, gray_noise, clip, rounds)
    return image


# poisson noise

def generate_poisson_noise(image: np.ndarray, scale: float=1.0, gray_noise: bool=False, cv2_cvtcolor_mode: int=cv2.COLOR_RGB2GRAY) -> np.ndarray:
    """generate poisson noise

    Args:
        image (ndarray): image with shape [H,W,C] in range [0,1]
        scale (float, optional): Default: 1.0.
        gray_noise (bool, optional): Default: False.
        cv2_cvtcolor_mode (int, optional): cv2 color convert mode. Default: cv2.COLOR_RGB2GRAY.

    Returns:
        np.ndarray:
    """
    if gray_noise:
        image = cv2.cvtColor(image, cv2_cvtcolor_mode)
    # round and clip image for counting vals correctly
    image = _clip_rounds(image, clip=True, rounds=True)
    vals = len(np.unique(image))
    vals = 2**np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(image * vals) / float(vals))
    noise = out - image
    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    return noise * scale


def add_poisson_noise(image: np.ndarray, scale: float=1.0, gray_noise: bool=False, clip: bool=True, rounds: bool=False, cv2_cvtcolor_mode: int=cv2.COLOR_RGB2GRAY) -> np.ndarray:
    """add poisson noise

    Args:
        image (np.ndarray): image with shape [H,W,C] in range [0,1]
        scale (float, optional): Default: 1.0.
        gray_noise (bool, optional): Default: False.
        clip (bool, optional): clip output. Default: True.
        rounds (bool, optional): round output before clipping. Default: False.
        cv2_cvtcolor_mode (int, optional): cv2 color convert mode. Default: cv2.COLOR_RGB2GRAY.

    Returns:
        np.ndarray:
    """
    noise = generate_poisson_noise(image, scale, gray_noise, cv2_cvtcolor_mode)
    image = image + noise
    image = _clip_rounds(image, clip, rounds)
    return image


def random_poisson_noise(image: np.ndarray, scale_range: tuple[float]=(0.0, 1.0), gray_prob: float=0.0, clip: bool=True, rounds: bool=True, cv2_cvtcolor_mode: int=cv2.COLOR_RGB2GRAY) -> np.ndarray:
    """add poisson noise with random scale

    Args:
        image (ndarray): image with shape [H,W,C] in range [0,1]
        scale_range (tuple[float], optional): range for scale. Default: (0.0, 1.0).
        gray_prob (float, optional): probability to use gray noise. Default: 0.0.
        clip (bool, optional): clip output. Default: True.
        rounds (bool, optional): round output before clipping. Default: False.
        cv2_cvtcolor_mode (int, optional): cv2 color convert mode. Default: cv2.COLOR_RGB2GRAY.

    Returns:
        np.ndarray:
    """
    scale = np.random.uniform(*scale_range)
    gray_noise = np.random.rand() < gray_prob
    image = add_poisson_noise(image, scale, gray_noise, clip, rounds, cv2_cvtcolor_mode)
    return image


# jpg compression

def add_jpg_compression(image: np.ndarray, quality: float=90) -> np.ndarray:
    """Add JPG compression artifacts.

    Args:
        image (ndarray): Input image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for best quality. Default: 90.

    Returns:
        ndarray:
    """
    image = np.clip(image, 0, 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), round(quality)]
    _, encimg = cv2.imencode('.jpg', image * 255., encode_param)
    image = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return image


def random_jpg_compression(image: np.ndarray, quality_range: tuple[float]=(90, 100)):
    """Randomly add JPG compression artifacts.

    Args:
        image (ndarray): Input image, shape (h, w, c), range [0, 1], float32.
        quality_range (tuple[float], optional): JPG compression quality
            range. 0 for lowest quality, 100 for best quality. Default: (90, 100).

    Returns:
        ndarray:
    """
    quality = np.random.uniform(*quality_range)
    return add_jpg_compression(image, quality)
