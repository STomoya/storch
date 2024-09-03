"""pytorch availability."""

import platform

import torch
import torchvision
from pkg_resources import parse_version

_PYTHON_VERSION = parse_version(platform.python_version())
_PYTORCH_VERSION = parse_version(torch.__version__)
_TORCHVISION_VERSION = parse_version(torchvision.__version__)


def _is_version_geq(current, minimum: str) -> bool:
    """Compair versions.

    Args:
        current (str): current version.
        minimum (str): minimum version required.

    Returns:
        bool: result.

    """
    if isinstance(minimum, str):
        minimum = parse_version(minimum)
    return current >= minimum


def is_python_version_geq(minimum) -> bool:
    """Is python version greater equal.

    Args:
        minimum (str): the minimum version.

    Returns:
        bool: result.

    """
    return _is_version_geq(_PYTHON_VERSION, minimum)


def is_torch_version_geq(minimum) -> bool:
    """Is torch version greater equal.

    Args:
        minimum (str): the minimum version.

    Returns:
        bool: result.

    """
    return _is_version_geq(_PYTORCH_VERSION, minimum)


def is_torchvision_version_geq(minimum: str) -> bool:
    """Is torchvision version greater equal.

    Args:
        minimum (str): the minimum version.

    Returns:
        bool: result.

    """
    return _is_version_geq(_TORCHVISION_VERSION, minimum)


def is_native_amp_available() -> bool:
    """Is native amp available.

    Returns:
        bool: result.

    """
    return is_torch_version_geq('1.6.0')


def is_multi_weight_api_available() -> bool:
    """Is multi weight API available.

    Returns:
        bool: result.

    """
    return is_torchvision_version_geq('0.13.0')


def is_compiler_available() -> bool:
    """Is torch.compile available.

    Returns:
        bool: result.

    """
    return is_torch_version_geq('2.0.0')


def is_v2_transforms_available() -> bool:
    """Is vs transforms available.

    NOTE: v2 namespace appeared in v0.15.0, but had some breaking changes in v0.16.0, so we skip v0.15.x series.

    Returns:
        bool: result.

    """
    return is_torchvision_version_geq('0.16.0')


def is_dist_state_dict_available() -> bool:
    """Is distributed get / set state_dict API available.

    Returns:
        bool: result.

    """
    return is_torch_version_geq('2.2.2')
