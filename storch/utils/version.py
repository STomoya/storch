"""pytorch availability"""

import platform

import torch
import torchvision
from pkg_resources import parse_version

_PYTHON_VERSION = parse_version(platform.python_version())
_PYTORCH_VERSION = parse_version(torch.__version__)
_TORCHVISION_VERSION = parse_version(torchvision.__version__)


def _is_version_geq(current, minimum):
    if isinstance(minimum, str):
        minimum = parse_version(minimum)
    return current >= minimum


def is_python_version_geq(minimum):
    return _is_version_geq(_PYTHON_VERSION, minimum)


def is_torch_version_geq(minimum):
    return _is_version_geq(_PYTORCH_VERSION, minimum)


def is_torchvision_version_geq(minimum):
    return _is_version_geq(_TORCHVISION_VERSION, minimum)


def is_native_amp_available():
    return is_torch_version_geq('1.6.0')


def is_multi_weight_api_available():
    return is_torchvision_version_geq('0.13.0')


def is_compiler_available():
    return is_torchvision_version_geq('2.0.0')
