"""pytorch availability"""

import torch
import torchvision
from pkg_resources import parse_version

_PYTORCH_VERSION = parse_version(torch.__version__)
_TORCHVISION_VERSION = parse_version(torchvision.__version__)


def is_native_amp_available():
    return _PYTORCH_VERSION >= parse_version('1.6.0')


def is_multi_weight_api_available():
    return _TORCHVISION_VERSION >= parse_version('0.13.0')


def is_compiler_available():
    return _PYTORCH_VERSION >= parse_version('2.0.0')
