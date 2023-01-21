"""pytorch availability"""

import torch
import torchvision
from pkg_resources import parse_version

__pytorch_version__ = parse_version(torch.__version__)
__ptvision_version__ = parse_version(torchvision.__version__)


def is_native_amp_available():
    return __pytorch_version__ >= parse_version('1.6.0')


def is_multi_weight_api_available():
    return __ptvision_version__ >= parse_version('0.13.0')


def is_compiler_available():
    return __pytorch_version__ >= parse_version('2.0.0')
