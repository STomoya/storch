"""pytorch availability"""

from pkg_resources import parse_version

import torch
import torchvision

__pytorch_version__ = parse_version(torch.__version__)


def is_native_amp_available():
    return __pytorch_version__ >= parse_version('1.6.0')
