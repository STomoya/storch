"""pytorch availability"""

from pkg_resources import parse_version

import torch
import torchvision

__pytorch_version__ = parse_version(torch.__version__)
