"""Inception v3."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.inception as inception
from torchvision.models import inception_v3

from storch.metrics.utils.download import download_url

JIT_INCEPTION_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
INCEPTION_URL = (
    'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
)


class InceptionV3JIT(nn.Module):
    """Torchscript version of InceptionV3 model.

    from: https://github.com/GaParmar/clean-fid/blob/main/cleanfid/inception_torchscript.py
    """

    def __init__(
        self, feature_dims=2048, weight_folder='./.cache/storch/metrics', filename='jit-inception-2015-12-05.torch'
    ) -> None:
        """Inception v3 model.

        Args:
            feature_dims (int, optional): deprecated.
            weight_folder (str, optional): Folder to save the weights. Default: './.cache/storch/metrics'.
            filename (str, optional): filename. Default: 'jit-inception-2015-12-05.torch'.

        """
        super().__init__()
        self.ckpt_path = download_url(JIT_INCEPTION_URL, filename, weight_folder)
        self.base = torch.jit.load(self.ckpt_path).eval()
        self.layers = self.base.layers

    def forward(self, x):  # noqa: D102
        assert x.size(-1) == x.size(-2) == 299, 'Input size of inceptioinv3 must be 299x299'  # noqa: PLR2004
        features = self.layers.forward(x).view(x.size(0), 2048)
        return features


class InceptionV3(nn.Module):
    """Pytorch implementation of InceptionV3 model with some modifications to fit tf implementation.

    from: https://github.com/mseitzer/pytorch-fid
    """

    def __init__(
        self, feature_dims=2048, weight_folder='./.cache/storch/metrics', filename='inception-2015-12-05.torch'
    ) -> None:
        """Inception v3.

        This class uses the official PyTorch implementation but modified to fit the Tensorflow implementation.

        Args:
            feature_dims (int, optional): Number of output feature dimension. Default: 2048.
            weight_folder (str, optional): Folder to save the weights. Default: './.cache/storch/metrics'.
            filename (str, optional): File to save the weights. Default: 'inception-2015-12-05.torch'.

        """
        super().__init__()
        feature_dim2block_index = {64: 0, 192: 1, 768: 2, 2048: 3}
        assert (
            feature_dims in feature_dim2block_index
        ), f'feature_dims must be one of {list(feature_dim2block_index.keys())}'
        self.block_index = feature_dim2block_index[feature_dims]

        self.ckpt_path = download_url(INCEPTION_URL, filename, weight_folder)

        inception = inception_v3(num_classes=1008, aux_logits=False, init_weights=True)
        # replace some layers to match tf implementation.
        inception.Mixed_5b = InceptionA(192, pool_features=32)
        inception.Mixed_5c = InceptionA(256, pool_features=64)
        inception.Mixed_5d = InceptionA(288, pool_features=64)
        inception.Mixed_6b = InceptionC(768, channels_7x7=128)
        inception.Mixed_6c = InceptionC(768, channels_7x7=160)
        inception.Mixed_6d = InceptionC(768, channels_7x7=160)
        inception.Mixed_6e = InceptionC(768, channels_7x7=192)
        inception.Mixed_7b = InceptionE_1(1280)
        inception.Mixed_7c = InceptionE_2(2048)
        state_dict = torch.load(self.ckpt_path, map_location='cpu')
        inception.load_state_dict(state_dict)

        self.blocks = nn.Sequential()
        self.blocks.append(
            nn.Sequential(
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
        )
        self.blocks.append(
            nn.Sequential(inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2))
        )
        self.blocks.append(
            nn.Sequential(
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            )
        )
        self.blocks.append(nn.Sequential(inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):  # noqa: D102
        assert x.size(-1) == x.size(-2) == 299, 'Input size of inceptioinv3 must be 299x299'  # noqa: PLR2004
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.block_index:
                x = self.avgpool(x).flatten(1)
                return x


class InceptionA(inception.InceptionA):  # noqa: D101
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs


class InceptionC(inception.InceptionC):  # noqa: D101
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs


class InceptionE_1(inception.InceptionE):  # noqa: D101
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs


class InceptionE_2(inception.InceptionE):  # noqa: D101
    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs
