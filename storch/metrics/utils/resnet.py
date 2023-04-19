
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

from storch.metrics.utils.download import download_url
from storch.utils.pt_version import is_multi_weight_api_available

SWAV_IN1K_URL = 'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar'


class _ResNet(nn.Module):
    """Base class for ResNet50 feature extractors."""
    def __init__(self,
        feature_dims=2048,
        weight_folder='./.cache/storch/metrics', filename=None
    ) -> None:
        super().__init__()
        feature_dim2block_index = {256: 0, 512: 1, 1024: 2, 2048: 3}
        assert feature_dims in feature_dim2block_index.keys(), f'feature_dims must be one of {list(feature_dim2block_index.keys())}'
        self.block_index = feature_dim2block_index[feature_dims]
        self.ckpt_path, resnet = self.load_checkpoint(weight_folder, filename)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def load_checkpoint(self, weight_folder, filename):
        raise NotImplementedError()

    def forward(self, x):
        # normalize input.
        x = x / 255.0
        x = x * 2 - 1

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.block_index == 0:
            x = self.avgpool(x).flatten(1)
            return x

        x = self.layer2(x)
        if self.block_index == 1:
            x = self.avgpool(x).flatten(1)
            return x

        x = self.layer3(x)
        if self.block_index == 2:
            x = self.avgpool(x).flatten(1)
            return x

        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return x


class ResNetIN(_ResNet):
    """Supervised ResNet50"""
    def load_checkpoint(self, weight_folder, filename):
        ckpt_path = None
        if is_multi_weight_api_available():
            from torchvision.models.resnet import ResNet50_Weights
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            resnet = resnet50(pretrained=True)
        return ckpt_path, resnet


class ResNetSwAVIN(_ResNet):
    """Self-supervised ResNet50 (SwAV)"""
    def __init__(self, feature_dims=2048, weight_folder='./.cache/storch/metrics', filename='swav-in1k-resnet50.torch') -> None:
        super().__init__(feature_dims, weight_folder, filename)

    def load_checkpoint(self, weight_folder, filename):
        ckpt_path = download_url(SWAV_IN1K_URL, filename, weight_folder)

        state_dict = torch.load(ckpt_path, map_location='cpu')
        new_state_dict = {}
        for key, value in state_dict.items():
            # erase 'module.' prefix
            key = key.replace('module.', '')
            # we don't need ssl specific weights.
            if not key.startswith('projection_head.') and not key.startswith('prototypes'):
                new_state_dict[key] = value
        resnet = resnet50()
        resnet.fc = None
        resnet.load_state_dict(new_state_dict)
        return ckpt_path, resnet
