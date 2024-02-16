"""Torch image ops."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.io.image import ImageReadMode, read_image
from torchvision.utils import save_image

from storch.imageops.utils import random_box
from storch.utils.layer_helpers import to_2tuple

__all__ = [
    'make_image_grid',
    'save_images',
    'to_tensor',
    'tensor2heatmap',
    'tensor2heatmap_cv2',
    'torch_load_image',
    'make_mask',
    'make_masks',
    'apply_mask',
    'gaussian_1d',
    'gaussian_2d',
    'pascal_1d',
    'pascal_2d',
    'laplacian_1d',
    'laplacian_2d',
    'sobel_2d',
]


def torch_load_image(path: str, mode: ImageReadMode = ImageReadMode.RGB):
    """Load image using torchvision.

    Args:
    ----
        path (str): path to image
        mode (ImageReadMode, optional): mode. Default: ImageReadMode.RGB.

    Returns:
    -------
        _type_: loaded image.

    """
    return read_image(path, mode)


def make_image_grid(*image_tensors, num_images: int | None = None):
    """Align images.

    Args:
    ----
        *image_tensors: image tensors.
        num_images (int, optional): _description_. Default: None.

    Returns:
    -------
        torch.Tensor: aligned images.

    Examples::
        >>> img1, img2 = [torch.randn(3, 3, 128, 128) for _ in range(2)]
        >>> aligned = make_image_grid(img1, img2)
        >>> # aligned.size() == [6, 3, 128, 128]
        >>> # aligned        == [img1[0], img2[0], img1[1], img2[1], img1[2], img2[2]]

        >>> # Any number of tensors can be passed to this function
        >>> # as long as the sizes are equal except for the batch dimension
        >>> img_tensors = [torch.randn(random.randint(1, 10), 3, 128, 128) for _ in range(24)]
        >>> aligned = make_image_grid(*img_tensors)

    """

    def _split(x):
        return x.chunk(x.size(0), 0)

    image_tensor_lists = map(_split, image_tensors)
    images = []
    for index, image_set in enumerate(zip(*image_tensor_lists, strict=False)):
        images.extend(list(image_set))
        if num_images is not None and index == num_images - 1:
            break
    return torch.cat(images, dim=0)


@torch.no_grad()
def save_images(
    *image_tensors,
    filename: str,
    num_images: int | None = None,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float] = (-1, 1),
    **kwargs,
) -> None:
    """Create image grid and save.

    Args:
    ----
        *image_tensors: Image tensors.
        filename (str): Where to save the images to.
        num_images (int, optional): Number of samples to save. Default: None.
        nrow (int, optional): torchvision.utils.save_image() argument. Default: 8.
        normalize (bool, optional): torchvision.utils.save_image() argument. Default: True.
        value_range (tuple[float], optional): torchvision.utils.save_image() argument. Default: (-1, 1).
        **kwargs: extra args for make_grid.

    """
    images = make_image_grid(*image_tensors, num_images=num_images)
    save_image(images, filename, nrow=nrow, normalize=normalize, value_range=value_range, **kwargs)


def to_tensor(image: Image.Image, mean: float | tuple = 0.5, std: float | tuple = 0.5) -> torch.Tensor:
    """Convert image to torch.Tensor.

    Args:
    ----
        image (Image.Image): Image to convert to torch.Tensor
        mean (float | tuple, optional): Mean used to normalize the image.. Default: 0.5.
        std (float | tuple, optional): Standard used to normalize the image.. Default: 0.5.

    Returns:
    -------
        torch.Tensor: image as tensor.

    """
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean, std)
    return image


@torch.no_grad()
def tensor2heatmap_cv2(tensor, size, normalize=True, cmap=cv2.COLORMAP_JET) -> torch.Tensor:
    """Apply color map to 1 channel batched image tensor using opencv.

    Slower than tensor2heatmap(), but supports various cmap.
    It converts the image to np.ndarray, apply colormap and convert back to torch.Tensor

    Args:
    ----
        tensor (_type_): image tensor of shape [BHW] or [B1HW]
        size (_type_): output size of heatmap
        normalize (bool, optional):  normalize output to [-1, 1]. Default: True.
        cmap (_type_, optional): colormap to apply. supports any colormap type implemented in opencv.
            Default: cv2.COLORMAP_JET.

    Returns:
    -------
        torch.Tensor: heatmap as image.

    """
    assert tensor.size(1) == 1 and tensor.ndim == 4, 'Expects 1 channel batched image tensor.'  # noqa: PLR2004
    b, dtype, device = tensor.size(0), tensor.dtype, tensor.device
    # numpy, cv2 image channel format
    arrays = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    # -> [0., 1.], np.float32
    arrays -= arrays.min(axis=(1, 2, 3), keepdims=True)
    arrays /= arrays.max(axis=(1, 2, 3), keepdims=True)
    # -> [0, 255], np.uint8
    arrays = (arrays * 255).astype(np.uint8)
    # batched array with target size
    new_array = np.zeros((b, *size, 3), dtype=np.uint8)
    # resize + apply color map
    for i in range(len(arrays)):
        array = arrays[i].copy()
        array = cv2.resize(array, size)
        array = cv2.applyColorMap(array, cmap)
        new_array[i] = array
    # -> [0., 1.], np.float32, BGR2RGB
    new_array = (new_array.astype(np.float32) / 255)[..., ::-1].copy()
    # torch.Tensor, channel first
    tensor = torch.from_numpy(new_array).to(device=device, dtype=dtype)
    tensor = tensor.permute(0, 3, 1, 2)
    # normalize
    if normalize:
        tensor = tensor.sub(0.5).div(0.5)
    return tensor


# cv2.COLORMAP_JET colormap as torch.Tensor object
# from https://github.com/opencv/opencv/blob/ebb6915e588fcee1e6664cce670f0253bac0e67b/modules/imgproc/src/colormap.cpp#L240-L242
# colormap_jet.size() >>> torch.Size([256, 3]) # RGB format
colormap_jet = torch.tensor(
    [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.00588235294117645,
            0.02156862745098032,
            0.03725490196078418,
            0.05294117647058827,
            0.06862745098039214,
            0.084313725490196,
            0.1000000000000001,
            0.115686274509804,
            0.1313725490196078,
            0.1470588235294117,
            0.1627450980392156,
            0.1784313725490196,
            0.1941176470588235,
            0.2098039215686274,
            0.2254901960784315,
            0.2411764705882353,
            0.2568627450980392,
            0.2725490196078431,
            0.2882352941176469,
            0.303921568627451,
            0.3196078431372549,
            0.3352941176470587,
            0.3509803921568628,
            0.3666666666666667,
            0.3823529411764706,
            0.3980392156862744,
            0.4137254901960783,
            0.4294117647058824,
            0.4450980392156862,
            0.4607843137254901,
            0.4764705882352942,
            0.4921568627450981,
            0.5078431372549019,
            0.5235294117647058,
            0.5392156862745097,
            0.5549019607843135,
            0.5705882352941174,
            0.5862745098039217,
            0.6019607843137256,
            0.6176470588235294,
            0.6333333333333333,
            0.6490196078431372,
            0.664705882352941,
            0.6803921568627449,
            0.6960784313725492,
            0.7117647058823531,
            0.7274509803921569,
            0.7431372549019608,
            0.7588235294117647,
            0.7745098039215685,
            0.7901960784313724,
            0.8058823529411763,
            0.8215686274509801,
            0.8372549019607844,
            0.8529411764705883,
            0.8686274509803922,
            0.884313725490196,
            0.8999999999999999,
            0.9156862745098038,
            0.9313725490196076,
            0.947058823529412,
            0.9627450980392158,
            0.9784313725490197,
            0.9941176470588236,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0.9862745098039216,
            0.9705882352941178,
            0.9549019607843139,
            0.93921568627451,
            0.9235294117647062,
            0.9078431372549018,
            0.892156862745098,
            0.8764705882352941,
            0.8607843137254902,
            0.8450980392156864,
            0.8294117647058825,
            0.8137254901960786,
            0.7980392156862743,
            0.7823529411764705,
            0.7666666666666666,
            0.7509803921568627,
            0.7352941176470589,
            0.719607843137255,
            0.7039215686274511,
            0.6882352941176473,
            0.6725490196078434,
            0.6568627450980391,
            0.6411764705882352,
            0.6254901960784314,
            0.6098039215686275,
            0.5941176470588236,
            0.5784313725490198,
            0.5627450980392159,
            0.5470588235294116,
            0.5313725490196077,
            0.5156862745098039,
            0.5,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.001960784313725483,
            0.01764705882352935,
            0.03333333333333333,
            0.0490196078431373,
            0.06470588235294117,
            0.08039215686274503,
            0.09607843137254901,
            0.111764705882353,
            0.1274509803921569,
            0.1431372549019607,
            0.1588235294117647,
            0.1745098039215687,
            0.1901960784313725,
            0.2058823529411764,
            0.2215686274509804,
            0.2372549019607844,
            0.2529411764705882,
            0.2686274509803921,
            0.2843137254901961,
            0.3,
            0.3156862745098039,
            0.3313725490196078,
            0.3470588235294118,
            0.3627450980392157,
            0.3784313725490196,
            0.3941176470588235,
            0.4098039215686274,
            0.4254901960784314,
            0.4411764705882353,
            0.4568627450980391,
            0.4725490196078431,
            0.4882352941176471,
            0.503921568627451,
            0.5196078431372548,
            0.5352941176470587,
            0.5509803921568628,
            0.5666666666666667,
            0.5823529411764705,
            0.5980392156862746,
            0.6137254901960785,
            0.6294117647058823,
            0.6450980392156862,
            0.6607843137254901,
            0.6764705882352942,
            0.692156862745098,
            0.7078431372549019,
            0.723529411764706,
            0.7392156862745098,
            0.7549019607843137,
            0.7705882352941176,
            0.7862745098039214,
            0.8019607843137255,
            0.8176470588235294,
            0.8333333333333333,
            0.8490196078431373,
            0.8647058823529412,
            0.8803921568627451,
            0.8960784313725489,
            0.9117647058823528,
            0.9274509803921569,
            0.9431372549019608,
            0.9588235294117646,
            0.9745098039215687,
            0.9901960784313726,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0.9901960784313726,
            0.9745098039215687,
            0.9588235294117649,
            0.943137254901961,
            0.9274509803921571,
            0.9117647058823528,
            0.8960784313725489,
            0.8803921568627451,
            0.8647058823529412,
            0.8490196078431373,
            0.8333333333333335,
            0.8176470588235296,
            0.8019607843137253,
            0.7862745098039214,
            0.7705882352941176,
            0.7549019607843137,
            0.7392156862745098,
            0.723529411764706,
            0.7078431372549021,
            0.6921568627450982,
            0.6764705882352944,
            0.6607843137254901,
            0.6450980392156862,
            0.6294117647058823,
            0.6137254901960785,
            0.5980392156862746,
            0.5823529411764707,
            0.5666666666666669,
            0.5509803921568626,
            0.5352941176470587,
            0.5196078431372548,
            0.503921568627451,
            0.4882352941176471,
            0.4725490196078432,
            0.4568627450980394,
            0.4411764705882355,
            0.4254901960784316,
            0.4098039215686273,
            0.3941176470588235,
            0.3784313725490196,
            0.3627450980392157,
            0.3470588235294119,
            0.331372549019608,
            0.3156862745098041,
            0.2999999999999998,
            0.284313725490196,
            0.2686274509803921,
            0.2529411764705882,
            0.2372549019607844,
            0.2215686274509805,
            0.2058823529411766,
            0.1901960784313728,
            0.1745098039215689,
            0.1588235294117646,
            0.1431372549019607,
            0.1274509803921569,
            0.111764705882353,
            0.09607843137254912,
            0.08039215686274526,
            0.06470588235294139,
            0.04901960784313708,
            0.03333333333333321,
            0.01764705882352935,
            0.001960784313725483,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0.5,
            0.5156862745098039,
            0.5313725490196078,
            0.5470588235294118,
            0.5627450980392157,
            0.5784313725490196,
            0.5941176470588235,
            0.6098039215686275,
            0.6254901960784314,
            0.6411764705882352,
            0.6568627450980392,
            0.6725490196078432,
            0.6882352941176471,
            0.7039215686274509,
            0.7196078431372549,
            0.7352941176470589,
            0.7509803921568627,
            0.7666666666666666,
            0.7823529411764706,
            0.7980392156862746,
            0.8137254901960784,
            0.8294117647058823,
            0.8450980392156863,
            0.8607843137254902,
            0.8764705882352941,
            0.892156862745098,
            0.907843137254902,
            0.9235294117647059,
            0.9392156862745098,
            0.9549019607843137,
            0.9705882352941176,
            0.9862745098039216,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0.9941176470588236,
            0.9784313725490197,
            0.9627450980392158,
            0.9470588235294117,
            0.9313725490196079,
            0.915686274509804,
            0.8999999999999999,
            0.884313725490196,
            0.8686274509803922,
            0.8529411764705883,
            0.8372549019607844,
            0.8215686274509804,
            0.8058823529411765,
            0.7901960784313726,
            0.7745098039215685,
            0.7588235294117647,
            0.7431372549019608,
            0.7274509803921569,
            0.7117647058823531,
            0.696078431372549,
            0.6803921568627451,
            0.6647058823529413,
            0.6490196078431372,
            0.6333333333333333,
            0.6176470588235294,
            0.6019607843137256,
            0.5862745098039217,
            0.5705882352941176,
            0.5549019607843138,
            0.5392156862745099,
            0.5235294117647058,
            0.5078431372549019,
            0.4921568627450981,
            0.4764705882352942,
            0.4607843137254903,
            0.4450980392156865,
            0.4294117647058826,
            0.4137254901960783,
            0.3980392156862744,
            0.3823529411764706,
            0.3666666666666667,
            0.3509803921568628,
            0.335294117647059,
            0.3196078431372551,
            0.3039215686274508,
            0.2882352941176469,
            0.2725490196078431,
            0.2568627450980392,
            0.2411764705882353,
            0.2254901960784315,
            0.2098039215686276,
            0.1941176470588237,
            0.1784313725490199,
            0.1627450980392156,
            0.1470588235294117,
            0.1313725490196078,
            0.115686274509804,
            0.1000000000000001,
            0.08431372549019622,
            0.06862745098039236,
            0.05294117647058805,
            0.03725490196078418,
            0.02156862745098032,
            0.00588235294117645,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    ],
    dtype=torch.float32,
).transpose(0, 1)


@torch.no_grad()
def tensor2heatmap(tensor: torch.Tensor, size: tuple[int], normalize: bool = True) -> torch.Tensor:
    """Apply cv2.COLORMAP_JET to 1 channel batched image tensor.

    Fast, but only supports JET colormap.

    Args:
    ----
        tensor (torch.Tensor): image tensor of shape [BHW] or [B1HW]
        size (tuple[int]): output size of heatmap
        normalize (bool, optional): normalize output to [-1, 1]. Default: True.

    Returns:
    -------
        torch.Tensor: heatmap as image.

    """
    assert tensor.ndim == 3 or (tensor.ndim == 4 and tensor.size(1) == 1)  # noqa: PLR2004
    if tensor.ndim == 4:  # noqa: PLR2004
        tensor = tensor.squeeze(1)
    B, device = tensor.size(0), tensor.device
    # -> [0, 1]
    tensor -= tensor.view(B, -1, 1).min(dim=1, keepdim=True)[0]
    tensor /= tensor.view(B, -1, 1).max(dim=1, keepdim=True)[0]
    # resize
    tensor = TF.resize(tensor, size)
    # -> [0, 255]
    index = tensor.mul(255).long()
    # -> apply color map -> [B3HW]
    output = colormap_jet.to(device)[index].permute(0, 3, 1, 2)
    if normalize:
        output = output.sub(0.5).div(0.5)
    return output


def make_mask(image_size: tuple[int], num_boxes: int = 1, min_size: float = 0.1, max_size: float = 0.5) -> torch.Tensor:
    """Make a randomly sampled mask.

    Args:
    ----
        image_size (tuple[int]): image size (H, W)
        num_boxes (int, optional): number of boxes to create. Default: 1.
        min_size (float, optional): relative minimum size of the box. Default: 0.1.
        max_size (float, optional): relative maximum size of the box. Default: 0.5.

    Returns:
    -------
        torch.Tensor: _description_

    """
    mask = torch.ones(1, 1, *image_size)
    margin = -int(min(*image_size) * min_size)
    for _ in range(num_boxes):
        yxyx = random_box(image_size, min_size, max_size, margin)
        mask[:, :, yxyx[0] : yxyx[2], yxyx[1] : yxyx[3]] = 0.0
    return mask


def make_masks(image: torch.Tensor, num_boxes: int = 1, min_size: float = 0.1, max_size: float = 0.5) -> torch.Tensor:
    """Make a batch of randomly sampled masks.

    Args:
    ----
        image (torch.Tensor): image tensor
        num_boxes (int, optional): number of boxes to create. Default: 1.
        min_size (float, optional): relative minimum size of the box. Default: 0.1.
        max_size (float, optional): relative maximum size of the box. Default: 0.5.

    Returns:
    -------
        torch.Tensor: _description_

    """
    B, _, H, W = image.size()
    masks = torch.ones(B, 1, H, W)
    for i in range(B):
        masks[i] = make_mask((H, W), num_boxes, min_size, max_size)
    masks = masks.to(image)
    return masks


def apply_mask(
    image: torch.Tensor, mask: torch.Tensor, mask_filler: Callable | torch.Tensor = torch.zeros
) -> torch.Tensor:
    """Apply generated mask to image.

    Args:
    ----
        image (torch.Tensor): image tensor
        mask (torch.Tensor): mask tensor
        mask_filler (Callable | torch.Tensor, optional): A callable which creates a tensor or another image tensor.
            Default: torch.zeros.

    Returns:
    -------
        torch.Tensor: _description_

    """
    if isinstance(mask_filler, Callable):
        mask_filler = mask_filler(image.size(), device=image.device)
    masked = image * mask + mask_filler * (1 - mask)
    return masked


"""filters"""


def gaussian_1d(kernel_size: int, sigma: float = 1.0) -> torch.Tensor:
    """1-D Gaussian filter.

    Args:
    ----
        kernel_size (int): Size of the filter.
        sigma (float, optional): Sigma. Default: 1.0.

    Returns:
    -------
        torch.Tensor: 1d-gaussian filter.

    """
    x = torch.arange(kernel_size) - kernel_size // 2
    if kernel_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma**2))
    return gauss / gauss.sum()


def gaussian_2d(kernel_size: int | tuple[int], sigma: float | tuple[float] = 1.0) -> torch.Tensor:
    """2-D Gaussian filter.

    Args:
    ----
        kernel_size (int | tuple[int]): Size of the filter.
        sigma (float | tuple[float], optional): Sigma. Default: 1.0.

    Returns:
    -------
        torch.Tensor: 2d-gaussian filter

    """
    kernel_size = to_2tuple(kernel_size)
    sigma = to_2tuple(sigma)
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma

    kernel_x = gaussian_1d(ksize_x, sigma_x)
    kernel_y = gaussian_1d(ksize_y, sigma_y)
    return torch.outer(kernel_x, kernel_y)


def _pascal_triangle(size: int):
    """Return binomial filter from size."""

    def c(n, k):
        if k <= 0 or n <= k:
            return 1
        else:
            return c(n - 1, k - 1) + c(n - 1, k)

    return [c(size - 1, j) for j in range(size)]


def pascal_1d(kernel_size: int) -> torch.Tensor:
    """1-D discrete aproximation of Gaussian filter.

    Args:
    ----
        kernel_size (int): filter size.

    Returns:
    -------
        torch.Tensor: 1-d gaussian filter.

    """
    kernel = torch.tensor(_pascal_triangle(kernel_size), dtype=torch.float32)
    return kernel / kernel.sum()


def pascal_2d(kernel_size: int | tuple[int]) -> torch.Tensor:
    """2-D discrete aproximation of Gaussian filter.

    Args:
    ----
        kernel_size (int | tuple[int]): Size of the filter.

    Returns:
    -------
        torch.Tensor: 2-d gaussian filter

    """
    kernel_size = to_2tuple(kernel_size)
    ksize_x, ksize_y = kernel_size
    kernel_x = pascal_1d(ksize_x)
    kernel_y = pascal_1d(ksize_y)
    return torch.outer(kernel_x, kernel_y)


def laplacian_1d(kernel_size: int) -> torch.Tensor:
    """1-D Laplacian filter.

    Args:
    ----
        kernel_size (int): size of the filter

    Raises:
    ------
        Exception: kernel_size is not an odd number.

    Returns:
    -------
        torch.Tensor: 1-D Laplacian filter

    """
    if kernel_size % 2 != 1:
        raise Exception(f'kernel_size must be odd but got {kernel_size}')
    kernel = torch.ones(kernel_size)
    kernel[kernel_size // 2] = -(kernel_size - 1)
    return kernel


def laplacian_2d(kernel_size: int | tuple[int]) -> torch.Tensor:
    """1-D Laplacian filter.

    Args:
    ----
        kernel_size (int | tuple[int]): size of the filter

    Returns:
    -------
        torch.Tensor: 2-D Laplacian filter

    """
    kernel_size = to_2tuple(kernel_size)
    kernel = torch.ones(kernel_size)
    kernel[kernel_size[0] // 2, kernel_size[1] // 2] = -(kernel.sum() - 1)
    return kernel


def sobel_2d(kernel_size: int, direction: int = 0) -> torch.Tensor:
    """2-D Sobel filter.

    Args:
    ----
        kernel_size (int): Size of the filter.
        direction (int, optional): 0: vertical 1: horizontal. Default: 0.

    Returns:
    -------
        torch.Tensor: 2-D Sobel filter

    """
    dx, dy = direction, 1 - direction
    kernel_x, kernel_y = cv2.getDerivKernels(dx, dy, kernel_size)
    kernel_x, kernel_y = torch.from_numpy(kernel_x), torch.from_numpy(kernel_y)
    return torch.outer(kernel_x.view(-1), kernel_y.view(-1))
