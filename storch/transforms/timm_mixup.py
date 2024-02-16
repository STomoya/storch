"""implementation of Mixup in pytorch-image-models.

from: https://github.com/rwightman/pytorch-image-models/blob/656757d26b78cbf35f526b454e4333b3cfda7012/timm/data/mixup.py
"""
# ruff: noqa: PLR2004

import warnings

import numpy as np
import torch
import torch.nn.functional as F


def rand_bbox(image_shape, lam, margin=0.0, count=None):
    """Create standard CutMix bounding-box.

    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
    ----
        image_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate

    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = image_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(image_shape, minmax, count=None):
    """Create Min-Max CutMix bounding-box.

    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.
    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
    ----
        image_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate

    """
    assert len(minmax) == 2
    img_h, img_w = image_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(image_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """Generate bbox and apply lambda correction."""
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(image_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(image_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(image_shape[-2] * image_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup(torch.nn.Module):
    """Mixup (+ CutMix) implementation in timm."""

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        correct_lambda=True,
    ) -> None:
        """Mixup.

        Args:
        ----
            mixup_alpha (float, optional): alpha used to sample from Beta dist. Default: 1.0.
            cutmix_alpha (float, optional): alpha used to sample from Beta dist. Default: 0.0.
            cutmix_minmax (_type_, optional): Min and max bbox ratios (as percent of image size). Default: None.
            prob (float, optional): Probability to apply Mixup. Default: 1.0.
            switch_prob (float, optional): Probability to switch to CutMix. Default: 0.5.
            mode (str, optional): Apply Mixup {batch,element}-wise. Default: 'batch'.
            correct_lambda (bool, optional): correct lambda. Default: True.

        """
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            if self.cutmix_alpha != 1.0:
                warnings.warn(
                    f'Mixup: cutmix_alpha will be forced to 1.0 when cutmix_minmax is given. (got {self.cutmix_alpha})',
                    UserWarning,
                    stacklevel=1,
                )
            self.cutmix_alpha = 1.0

        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.correct_lambda = correct_lambda
        self.enabled = True  # make it easy to disable

    def _params_per_element(self, batch_size):
        """Create parameters for Mixup applied element-wise."""
        lambdas = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool8)
        if self.enabled:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lambdas_mixed = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size),
                )
            elif self.mixup_alpha > 0.0:
                lambdas_mixed = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = np.ones(batch_size, dtype=np.bool8)
                lambdas_mixed = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            lambdas = np.where(np.random.rand(batch_size) < self.mix_prob, lambdas_mixed.astype(np.float32), lambdas)
        return lambdas, use_cutmix

    def _params_per_batch(self):
        """Create parameters for Mixup applied batch-wise."""
        lam = 1.0
        use_cutmix = False
        if self.enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mixed = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mixed = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mixed = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = float(lam_mixed)
        return lam, use_cutmix

    def _mix_element(self, x):
        """element-wise mixing."""
        batch_size, *image_shape = x.size()
        lambdas, use_cutmix = self._params_per_element(batch_size)
        x_org = x.clone()
        for i, lam in enumerate(lambdas):
            j = batch_size - i - 1
            if lam != 1:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(  # noqa: PLW2901
                        image_shape, lam, self.cutmix_minmax, self.correct_lambda
                    )
                    x[i, :, yl:yh, xl:xh] = x_org[j, :, yl:yh, xl:xh]
                    lambdas[i] = lam
                else:
                    x[i] = x[i] * lam + x_org[j] * (1 - lam)
        return torch.tensor(lambdas, device=x.device, dtype=x.dtype)

    def _mix_batch(self, x):
        """batch-wise mixing."""
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam
            )
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def forward(self, x):  # noqa: D102
        if self.mode == 'batch':
            lambdas = self._mix_batch(x)
        elif self.mode == 'element':
            lambdas = self._mix_element(x)
        return x, lambdas

    def mixed_cross_entropy_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, lambdas: torch.Tensor, label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """Cross entropy loss for mixed images. Use this function instead of nn.CrossEntropy.

        Args:
        ----
            logits (torch.Tensor): Output logits of the model.
            targets (torch.Tensor): targets.
            lambdas (torch.Tensor): lambdas returned by mixup()
            label_smoothing (float): Amount of smoothing.

        Returns:
        -------
            torch.Tensor: the loss

        """
        ce_loss_a = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing) * lambdas
        ce_loss_b = F.cross_entropy(logits, targets.flip(0), reduction='none', label_smoothing=label_smoothing) * (
            1 - lambdas
        )
        loss = (ce_loss_a + ce_loss_b).mean()
        return loss
