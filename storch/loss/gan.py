"""GAN losses."""
# ruff: noqa:D102 D107

import torch
import torch.nn as nn
import torch.nn.functional as F

from storch.loss._base import Loss


class Adversarial(Loss):
    """Base class for adversarial losses."""

    def real_loss(self, prob: torch.Tensor) -> torch.Tensor:
        """Classify logits as real images.

        Args:
            prob (torch.Tensor): D output to classify as real.

        Returns:
            torch.Tensor: The loss

        """
        raise NotImplementedError()

    def fake_loss(self, prob: torch.Tensor) -> torch.Tensor:
        """Classify logits as fake images.

        Args:
            prob (torch.Tensor): D output to classify as fake.

        Returns:
            torch.Tensor: The loss

        """
        raise NotImplementedError()

    def d_loss(self, real_prob: torch.Tensor, fake_prob: torch.Tensor) -> torch.Tensor:
        """Calc discriminator loss.

        Args:
            real_prob (torch.Tensor): D output of real inputs
            fake_prob (torch.Tensor): D output of fake inputs

        Returns:
            torch.Tensor: The loss

        """
        rl = self.real_loss(real_prob)
        fl = self.fake_loss(fake_prob)
        loss = rl + fl
        if self._return_all:
            return loss, rl, fl
        return loss

    def g_loss(self, fake_prob: torch.Tensor) -> torch.Tensor:
        """Calc generator loss.

        Args:
            fake_prob (torch.Tensor): D output of fake inputs.

        Returns:
            torch.Tensor: The loss

        """
        return self.real_loss(fake_prob)


class GANLoss(Adversarial):
    """Original GAN loss.

    Ld = E[log(D(x)) + log(1 - D(G(z)))]
    Lg = E[log(1 - D(G(z)))]
    """

    def __init__(self, return_all: bool = False) -> None:
        super().__init__(return_all=return_all)
        self.criterion = nn.BCEWithLogitsLoss()

    def real_loss(self, prob: torch.Tensor) -> torch.Tensor:
        valid = torch.ones(prob.size(), device=prob.device)
        return self.criterion(prob, valid)

    def fake_loss(self, prob: torch.Tensor) -> torch.Tensor:
        fake = torch.zeros(prob.size(), device=prob.device)
        return self.criterion(prob, fake)


class LSGANLoss(GANLoss):
    """least square GAN loss (a,b,c = 0,1,1).

    Ld = 1/2*E[(D(x) - 1)^2] + 1/2*E[D(G(z))^2]
    Lg = 1/2*E[(D(G(z)) - 1)^2]
    """

    def __init__(self, return_all: bool = False) -> None:
        super().__init__(return_all)
        self.criterion = nn.MSELoss()

    def d_loss(self, real_prob: torch.Tensor, fake_prob: torch.Tensor) -> torch.Tensor:
        rl = self.real_loss(real_prob) * 0.5
        fl = self.fake_loss(fake_prob) * 0.5
        loss = rl + fl
        if self._return_all:
            return loss, rl, fl
        return loss

    def g_loss(self, fake_prob: torch.Tensor) -> torch.Tensor:
        return self.real_loss(fake_prob) * 0.5


class NonSaturatingLoss(Adversarial):
    """non-saturating GAN loss.

    Ld = E[log(D(x)) + log(1 - D(G(z)))]
    Lg = E[log(D(G(z)))]
    """

    def real_loss(self, prob: torch.Tensor) -> torch.Tensor:
        return F.softplus(-prob).mean()

    def fake_loss(self, prob: torch.Tensor) -> torch.Tensor:
        return F.softplus(prob).mean()


class WGANLoss(Adversarial):
    """WGAN loss.

    Ld = E[D(G(z))] - E[D(x)]
    Lg = -E[D(G(z))]
    """

    def real_loss(self, prob: torch.Tensor) -> torch.Tensor:
        return -prob.mean()

    def fake_loss(self, prob: torch.Tensor) -> torch.Tensor:
        return prob.mean()


class HingeLoss(Adversarial):
    """Hinge loss.

    Ld = - E[min(0, 1 + D(x))] - E[min(0, -1 - D(G(z)))]
    Lg = - E[D(G(z))]
    """

    def real_loss(self, prob: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 - prob).mean()

    def fake_loss(self, prob: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 + prob).mean()

    def g_loss(self, fake_prob: torch.Tensor) -> torch.Tensor:
        return -fake_prob.mean()
