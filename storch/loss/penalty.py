
from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.cuda.amp import GradScaler, autocast

from storch.loss._base import Loss


@autocast(enabled=False)
def calc_grad(
    outputs: torch.Tensor, inputs: torch.Tensor,
    scaler: Optional[GradScaler]=None
) -> torch.Tensor:
    """calculate gradients with AMP support

    Args:
        outputs (torch.Tensor): Output tensor from a model
        inputs (torch.Tensor): Input tensor to a model
        scaler (Optional[GradScaler], optional): GradScaler object if using AMP. Defaults to None.

    Returns:
        torch.Tensor: The gradients of the input.
    """
    if isinstance(scaler, GradScaler):
        outputs = scaler.scale(outputs)
    ones = torch.ones(outputs.size(), device=outputs.device)
    gradients = grad(
        outputs=outputs, inputs=inputs, grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    if isinstance(scaler, GradScaler):
        gradients = gradients / scaler.get_scale()
    return gradients

class Penalty(Loss):
    """Penalty

    Examples:
        >>> gp = Penalty()

        >>> # Examples for when your D has multiple outputs
        >>> # 1. Use only first output
        >>> gp.filter_output = lambda d_output: d_output[0]
        >>> # 2. Sum all outputs
        >>> gp.filter_output = lambda d_output: sum(list(map(lambda x:x.sum(), d_output)))
        >>> # ...etc.
        >>> # It should be a callable that receives outputs from D and outputs a scalar torch.Tensor.
        >>> # Penalty().filter_output defaults to lambda x:x

        >>> # calculate penalty
        >>> loss = gp(...)
        >>> loss.backward()
    """
    def __init__(self, return_all: bool = False) -> None:
        super().__init__(return_all=return_all)
        self.filter_output = lambda x: x

class gradient_penalty(Penalty):
    def __call__(self,
        real: torch.Tensor, fake: torch.Tensor,
        D: nn.Module, scaler: Optional[GradScaler]=None, center: float=1.,
        d_aux_input: tuple=tuple()
    ) -> torch.Tensor:
        """gradient penalty + 0-centered gradient penalty

        Args:
            real (torch.Tensor): Real samples
            fake (torch.Tensor): Fake samples
            D (nn.Module): Discriminator
            scaler (Optional[GradScaler], optional): GradScaler object if using AMP. Default: None.
            center (float, optional): <center>-centered gradient penalty. Default: 1.
            d_aux_input (tuple, optional): Auxiliary inputs to discriminator. Default: tuple().

        Returns:
            torch.Tensor: The loss
        """

        assert center in [1., 0.]

        device = real.device

        alpha = torch.rand(1, device=device)
        x_hat = real * alpha + fake * (1 - alpha)
        x_hat = Variable(x_hat, requires_grad=True)

        d_x_hat = self.filter_output(D(x_hat, *d_aux_input))

        gradients = calc_grad(d_x_hat, x_hat, scaler)

        gradients = gradients.reshape(gradients.size(0), -1)
        penalty = (gradients.norm(2, dim=1) - center).pow(2).mean()

        return penalty

class dragan_penalty(Penalty):
    def __call__(self,
        real: torch.Tensor, D: nn.Module,
        scaler: Optional[GradScaler]=None, center: float=1.,
        d_aux_input: tuple=tuple()
    ) -> torch.Tensor:
        """DRAGAN gradient penalty

        Args:
            real (torch.Tensor): Real samples
            D (nn.Module): Discriminator
            scaler (Optional[GradScaler], optional): GradScaler object if using AMP. Default: None.
            center (float, optional): <center>-centered gradient penalty. Default: 1.
            d_aux_input (tuple, optional): Auxiliary inputs to discriminator. Default: tuple().

        Returns:
            torch.Tensor: The loss.
        """

        device = real.device

        alpha = torch.rand((real.size(0), 1, 1, 1), device=device).expand(real.size())
        beta = torch.rand(real.size(), device=device)
        x_hat = real * alpha + (1 - alpha) * (real + 0.5 * real.std() * beta)
        x_hat = Variable(x_hat, requires_grad=True)

        d_x_hat = self.filter_output(D(x_hat, *d_aux_input))

        gradients = calc_grad(d_x_hat, x_hat, scaler)

        gradients = gradients.reshape(gradients.size(0), -1)
        penalty = (gradients.norm(2, dim=1) - center).pow(2).mean()

        return penalty

class r1_regularizer(Penalty):
    def __call__(self,
        real: torch.Tensor, D: nn.Module,
        scaler: Optional[GradScaler]=None, d_aux_input: tuple=tuple()
    ) -> torch.Tensor:
        """R1 Regularizer

        Args:
            real (torch.Tensor): Real samples
            D (nn.Module): Discriminator
            scaler (Optional[GradScaler], optional): GradScaler object if using AMP. Default: None.
            d_aux_input (tuple, optional): Auxiliary inputs to discriminator. Default: tuple().

        Returns:
            torch.Tensor: The loss.
        """
        real_loc = Variable(real, requires_grad=True)

        d_real_loc = self.filter_output(D(real_loc, *d_aux_input))

        gradients = calc_grad(d_real_loc, real_loc, scaler)

        gradients = gradients.reshape(gradients.size(0), -1)
        penalty = gradients.norm(2, dim=1).pow(2).mean() / 2.

        return penalty

class r2_regularizer(r1_regularizer):
    def __call__(self,
        fake: torch.Tensor, D: nn.Module,
        scaler: Optional[GradScaler], d_aux_input: tuple=tuple()
    ) -> torch.Tensor:
        """R2 Regularizer

        Args:
            fake (torch.Tensor): Fake samples
            D (nn.Module): Discriminator
            scaler (Optional[GradScaler], optional): GradScaler object if using AMP. Default: None.
            d_aux_input (tuple, optional): Auxiliary inputs to discriminator. Default: tuple().

        Returns:
            torch.Tensor: The loss.
        """
        return super().__call__(fake, D, scaler, d_aux_input)
