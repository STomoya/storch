"""Metric utils."""

import torch


def reduce_dimension(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Dimension reduction of calculated loss.

    Args:
    ----
        tensor (torch.Tensor): Loss.
        mode (str): reduction mode.

    Returns:
    -------
        torch.Tensor: reduced loss.
    """
    if mode == 'sum':
        return tensor.flatten(1).sum()
    elif mode == 'mean':
        return tensor.flatten(1).mean()
    elif mode == 'elementwise_mean':
        return tensor.flatten(1).mean(dim=1)
    elif mode == 'none':
        return tensor
