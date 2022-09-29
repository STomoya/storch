import torch


def reduce_dimension(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'sum':
        return tensor.flatten(1).sum()
    elif mode == 'mean':
        return tensor.flatten(1).mean()
    elif mode == 'elementwise_mean':
        return tensor.flatten(1).mean(dim=1)
    elif mode == 'none':
        return tensor
