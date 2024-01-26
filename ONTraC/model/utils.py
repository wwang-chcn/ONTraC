import torch


def apply_along_axis(function, x, dim: int = 0):
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=dim)], dim=dim)
