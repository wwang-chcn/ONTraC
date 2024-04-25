import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from ..log import *
from ..utils.decorators import epoch_filter_decorator, selective_args_decorator


def loss_record(epoch: int, batch: int, loss: Tensor, **kwargs):
    """
    Loss record function
    :param epoch: epoch number
    :param loss: loss tensor
    :param r_loss: reconstruction loss tensor
    :param g_loss: graph smooth loss tensor
    :return: None
    """
    other_loss_text = ''
    for key, value in kwargs.items():
        if 'loss' in key:
            other_loss_text += f', {key}: {value}'
    info(f'epoch: {epoch}, batch: {batch}, loss: {loss}{other_loss_text}')


def _moran_I_factor_tensor(X: Tensor, W: Tensor, mask: Tensor) -> Tensor:
    r"""
    Calculate Moran's I.
    :math:`I = \frac{n}{\sum_{i=1}^n\sum_{j=1}^n w_{ij}}\frac{(X-\bar{X})^T W (X-\bar{X})}{\sum_{i=1}^n (X_i-\bar{X})^2}`
    :param X: np.ndarray, shape: (N, F)
    :param W: np.ndarray, shape: (N, N)
    :param mask: np.ndarray, shape: (N, )
    :return: moran_I
    """
    # --- input shape check ---
    X = X.unsqueeze(1) if X.dim() == 1 else X

    assert X.shape[0] == W.shape[0] == mask.shape[0]
    assert W.shape[0] == W.shape[1]

    F = X.shape[1]

    # --- mask ---
    X_masked = X[mask].reshape((-1, F))
    W_masked = W[mask][:, mask]
    # debug(f'X_masked shape: {X_masked.shape}.')
    # debug(f'W_masked shape: {W_masked.shape}.')

    # --- calculate ---
    n = X_masked.shape[0]
    X_masked_ = X_masked - X_masked.mean(dim=0, keepdim=True)
    W_masked_sum = W_masked.sum()
    # debug(f'X_masked_ shape: {X_masked_.shape}.')
    # debug(f'n: {n}.')
    # debug(f'W_masked_sum: {W_masked_sum}.')
    # debug(f'numerator: {torch.diagonal(X_masked_.T @ W_masked @ X_masked_)}.')
    # debug(f'denominator shape: {(X_masked_**2).sum(dim = 0, keepdim=True).shape}.')
    # debug(f'denominator: {(X_masked_**2).sum(dim = 0, keepdim=True)}.')
    return n / W_masked_sum * torch.diagonal(X_masked_.T @ W_masked @ X_masked_) / (X_masked_**2).sum(dim=0,
                                                                                                      keepdim=True)


@selective_args_decorator
def moran_I(output_dir: str, step: int, epoch: int, data: Data, z: Tensor) -> None:
    """
    Moran's I record function
    :param epoch: epoch number
    :param data: torch_geometric.data.Data
    :param z: hidden embedding tensor
    :return: None
    """

    # --- check if epoch is multiple of step ---
    if epoch % step != 0:
        return

    # --- inputs shape check ---
    z = z.unsqueeze(0) if z.dim() == 2 else z  # B x N x F
    adj = data.adj.unsqueeze(0) if data.adj.dim() == 2 else data.adj  # B x N x N
    mask = data.mask.unsqueeze(0) if data.mask.dim() == 1 else data.mask  # B x N
    B, N, F = z.size()

    # --- calculate moran's I ---
    MI_list = np.array(
        list(_moran_I_factor_tensor(z[i, :, :], adj[i], mask[i]).detach().cpu().numpy()
             for i in range(B))).reshape(B, F)

    np.savetxt(fname=f'{output_dir}/moran_I_{epoch}.csv', X=MI_list, delimiter=',')


@selective_args_decorator
@epoch_filter_decorator
def z_record(output_dir: str, epoch: int, z: Tensor, data: Batch) -> None:
    """
    Hidden embedding record function
    :param epoch: epoch number
    :param batch: batch number
    :param z: hidden embedding tensor
    :return: None
    """
    # --- check whether record ---

    # --- inputs shape check ---
    z = z.unsqueeze(0) if z.dim() == 2 else z
    B, N, F = z.size()
    assert B == len(data.x)  # type: ignore

    for index, name in enumerate(data.name):  # type: ignore
        np.savetxt(fname=f'{output_dir}/Epoch_{epoch}/{name}_z.csv.gz',
                   X=z[index].detach().cpu().numpy(),
                   delimiter=',')


@selective_args_decorator
@epoch_filter_decorator
def s_record(output_dir: str, epoch: int, s: Tensor, data: Batch) -> None:
    """
    Assignment record function
    :param epoch: epoch number
    :param batch: batch number
    :param s: assignment tensor
    :return: None
    """
    # --- check whether record ---

    # --- inputs shape check ---
    s = s.unsqueeze(0) if s.dim() == 2 else s
    B, N, F = s.size()
    assert B == len(data.x)  # type: ignore

    for index, name in enumerate(data.name):  # type: ignore
        np.savetxt(fname=f'{output_dir}/Epoch_{epoch}/{name}_s.csv.gz',
                   X=s[index].detach().cpu().numpy(),
                   delimiter=',')


@selective_args_decorator
@epoch_filter_decorator
def out_record(output_dir: str, epoch: int, out: Tensor, data: Batch) -> None:
    """
    Output record function
    :param epoch: epoch number
    :param batch: batch number
    :param out: output tensor
    :return: None
    """
    # --- check whether record ---

    # --- inputs shape check ---
    out = out.unsqueeze(0) if out.dim() == 2 else out
    B, N, F = out.size()
    assert B == len(data.x)  # type: ignore

    for index, name in enumerate(data.name):  # type: ignore
        np.savetxt(fname=f'{output_dir}/Epoch_{epoch}/{name}_out.csv.gz',
                   X=out[index].detach().cpu().numpy(),
                   delimiter=',')


@selective_args_decorator
@epoch_filter_decorator
def out_adj_record(output_dir: str, epoch: int, out_adj: Tensor, data: Batch) -> None:
    """
    Output adj record function
    :param epoch: epoch number
    :param batch: batch number
    :param out_adj: output adj tensor
    :return: None
    """
    # --- check whether record ---

    # --- inputs shape check ---
    out_adj = out_adj.unsqueeze(0) if out_adj.dim() == 2 else out_adj
    B, N, F = out_adj.size()
    assert B == len(data.x)  # type: ignore

    for index, name in enumerate(data.name):  # type: ignore
        np.savetxt(fname=f'{output_dir}/Epoch_{epoch}/{name}_out_adj.csv.gz',
                   X=out_adj[index].detach().cpu().numpy(),
                   delimiter=',')
