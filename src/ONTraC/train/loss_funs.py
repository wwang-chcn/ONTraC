import torch
from torch import Tensor

from ..log import debug


def moran_I_features(X: Tensor, W: Tensor, mask: Tensor) -> Tensor:
    r"""
    Calculate Moran's I.
    :math:`I = \frac{n}{\sum_{i=1}^n\sum_{j=1}^n w_{ij}}\frac{(X-\bar{X})^T W (X-\bar{X})}{\sum_{i=1}^n (X_i-\bar{X})^2}`
    :param X: Tensor, shape: (N, F)
    :param W: Tensor, shape: (N, N)
    :param mask: Tensor, shape: (N, )
    :return: moran_I: Tensor, shape: (F, )
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
    # debug(f'numerator: {torch.diagonal(X_masked_.T @ W_masked @ X_masked_)}.')  # F x 1
    # debug(f'denominator: {(X_masked_**2).sum(dim = 0, keepdim=True)}.')  # F x 1
    return n / W_masked_sum * torch.diagonal(X_masked_.T @ W_masked @ X_masked_) / (X_masked_**2).sum(dim=0,
                                                                                                      keepdim=True)


def graph_smooth_loss(z: Tensor, adj: Tensor, mask: Tensor) -> Tensor:
    r"""
    Graph smooth loss using -1 * moran's I
    :math: `L_{smooth} = \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}w_{ij}(z_i - z_j)^2`
    
    :param z: hidden embedding tensor
        :math:`\mathbf{Z} \in \mathbb{R}^{B \times N \times F}`, with
        batch-size :math:`B`, (maximum) number of nodes :math:`N` for
        each graph, and feature dimension :math:`F`.
    :param adj: adjacency tensor
        :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
    :param mask: mask tensor
        :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}`
    :return: loss tensor
    """
    # --- inputs shape check ---
    z = z.unsqueeze(0) if z.dim() == 2 else z  # B x N x F
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # B x N x N
    mask = mask.unsqueeze(0) if mask.dim() == 1 else mask  # B x N
    B, N, F = z.size()

    # --- calculate moran's I ---
    MI_list = torch.cat(list(moran_I_features(z[i, :, :], adj[i], mask[i]) for i in range(B)))
    # debug(f'MI_list: {MI_list}')

    # --- calculate loss ---
    MI_loss = -1 * MI_list.mean()
    MI_loss = MI_loss + 1  # make the loss positive

    return MI_loss


def within_cluster_variance_loss(x: Tensor, s: Tensor, mask: Tensor) -> Tensor:
    """
    Calculate within cluster variance loss
    Args:
        x: input tensor, shape: (B, N, F)
        s: soft cluster assignment matrix, shape: (B, N, C)
        mask: mask tensor, shape: (B, N)
    Returns:
        loss: within cluster variance loss
    """

    # --- Extend mask to match the dimensions of x and s ---
    mask_extended_x = mask.unsqueeze(-1).expand_as(x)  # B x N x F
    mask_extended_s = mask.unsqueeze(-1).expand_as(s)  # B x N x C

    # --- Apply mask to x and s ---
    masked_x = x * mask_extended_x  # B x N x F
    masked_s = s * mask_extended_s  # B x N x C

    # --- Compute the cluster centroids ---
    sum_x = torch.einsum('bnf,bnc->cf', masked_x, masked_s)  # C x F, sum of each cluster
    num_points = torch.sum(masked_s, dim=(0, 1)) + 1e-10  # avoid divide by zero
    centroids = sum_x / num_points.unsqueeze(-1)  # C x F, mean of each cluster

    # --- Compute the squared distance from the centroids ---
    expanded_centroids = torch.einsum('cf,bnc->bnf', centroids, masked_s)  # B x N x F
    squared_distance = (masked_x - expanded_centroids)**2  # B x N x F

    # --- Compute the loss ---
    loss = torch.sum(squared_distance * mask_extended_x) / mask.sum() / x.shape[-1]  # average over all nodes with mask

    return loss


def masked_variance(x, mask):
    """
    Args:
        x: input tensor, shape: (B, N, F)
        mask: mask tensor, shape: (B, N)
    Returns:
        variance
    """

    # Apply the mask
    x_masked = x * mask.unsqueeze(-1)

    # Calculate the mean
    sum_x = x_masked.sum()
    sum_mask = mask.sum() * x.shape[-1]
    mean_x = sum_x / sum_mask

    # Calculate the variance
    diff = x_masked - mean_x.unsqueeze(-1).unsqueeze(-1)
    var_x = torch.sum(diff * diff * mask.unsqueeze(-1)) / sum_mask

    return var_x
