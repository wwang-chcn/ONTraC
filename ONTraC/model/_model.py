from typing import Optional, Tuple

import torch
from torch import Tensor

from ..log import *
from .dmon_exp_pool import DMoNPooling
from .norm_dense_gcn_conv import NormDenseGCNConv


class NodePooling(torch.nn.Module):
    """
    NodePooling
    """

    def __init__(self, input_feats, k: int, dropout: float = 0, exponent: float = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.exponent = exponent
        self.pool = DMoNPooling(channels=input_feats, k=k, dropout=0, exponent=self.exponent)
        self.k = k

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pool.reset_parameters()

    def forward(self,
                x: Tensor,
                adj: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        forward function
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        Returns:
            Tensor: output feature matrix
        """
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def predict(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        s, out, out_adj, *_ = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj


class GraphPooling(torch.nn.Module):
    """
    GNN with Node Pooling
    """

    def __init__(self,
                 input_feats: int,
                 hidden_feats: int,
                 k: int,
                 dropout: float = 0,
                 exponent: float = 1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gcn1 = NormDenseGCNConv(input_feats, hidden_feats)
        self.activation1 = torch.nn.SELU()
        self.gcn2 = NormDenseGCNConv(hidden_feats, hidden_feats)
        self.activation2 = torch.nn.SELU()
        self.pool = NodePooling(input_feats=hidden_feats, k=k, dropout=dropout, exponent=exponent)
        self.k = k

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.pool.reset_parameters()

    def forward(self,
                x: Tensor,
                adj: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        forward function
        X' = \mathbf{\hat{L}}X\mathbf{\Theta}
        \mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}
        \mathbf(\hat{D}) = \sum_{j=1}^N \mathbf{\hat{A}}_{ij}
        \mathbf{\hat{L}} = \mathbf{\hat{D}}^{-1/2}\mathbf{\hat{A}}\mathbf{\hat{D}}^{-1/2}
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        Returns:
            Tensor: output feature matrix
        """
        x = self.activation1(self.gcn1(x=x, adj=adj, mask=mask))
        x = self.activation2(self.gcn2(x=x, adj=adj, mask=mask))
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def evaluate(self,
                 x: Tensor,
                 adj: Tensor,
                 mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = self.activation1(self.gcn1(x=x, adj=adj, mask=mask))
        x = self.activation2(self.gcn2(x=x, adj=adj, mask=mask))
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def predict(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        predict function
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (torch.Tensor, optional): Mask matrix\
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating\
                the valid nodes for each graph. (default: :obj:`None`)
        Returns:
            s (torch.Tensor): Node assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}`
            out (torch.Tensor): Output feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{B \times K \times H}`
            out_adj (torch.Tensor): Output adjacency matrix
                :math:`\mathbf{A} \in \mathbb{R}^{B \times K \times K}`
            """
        x = self.activation1(self.gcn1(x=x, adj=adj, mask=mask))
        x = self.activation2(self.gcn2(x=x, adj=adj, mask=mask))
        s, out, out_adj, *_ = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj

    def predict_embed(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.activation1(self.gcn1(x=x, adj=adj, mask=mask))
        x = self.activation2(self.gcn2(x=x, adj=adj, mask=mask))
        return x


__all__ = ['NodePooling', 'GraphPooling']
