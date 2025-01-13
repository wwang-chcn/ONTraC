from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.typing import OptTensor

from ..log import *
from .dmon_exp_pool import SparseDMoNPooling
from .norm_gcn_conv import NormGCNConv


class GraphPooling(torch.nn.Module):
    """
    GraphPooling
    """

    def __init__(self, input_feats, k: int, dropout: float = 0, exponent: float = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.exponent = exponent
        self.pool = SparseDMoNPooling(channels=input_feats, k=k, dropout=0, exponent=self.exponent)
        self.k = k

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.pool.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        ptr: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        forward function
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            edge_index (torch.Tensor): Edge index tensor
                :math:`\mathbf{E} \in \mathbb{R}^{2 \times E}`.
            edge_weight (torch.Tensor, optional): Edge weight tensor
                :math:`\mathbf{E} \in \mathbb{R}^{E}`.
            batch (torch.Tensor, optional): Batch vector with shape :obj:`[N]` and type
                :obj:`torch.long`.
            ptr (torch.Tensor, optional): Pointer vector holding the start
                and end (exclusive) index of node of each graph in a
                mini-batch. Its shape must be of shape :obj:`[V + 1]` with
                :obj:`V` referring to the number of graphs in the mini-batch.
        Returns:
            Tensor: output feature matrix
        """
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x,
                                                                             edge_index=edge_index,
                                                                             edge_weight=edge_weight,
                                                                             batch=batch,
                                                                             ptr=ptr)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        ptr: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        s, out, out_adj, *_ = self.pool(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch, ptr=ptr)
        return s, out, out_adj


class GNN(torch.nn.Module):
    """
    GCN + GraphPooling
    """

    def __init__(self,
                 input_feats: int,
                 hidden_feats: int,
                 k: int,
                 n_gcn_layers: int = 2,
                 dropout: float = 0,
                 exponent: float = 1,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_gcn_layers = n_gcn_layers
        self.gcns = nn.ModuleList(
            [NormGCNConv(input_feats if i == 0 else hidden_feats, hidden_feats) for i in range(self.n_gcn_layers)])
        self.activations = nn.ModuleList([torch.nn.SELU() for _ in range(self.n_gcn_layers)])
        self.pool = GraphPooling(input_feats=hidden_feats, k=k, dropout=dropout, exponent=exponent)
        self.k = k

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for gcn in self.gcns:
            gcn.reset_parameters()
        self.pool.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        ptr: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            edge_index (torch.Tensor): Edge index tensor
                :math:`\mathbf{E} \in \mathbb{R}^{2 \times E}`.
            edge_weight (torch.Tensor, optional): Edge weight tensor
                :math:`\mathbf{E} \in \mathbb{R}^{E}`.
            batch (torch.Tensor): Batch vector with shape :obj:`[N]` and type
                :obj:`torch.long`.
            ptr (torch.Tensor): Pointer vector holding the start
                and end (exclusive) index of node of each graph in a
                mini-batch. Its shape must be of shape :obj:`[V + 1]` with
                :obj:`V` referring to the number of graphs in the mini-batch.
        Returns:
            Tensor: output feature matrix
        """
        for i in range(self.n_gcn_layers):
            x = self.activations[i](self.gcns[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x,
                                                                             edge_index=edge_index,
                                                                             edge_weight=edge_weight,
                                                                             batch=batch,
                                                                             ptr=ptr)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def evaluate(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        ptr: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        for i in range(self.n_gcn_layers):
            x = self.activations[i](self.gcns[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x,
                                                                             edge_index=edge_index,
                                                                             edge_weight=edge_weight,
                                                                             batch=batch,
                                                                             ptr=ptr)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        ptr: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        predict function
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            edge_index (torch.Tensor): Edge index tensor
                :math:`\mathbf{E} \in \mathbb{R}^{2 \times E}`.
            edge_weight (torch.Tensor, optional): Edge weight tensor
                :math:`\mathbf{E} \in \mathbb{R}^{E}`.
            batch (torch.Tensor): Batch vector with shape :obj:`[N]` and type
                :obj:`torch.long`.
            ptr (torch.Tensor, optional): Pointer vector holding the start
                and end (exclusive) index of node of each graph in a
                mini-batch. Its shape must be of shape :obj:`[V + 1]` with
                :obj:`V` referring to the number of graphs in the mini-batch.
        Returns:
            s (torch.Tensor): Node assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}`
            out (torch.Tensor): Output feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{B \times K \times H}`
            out_adj (torch.Tensor): Output adjacency matrix
                :math:`\mathbf{A} \in \mathbb{R}^{B \times K \times K}`
            """

        for i in range(self.n_gcn_layers):
            x = self.activations[i](self.gcns[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
        s, out, out_adj, *_ = self.pool(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch, ptr=ptr)
        return s, out, out_adj

    def predict_embed(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:
        for i in range(self.n_gcn_layers):
            x = self.activations[i](self.gcns[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
        return x


__all__ = ['GraphPooling', 'GNN']
