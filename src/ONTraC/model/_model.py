"""This module contains model level classes for ONTraC, including the GNN encoder and pooling layers."""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from copy import deepcopy

from ..log import *
from .dmon_exp_pool import DMoNPooling
from .norm_dense_gcn_conv import NormDenseGCNConv


class GraphPooling(torch.nn.Module):
    """Wrapper around :class:`DMoNPooling` used by ONTraC."""

    def __init__(self, input_feats, k: int, dropout: float = 0, exponent: float = 1, *args, **kwargs) -> None:
        """Initialize graph pooling module.

                Parameters
                ----------
        input_feats :
            int or list[int]
                    Input feature dimension (or MLP channel list) for assignment MLP.
        k :
            int
                    Number of pooled clusters.
        dropout :
            float, default=0
                    Dropout used inside the pooling assignment head.
        exponent :
            float, default=1
                    Assignment logit temperature scaling factor.
        """
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.exponent = exponent
        self.pool = DMoNPooling(channels=input_feats, k=k, dropout=0, exponent=self.exponent)
        self.k = k

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters in the internal pooling layer."""
        self.pool.reset_parameters()

    def forward(
        self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
                forward function
        Parameters
        ----------
                    x (torch.Tensor): Node feature tensor
                        :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times F}`, with
                        batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                        each graph, and feature dimension :math:`F`.
                    adj (torch.Tensor): Adjacency tensor
                        :math:`\\mathbf{A} \\in \\mathbb{R}^{B \\times N \\times N}`.
                    mask (torch.Tensor, optional): Mask matrix
                        :math:`\\mathbf{M} \\in {\\{ 0, 1 \\}}^{B \\times N}` indicating
                        the valid nodes for each graph. (default: :obj:`None`)
        Returns
        -------
        Tensor
            output feature matrix
        """
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def predict(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Run pooling inference and return assignment, pooled features, and adjacency."""
        s, out, out_adj, *_ = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj


class GNN(torch.nn.Module):
    """Stacked dense-GCN encoder followed by DMoN graph pooling."""

    def __init__(
        self,
        input_feats: int,
        hidden_feats: int,
        k: int,
        n_gcn_layers: int = 2,
        dropout: float = 0,
        exponent: float = 1,
        *args,
        **kwargs,
    ) -> None:
        """Initialize ONTraC graph clustering model.

                Parameters
                ----------
        input_feats :
            int
                    Input node-feature dimension.
        hidden_feats :
            int
                    Hidden feature dimension used by all GCN layers.
        k :
            int
                    Number of niche clusters.
        n_gcn_layers :
            int, default=2
                    Number of dense GCN encoder layers.
        dropout :
            float, default=0
                    Dropout probability used inside pooling assignment head.
        exponent :
            float, default=1
                    Assignment logit temperature scaling factor for pooling.
        """
        super().__init__(*args, **kwargs)
        self.n_gcn_layers = n_gcn_layers
        self.gcns = nn.ModuleList(
            [NormDenseGCNConv(input_feats if i == 0 else hidden_feats, hidden_feats) for i in range(self.n_gcn_layers)]
        )
        self.activations = nn.ModuleList([torch.nn.SELU() for _ in range(self.n_gcn_layers)])
        self.pool = GraphPooling(input_feats=hidden_feats, k=k, dropout=dropout, exponent=exponent)
        self.k = k

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all encoder and pooling parameters."""
        for gcn in self.gcns:
            gcn.reset_parameters()
        self.pool.reset_parameters()

    def forward(
        self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
                forward function
                X' = \\mathbf{\\hat{L}}X\\mathbf{\\Theta}
                \\mathbf{\\hat{A}} = \\mathbf{A} + \\mathbf{I}
                \\mathbf(\\hat{D}) = \\sum_{j=1}^N \\mathbf{\\hat{A}}_{ij}
                \\mathbf{\\hat{L}} = \\mathbf{\\hat{D}}^{-1/2}\\mathbf{\\hat{A}}\\mathbf{\\hat{D}}^{-1/2}
        Parameters
        ----------
                    x (torch.Tensor): Node feature tensor
                        :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times F}`, with
                        batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                        each graph, and feature dimension :math:`F`.
                    adj (torch.Tensor): Adjacency tensor
                        :math:`\\mathbf{A} \\in \\mathbb{R}^{B \\times N \\times N}`.
                    mask (torch.Tensor, optional): Mask matrix
                        :math:`\\mathbf{M} \\in {\\{ 0, 1 \\}}^{B \\times N}` indicating
                        the valid nodes for each graph. (default: :obj:`None`)
        Returns
        -------
        Tensor
            output feature matrix
        """
        for i in range(self.n_gcn_layers):
            x = self.activations[i](self.gcns[i](x=x, adj=adj, mask=mask))
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def evaluate(
        self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run a forward-style pass intended for evaluation loops.

        Returns the same tuple as :meth:`forward`.
        """
        for i in range(self.n_gcn_layers):
            x = self.activations[i](self.gcns[i](x=x, adj=adj, mask=mask))
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def predict(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
                predict function
        Parameters
        ----------
                    x (torch.Tensor): Node feature tensor
                        :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times F}`, with
                        batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                        each graph, and feature dimension :math:`F`.
                    adj (torch.Tensor): Adjacency tensor
                        :math:`\\mathbf{A} \\in \\mathbb{R}^{B \\times N \\times N}`.
                    mask (torch.Tensor, optional): Mask matrix\\
                        :math:`\\mathbf{M} \\in {\\{ 0, 1 \\}}^{B \\times N}` indicating\\
                        the valid nodes for each graph. (default: :obj:`None`)
        Returns
        -------
        s (torch.Tensor)
            Node assignment matrix

            math:`\\mathbf{S} \\in \\mathbb{R}^{B \\times N \\times K}`
        out (torch.Tensor)
            Output feature matrix

            math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times K \\times H}`
        out_adj (torch.Tensor)
            Output adjacency matrix

            math:`\\mathbf{A} \\in \\mathbb{R}^{B \\times K \\times K}`
                    """

        for i in range(self.n_gcn_layers):
            x = self.activations[i](self.gcns[i](x=x, adj=adj, mask=mask))
        s, out, out_adj, *_ = self.pool(x=x, adj=adj, mask=mask)
        return s, out, out_adj

    def predict_embed(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Return encoder embeddings before pooling."""
        for i in range(self.n_gcn_layers):
            x = self.activations[i](self.gcns[i](x=x, adj=adj, mask=mask))
        return x


__all__ = ["GraphPooling", "GNN"]
