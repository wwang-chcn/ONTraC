from networkx import laplacian_matrix
import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor

from ..log import *


class NormDenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, adj: Tensor, mask: OptTensor = None) -> Tensor:
        r"""
        :math: `\mathbf{\hat{L}}X\mathbf{\Theta}`, with
               `\mathbf{\hat{L}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2} + I`

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """

        # inputs shape check
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        # X\mathbf{\Theta}
        out = self.lin(x)

        # calculate \mathbf{\hat{L}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2} + I
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        eye_matrix = torch.eye(N).to(adj.device).unsqueeze(0).repeat(B,1,1)
        laplacian = adj + eye_matrix if not self.improved else adj + 2 * eye_matrix

        # \mathbf{\hat{L}}X\mathbf{\Theta}
        out = torch.matmul(laplacian, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
