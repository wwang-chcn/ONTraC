from typing import Optional
from networkx import laplacian_matrix

from torch_geometric.nn import DenseGCNConv

import torch
from torch import Tensor
from torch.nn import Linear

from utils import apply_along_axis


class NormDenseGraphConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
    ):
        assert aggr in ['add', 'mean', 'max']
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        r"""
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
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, C = x.size()

        # calculate normalized adjacency matrix
        # .. math:: \mathbf{\hat{L}}X\mathbf{\Theta}
        # .. math:: \mathbf{\hat{L}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}

        deg_vector = torch.sum(adj, dim=-1)
        deg_matrix = apply_along_axis(torch.diag, deg_vector, dim=1)
        norm_deg_matrix = deg_matrix.pow(-0.5)
        norm_deg_matrix[norm_deg_matrix == float('inf')] = 0
        norm_adj_matrix = torch.matmul(torch.matmul(norm_deg_matrix, adj), norm_deg_matrix) 
        eye_matrix = torch.eye(N).to(adj.device).unsqueeze(0).repeat(B,1,1)
        norm_laplacian_matrix = norm_adj_matrix + eye_matrix

        if self.aggr == 'add':
            out = torch.matmul(norm_laplacian_matrix, x)
        elif self.aggr == 'mean':
            out = torch.matmul(norm_laplacian_matrix, x)
            out = out / norm_laplacian_matrix.sum(dim=-1, keepdim=True).clamp_(min=1)
        elif self.aggr == 'max':
            out = x.unsqueeze(-2).repeat(1, 1, N, 1)
            norm_laplacian_matrix = norm_laplacian_matrix.unsqueeze(-1).expand(B, N, N, C)
            out[norm_laplacian_matrix == 0] = float('-inf')
            out = out.max(dim=-3)[0]
            out[out == float('-inf')] = 0.
        else:
            raise NotImplementedError

        out = self.lin_rel(out)
        out = out + self.lin_root(x)

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
