from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.mincut_pool import _rank3_trace
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.typing import OptTensor

EPS = 1e-15


class SparseDMoNPooling(torch.nn.Module):
    r"""The spectral modularity pooling operator from the `"Graph Clustering
    with Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the learned cluster assignment matrix, the pooled node feature
    matrix, the coarsened symmetrically normalized adjacency matrix, and three
    auxiliary objectives: (1) The spectral loss

    .. math::
        \mathcal{L}_s = - \frac{1}{2m}
        \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})}

    where :math:`\mathbf{B}` is the modularity matrix, (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F

    where :math:`C` is the number of clusters, and (3) the cluster loss

    .. math::
        \mathcal{L}_c = \frac{\sqrt{C}}{n}
        {\left\|\sum_i\mathbf{C_i}^{\top}\right\|}_F - 1.

    .. note::

        For an example of using :class:`DMoNPooling`, see
        `examples/proteins_dmon_pool.py
        <https://github.com/pyg-team/pytorch_geometric/blob
        /master/examples/proteins_dmon_pool.py>`_.

    .. note::

        correction of original DMONPooing implemente have been merged into PyG #8285 <https://github.com/pyg-team/pytorch_geometric/pull/8285>

    Args:
        channels (int or List[int]): Size of each input sample. If given as a
            list, will construct an MLP based on the given feature sizes.
        k (int): The number of clusters.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """

    def __init__(self, channels: Union[int, List[int]], k: int, dropout: float = 0.0, exponent: float = 1.0):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        self.mlp = MLP(channels + [k], act=None, norm=None)

        self.dropout = dropout
        self.exponent = exponent

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: OptTensor = None,
        ptr: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in {N \times F}`, with
                number of nodes in all input graph :math:`N`, and feature dimension :math:`F`.
                Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{N \times C}` is
                being created within this method.
            edge_index (torch.Tensor): Graph connectivity in COO format with shape
                :obj:`[2, E]` and type :obj:`torch.long`.
            edge_weight (torch.Tensor, optional): Edge weight vector with shape
                :obj:`[E]` and type :obj:`torch.float`.
            batch (torch.Tensor): Batch vector with shape :obj:`[N]` and type
                :obj:`torch.long`.
            ptr (torch.Tensor, optional): Pointer vector holding the start
                and end (exclusive) index of node for each graph in a
                mini-batch. Its shape must be of shape :obj:`[V + 1]` with
                :obj:`V` referring to the number of graphs in the mini-batch.
                If given, computes the mini-batch adjacency matrix based on
                these edge pointers.
        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`)
        """

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if ptr is None:
            ptr = torch.tensor([0, x.size(0)], dtype=torch.long, device=x.device)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=edge_index.device)

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)
        s = torch.softmax(s * self.exponent, dim=-1)  # N x C

        num_graphs = ptr.size(0) - 1
        device = x.device
        max_nodes = (ptr[1:] - ptr[:-1]).max().item()

        # Split and pad s, adj, and degrees
        x_padded = torch.zeros((num_graphs, max_nodes, x.size(1)), device=device)  # B x N x F
        s_padded = torch.zeros((num_graphs, max_nodes, s.size(1)), device=device)  # B x N x C
        edge_index_batch = batch[edge_index[0, :]]  # E
        edge_index_padded = edge_index - ptr[edge_index_batch].repeat(2).view(2, -1)

        for i in range(num_graphs):
            start, end = ptr[i], ptr[i + 1]
            s_padded[i, :end - start] = s[start:end]
            x_padded[i, :end - start] = x[start:end]

        indices = torch.stack([edge_index_batch, edge_index_padded[0, :], edge_index_padded[1, :]], dim=0)
        adj_padded = torch.sparse_coo_tensor(indices, edge_weight, torch.Size([num_graphs, max_nodes, max_nodes]))
        degrees_padded = adj_padded.sum(dim=1).to_dense().view(num_graphs, max_nodes, 1)  # B x N x 1

        out_adj_padded = torch.matmul(s_padded.transpose(1, 2), torch.bmm(adj_padded, s_padded))  # B x C x C

        # Calculate losses in a batched manner
        # Spectral loss:
        # -Tr(S^T B S) / 2m
        degrees_t = degrees_padded.transpose(1, 2)  # B x 1 x N
        m = torch.einsum('ijk->i', degrees_padded) / 2  # B
        m_expand = m.unsqueeze(-1).unsqueeze(-1).expand(-1, out_adj_padded.size(1), out_adj_padded.size(1))  # B x C x C
        ca = torch.matmul(s_padded.transpose(1, 2), degrees_padded)  # B x C x 1
        cb = torch.matmul(degrees_t, s_padded)  # B x 1 x C
        normalizer = torch.matmul(ca, cb) / 2 / m_expand
        decompose = out_adj_padded - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = torch.mean(spectral_loss)

        # Orthogonality regularization:
        # ||S^T S / ||S^T S||_F - I / N ||_F
        ss = torch.matmul(s_padded.transpose(1, 2), s_padded)
        i_s = torch.eye(s_padded.size(2)).type_as(ss)
        ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        # Cluster loss:
        # || sum_i S_ij ||_F / N * sqrt(k)
        cluster_size = torch.einsum('ijk->ik', s_padded)  # B x C
        cluster_loss = torch.norm(input=cluster_size, dim=1) / (ptr[1:] - ptr[:-1]) * torch.norm(i_s)
        cluster_loss = torch.mean(cluster_loss)

        # Calculate the output feature matrix
        out = F.selu(torch.matmul(s_padded.transpose(1, 2), x_padded))
        # Fix and normalize coarsened adjacency matrix
        ind = torch.arange(s.size(1), device=out_adj_padded.device)
        out_adj_padded[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj_padded)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj_padded / d) / d.transpose(1, 2)  # B x C x C

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mlp.in_channels}, '
                f'num_clusters={self.mlp.out_channels})')


class DMoNPooling(torch.nn.Module):
    r"""The spectral modularity pooling operator from the `"Graph Clustering
    with Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the learned cluster assignment matrix, the pooled node feature
    matrix, the coarsened symmetrically normalized adjacency matrix, and three
    auxiliary objectives: (1) The spectral loss

    .. math::
        \mathcal{L}_s = - \frac{1}{2m}
        \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})}

    where :math:`\mathbf{B}` is the modularity matrix, (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F

    where :math:`C` is the number of clusters, and (3) the cluster loss

    .. math::
        \mathcal{L}_c = \frac{\sqrt{C}}{n}
        {\left\|\sum_i\mathbf{C_i}^{\top}\right\|}_F - 1.

    .. note::

        For an example of using :class:`DMoNPooling`, see
        `examples/proteins_dmon_pool.py
        <https://github.com/pyg-team/pytorch_geometric/blob
        /master/examples/proteins_dmon_pool.py>`_.

    .. note::

        correction of original DMONPooing implemente have been merged into PyG #8285 <https://github.com/pyg-team/pytorch_geometric/pull/8285>

    Args:
        channels (int or List[int]): Size of each input sample. If given as a
            list, will construct an MLP based on the given feature sizes.
        k (int): The number of clusters.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """

    def __init__(self, channels: Union[int, List[int]], k: int, dropout: float = 0.0, exponent: float = 1.0):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        self.mlp = MLP(channels + [k], act=None, norm=None)

        self.dropout = dropout
        self.exponent = exponent

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                being created within this method.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)
        s = torch.softmax(s * self.exponent, dim=-1)

        (batch_size, num_nodes, _), k = x.size(), s.size(-1)

        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=s.device)
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

        out = F.selu(torch.matmul(s.transpose(1, 2), x))
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # Spectral loss:
        # -Tr(S^T B S) / 2m
        degrees = torch.einsum('ijk->ij', adj).unsqueeze(-1) * mask  # B x N x 1
        degrees_t = degrees.transpose(1, 2)  # B x 1 x N
        m = torch.einsum('ijk->i', degrees) / 2  # B
        m_expand = m.unsqueeze(-1).unsqueeze(-1).expand(-1, k, k)  # B x k x k
        # print(f'm: {m}')

        ca = torch.matmul(s.transpose(1, 2), degrees)  # B x k x 1
        cb = torch.matmul(degrees_t, s)  # B x 1 x k

        normalizer = torch.matmul(ca, cb) / 2 / m_expand
        # print(f'out_adj: {out_adj}')
        # print(f'normalizer: {normalizer}')
        decompose = out_adj - normalizer
        # print(f'decompose: {decompose}')
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        # print(f'spectral_loss: {spectral_loss}')
        spectral_loss = torch.mean(spectral_loss)

        # Orthogonality regularization:
        # ||S^T S / ||S^T S||_F - I / N ||_F
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(k).type_as(ss)
        ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)

        # Cluster loss:
        # || sum_i S_ij ||_F / N * sqrt(k)
        cluster_size = torch.einsum('ijk->ik', s)  # B x C
        cluster_loss = torch.norm(input=cluster_size, dim=1) / mask.sum(1).view(-1) * torch.norm(i_s)
        cluster_loss = torch.mean(cluster_loss)

        # Fix and normalize coarsened adjacency matrix:
        ind = torch.arange(k, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum('ijk->ij', out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mlp.in_channels}, '
                f'num_clusters={self.mlp.out_channels})')
