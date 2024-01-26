import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNNormConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index) -> torch.Tensor:
        r"""
        .. math:: \mathbf{\hat{L}}X\mathbf{\Theta}
        .. math:: \mathbf{\hat{L}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2} + \mathbf{I}

        Args:
            x: Node feature matrix with shape [N, in_channels]
            edge_index: Graph connectivity in COO format with shape [2, E]
        """

        # Step 1: Compute normalized adjacency matrix.
        #         .. math:: \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # .. math:: \mathbf{D}^{-1/2}
        edge_attr = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 2: Add self-loops to the normalized adjacency matrix.
        #         .. math:: \mathbf{I} + \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}
        edge_index, edge_attr = add_self_loops(edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))

        # Step 3: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=edge_attr)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
