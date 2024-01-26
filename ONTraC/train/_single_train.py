from optparse import Values
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..log import debug
from .loss_funs import graph_smooth_loss, recon_loss


class SingleTrain(object):
    """docstring for batch_train"""

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 x: Tensor,
                 adj: Tensor,
                 mask: Optional[Tensor] = None):
        super(SingleTrain, self).__init__()
        self.model = model
        self.device = device
        self.x = x.to(self.device)
        self.adj = adj.to(self.device)
        assert self.x.shape[0] == self.adj.shape[0] == self.adj.shape[1]
        if mask is None:
            self.mask = torch.ones((self.x.shape[0], ), dtype=torch.bool, device=self.device)
        self.data = Data(x=self.x, adj=self.adj, mask=self.mask)
        debug(f'x shape: {self.data.x.shape}.')
        debug(f'adj shape: {self.data.adj.shape}.')
        debug(f'mask shape: {self.data.mask.shape}.')

    def train(self,
              recon_loss_weight: float,
              graph_smooth_loss_weight: float,
              optimizer: torch.optim.Optimizer,
              epoch: int,
              inspect_funcs: Optional[List[Callable]] = None) -> None:
        self.model.train()
        for i in range(epoch):
            # debug(f'epoch {i+1} start.')
            recon_x, z = self.model(x=self.data.x, adj=self.data.adj, mask=self.data.mask)
            # debug(f'epoch {i+1} forward step complete.')
            r_loss = recon_loss(x=self.data.x, recon_x=recon_x, mask=self.data.mask)
            g_loss = graph_smooth_loss(z=z, adj=self.data.adj, mask=self.data.mask)
            loss = recon_loss_weight * r_loss + graph_smooth_loss_weight * g_loss
            # debug(f'epoch {i+1} loss calculation complete.')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if inspect_funcs is not None:
                for inspect_func in inspect_funcs:
                    inspect_func(epoch=i,
                                 data=self.data,
                                 recon_x=recon_x,
                                 z=z,
                                 loss=loss,
                                 r_loss=r_loss,
                                 g_loss=g_loss)

    def test(self, data: Data, options: Values) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Test function
        :param data: torch_geometric.data.Data
        :return: z, recon_x, loss, r_loss, g_loss
        """
        self.model.eval()
        data = data.to(self.device)  # type: ignore
        recon_x, z = self.model.test(x=data.x, adj=data.adj, mask=data.mask)  # type: ignore
        r_loss = recon_loss(x=data.x, recon_x=recon_x, mask=data.mask)
        g_loss = graph_smooth_loss(z=z, adj=data.adj, mask=data.mask)
        loss = options.recon_loss_weight * r_loss + options.graph_smooth_loss_weight * g_loss

        return recon_x, z, loss, r_loss, g_loss
