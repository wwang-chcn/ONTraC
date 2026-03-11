"""This module contains the BatchTrain class, which provides an abstract interface for mini-batch training of ONTraC models, as well as a concrete implementation for GNN-based models."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ONTraC.utils.decorators import selective_args_decorator

from ..log import debug
from ..utils import round_epoch_filter
from ..utils.decorators import selective_args_decorator
from .loss_funs import masked_variance, within_cluster_variance_loss


class BatchTrain(ABC):
    """Abstract minibatch training interface for ONTraC models."""

    def __init__(self, model: torch.nn.Module, device: torch.device, data_loader: DataLoader) -> None:
        """Initialize trainer state.
        
                Parameters
                ----------
        model :
            torch.nn.Module
                    Model to optimize.
        device :
            torch.device
                    Compute device used for training and inference.
        data_loader :
            DataLoader
                    Mini-batch loader that yields graph ``Data`` objects.
                """
        super(BatchTrain, self).__init__()
        self.model: torch.nn.Module = model
        self.device: torch.device = device
        self.data_loader: DataLoader = data_loader
        self.model = self.model.to(device=self.device)

    def __str__(self):
        """Return human-readable trainer summary."""
        return f"{self.__class__.__name__}(model='{self.model}', device='{self.device}', data_loader='{self.data_loader}')"

    def __repr__(self):
        """Return unambiguous trainer representation."""
        return self.__str__()

    def train(self,
              max_epochs: int = 100,
              max_patience: int = 50,
              min_delta: float = 0,
              min_epochs: int = 100,
              *args,
              **kwargs) -> None:
        """Train model with optional early stopping and checkpoint snapshots.
        
                Parameters
                ----------
        max_epochs :
            int, default=100
                    Maximum number of training epochs.
        max_patience :
            int, default=50
                    Early-stopping patience in epochs. Use ``0`` to disable.
        min_delta :
            float, default=0
                    Relative improvement threshold for resetting patience.
        min_epochs :
            int, default=100
                    Minimum number of epochs before early stopping is considered.
                *args, **kwargs
                    Passed to :meth:`set_train_args`. If ``output`` is provided in
                    ``kwargs``, periodic snapshots are saved.
                """

        self.set_train_args(*args, **kwargs)

        self.model.train()
        min_loss = np.inf
        patience = 0
        best_params = self.model.state_dict()

        for epoch in range(max_epochs):
            train_loss = self.train_epoch(epoch=epoch)
            if np.isnan(train_loss):  # unexpected situation
                best_params = self.model.state_dict()
                break
            elif max_patience == 0:  # no early stopping
                best_params = self.model.state_dict()
            elif min_loss - train_loss < min_loss * min_delta:  # no improvement
                patience += 1
            else:  # improvement
                min_loss = train_loss
                patience = 0
                best_params = self.model.state_dict()
            # max_patience == 0 means no early stopping
            if max_patience != 0 and patience >= max_patience and epoch >= min_epochs:
                break
            if round_epoch_filter(epoch) and 'output' in kwargs:
                output_dir = kwargs['output']
                self.save(f'{output_dir}/epoch_{epoch + 1}.pt')
        self.model.load_state_dict(best_params)

    @abstractmethod
    def set_train_args(self) -> None:
        """Method that should be implemented by all derived classes."""
        raise NotImplementedError("The set_train_args method should be implemented by subclasses.")

    @abstractmethod
    def train_epoch(self, epoch: int) -> float:
        """Method that should be implemented by all derived classes."""
        raise NotImplementedError("The train_epoch method should be implemented by subclasses.")

    @abstractmethod
    def evaluate(self) -> Dict[str, np.floating]:
        """Method that should be implemented by all derived classes."""
        raise NotImplementedError("The evaluate method should be implemented by subclasses.")

    def predict(self, data: Data) -> Tuple[Tensor, ...] | Tensor:
        """Run model inference on one graph batch object."""
        self.model.eval()
        with torch.no_grad():
            res = self.model.predict(data.x, data.adj, data.mask)  # type: ignore
        return res

    @abstractmethod
    def predict_dict(self, data: Data) -> Dict[str, Tensor]:
        """Method that should be implemented by all derived classes."""
        raise NotImplementedError("The predict_dict method should be implemented by subclasses.")

    def save(self, path: str) -> None:
        """Serialize model parameters to disk."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model parameters from disk."""
        self.model.load_state_dict(torch.load(path))


class SubBatchTrainProtocol(Protocol):
    """Structural protocol for trainer-like objects used in this package."""

    def train(self, *args, **kwargs) -> None:
        """Fit model parameters on training data."""
        ...

    def evaluate(self) -> Dict[str, np.floating]:
        """Compute evaluation metrics on held-out data."""
        ...

    def predict(self, data: Data) -> Tuple[Tensor, ...] | Tensor:
        """Run prediction on a single ``Data`` instance."""
        ...

    def predict_dict(self, data: Data) -> Dict[str, Tensor]:
        """Return prediction outputs in a named dictionary."""
        ...

    def save(self, path: str) -> None:
        """Persist model parameters to ``path``."""
        ...

    def load(self, path: str) -> None:
        """Restore model parameters from ``path``."""
        ...


class GNNBatchTrain(BatchTrain):
    """
    GNN batch training class.
    """

    @selective_args_decorator
    def set_train_args(self,
                       optimizer: torch.optim.Optimizer,
                       modularity_loss_weight: float = 1,
                       purity_loss_weight: float = 0,
                       regularization_loss_weight: float = 1,
                       ortho_loss_weight: float = 0,
                       inspect_funcs: Optional[List[Callable]] = None) -> None:
        """Configure optimizer and weighted objective coefficients."""
        self.optimizer = optimizer
        self.spectral_loss_weight = modularity_loss_weight
        self.ortho_loss_weight = ortho_loss_weight
        self.cluster_loss_weight = regularization_loss_weight
        self.feat_similarity_loss_weight = purity_loss_weight
        self.inspect_funcs = inspect_funcs

    def cal_loss(self, spectral_loss, ortho_loss, cluster_loss, data, s) -> Tuple[Tensor, ...]:
        """Compose weighted training objective from model and regularization terms."""
        spectral_loss = self.spectral_loss_weight * spectral_loss
        ortho_loss = self.ortho_loss_weight * ortho_loss * np.sqrt(2)
        cluster_loss = self.cluster_loss_weight * cluster_loss / (np.sqrt(self.model.k) - 1)
        feat_similarity_loss = within_cluster_variance_loss(x=data.x, s=s, mask=data.mask)
        total_var = masked_variance(x=data.x, mask=data.mask)
        feat_similarity_loss = self.feat_similarity_loss_weight * feat_similarity_loss
        loss = spectral_loss + ortho_loss + cluster_loss + feat_similarity_loss

        return loss, spectral_loss, ortho_loss, cluster_loss, feat_similarity_loss

    def train_epoch(self, epoch: int) -> float:
        """Execute one epoch across the data loader and return summed loss."""
        self.model.train()
        train_loss = 0
        for batch, data in enumerate(self.data_loader):
            # debug(f'epoch {epoch+1}, batch {batch+1} start.')
            data = data.to(self.device)
            s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.model(data.x, data.adj, data.mask)
            loss, spectral_loss, ortho_loss, cluster_loss, feat_similarity_loss = self.cal_loss(
                spectral_loss, ortho_loss, cluster_loss, data, s)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # debug(f'epoch {epoch+1}, batch {batch+1} end.')
            if self.inspect_funcs is not None:
                for inspect_func in self.inspect_funcs:
                    inspect_func(
                        epoch=epoch + 1,
                        batch=batch + 1,
                        data=data,
                        s=s,
                        out=out,
                        out_adj=out_adj,
                        loss=loss,
                        modularity_loss=spectral_loss,
                        # ortho_loss=ortho_loss,
                        purity_loss=feat_similarity_loss,
                        regularization_loss=cluster_loss,
                    )
            train_loss += loss.item()
        return train_loss

    def evaluate(self) -> Dict[str, np.floating]:
        """Evaluate average objective terms over the full data loader."""
        spectral_loss_list, ortho_loss_list, cluster_loss_list = [], [], []
        # bin_spectral_loss_list, bin_ortho_loss_list, bin_cluster_loss_list = [], [], []
        feat_similarity_loss_list = []
        loss_list = []
        self.model.eval()
        with torch.no_grad():
            for data in self.data_loader:
                data = data.to(self.device)
                s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.model.evaluate(
                    data.x, data.adj, data.mask)
                loss, spectral_loss, ortho_loss, cluster_loss, feat_similarity_loss = self.cal_loss(
                    spectral_loss, ortho_loss, cluster_loss, data, s)

                spectral_loss_list.append(spectral_loss.item())
                # ortho_loss_list.append(ortho_loss.item())
                cluster_loss_list.append(cluster_loss.item())
                feat_similarity_loss_list.append(feat_similarity_loss.item())
                loss_list.append(loss.item())
        spectral_loss = np.mean(spectral_loss_list)
        # ortho_loss = np.mean(ortho_loss_list)
        cluster_loss = np.mean(cluster_loss_list)
        feat_similarity_loss = np.mean(feat_similarity_loss_list)
        loss = np.mean(loss_list)
        results_dict = {
            'modularity_loss': spectral_loss,
            'purity_loss': feat_similarity_loss,
            'regularization_loss': cluster_loss,
            # 'ortho_loss': ortho_loss,
            'total_loss': loss
        }
        return results_dict

    def predict_dict(self, data: Data) -> Dict[str, Tensor]:
        """Predict and return embeddings, assignments, pooled features, and adjacency."""
        self.model.eval()
        with torch.no_grad():
            s, out, out_adj = self.model.predict(data.x, data.adj, data.mask)
            z = self.model.predict_embed(data.x, data.adj, data.mask)
        return {'z': z, 's': s, 'out': out, 'out_adj': out_adj}

    def predict_embed(self, data: Data) -> Tensor:
        """Return node embeddings before pooling for a ``Data`` object."""
        self.model.eval()
        with torch.no_grad():
            z = self.model.predict_embed(data.x, data.adj, data.mask)  # type: ignore
        return z


__all__ = ['SubBatchTrainProtocol', 'GNNBatchTrain']
