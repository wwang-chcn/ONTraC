from optparse import Values

import pytest
import torch
from torch_geometric.loader import DenseDataLoader

from ONTraC.data import load_dataset
from ONTraC.GNN._GNN import SpatailOmicsDataset
from ONTraC.model import GraphPooling
from ONTraC.train import GPBatchTrain


@pytest.fixture
def options() -> Values:
    # Create an options object for testing
    _options = Values()
    _options.preprocessing_dir = 'tests/_data/preprocessing'
    _options.GNN_dir = 'tests/_data/GNN'
    _options.batch_size = 5
    _options.lr = 0.03
    _options.hidden_feats = 4
    _options.k = 6
    _options.modularity_loss_weight = 0.3
    _options.purity_loss_weight = 300
    _options.regularization_loss_weight = 0.1
    _options.beta = 0.03
    return _options


@pytest.fixture()
def dataset(options: Values) -> SpatailOmicsDataset:
    return load_dataset(options=options)


@pytest.fixture()
def sample_loader(options: Values, dataset: SpatailOmicsDataset) -> DenseDataLoader:
    batch_size = options.batch_size if options.batch_size > 0 else len(dataset)
    sample_loader = DenseDataLoader(dataset, batch_size=batch_size)
    return sample_loader


@pytest.fixture()
def nn_model(options: Values, dataset: SpatailOmicsDataset) -> torch.nn.Module:
    model = GraphPooling(input_feats=dataset.num_features,
                         hidden_feats=options.hidden_feats,
                         k=options.k,
                         exponent=options.beta)
    model.load_state_dict(torch.load(f'{options.GNN_dir}/epoch_0.pt', map_location=torch.device('cpu')))
    return model


def test_train(options: Values, sample_loader: DenseDataLoader, nn_model: torch.nn.Module) -> None:
    batch_train = GPBatchTrain(model=nn_model, device=torch.device('cpu'), data_loader=sample_loader)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=options.lr)
    batch_train.set_train_args(optimizer=optimizer,
                               modularity_loss_weight=options.modularity_loss_weight,
                               purity_loss_weight=options.purity_loss_weight,
                               regularization_loss_weight=options.regularization_loss_weight,
                               beta=options.beta)
    batch_train.train_epoch(epoch=1)
    trained_params = torch.load(f'{options.GNN_dir}/epoch_1.pt', map_location=torch.device('cpu'))
    for k, v in nn_model.named_parameters():
        assert torch.allclose(v, trained_params[k], rtol=0.05)  # there are some difference between linux and macOS (may be caused by chip?)
