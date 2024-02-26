import os
from optparse import Values
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from numpy import ndarray
from torch_geometric.loader import DenseDataLoader

from ONTraC.data import SpatailOmicsDataset, create_torch_dataset
from ONTraC.log import debug, info, warning
from ONTraC.train import SubBatchTrainProtocol
from ONTraC.utils import get_rel_params, read_yaml_file
from ONTraC.utils.pseudo_time import get_pseudo_time_line, get_niche_trajectory


def load_parameters(opt_validate_func: Callable, prepare_optparser_func: Callable) -> Tuple[Values, Dict]:
    """
    Load parameters
    :param opt_validate_func: validate function
    :param prepare_optparser_func: prepare optparser function
    :return: options, rel_params
    """
    options = opt_validate_func(prepare_optparser_func())
    params = read_yaml_file(f'{options.input}/samples.yaml')
    rel_params = get_rel_params(options, params)
    os.makedirs(options.output, exist_ok=True)

    return options, rel_params


def load_data(options: Values, rel_params: Dict) -> Tuple[SpatailOmicsDataset, DenseDataLoader]:
    """
    Load data
    :param options: options
    :param rel_params: rel_params
    :return: dataset, sample_loader
    """
    dataset = create_torch_dataset(options, rel_params)
    batch_size = options.batch_size if options.batch_size > 0 else len(dataset)
    sample_loader = DenseDataLoader(dataset, batch_size=batch_size)
    return dataset, sample_loader


def tain_prepare(options) -> torch.device:
    if options.device.startswith('cuda') and not torch.cuda.is_available():
        warning('CUDA is not available, use CPU instead.')
        options.device = 'cpu'
    if options.device.startswith('mps') and not torch.backends.mps.is_available():
        warning('MPS is not available, use CPU instead.')
        options.device = 'cpu'
    device = torch.device(options.device)

    return device


def train(nn_model: Type[torch.nn.Module], options: Values, BatchTrain: Type[SubBatchTrainProtocol],
          device: torch.device, dataset: SpatailOmicsDataset, sample_loader: DenseDataLoader,
          inspect_funcs: Optional[List[Callable]], model_name: str) -> SubBatchTrainProtocol:
    info(message=f'{model_name} train start.')
    model = nn_model(input_feats=dataset.num_features,
                     hidden_feats=options.hidden_feats,
                     k=options.k,
                     dropout=options.dropout,
                     exponent=options.assign_exponent)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    batch_train = BatchTrain(model=model, device=device, data_loader=sample_loader)  # type: ignore

    loss_weight_args: dict[str, float] = {
        key: value
        for key, value in options.__dict__.items() if key.endswith('loss_weight')
    }

    batch_train.train(optimizer=optimizer,
                      inspect_funcs=inspect_funcs,
                      max_epochs=options.epochs,
                      max_patience=options.patience,
                      min_delta=options.min_delta,
                      min_epochs=options.min_epochs,
                      output=options.output,
                      **loss_weight_args)
    batch_train.save(path=f'{options.output}/model_state_dict.pt')
    return batch_train


def evaluate(batch_train: SubBatchTrainProtocol, model_name: str) -> None:
    """
    Evaluate process
    :return: None
    """
    info(message=f'{model_name} eval start.')
    loss_dict: Dict[str, np.floating] = batch_train.evaluate()  # type: ignore
    info(message=f'Evaluate loss, {repr(loss_dict)}')


def predict(output_dir: str, batch_train: SubBatchTrainProtocol, dataset: SpatailOmicsDataset,
            model_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    info(f'{model_name} predict start.')
    each_sample_loader = DenseDataLoader(dataset, batch_size=1)
    consolidate_flag = False
    consolidate_s_list = []
    consolidate_out = None
    consolidate_out_adj = None
    for data in each_sample_loader:  # type: ignore
        data = data.to(batch_train.device)  # type: ignore
        predict_result = batch_train.predict_dict(data=data)  # type: ignore
        for key, value in predict_result.items():
            np.savetxt(fname=f'{output_dir}/{data.name[0]}_{key}.csv.gz',
                       X=value.squeeze(0).detach().cpu().numpy(),
                       delimiter=',')

        # consolidate results
        if not consolidate_flag and ('s' in predict_result and 'out' in predict_result and 'out_adj' in predict_result):
            consolidate_flag = True
        if consolidate_flag:
            s = predict_result['s']
            out = predict_result['out']
            s = s.squeeze(0)
            consolidate_s_list.append(s)
            out_adj_ = torch.matmul(torch.matmul(s.T, data.adj.squeeze(0)), s)
            consolidate_out_adj = out_adj_ if consolidate_out_adj is None else consolidate_out_adj + out_adj_
            consolidate_out = out.squeeze(
                0) * data.mask.sum() if consolidate_out is None else consolidate_out + out.squeeze(0) * data.mask.sum()

    if consolidate_flag:
        # consolidate out
        nodes_num = 0
        for data in each_sample_loader:  # type: ignore
            nodes_num += data.mask.sum()
        consolidate_out = consolidate_out / nodes_num  # type: ignore
        consolidate_out_array = consolidate_out.detach().cpu().numpy()
        np.savetxt(fname=f'{output_dir}/consolidate_out.csv.gz', X=consolidate_out_array, delimiter=',')
        # consolidate s
        consolidate_s = torch.cat(consolidate_s_list, dim=0)
        # consolidate out_adj
        consolidate_out_adj = consolidate_out_adj / len(dataset)  # type: ignore
        ind = torch.arange(consolidate_s.shape[-1], device=consolidate_out_adj.device)
        consolidate_out_adj[ind, ind] = 0
        d = torch.einsum('ij->i', consolidate_out_adj)
        d = torch.sqrt(d)[:, None] + 1e-15
        consolidate_out_adj = (consolidate_out_adj / d) / d.transpose(0, 1)
        consolidate_s_array = consolidate_s.detach().cpu().numpy()
        consolidate_out_adj_array = consolidate_out_adj.detach().cpu().numpy()
        np.savetxt(fname=f'{output_dir}/consolidate_s.csv.gz', X=consolidate_s_array, delimiter=',')
        np.savetxt(fname=f'{output_dir}/consolidate_out_adj.csv.gz', X=consolidate_out_adj_array, delimiter=',')

        return consolidate_s_array, consolidate_out_adj_array
    else:
        return None, None


def pseudotime(options: Values, dataset: SpatailOmicsDataset, consolidate_s_array: ndarray,
               consolidate_out_adj_array: ndarray) -> None:
    """
    Pseudotime calculateion process
    :param options: options
    :param dataset: dataset
    :return: None
    """

    # all_sample_loader = DenseDataLoader(dataset, batch_size=len(dataset))
    # data = next(iter(all_sample_loader))
    # pseudotime_cluster, pseudotime_node = get_pseudo_time_line(data=data,
    #                                                            out_adj=consolidate_out_adj_array,
    #                                                            s=consolidate_s_array,
    #                                                            init_node_label=options.init_node_label)
    pseudotime_cluster, pseudotime_node = get_niche_trajectory(niche_cluster_loading=consolidate_s_array,
                                                               niche_adj_matrix=consolidate_out_adj_array)
    np.savetxt(fname=f'{options.output}/NT_niche_cluster.csv.gz', X=pseudotime_cluster, delimiter=',')
    np.savetxt(fname=f'{options.output}/NT_niche.csv.gz', X=pseudotime_node, delimiter=',')
