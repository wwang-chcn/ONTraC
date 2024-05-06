from optparse import Values
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from scipy.sparse import load_npz
from torch_geometric.loader import DenseDataLoader

from ONTraC.data import SpatailOmicsDataset, create_torch_dataset
from ONTraC.log import *
from ONTraC.train import SubBatchTrainProtocol
from ONTraC.utils import get_rel_params, read_yaml_file
from ONTraC.utils.NTScore import (NTScore_table, get_niche_NTScore,
                                  niche_to_cell_NTScore)


def load_parameters(opt_validate_func: Callable, prepare_optparser_func: Callable) -> Values:
    """
    Load parameters
    :param opt_validate_func: validate function
    :param prepare_optparser_func: prepare optparser function
    :return: options
    """
    options = opt_validate_func(prepare_optparser_func())

    return options


def load_data(options: Values) -> Tuple[SpatailOmicsDataset, DenseDataLoader]:
    """
    Load data
    :param options: options
    :return: dataset, sample_loader
    """
    params = read_yaml_file(f'{options.preprocessing_dir}/samples.yaml')
    rel_params = get_rel_params(options, params)
    dataset = create_torch_dataset(options, rel_params)
    batch_size = options.batch_size if options.batch_size > 0 else len(dataset)
    sample_loader = DenseDataLoader(dataset, batch_size=batch_size)
    return dataset, sample_loader


def train(nn_model: Type[torch.nn.Module], options: Values, BatchTrain: Type[SubBatchTrainProtocol],
          device: torch.device, dataset: SpatailOmicsDataset, sample_loader: DenseDataLoader,
          inspect_funcs: Optional[List[Callable]], model_name: str) -> SubBatchTrainProtocol:
    info(message=f'{model_name} train start.')
    model = nn_model(input_feats=dataset.num_features,
                     hidden_feats=options.hidden_feats,
                     k=options.k,
                     exponent=options.beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    batch_train = BatchTrain(model=model, device=device, data_loader=sample_loader)  # type: ignore
    batch_train.save(path=f'{options.GNN_dir}/epoch_0.pt')

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
                      output=options.GNN_dir,
                      **loss_weight_args)
    batch_train.save(path=f'{options.GNN_dir}/model_state_dict.pt')
    return batch_train


def evaluate(batch_train: SubBatchTrainProtocol, model_name: str) -> None:
    """
    Evaluate process
    :return: None
    """
    info(message=f'{model_name} eval start.')
    loss_dict: Dict[str, np.floating] = batch_train.evaluate()  # type: ignore
    info(message=f'Evaluate loss, {repr(loss_dict)}')


def graph_pooling_output(ori_data_df: pd.DataFrame, dataset: SpatailOmicsDataset, rel_params: Dict,
                         consolidate_s_array: np.ndarray, output_dir: str) -> None:
    """
    Write the graph pooling results as the Niche cluster (max probability for each niche & cell).
    :param ori_data_df: pd.DataFrame, original data
    :param consolidate_s_array: np.ndarray, consolidate s array
    :return: None
    """

    consolidate_s_niche_df = pd.DataFrame()
    consolidate_s_cell_df = pd.DataFrame()
    for i, data in enumerate(dataset):
        # the slice of data in each sample
        slice_ = slice(i * data.x.shape[0], i * data.x.shape[0] + data.mask.sum())
        consolidate_s = consolidate_s_array[slice_]  # N x C
        consolidate_s_df_ = pd.DataFrame(consolidate_s,
                                         columns=[f'NicheCluster_{i}' for i in range(consolidate_s.shape[1])])
        consolidate_s_df_['Cell_ID'] = ori_data_df[ori_data_df['Sample'] == data.name]['Cell_ID'].values
        consolidate_s_niche_df = pd.concat([consolidate_s_niche_df, consolidate_s_df_], axis=0)

        # niche to cell matrix
        niche_weight_matrix = load_npz(rel_params['Data'][i]['NicheWeightMatrix'])
        niche_to_cell_matrix = (
            niche_weight_matrix /
            niche_weight_matrix.sum(axis=0)).T  # normalize by the all niches associated with each cell, N x N

        consolidate_s_cell = niche_to_cell_matrix @ consolidate_s
        consolidate_s_cell_df_ = pd.DataFrame(consolidate_s_cell,
                                              columns=[f'NicheCluster_{i}' for i in range(consolidate_s_cell.shape[1])])
        consolidate_s_cell_df_['Cell_ID'] = ori_data_df[ori_data_df['Sample'] == data.name]['Cell_ID'].values
        consolidate_s_cell_df = pd.concat([consolidate_s_cell_df, consolidate_s_cell_df_], axis=0)

    consolidate_s_niche_df = consolidate_s_niche_df.set_index('Cell_ID')
    consolidate_s_niche_df = consolidate_s_niche_df.loc[ori_data_df['Cell_ID'], :]
    consolidate_s_niche_df.to_csv(f'{output_dir}/niche_level_niche_cluster.csv.gz', index=True, index_label='Cell_ID', header=True)
    consolidate_s_niche_df['Niche_Cluster'] = consolidate_s_niche_df.values.argmax(axis=1)
    consolidate_s_niche_df['Niche_Cluster'].to_csv(f'{output_dir}/niche_level_max_niche_cluster.csv.gz',
                                                   index=True,
                                                   header=True)
    consolidate_s_cell_df = consolidate_s_cell_df.set_index('Cell_ID')
    consolidate_s_cell_df = consolidate_s_cell_df.loc[ori_data_df['Cell_ID'], :]
    consolidate_s_cell_df.to_csv(f'{output_dir}/cell_level_niche_cluster.csv.gz', index=True, index_label='Cell_ID', header=True)
    consolidate_s_cell_df['Niche_Cluster'] = consolidate_s_cell_df.values.argmax(axis=1)
    consolidate_s_cell_df['Niche_Cluster'].to_csv(f'{output_dir}/cell_level_max_niche_cluster.csv.gz',
                                                  index=True,
                                                  header=True)


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


def NTScore(options: Values, dataset: SpatailOmicsDataset, consolidate_s_array: ndarray,
            consolidate_out_adj_array: ndarray) -> None:
    """
    Pseudotime calculateion process
    :param options: options
    :param consolidate_s_array: consolidate s array
    :param consolidate_out_adj_array: consolidate out adj array
    :return: None
    """

    params = read_yaml_file(f'{options.preprocessing_dir}/samples.yaml')
    rel_params = get_rel_params(options, params)

    niche_cluster_score, niche_level_NTScore = get_niche_NTScore(niche_cluster_loading=consolidate_s_array,
                                                                 niche_adj_matrix=consolidate_out_adj_array)
    cell_level_NTScore, all_niche_level_NTScore_dict, all_cell_level_NTScore_dict = niche_to_cell_NTScore(
        dataset=dataset, rel_params=rel_params, niche_level_NTScore=niche_level_NTScore)

    np.savetxt(fname=f'{options.NTScore_dir}/niche_cluster_score.csv.gz', X=niche_cluster_score, delimiter=',')
    np.savetxt(fname=f'{options.NTScore_dir}/niche_NTScore.csv.gz', X=niche_level_NTScore, delimiter=',')
    np.savetxt(fname=f'{options.NTScore_dir}/cell_NTScore.csv.gz', X=cell_level_NTScore, delimiter=',')

    NTScore_table(options=options,
                  rel_params=rel_params,
                  all_niche_level_NTScore_dict=all_niche_level_NTScore_dict,
                  all_cell_level_NTScore_dict=all_cell_level_NTScore_dict)
