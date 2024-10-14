import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import load_npz

from ..data import SpatailOmicsDataset
from ..log import error, info
from .algorithm import brute_force, held_karp


def load_consolidate_data(GNN_dir: Union[str, Path]) -> Tuple[ndarray, ndarray]:
    """
    Load consolidate s_array and out_adj_array
    :param GNN_dir: Union[str, Path], the directory of GNN
    :return: Tuple[ndarray, ndarray], the consolidate s_array and out_adj_array
    """

    info('Loading consolidate s_array and out_adj_array...')

    if not os.path.exists(f'{GNN_dir}/consolidate_s.csv.gz') or not os.path.exists(
            f'{GNN_dir}/consolidate_out_adj.csv.gz'):
        error(f'consolidate_s.csv.gz or consolidate_out_adj.csv.gz does not exist in {GNN_dir} directory.')
    consolidate_s_array = np.loadtxt(fname=f'{GNN_dir}/consolidate_s.csv.gz', delimiter=',')
    consolidate_out_adj_array = np.loadtxt(fname=f'{GNN_dir}/consolidate_out_adj.csv.gz', delimiter=',')

    return consolidate_s_array, consolidate_out_adj_array


def get_niche_trajectory_path(trajectory_construct_method: str, niche_adj_matrix: ndarray) -> List[int]:
    """
    Get niche trajectory path
    :param trajectory_construct_method: str, the method to construct trajectory
    :param adj_matrix: non-negative ndarray, adjacency matrix of the graph
    :return: List[int], the niche trajectory
    """

    niche_adj_matrix = (niche_adj_matrix + niche_adj_matrix.T) / 2
    
    if trajectory_construct_method == 'BF':
        info('Finding niche trajectory with maximum connectivity using Brute Force.')

        niche_trajectory_path = brute_force(niche_adj_matrix)

    elif trajectory_construct_method == 'TSP':
        info('Finding niche trajectory with maximum connectivity using TSP.')

        niche_trajectory_path = held_karp(niche_adj_matrix)

    return niche_trajectory_path


def trajectory_path_to_NC_score(niche_trajectory_path: List[int]) -> ndarray:
    """
    Convert niche cluster trajectory path to NTScore
    :param niche_trajectory_path: List[int], the niche trajectory path
    :return: ndarray, the NTScore
    """

    info('Calculating NTScore for each niche cluster based on the trajectory path.')

    niche_NT_score = np.zeros(len(niche_trajectory_path))
    values = np.linspace(0, 1, len(niche_trajectory_path))

    for i, index in enumerate(niche_trajectory_path):
        # debug(f'i: {i}, index: {index}')
        niche_NT_score[index] = values[i]
    return niche_NT_score


def get_niche_NTScore(trajectory_construct_method: str, niche_cluster_loading: ndarray,
                      niche_adj_matrix: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Get niche-level niche trajectory and cell-level niche trajectory
    :param trajectory_construct_method: str, the method to construct trajectory
    :param niche_cluster_loading: ndarray, the loading of cell x niche clusters
    :param adj_matrix: ndarray, the adjacency matrix of the graph
    :return: Tuple[ndarray, ndarray], the niche-level niche trajectory and cell-level niche trajectory
    """

    info('Calculating NTScore for each niche.')

    niche_trajectory_path = get_niche_trajectory_path(trajectory_construct_method=trajectory_construct_method,
                                                      niche_adj_matrix=niche_adj_matrix)

    niche_cluster_score = trajectory_path_to_NC_score(niche_trajectory_path)
    niche_level_NTScore = niche_cluster_loading @ niche_cluster_score
    return niche_cluster_score, niche_level_NTScore


def niche_to_cell_NTScore(dataset: SpatailOmicsDataset, rel_params: Dict,
                          niche_level_NTScore: ndarray) -> Tuple[ndarray, Dict[str, ndarray], Dict[str, ndarray]]:
    """
    get cell-level NTScore
    :param dataset: SpatailOmicsDataset, dataset
    :param rel_params: Dict, relative paths
    :param niche_level_NTScore: ndarray, niche-level NTScore
    :return: Tuple[ndarray, Dict[str, ndarray], Dict[str, ndarray]], the cell-level NTScore, all niche-level NTScore dict,
    and all cell-level NTScore dict
    """

    info('Projecting NTScore from niche-level to cell-level.')

    cell_level_NTScore = np.zeros(niche_level_NTScore.shape[0])

    all_niche_level_NTScore_dict: Dict[str, ndarray] = {}
    all_cell_level_NTScore_dict: Dict[str, ndarray] = {}

    for i, data in enumerate(dataset):
        # the slice of data in each sample
        mask = data.mask
        slice_ = slice(i * data.x.shape[0], i * data.x.shape[0] + mask.sum())

        # niche to cell matrix
        niche_weight_matrix = load_npz(rel_params['Data'][i]['NicheWeightMatrix'])
        niche_to_cell_matrix = (
            niche_weight_matrix /
            niche_weight_matrix.sum(axis=0)).T  # normalize by the all niches associated with each cell, N x N

        # cell-level NTScore
        niche_level_NTScore_ = niche_level_NTScore[slice_].reshape(-1, 1)  # N x 1
        cell_level_NTScore_ = niche_to_cell_matrix @ niche_level_NTScore_
        cell_level_NTScore[slice_] = cell_level_NTScore_.reshape(-1)

        all_niche_level_NTScore_dict[data.name] = niche_level_NTScore_
        all_cell_level_NTScore_dict[data.name] = cell_level_NTScore_

    return cell_level_NTScore, all_niche_level_NTScore_dict, all_cell_level_NTScore_dict


def NTScore_table(save_dir: Union[str, Path], rel_params: Dict, all_niche_level_NTScore_dict: Dict[str, ndarray],
                  all_cell_level_NTScore_dict: Dict[str, ndarray]) -> None:
    """
    Generate NTScore table and save it
    :param save_dir: Union[str, Path], the directory to save NTScore table
    :param rel_params: Dict, relative paths
    :param all_niche_level_NTScore_dict: Dict[str, ndarray], all niche-level NTScore dict
    :param all_cell_level_NTScore_dict: Dict[str, ndarray], all cell-level NTScore dict
    :return: None
    """

    info('Output NTScore tables.')

    NTScore_table = pd.DataFrame()
    for sample in rel_params['Data']:
        coordinates_df = pd.read_csv(sample['Coordinates'], index_col=0)
        coordinates_df['Niche_NTScore'] = all_niche_level_NTScore_dict[sample['Name']]
        coordinates_df['Cell_NTScore'] = all_cell_level_NTScore_dict[sample['Name']]
        coordinates_df.to_csv(f'{save_dir}/{sample["Name"]}_NTScore.csv.gz')
        NTScore_table = pd.concat([NTScore_table, coordinates_df])

    NTScore_table.to_csv(f'{save_dir}/NTScore.csv.gz')
