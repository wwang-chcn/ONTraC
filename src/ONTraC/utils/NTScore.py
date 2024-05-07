import itertools
from optparse import Values
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import load_npz

from ONTraC.data import SpatailOmicsDataset


def get_niche_trajectory_path(niche_adj_matrix: ndarray) -> List[int]:
    """
    Find niche level trajectory with maximum connectivity using Brute Force
    :param adj_matrix: non-negative ndarray, adjacency matrix of the graph
    :return: List[int], the niche trajectory
    """
    max_connectivity = float('-inf')
    niche_trajectory_path = []
    for path in itertools.permutations(range(len(niche_adj_matrix))):
        connectivity = 0
        for i in range(len(path) - 1):
            connectivity += niche_adj_matrix[path[i], path[i + 1]]
        if connectivity > max_connectivity:
            max_connectivity = connectivity
            niche_trajectory_path = list(path)

    return niche_trajectory_path


def trajectory_path_to_NC_score(niche_trajectory_path: List[int]) -> ndarray:
    """
    Convert niche cluster trajectory path to NTScore
    :param niche_trajectory_path: List[int], the niche trajectory path
    :return: ndarray, the NTScore
    """

    niche_NT_score = np.zeros(len(niche_trajectory_path))
    values = np.linspace(0, 1, len(niche_trajectory_path))

    for i, index in enumerate(niche_trajectory_path):
        # debug(f'i: {i}, index: {index}')
        niche_NT_score[index] = values[i]
    return niche_NT_score


def get_niche_NTScore(niche_cluster_loading: ndarray, niche_adj_matrix: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Get niche-level niche trajectory and cell-level niche trajectory
    :param niche_cluster_loading: ndarray, the loading of cell x niche clusters
    :param adj_matrix: ndarray, the adjacency matrix of the graph
    :return: Tuple[ndarray, ndarray], the niche-level niche trajectory and cell-level niche trajectory
    """

    niche_trajectory_path = get_niche_trajectory_path(niche_adj_matrix=niche_adj_matrix)

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


def NTScore_table(options: Values, rel_params: Dict, all_niche_level_NTScore_dict: Dict[str, ndarray],
                  all_cell_level_NTScore_dict: Dict[str, ndarray]) -> None:
    """
    Generate NTScore table and save it
    :param options: Values, options
    :param rel_params: Dict, relative paths
    :param all_niche_level_NTScore_dict: Dict[str, ndarray], all niche-level NTScore dict
    :param all_cell_level_NTScore_dict: Dict[str, ndarray], all cell-level NTScore dict
    :return: pd.DataFrame, NTScore table
    """

    NTScore_table = pd.DataFrame()
    for sample in rel_params['Data']:
        coordinates_df = pd.read_csv(sample['Coordinates'], index_col=0)
        coordinates_df['Niche_NTScore'] = all_niche_level_NTScore_dict[sample['Name']]
        coordinates_df['Cell_NTScore'] = all_cell_level_NTScore_dict[sample['Name']]
        coordinates_df.to_csv(f'{options.NTScore_dir}/{sample["Name"]}_NTScore.csv.gz')
        NTScore_table = pd.concat([NTScore_table, coordinates_df])

    NTScore_table.to_csv(f'{options.NTScore_dir}/NTScore.csv.gz')
