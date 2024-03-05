import itertools
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment

from ONTraC.data import SpatailOmicsDataset

from ..log import debug, info


def get_niche_trajectory_path(niche_adj_matrix: ndarray) -> List[int]:
    """
    Find niche level trajectory with maximum connectivity
    :param adj_matrix: non-negative ndarray, adjacency matrix of the graph with shape (n, n), n >= 2
    :return: List[int], the niche trajectory

    1) Find a circle with maximum connectivity
    2) Remove the edge with lowest weight in the circle
    3) Get the path
    """

    row_ind, col_ind = linear_sum_assignment(niche_adj_matrix, maximize=True)  # Find a circle with maximum connectivity
    edges = dict(zip(row_ind, col_ind))

    # remove the edge with lowest weight in the circle
    min_connectivity = niche_adj_matrix.max()
    for i in range(niche_adj_matrix.shape[0]):
        if niche_adj_matrix[i, edges[i]] < min_connectivity:
            min_connectivity = niche_adj_matrix[i, edges[i]]
            min_edge = (i, edges[i])

    # get the path
    niche_trajectory_path = []
    start_node = min_edge[1]
    while True:
        niche_trajectory_path.append(start_node)
        start_node = edges[start_node]
        if start_node == min_edge[0]:
            break
    
    if len(niche_trajectory_path) != niche_adj_matrix.shape[0]:
        raise ValueError('No niche trajectory path found. Please adjust the parameters.')

    return niche_trajectory_path


def trajectory_path_to_NC_score(niche_trajectory_path: List[int]) -> ndarray:
    """
    Convert niche trajectory path to NTScore
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


def niche_to_cell_NTScore(dataset: SpatailOmicsDataset, rel_params: Dict, niche_level_NTScore: ndarray) -> ndarray:
    """
    Get cell-level NTScore
    :param dataset: SpatailOmicsDataset, the dataset
    :param real_param: Dict, the real parameters
    :param niche_level_NTScore: ndarray, the niche-level NTScore
    :return: ndarray, the cell-level NTScore
    """

    cell_level_NTScore = np.zeros(niche_level_NTScore.shape[0])

    node_sum = 0
    for i, data in enumerate(dataset):
        name = data.name
        mask = data.mask
        niche_weight_matrix = np.loadtxt(rel_params['Data'][i]['NicheWeightMatrix'], delimiter=',')
        niche_weight_matrix_norm = niche_weight_matrix / niche_weight_matrix.sum(axis=1, keepdims=True)  # normalize
        neighbor_indices_matrix = np.loadtxt(rel_params['Data'][i]['NeighborIndicesMatrix'], delimiter=',').astype(int)
        niche_level_NTScore_ = niche_level_NTScore[node_sum:node_sum + mask.sum()]
        neighbor_niche_level_NTScore = niche_level_NTScore_[neighbor_indices_matrix]
        cell_level_NTScore_ = (neighbor_niche_level_NTScore * niche_weight_matrix_norm).sum(axis=1)
        cell_level_NTScore[node_sum:node_sum + mask.sum()] = cell_level_NTScore_

    return cell_level_NTScore
