import itertools
from typing import List, Tuple

import numpy as np
from numpy import ndarray

from ..log import debug, info


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


def trajectory_path_to_NTScore(niche_trajectory_path: List[int]) -> ndarray:
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


def get_niche_trajectory(niche_cluster_loading: ndarray, niche_adj_matrix: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Get niche-level niche trajectory and cell-level niche trajectory
    :param niche_cluster_loading: ndarray, the loading of cell x niche clusters
    :param adj_matrix: ndarray, the adjacency matrix of the graph
    :return: Tuple[ndarray, ndarray], the niche-level niche trajectory and cell-level niche trajectory
    """

    niche_trajectory_path = get_niche_trajectory_path(niche_adj_matrix=niche_adj_matrix)

    niche_level_NTScore = trajectory_path_to_NTScore(niche_trajectory_path)
    cell_level_NTScore = niche_cluster_loading @ niche_level_NTScore
    return niche_level_NTScore, cell_level_NTScore
