import itertools
from typing import List

import numpy as np


def brute_force(conn_matrix: np.ndarray) -> List[int]:
    """
    Brute force method to find the optimal path with the highest connectivity.
    """
    max_connectivity = float('-inf')
    optimal_path = []
    for path in itertools.permutations(range(len(conn_matrix))):
        connectivity = 0
        for i in range(len(path) - 1):
            connectivity += conn_matrix[path[i], path[i + 1]]
        if connectivity > max_connectivity:
            max_connectivity = connectivity
            optimal_path = list(path)
    return optimal_path
