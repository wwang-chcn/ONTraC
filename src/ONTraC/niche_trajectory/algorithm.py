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


def held_karp(conn_matrix: np.ndarray) -> List[int]:
    """
    Held-Karp algorithm to find the optimal path with the highest connectivity.
    """
    n = conn_matrix.shape[0]
    max_connectivity = float('-inf')
    optimal_path = []

    # --- Held-Karp algorithm ---
    # Test each node as the starting node
    for start_node in range(n):
        C = {}

        # Initial state
        for k in range(n):
            if k != start_node:
                C[(1 << k, k)] = (conn_matrix[start_node][k], [start_node, k])

        # Iterate subsets of increasing length and store the maximum path
        for subset_size in range(2, n):
            for subset in itertools.combinations(range(n), subset_size):
                if start_node in subset:
                    continue
                bits = 0
                for bit in subset:
                    bits |= 1 << bit

                for k in subset:
                    prev_bits = bits & ~(1 << k)
                    res = []
                    for m in subset:
                        if m == k:
                            continue
                        res.append((C[(prev_bits, m)][0] + conn_matrix[m][k], C[(prev_bits, m)][1] + [k]))
                    C[(bits, k)] = max(res)

        # Find the best path that ends at any node excluding the start node
        bits = (2**n - 1) - (1 << start_node)
        for k in range(n):
            if k != start_node and (bits, k) in C:
                if C[(bits, k)][0] > max_connectivity:
                    max_connectivity = C[(bits, k)][0]
                    optimal_path = C[(bits, k)][1]

    return optimal_path
