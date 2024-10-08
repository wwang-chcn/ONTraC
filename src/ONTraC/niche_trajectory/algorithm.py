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
    C = {}

    # --- Held-Karp algorithm ---
    # Initial state
    for k in range(1, n):
        C[(1 << k, k)] = (conn_matrix[0][k], [0, k])

    # Iterate subsets of increasing length and store the maximum path
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
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

    # We're interested in all bits but the least significant (the start city)
    bits = (2**n - 1) - 1

    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + conn_matrix[k][0], C[(bits, k)][1]))

    _, optimal_circle = max(res)
    optimal_circle.append(0)  # complete the cycle

    # --- make the circle become a path ---
    # Cut the shortest edge out from the cycle
    min_edge_index = 0
    min_edge_conn = conn_matrix[optimal_circle[0]][optimal_circle[1]]
    for (i, start_node), end_node in zip(enumerate(optimal_circle[:-1]), optimal_circle[1:]):
        if conn := conn_matrix[start_node][end_node] < min_edge_conn:
            min_edge_conn = conn
            min_edge_index = i

    optimal_path = optimal_circle[min_edge_index + 1:-1] + optimal_circle[:min_edge_index + 1]

    # Reverse the path if the start node index is smaller than the end node index
    # This is to make sure the path is the same as Brute force method
    if optimal_path[0] > optimal_path[-1]:
        optimal_path.reverse()

    return optimal_path
