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

    max_cost, optimal_path = max(res)
    optimal_path.append(0)  # complete the cycle

    # --- Cut the shortest edge out from the cycle ---
    dists = []
    for i in range(len(optimal_path)-1):
        dists.append(conn_matrix[optimal_path[i]][optimal_path[i+1]])

    cut_index = dists.index(min(dists))
    if optimal_path[cut_index] < optimal_path[cut_index + 1]:
        start_index = cut_index
        end_index = cut_index + 1
    else:
        start_index = cut_index + 1
        end_index = cut_index

    if start_index == len(optimal_path) - 1:
        optimal_path.pop(start_index)
    elif start_index == 0:
        optimal_path.pop(start_index)
        optimal_path.reverse()
    elif start_index < end_index:
        optimal_path.pop(len(optimal_path)-1)
        seg1 = optimal_path[:start_index + 1]
        seg1.reverse()
        seg2 = optimal_path[end_index:]
        seg2.reverse()
        optimal_path = seg1 + seg2
    else:
        optimal_path.pop(len(optimal_path)-1)
        seg1 = optimal_path[start_index:]
        seg2 = optimal_path[:end_index + 1]
        optimal_path = seg1 + seg2

    return optimal_path
