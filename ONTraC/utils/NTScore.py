from typing import List, Tuple

import numpy as np
import torch
import itertools
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Data

from ..log import debug, info


def primMST(adj_matrix: ndarray) -> List[Tuple[int, int]]:
    num_vertices = len(adj_matrix)
    visited: List[bool] = [False] * num_vertices
    # first edge
    dis = float('-inf')
    x, y = 0, 0  # define, will be changed
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if dis < adj_matrix[i, j]:
                x, y = i, j
                dis = adj_matrix[i, j]
    num_of_edges = 2
    MST_edges = [(x, y)]

    visited[x] = True
    visited[y] = True

    while num_of_edges < num_vertices:
        # print(visited)
        min_distance = float('-inf')
        x, y = 0, 0

        for i in range(num_vertices):
            if visited[i]:
                for j in range(num_vertices):
                    if not visited[j] and adj_matrix[i][j] > min_distance:
                        min_distance = adj_matrix[i][j]
                        x, y = i, j

        MST_edges.append((x, y))
        visited[y] = True
        num_of_edges += 1

    return MST_edges


def find_end_nodes(edges: List[tuple[int, int]], node_num: int):
    count = np.array([0] * node_num)
    for edge in edges:
        for node in edge:
            count[node] += 1
    return np.where(count == 1)


def DFS(adj_matrix, visited, current, target, edges, path, s):
    # debug(f'path: {path}')
    for edge in edges:
        if current == edge[0] and not visited[edge[1]]:
            new_visited = visited[:]
            new_visited[edge[1]] = True
            new_path = path[:]
            new_path.append(edge[1])
            if edge[1] == target:
                return sum(new_visited), sum(sum(s[:, new_visited])), new_path  # type: ignore
            forward_res = DFS(adj_matrix, new_visited, edge[1], target, edges, new_path, s)
            if forward_res is None:
                continue
            else:
                return forward_res
        if current == edge[1] and not visited[edge[0]]:
            new_visited = visited[:]
            new_visited[edge[0]] = True
            new_path = path[:]
            new_path.append(edge[0])
            if edge[0] == target:
                return sum(new_visited), sum(sum(s[:, new_visited])), new_path  # type: ignore
            forward_res = DFS(adj_matrix, new_visited, edge[0], target, edges, new_path, s)
            if forward_res is None:
                continue
            else:
                return forward_res


def end_node_dis_matrix(adj_matrix: np.ndarray, end_nodes, edges, s):
    # debug(f'edges: {edges}')
    results = []
    end_node_num = len(end_nodes)
    for i in range(end_node_num):
        visited = [False] * len(adj_matrix)
        visited[end_nodes[i]] = True
        for j in range(i + 1, end_node_num):
            # debug(f'i, j, {i}, {j}')
            res = DFS(adj_matrix, visited, end_nodes[i], end_nodes[j], edges, [end_nodes[i]], s)
            results.append([(end_nodes[i], end_nodes[j]), *res])  # type: ignore

    return sorted(results, key=lambda x: (x[1], x[2]))[0]  # with maximum connectivity


def get_new_coord(path: List[int], adj_matrix) -> np.ndarray:
    # deprecated
    # assign 0-1 coordinate to each cluster according to the connectivity between clusters

    # coord = [0] * len(adj_matrix)
    # cum_norm_dis = 0
    # for index in range(1, len(path)):
    #     cum_norm_dis += adj_matrix[path[index - 1], path[index]]
    #     coord[path[index]] = cum_norm_dis

    # for node in range(len(adj_matrix)):
    #     if node in path:
    #         continue
    #     max_connect = float('-inf')
    #     for i in range(len(adj_matrix)):
    #         if i in path and adj_matrix[node, i] > max_connect:
    #             max_connect = adj_matrix[node, i]
    #             coord[node] = coord[i]

    pseudo_time_axis = np.zeros(len(adj_matrix))
    linspace = np.linspace(0, 1, len(path))

    for i, index in enumerate(path):
        # debug(f'i: {i}, index: {index}')
        pseudo_time_axis[index] = linspace[i]

    return pseudo_time_axis


def get_pseudo_time_branch(out_adj: ndarray, s: ndarray) -> ndarray:
    MST_edges = primMST(out_adj)
    end_nodes = find_end_nodes(MST_edges, len(out_adj))[0].tolist()
    (start, end), cluster_number, node_number, path = end_node_dis_matrix(out_adj, end_nodes, MST_edges, s)
    pesudo_time_axis = get_new_coord(path, out_adj)
    pseudo_time_per_node = s @ pesudo_time_axis.reshape((-1, 1))

    return pseudo_time_per_node


def get_init_cluster(data: Data, s: ndarray, init_node_label: int) -> List[int]:
    init_node_label_index = (data.y == init_node_label).flatten().detach().cpu().numpy()
    init_node_label_index_loading = init_node_label_index @ s
    # return init_node_label_index_loading.argmax()
    ascending_order = np.argsort(init_node_label_index_loading)
    return [ascending_order[-1]]


def get_pseudo_time_line(data: Data, out_adj: ndarray, s: ndarray, init_node_label: int) -> Tuple[ndarray, ndarray]:
    init_cluster: List[int] = get_init_cluster(data, s, init_node_label)
    info(message=f'Initial cluster was set to {init_cluster}.')

    path: list[int] = init_cluster
    visited: list[bool] = [False] * len(out_adj)
    for x in path:
        visited[x] = True
    while True:
        max_connect = float('-inf')
        next_cluster = -1
        for i in range(len(out_adj)):
            if not visited[i] and out_adj[path[-1], i] > max_connect:
                max_connect = out_adj[path[-1], i]
                next_cluster = i
        if next_cluster == -1:
            break
        path.append(next_cluster)
        visited[next_cluster] = True

    pseudo_time_axis = get_new_coord(path, out_adj)
    pseudo_time_per_node = s @ pseudo_time_axis.reshape((-1, 1))

    return pseudo_time_axis, pseudo_time_per_node


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
