from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.sparse import load_npz

from ..log import info
from .algorithm import brute_force


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

    return niche_trajectory_path


def trajectory_path_to_NC_score(niche_trajectory_path: List[int],
                                niche_clustering_sum: ndarray) -> ndarray:
    """
    Convert niche cluster trajectory path to NTScore
    :param niche_trajectory_path: List[int], the niche trajectory path
    :param niche_clustering_sum: ndarray, the sum of each niche cluster
    :param equal_space: bool, whether the niche clusters are equally spaced in the trajectory
    :return: ndarray, the NTScore
    """

    info('Calculating NTScore for each niche cluster based on the trajectory path.')

    niche_NT_score = np.zeros(len(niche_trajectory_path))
    
    values = np.linspace(0, 1, len(niche_trajectory_path))
    for i, index in enumerate(niche_trajectory_path):
        # debug(f'i: {i}, index: {index}')
        niche_NT_score[index] = values[i]
    return niche_NT_score


def get_niche_NTScore(trajectory_construct_method: str,
                      niche_level_niche_cluster_assign_df: DataFrame,
                      niche_adj_matrix: ndarray) -> Tuple[ndarray, DataFrame]:
    """
    Get niche-level niche trajectory and cell-level niche trajectory
    :param trajectory_construct_method: str, the method to construct trajectory
    :param niche_level_niche_cluster_assign_df: DataFrame, the niche-level niche cluster assignment. #niche x #niche_cluster
    :param adj_matrix: ndarray, the adjacency matrix of the graph
    :return: Tuple[ndarray, DataFrame], the niche-level niche trajectory and cell-level niche trajectory
    """

    info('Calculating NTScore for each niche.')

    niche_trajectory_path = get_niche_trajectory_path(trajectory_construct_method=trajectory_construct_method,
                                                      niche_adj_matrix=niche_adj_matrix)

    niche_clustering_sum = niche_level_niche_cluster_assign_df.values.sum(axis=0)
    niche_cluster_score = trajectory_path_to_NC_score(niche_trajectory_path=niche_trajectory_path,
                                                      niche_clustering_sum=niche_clustering_sum)
    niche_level_NTScore_df = pd.DataFrame(niche_level_niche_cluster_assign_df.values @ niche_cluster_score,
                                          index=niche_level_niche_cluster_assign_df.index,
                                          columns=['Niche_NTScore'])
    return niche_cluster_score, niche_level_NTScore_df


def niche_to_cell_NTScore(meta_data_df: DataFrame, niche_level_NTScore_df: DataFrame, rel_params: Dict) -> DataFrame:
    """
    get cell-level NTScore
    :param meta_data_df: DataFrame, the meta data
    :param niche_level_NTScore_df: DataFrame, the niche-level NTScore
    :param rel_params: Dict, relative paths
    :return: DataFrame, cell-level NTScore
    """

    info('Projecting NTScore from niche-level to cell-level.')

    # prepare
    id_name: str = meta_data_df.columns[0]
    samples = meta_data_df['Sample'].cat.categories
    sample_files_by_name_dict = {
        sample_files_dict['Name']: sample_files_dict
        for sample_files_dict in rel_params['Data']
    }
    sample_cell_level_NTScore_list = []

    for sample in samples:
        sample_niche_level_NTScore_df = niche_level_NTScore_df.loc[meta_data_df[meta_data_df['Sample'] == sample][id_name].values]
        niche_weight_matrix = load_npz(sample_files_by_name_dict[sample]['NicheWeightMatrix'])
        if niche_weight_matrix.shape[0] != sample_niche_level_NTScore_df.shape[0]:
            raise ValueError(f'Inconsistent number of niches in {sample} sample. '
                             f'Please check the niche weight matrix and the niche-level NTScore.')
        if niche_weight_matrix.shape[1] != sample_niche_level_NTScore_df.shape[0]:
            raise ValueError(f'Inconsistent number of cells in {sample} sample. '
                             f'Please check the niche weight matrix and the niche-level NTScore.')
        niche_to_cell_matrix = (niche_weight_matrix / niche_weight_matrix.sum(axis=0)
                                ).T  # normalize by the all niches associated with each cell, N (#cell) x N (#niche)

        # cell-level NTScore
        niche_level_NTScore_ = sample_niche_level_NTScore_df.values.reshape(-1, 1)  # N x 1
        cell_level_NTScore_ = niche_to_cell_matrix @ niche_level_NTScore_
        sample_cell_level_NTScore_list.append(
            pd.DataFrame(cell_level_NTScore_.reshape(-1),
                         index=sample_niche_level_NTScore_df.index,
                         columns=['Cell_NTScore']))

    cell_level_NTScore_df = pd.concat(sample_cell_level_NTScore_list).loc[meta_data_df[id_name]]

    return cell_level_NTScore_df


def NTScore_table(save_dir: Union[str, Path], meta_data_df: DataFrame, niche_level_NTScore_df: DataFrame,
                  cell_level_NTScore_df: DataFrame, rel_params: Dict) -> None:
    """
    Generate NTScore table and save it
    :param save_dir: Union[str, Path], the directory to save NTScore table
    :param meta_data_df: DataFrame, the meta data
    :param niche_level_NTScore_df: DataFrame, the niche-level NTScore
    :param cell_level_NTScore_df: DataFrame, the cell-level NTScore
    :param rel_params: Dict, relative paths
    :return: None
    """

    info('Output NTScore tables.')

    # prepare
    id_name: str = meta_data_df.columns[0]
    samples = meta_data_df['Sample'].cat.categories
    sample_files_by_name_dict = {
        sample_files_dict['Name']: sample_files_dict
        for sample_files_dict in rel_params['Data']
    }
    NTScore_table = pd.DataFrame()

    for sample in samples:
        coordinates_df = pd.read_csv(sample_files_by_name_dict[sample]['Coordinates'], index_col=0)
        sample_niche_level_NTScore_df = niche_level_NTScore_df.loc[meta_data_df[meta_data_df['Sample'] == sample][id_name].values]
        sample_cell_level_NTScore_df = cell_level_NTScore_df.loc[meta_data_df[meta_data_df['Sample'] == sample][id_name].values]
        coordinates_df = coordinates_df.join(sample_niche_level_NTScore_df).join(sample_cell_level_NTScore_df)
        coordinates_df.to_csv(f'{save_dir}/{sample}_NTScore.csv.gz')
        NTScore_table = pd.concat([NTScore_table, coordinates_df])

    NTScore_table.loc[meta_data_df[id_name]].to_csv(f'{save_dir}/NTScore.csv.gz')
