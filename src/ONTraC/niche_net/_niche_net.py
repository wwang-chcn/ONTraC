from optparse import Values
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial import cKDTree

from ..log import *


def build_knn_network(options: Values, sample_data_df: pd.DataFrame,
                      sample_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build KNN network for a sample
    :param options: Values, options
    :param sample_data_df: pd.DataFrame, sample data
    :param sample_name: str, sample name
    :param save: bool, save indices matrix or not
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray] (coordinates, dis_matrix, indices_matrix)
    """

    info(f'Building KNN network for sample: {sample_name}...')

    # build KDTree
    coordinates = sample_data_df[['x', 'y']].values
    kdtree = cKDTree(data=coordinates)
    dis_matrix, indices_matrix = kdtree.query(x=coordinates,
                                              k=np.max([options.n_neighbors, options.n_local]) + 1)  # include self

    return coordinates, dis_matrix, indices_matrix


def calc_edge_index(options: Values, sample_data_df: pd.DataFrame, indices_matrix: np.ndarray,
                    sample_name: str) -> np.ndarray:
    """
    Calculate edge index
    :param options: Values, options
    :param sample_data_df: pd.DataFrame, sample data
    :param coordinates: np.ndarray, coordinates
    :param indices_matrix: np.ndarray, indices matrix
    :param sample_name: str, sample name
    :return: np.ndarray, edge index
    """

    info(f'Calculating edge index for sample: {sample_name}...')

    N = sample_data_df.shape[0]

    # save edge index file
    # 1) convert edge index to csr_matrix
    # 2) make it bidirectional
    # 3) convert it to edge index back
    # 4) save it
    src_indices = np.repeat(np.arange(sample_data_df.shape[0]), options.n_neighbors)
    dst_indices = indices_matrix[:, 1:options.n_neighbors + 1].flatten()  # remove self, N x k, #niche x #cell
    adj_matrix = csr_matrix((np.ones(dst_indices.shape[0]), (src_indices, dst_indices)),
                            shape=(N, N))  # convert to csr_matrix
    adj_matrix = adj_matrix + adj_matrix.transpose()  # make it bidirectional
    edge_index = np.argwhere(adj_matrix > 0)  # convert it to edge index back

    return edge_index


def gauss_dist_1d(dist: np.ndarray, n_local: int) -> float:
    """
    Compute gaussian affinity between two cells (a cell and its KNN)
    :param dist_use: knn spatial distance to be used
    :param n_local: index of distance used for normalization
    :return: gaussian distance
    """
    return np.exp(-(dist / dist[n_local])**2)


def calc_niche_weight_matrix(options: Values, sample_data_df: pd.DataFrame, dis_matrix: np.ndarray,
                             indices_matrix: np.ndarray, sample_name: str) -> csr_matrix:
    """
    Calculate niche_weight_matrix and normalize it using self node and n_local-th neighbor using a gaussian kernel
    :param options: Values, options
    :param sample_data_df: pd.DataFrame, sample data
    :param coordinates: np.ndarray, coordinates
    :param dis_matrix: np.ndarray, distance matrix
    :param indices_matrix: np.ndarray, indices matrix
    :param sample_name: str, sample name
    :param save: bool, save weight matrix or not
    :param output: bool, return weight matrix or not
    :return: csr_matrix, niche weight matrix
    """

    info(f'Calculating niche weight matrix for sample: {sample_name}...')

    N = sample_data_df.shape[0]
    niche_weight_matrix = np.apply_along_axis(func1d=gauss_dist_1d, axis=1, arr=dis_matrix,
                                              n_local=options.n_local)[:, :options.n_neighbors +
                                                                       1]  # N x (k + 1), #niche x #cell
    src_indices = np.repeat(np.arange(sample_data_df.shape[0]), options.n_neighbors + 1)
    dst_indices = indices_matrix[:, :options.n_neighbors + 1].flatten()  # include self, N x (k + 1), #niche x #cell
    niche_weight_matrix_csr = csr_matrix((niche_weight_matrix.flatten(), (src_indices, dst_indices)),
                                         shape=(N, N))  # convert to csr_matrix, N x N, #niche x #cell

    return niche_weight_matrix_csr


def calc_cell_type_composition(sample_data_df: pd.DataFrame, niche_weight_matrix: csr_matrix,
                               sample_name: str) -> np.ndarray:
    """
    Calculate cell type composition
    :param sample_data_df: pd.DataFrame, sample data
    :param niche_weight_matrix_csr: csr_matrix, niche weight matrix
    :param save: bool, save cell type composition or not
    :return: np.ndarray, cell type composition
    """

    info(f'Calculating cell type composition for sample: {sample_name}...')

    N = sample_data_df.shape[0]

    # calculate cell type composition
    cell_to_niche_matrix = niche_weight_matrix / niche_weight_matrix.sum(axis=1)  # N x N, #niche x #cell
    one_hot_matrix = np.zeros(shape=(N, sample_data_df['Cell_Type'].cat.categories.shape[0]))  # N x #cell_type
    one_hot_matrix[np.arange(N), sample_data_df.Cell_Type.cat.codes.values] = 1
    cell_type_composition = cell_to_niche_matrix @ one_hot_matrix  # N x n_cell_type

    return cell_type_composition


def save_niche_network(options: Values, sample_data_df: pd.DataFrame, sample_name: str, indices_matrix: np.ndarray,
                       edge_index: np.ndarray, niche_weight_matrix: csr_matrix,
                       cell_type_composition: np.ndarray) -> None:
    """
    Save the results to disk.
    :param options: Values, options
    :param ori_data_df: pd.DataFrame, original data
    :param sample_name: str, sample name
    """

    # save coordinates
    # TODO: support 3D coordinates
    coord_df = sample_data_df[['Cell_ID', 'x', 'y']]
    coord_df.to_csv(f'{options.preprocessing_dir}/{sample_name}_Coordinates.csv', index=False)

    # save the kNN network
    neighbor_indices_file = f'{options.preprocessing_dir}/{sample_name}_NeighborIndicesMatrix.csv.gz'
    np.savetxt(fname=neighbor_indices_file, X=indices_matrix, delimiter=',')  # save indices matrix
    edge_index_file = f'{options.preprocessing_dir}/{sample_name}_EdgeIndex.csv.gz'
    np.savetxt(edge_index_file, edge_index, delimiter=',', fmt='%d')  # save edge index

    # save niche_weight_matrix
    niche_weight_file = f'{options.preprocessing_dir}/{sample_name}_NicheWeightMatrix.npz'
    save_npz(file=niche_weight_file, matrix=niche_weight_matrix)  # save weight matrix

    # save cell type composition
    cell_type_composition_file = f'{options.preprocessing_dir}/{sample_name}_CellTypeComposition.csv.gz'
    np.savetxt(fname=cell_type_composition_file, X=cell_type_composition, delimiter=',')


def construct_niche_network_sample(options: Values, sample_data_df: pd.DataFrame, sample_name: str) -> None:
    """
    Construct niche network for a sample
    :param options: Values, options
    :param sample_data_df: pd.DataFrame, sample data
    :param sample_name: str, sample name
    :return: None

    1) get coordinates and save it.
    2) save the celltype information
    3) build KDTree
        1. save edge index file
        2. calculate weight matrix
        3. calculate cell type composition and save it
    """

    info(f'Constructing niche network for sample: {sample_name}.')

    # build kNN network
    coordinates, dis_matrix, indices_matrix = build_knn_network(options=options,
                                                                sample_data_df=sample_data_df,
                                                                sample_name=sample_name)

    # calculate edge index
    edge_index = calc_edge_index(options=options,
                                 sample_data_df=sample_data_df,
                                 indices_matrix=indices_matrix,
                                 sample_name=sample_name)

    # calculate niche_weight_matrix
    niche_weight_matrix = calc_niche_weight_matrix(options=options,
                                                   sample_data_df=sample_data_df,
                                                   dis_matrix=dis_matrix,
                                                   indices_matrix=indices_matrix,
                                                   sample_name=sample_name)  # type: ignore

    # calculate cell type composition
    cell_type_composition = calc_cell_type_composition(sample_data_df=sample_data_df,
                                                       niche_weight_matrix=niche_weight_matrix,
                                                       sample_name=sample_name)

    save_niche_network(options=options,
                       sample_data_df=sample_data_df,
                       sample_name=sample_name,
                       indices_matrix=indices_matrix,
                       edge_index=edge_index,
                       niche_weight_matrix=niche_weight_matrix,
                       cell_type_composition=cell_type_composition)


def construct_niche_network(options: Values, ori_data_df: pd.DataFrame) -> None:
    """
    Construct niche network
    :param ori_data_df: pd.DataFrame, original data
    :return: None
    """

    # get samples
    samples = ori_data_df['Sample'].unique()

    # construct niche network for each sample
    for sample in samples:
        sample_data_df = ori_data_df[ori_data_df['Sample'] == sample]
        construct_niche_network_sample(options=options, sample_data_df=sample_data_df, sample_name=sample)


def gen_samples_yaml(options: Values, ori_data_df: pd.DataFrame) -> None:
    """
    Generate samples.yaml
    :param ori_data_df: pd.DataFrame, original data
    :return: None
    """

    info('Generating samples.yaml file.')

    data: Dict[str, List[Any]] = {'Data': []}
    for sample in ori_data_df['Sample'].unique():
        data['Data'].append({
            'Name': f'{sample}',
            'Coordinates': f'{sample}_Coordinates.csv',
            'EdgeIndex': f'{sample}_EdgeIndex.csv.gz',
            'Features': f'{sample}_CellTypeComposition.csv.gz',
            'NicheWeightMatrix': f'{sample}_NicheWeightMatrix.npz',
            'NeighborIndicesMatrix': f'{sample}_NeighborIndicesMatrix.csv.gz'
        })

    yaml_file = f'{options.preprocessing_dir}/samples.yaml'
    with open(yaml_file, 'w') as fhd:
        yaml.dump(data, fhd)
