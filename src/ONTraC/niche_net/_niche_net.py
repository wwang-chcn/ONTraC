import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import cholesky, eigh
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial import cKDTree, distance

from ..log import *


def build_knn_network(sample_name: str,
                      sample_meta_df: pd.DataFrame,
                      n_neighbors: int = 50,
                      n_local: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build KNN network for a sample.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param n_neighbors: int, number of neighbors.
    :param n_local: int, index of distance used for normalization.
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray] (coordinates, dis_matrix, indices_matrix).
    """

    info(f'Building KNN network for sample: {sample_name}...')

    # build KDTree
    coordinates = sample_meta_df[['x', 'y']].values
    kdtree = cKDTree(data=coordinates)
    dis_matrix, indices_matrix = kdtree.query(x=coordinates, k=np.max([n_neighbors, n_local]) + 1)  # include self

    return coordinates, dis_matrix, indices_matrix


def calc_edge_index(sample_name: str,
                    sample_meta_df: pd.DataFrame,
                    indices_matrix: np.ndarray,
                    n_neighbors: int = 50) -> np.ndarray:
    """
    Calculate edge index.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param indices_matrix: np.ndarray, indices matrix.
    :param n_neighbors: int, number of neighbors.
    :return: np.ndarray, edge index.
    """

    info(f'Calculating edge index for sample: {sample_name}...')

    N = sample_meta_df.shape[0]

    # save edge index file
    # 1) convert edge index to csr_matrix
    # 2) make it bidirectional
    # 3) convert it to edge index back
    # 4) save it
    src_indices = np.repeat(np.arange(sample_meta_df.shape[0]), n_neighbors)
    dst_indices = indices_matrix[:, 1:n_neighbors + 1].flatten()  # remove self, N x k, #niche x #cell
    adj_matrix = csr_matrix((np.ones(dst_indices.shape[0]), (src_indices, dst_indices)),
                            shape=(N, N))  # convert to csr_matrix
    adj_matrix = adj_matrix + adj_matrix.transpose()  # make it bidirectional
    edge_index = np.argwhere(adj_matrix > 0)  # convert it to edge index back

    return edge_index


def gauss_dist_1d(dist: np.ndarray, n_local: int) -> float:
    """
    Compute gaussian affinity between two cells (a cell and its KNN).
    :param dist: np.ndarray, distance.
    :param n_local: int, index of distance used for normalization.
    :return: float, gaussian affinity.
    """
    return np.exp(-(dist / dist[n_local])**2)


def calc_niche_weight_matrix(sample_name: str,
                             sample_meta_df: pd.DataFrame,
                             dis_matrix: np.ndarray,
                             indices_matrix: np.ndarray,
                             n_neighbors: int = 50,
                             n_local: int = 20) -> csr_matrix:
    """
    Calculate niche_weight_matrix and normalize it using self node and n_local-th neighbor using a gaussian kernel.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param dis_matrix: np.ndarray, distance matrix.
    :param indices_matrix: np.ndarray, indices matrix.
    :param n_neighbors: int, number of neighbors.
    :param n_local: int, index of distance used for normalization.
    :return: csr_matrix, niche weight matrix.
    """

    info(f'Calculating niche weight matrix for sample: {sample_name}...')

    N = sample_meta_df.shape[0]
    niche_weight_matrix = np.apply_along_axis(func1d=gauss_dist_1d, axis=1, arr=dis_matrix,
                                              n_local=n_local)[:, :n_neighbors + 1]  # N x (k + 1), #niche x #cell
    src_indices = np.repeat(np.arange(sample_meta_df.shape[0]), n_neighbors + 1)
    dst_indices = indices_matrix[:, :n_neighbors + 1].flatten()  # include self, N x (k + 1), #niche x #cell
    niche_weight_matrix_csr = csr_matrix((niche_weight_matrix.flatten(), (src_indices, dst_indices)),
                                         shape=(N, N))  # convert to csr_matrix, N x N, #niche x #cell

    return niche_weight_matrix_csr


def calc_cell_type_composition(niche_weight_matrix: csr_matrix,
                               ct_coding_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate cell type composition.
    :param niche_weight_matrix: csr_matrix, niche weight matrix.
    :param ct_coding_matrix: Optional[np.ndarray], decompsited cell type.
    :return: np.ndarray, cell type composition.
    """

    # calculate cell type composition
    cell_to_niche_matrix = niche_weight_matrix / niche_weight_matrix.sum(axis=1)  # N x N, #niche x #cell
    cell_type_composition = cell_to_niche_matrix @ ct_coding_matrix  # N x n_cell_type

    return cell_type_composition


def save_niche_network(sample_name: str, sample_meta_df: pd.DataFrame, indices_matrix: np.ndarray,
                       edge_index: np.ndarray, niche_weight_matrix: csr_matrix, cell_type_composition: np.ndarray,
                       save_dir: Union[str, Path]) -> None:
    """
    Save the results to disk.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param indices_matrix: np.ndarray, indices matrix.
    :param edge_index: np.ndarray, edge index.
    :param niche_weight_matrix: csr_matrix, niche weight matrix.
    :param cell_type_composition: np.ndarray, cell type composition.
    :param save_dir: str, save directory.
    :return: None.
    """

    # save coordinates
    # TODO: support 3D coordinates
    id_name = sample_meta_df.columns[0]
    coord_df = sample_meta_df[[id_name, 'x', 'y']]
    coord_df.to_csv(f'{save_dir}/{sample_name}_Coordinates.csv', index=False)

    # save the kNN network
    neighbor_indices_file = f'{save_dir}/{sample_name}_NeighborIndicesMatrix.csv.gz'
    np.savetxt(fname=neighbor_indices_file, X=indices_matrix, delimiter=',')  # save indices matrix
    edge_index_file = f'{save_dir}/{sample_name}_EdgeIndex.csv.gz'
    np.savetxt(edge_index_file, edge_index, delimiter=',', fmt='%d')  # save edge index

    # save niche_weight_matrix
    niche_weight_file = f'{save_dir}/{sample_name}_NicheWeightMatrix.npz'
    save_npz(file=niche_weight_file, matrix=niche_weight_matrix)  # save weight matrix

    # save cell type composition
    cell_type_composition_file = f'{save_dir}/{sample_name}_CellTypeComposition.csv.gz'
    np.savetxt(fname=cell_type_composition_file, X=cell_type_composition, delimiter=',')


def construct_niche_network_sample(sample_name: str,
                                   sample_meta_df: pd.DataFrame,
                                   sample_ct_coding: pd.DataFrame,
                                   save_dir: Union[str, Path],
                                   n_neighbors: int = 50,
                                   n_local: int = 20) -> None:
    """
    Construct niche network for a sample.
    :param sample_name: str, sample name.
    :param sample_meta_df: pd.DataFrame, sample data.
    :param sample_ct_coding: pd.DataFrame, sample cell type coding.
    :param save_dir: str, save directory.
    :param n_neighbors: int, number of neighbors.
    :param n_local: int, index of distance used for normalization.
    :return: None.

    1) get coordinates and save it.
    2) save the celltype information.
    3) build KDTree.
        1. save edge index file.
        2. calculate weight matrix.
        3. calculate cell type composition and save it.
    """

    info(f'Constructing niche network for sample: {sample_name}.')

    # build kNN network
    coordinates, dis_matrix, indices_matrix = build_knn_network(sample_name=sample_name,
                                                                sample_meta_df=sample_meta_df,
                                                                n_neighbors=n_neighbors,
                                                                n_local=n_local)

    # calculate edge index
    edge_index = calc_edge_index(sample_name=sample_name,
                                 sample_meta_df=sample_meta_df,
                                 indices_matrix=indices_matrix,
                                 n_neighbors=n_neighbors)

    # calculate niche_weight_matrix
    niche_weight_matrix = calc_niche_weight_matrix(sample_name=sample_name,
                                                   sample_meta_df=sample_meta_df,
                                                   dis_matrix=dis_matrix,
                                                   indices_matrix=indices_matrix,
                                                   n_neighbors=n_neighbors,
                                                   n_local=n_local)

    # calculate cell type composition
    info(f'Calculating cell type composition for sample: {sample_name}...')
    cell_type_composition = calc_cell_type_composition(niche_weight_matrix=niche_weight_matrix,
                                                       ct_coding_matrix=sample_ct_coding.values)

    save_niche_network(sample_meta_df=sample_meta_df,
                       sample_name=sample_name,
                       indices_matrix=indices_matrix,
                       edge_index=edge_index,
                       niche_weight_matrix=niche_weight_matrix,
                       cell_type_composition=cell_type_composition,
                       save_dir=save_dir)


def construct_niche_network(meta_data_df: pd.DataFrame,
                            ct_coding_df: pd.DataFrame,
                            save_dir: Union[str, Path],
                            n_neighbors: int = 50,
                            n_local: int = 20) -> None:
    """
    Construct niche network.
    :param meta_data_df: pd.DataFrame, meta data.
    :param ct_coding_df: pd.DataFrame, cell type coding.
    :param save_dir: str, save directory.
    :param n_neighbors: int, number of neighbors.
    :param n_local: int, index of distance used for normalization.
    :return: None.
    """

    # get samples
    samples = meta_data_df['Sample'].unique().tolist()

    # construct niche network for each sample
    for sample_name in samples:
        sample_meta_df = meta_data_df[meta_data_df['Sample'] == sample_name]
        sample_ct_coding = ct_coding_df.loc[meta_data_df[meta_data_df['Sample'] == sample_name].iloc[:, 0]]
        construct_niche_network_sample(sample_name=sample_name,
                                       sample_meta_df=sample_meta_df,
                                       sample_ct_coding=sample_ct_coding,
                                       save_dir=save_dir,
                                       n_neighbors=n_neighbors,
                                       n_local=n_local)


def gen_samples_yaml(meta_data_df: pd.DataFrame, save_dir: Union[str, Path]) -> None:
    """
    Generate samples.yaml.
    :param meta_data_df: pd.DataFrame, meta data.
    :param save_dir: str, save directory.
    :return: None.
    """

    info('Generating samples.yaml file.')

    data: Dict[str, List[Any]] = {'Data': []}
    for sample in meta_data_df['Sample'].unique():
        data['Data'].append({
            'Name': f'{sample}',
            'Coordinates': f'{sample}_Coordinates.csv',
            'EdgeIndex': f'{sample}_EdgeIndex.csv.gz',
            'Features': f'{sample}_CellTypeComposition.csv.gz',
            'NicheWeightMatrix': f'{sample}_NicheWeightMatrix.npz',
            'NeighborIndicesMatrix': f'{sample}_NeighborIndicesMatrix.csv.gz'
        })

    yaml_file = f'{save_dir}/samples.yaml'
    with open(yaml_file, 'w') as fhd:
        yaml.dump(data, fhd)


def ct_coding_adjust(NN_dir: Union[str, Path],
                     meta_data_df: pd.DataFrame,
                     embedding_df: pd.DataFrame,
                     deconvoluted_exp_input: Optional[Union[str, Path]],
                     sigma: float = 1.0) -> None:
    """
    Adjust the cell type coding according to embeddings

    1) check the embedding info in the original data
    2) calculate embedding postion for each cell type
    3) calculate distance between each cell type
    4) calculate the M
    5) got the new basis for cell type coding
    6) adjust the cell type coding

    :param options: Values, options
    :param meta_df: pd.DataFrame, meta data
    :return: None
    """

    if deconvoluted_exp_input is None:
        # check the embedding info in the original data
        if embedding_df.shape[1] < 2:
            warning('At least two (Embedding_1 and Embedding_2) should be in the original data. Skip the adjustment.')
            return

        # calculate embedding postion for each cell type
        ct_embedding = pd.concat(
            [meta_data_df.set_index(meta_data_df.columns[0])['Cell_Type'], embedding_df.iloc[:, 1:]],  # remove Cell_ID
            axis=1).groupby('Cell_Type').mean()

    else:  # spot-based data
        ct_embedding = pd.read_csv(deconvoluted_exp_input, index_col=0)  # #cell_type x #gene

    # calculate distance between each cell type
    ct_embedding.to_csv(f'{NN_dir}/ct_embedding.csv', index=False)
    raw_distance = distance.cdist(ct_embedding.values, ct_embedding.values, 'euclidean')

    median_distance = np.median(raw_distance[np.triu_indices(raw_distance.shape[0], k=1)])
    info(f'Median distance between cell types: {median_distance}')

    # calculate the M
    M = np.exp(-(raw_distance / (sigma * median_distance))**2)

    # got the new basis for cell type coding
    eig_val, eig_vec = eigh(M)
    if np.any(eig_val < 0):
        warning('Negative eigenvalues found. Skip the adjustment.')
        return
    L = cholesky(M)

    # adjust the cell type coding
    for sample in meta_data_df['Sample'].unique():
        feat_file = f'{NN_dir}/{sample}_CellTypeComposition.csv.gz'
        ctc_raw = np.loadtxt(feat_file, delimiter=',')
        ctc_new = ctc_raw @ L.T
        os.rename(feat_file, f'{NN_dir}/{sample}_Raw_CellTypeComposition.csv.gz')
        np.savetxt(feat_file, ctc_new, delimiter=',')
