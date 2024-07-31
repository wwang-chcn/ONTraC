import os
from optparse import Values
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import cholesky, eigh
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial import cKDTree, distance

from ..log import *


def gauss_dist_1d(dist: np.ndarray, n_local: int) -> float:
    """
    Compute gaussian affinity between two cells (a cell and its KNN)
    :param dist_use: knn spatial distance to be used
    :param n_local: index of distance used for normalization
    :return: gaussian distance
    """
    return np.exp(-(dist / dist[n_local])**2)


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

    n_local = options.n_local
    N = sample_data_df.shape[0]

    # get coordinates
    # TODO: support 3D coordinates
    coord_df = sample_data_df[['Cell_ID', 'x', 'y']]
    coord_df.to_csv(f'{options.preprocessing_dir}/{sample_name}_Coordinates.csv', index=False)

    # build KDTree
    coordinates = sample_data_df[['x', 'y']].values
    kdtree = cKDTree(data=coordinates)
    dis_matrix, indices_matrix = kdtree.query(x=coordinates,
                                              k=np.max([options.n_neighbors, options.n_local]) + 1)  # include self
    np.savetxt(f'{options.preprocessing_dir}/{sample_name}_NeighborIndicesMatrix.csv.gz', indices_matrix,
               delimiter=',')  # save indices matrix

    # save edge index file
    # 1) convert edge index to csr_matrix
    # 2) make it bidirectional
    # 3) convert it to edge index back
    # 4) save it
    src_indices = np.repeat(np.arange(coordinates.shape[0]), options.n_neighbors)
    dst_indices = indices_matrix[:, 1:options.n_neighbors + 1].flatten()  # remove self, N x k, #niche x #cell
    adj_matrix = csr_matrix((np.ones(dst_indices.shape[0]), (src_indices, dst_indices)),
                            shape=(N, N))  # convert to csr_matrix
    adj_matrix = adj_matrix + adj_matrix.transpose()  # make it bidirectional
    edge_index = np.argwhere(adj_matrix > 0)  # convert it to edge index back
    edge_index_file = f'{options.preprocessing_dir}/{sample_name}_EdgeIndex.csv.gz'
    np.savetxt(edge_index_file, edge_index, delimiter=',', fmt='%d')

    # calculate niche_weight_matrix and normalize it using self node and n_local-th neighbor using a gaussian kernel
    # calculate cell_to_niche_matrix
    niche_weight_matrix = np.apply_along_axis(func1d=gauss_dist_1d, axis=1, arr=dis_matrix,
                                              n_local=n_local)[:, :options.n_neighbors +
                                                               1]  # N x (k + 1), #niche x #cell
    src_indices = np.repeat(np.arange(coordinates.shape[0]), options.n_neighbors + 1)
    dst_indices = indices_matrix[:, :options.n_neighbors + 1].flatten()  # include self, N x (k + 1), #niche x #cell
    niche_weight_matrix_csr = csr_matrix((niche_weight_matrix.flatten(), (src_indices, dst_indices)),
                                         shape=(N, N))  # convert to csr_matrix, N x N, #niche x #cell
    save_npz(file=f'{options.preprocessing_dir}/{sample_name}_NicheWeightMatrix.npz',
             matrix=niche_weight_matrix_csr)  # save weight matrix
    cell_to_niche_matrix = niche_weight_matrix_csr / niche_weight_matrix_csr.sum(axis=1)  # N x N, #niche x #cell

    # calculate cell type composition
    if options.decomposition_cell_type_composition_input is None:
        one_hot_matrix = np.zeros(shape=(N, sample_data_df['Cell_Type'].cat.categories.shape[0]))  # N (#cell) x #cell_type
        one_hot_matrix[np.arange(N), sample_data_df.Cell_Type.cat.codes.values] = 1
        cell_type_composition = cell_to_niche_matrix @ one_hot_matrix  # N (#niche) x #cell_type
    else: # use the provided cell type composition, spot-based data
        decomposition_cell_type_composition_df = pd.read_csv(options.decomposition_cell_type_composition_input,
                                                                 header=0,
                                                                 index_col=0)  # N (#cell) x #cell_type
        cell_type_composition = cell_to_niche_matrix @ decomposition_cell_type_composition_df.values  # N (#niche) x #cell_type

    # save cell type composition
    np.savetxt(f'{options.preprocessing_dir}/{sample_name}_CellTypeComposition.csv.gz',
               cell_type_composition,
               delimiter=',')


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


def get_embedding_columns(ori_data_df: pd.DataFrame) -> List[str]:
    """
    Get embedding columns from the original data
    :param ori_data_df: pd.DataFrame, original data
    :return: List[str], embedding columns
    """
    embedding_start_column = [x for x in ori_data_df.columns if x.startswith('Embedding_')]
    embedding_columns = []
    i = 0
    while True:
        i += 1
        if f'Embedding_{i}' in embedding_start_column:
            embedding_columns.append(f'Embedding_{i}')
        else:
            break

    return embedding_columns


def ct_coding_adjust(options: Values, ori_data_df: pd.DataFrame):
    """
    Adjust the cell type coding according to embeddings

    1) check the embedding info in the original data
    2) calculate embedding postion for each cell type
    3) calculate distance between each cell type
    4) calculate the M
    5) got the new basis for cell type coding
    6) adjust the cell type coding

    :param options: Values, options
    :param ori_data_df: pd.DataFrame, original data
    :return: None
    """

    if options.decomposition_cell_type_composition_input is None:
        # check the embedding info in the original data
        embedding_columns = get_embedding_columns(ori_data_df)
        if len(embedding_columns) < 2:
            warning('At least two (Embedding_1 and Embedding_2) should be in the original data. Skip the adjustment.')
            return

        # calculate embedding postion for each cell type
        ct_embedding = ori_data_df[embedding_columns + ['Cell_Type']].groupby('Cell_Type').mean()
        # calculate distance between each cell type
        raw_distance = distance.cdist(ct_embedding[embedding_columns].values, ct_embedding[embedding_columns].values,
                                  'euclidean')
    
    else:  # spot-based data
        ct_embedding = np.loadtxt(f'{options.preprocessing_dir}/PCA_embedding.csv', delimiter=',')
        raw_distance = distance.cdist(ct_embedding, ct_embedding, 'euclidean')

    median_distance = np.median(raw_distance[np.triu_indices(raw_distance.shape[0], k=1)])
    info(f'Median distance between cell types: {median_distance}')

    # calculate the M
    M = np.exp(-(raw_distance / (options.sigma * median_distance)) **2)

    # got the new basis for cell type coding
    eig_val, eig_vec = eigh(M)
    if np.any(eig_val < 0):
        warning('Negative eigenvalues found. Skip the adjustment.')
        return
    L = cholesky(M)

    # adjust the cell type coding
    for sample in ori_data_df['Sample'].unique():
        feat_file = f'{options.preprocessing_dir}/{sample}_CellTypeComposition.csv.gz'
        ctc_raw = np.loadtxt(feat_file, delimiter=',')
        ctc_new = ctc_raw @ L.T
        os.rename(feat_file, f'{options.preprocessing_dir}/{sample}_Raw_CellTypeComposition.csv.gz')
        np.savetxt(feat_file, ctc_new, delimiter=',')
