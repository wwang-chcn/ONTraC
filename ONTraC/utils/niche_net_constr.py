from optparse import Values
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


def load_original_data(options: Values, data_file: str) -> pd.DataFrame:
    """
    Load original data
    :param data_file: str, original data file
    :return: pd.DataFrame, original data

    1) read original data file (csv format)
    2) check if Cell_ID, Sample, Cell_Type, x, and y columns in the original data
    3) make the Cell_Type column categorical
    4) return
        1. original data with Cell_ID, Sample, Cell_Type, x, and y columns
        2. samples
    """

    # read original data file
    ori_data_df = pd.read_csv(data_file)

    # check if Cell_ID, Sample, Cell_Type, x, and y columns in the original data
    if 'Cell_ID' not in ori_data_df.columns:
        raise ValueError('Cell_ID column is missing in the original data.')
    if 'Sample' not in ori_data_df.columns:
        raise ValueError('Sample column is missing in the original data.')
    if 'Cell_Type' not in ori_data_df.columns:
        raise ValueError('Cell_Type column is missing in the original data.')
    if 'x' not in ori_data_df.columns:
        raise ValueError('x column is missing in the original data.')
    if 'y' not in ori_data_df.columns:
        raise ValueError('y column is missing in the original data.')

    ori_data_df = ori_data_df.dropna(subset=['Cell_ID', 'Sample', 'Cell_Type', 'x', 'y'])

    # make the Cell_Type column categorical
    ori_data_df['Cell_Type'] = ori_data_df['Cell_Type'].astype('category')
    # save mappings of the categorical data
    cell_type_code = pd.DataFrame(enumerate(ori_data_df['Cell_Type'].cat.categories), columns=['Code', 'Cell_Type'])
    cell_type_code.to_csv(f'{options.output}/cell_type_code.csv', index=False)

    return ori_data_df


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

    n_local = 20
    N = sample_data_df.shape[0]

    # get coordinates
    # TODO: support 3D coordinates
    coord_df = sample_data_df[['Cell_ID', 'x', 'y']]
    coord_df.to_csv(f'{options.output}/{sample_name}_Coordinates.csv', index=False)

    # build KDTree
    coordinates = sample_data_df[['x', 'y']].values
    kdtree = cKDTree(data=coordinates)
    dis_matrix, indices_matrix = kdtree.query(x=coordinates, k=options.n_neighbors + 1)  # include self
    np.savetxt(f'{options.output}/{sample_name}_NeighborIndicesMatrix.csv.gz', indices_matrix, delimiter=',')  # save indices matrix

    # save edge index file
    # 1) convert edge index to csr_matrix
    # 2) make it bidirectional
    # 3) convert it to edge index back
    # 4) save it
    edge_index = np.argwhere(indices_matrix[:, 1:] > -1)  # remove self
    adj_matrix = csr_matrix((np.ones(edge_index.shape[0]), (edge_index[:, 0], edge_index[:, 1])), shape=(N, N))
    adj_matrix = adj_matrix + adj_matrix.transpose()
    edge_index = np.argwhere(adj_matrix.todense() > 0)
    edge_index_file = f'{options.output}/{sample_name}_EdgeIndex.csv.gz'
    np.savetxt(edge_index_file, edge_index, delimiter=',', fmt='%d')

    # calculate weight matrix
    # normalize the distance matrix using self node and 20-th neighbor using a gaussian kernel
    norm_dis_matrix = np.apply_along_axis(func1d=gauss_dist_1d, axis=1, arr=dis_matrix, n_local=n_local)  # N x (k + 1)
    np.savetxt(f'{options.output}/{sample_name}_NicheWeightMatrix.csv.gz', norm_dis_matrix,
               delimiter=',')  # save weight matrix

    # calculate cell type composition
    sample_data_df.Cell_Type.cat.codes.values
    one_hot_matrix = np.zeros(shape=(N, sample_data_df['Cell_Type'].cat.categories.shape[0]))  # N x n_cell_type
    one_hot_matrix[np.arange(N), sample_data_df.Cell_Type.cat.codes.values] = 1

    # extract non-weighted cell type composition
    raw_cell_type_composition = one_hot_matrix[indices_matrix]  # N x (k + 1) x n_cell_type
    cell_type_composition = (raw_cell_type_composition * norm_dis_matrix[..., np.newaxis]).sum(
        axis=1)  # N x n_cell_type

    # save cell type composition
    np.savetxt(f'{options.output}/{sample_name}_CellTypeComposition.csv.gz', cell_type_composition, delimiter=',')


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

    data: Dict[str, List[Any]] = {'Data': []}
    for sample in ori_data_df['Sample'].unique():
        data['Data'].append({
            'Name': sample,
            'Coordinates': f'{sample}_Coordinates.csv',
            'EdgeIndex': f'{sample}_EdgeIndex.csv.gz',
            'Features': f'{sample}_CellTypeComposition.csv.gz',
            'NicheWeightMatrix': f'{sample}_NicheWeightMatrix.csv.gz',
            'NeighborIndicesMatrix': f'{sample}_NeighborIndicesMatrix.csv.gz'
        })

    yaml_file = f'{options.output}/samples.yaml'
    with open(yaml_file, 'w') as fhd:
        yaml.dump(data, fhd)
