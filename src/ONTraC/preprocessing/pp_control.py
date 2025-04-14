import os
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from torch_geometric.loader import DenseDataLoader

from ..data import SpatailOmicsDataset, load_dataset
from ..log import error, info
from ..utils import get_meta_data_file
from .data import load_meta_data, save_cell_type_code


def load_input_data(
    meta_input: Union[str, Path],
    NN_dir: Union[str, Path],
) -> Dict[str, pd.DataFrame]:
    """
    Load data from original inputs.
    :param meta_input: str or Path, meta data file.
    :param NN_dir: str or Path, save directory.
    :return: Dict[str, pd.DataFrame], loaded data.
    """

    output = {}

    # load meta_data
    meta_data_df = load_meta_data(save_dir=NN_dir, meta_data_file=meta_input)
    output['meta_data'] = meta_data_df

    # cell-level data, validate the meta_data
    if 'Cell_Type' not in meta_data_df.columns:
        raise ValueError('Cell_Type in metadata is required for cell-level data.')
    if 'Cell_ID' not in meta_data_df.columns:
        raise ValueError('Cell_ID in metadata is required for cell-level data.')

    return output


def preprocessing_nn(meta_input: Union[str, Path], NN_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocessing for niche network.
    :param meta_input: str or Path, meta data file.
    :param NN_dir: str or Path, save directory.
    :return: Tuple[pd.DataFrame, pd.DataFrame], meta data and cell type coding matrix.
    """

    input_data = load_input_data(meta_input=meta_input, NN_dir=NN_dir)

    if 'meta_data' not in input_data:
        raise ValueError('meta_data is required for preprocessing.')
    meta_data_df = input_data['meta_data']

    # generate cell type coding matrix
    meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype('category')
    ct_coding_matrix = np.zeros(shape=(meta_data_df.shape[0],
                                       meta_data_df['Cell_Type'].cat.categories.shape[0]))  # N x #cell_type
    ct_coding_matrix[np.arange(meta_data_df.shape[0]), meta_data_df.Cell_Type.cat.codes.values] = 1
    ct_coding = pd.DataFrame(data=ct_coding_matrix,
                             columns=meta_data_df['Cell_Type'].cat.categories,
                             index=meta_data_df.index)

    # save cell type code
    save_cell_type_code(save_dir=NN_dir, cell_types=meta_data_df['Cell_Type'])

    return meta_data_df, ct_coding


def load_data(NN_dir: Union[str, Path], batch_size: int = 0) -> Tuple[SpatailOmicsDataset, DenseDataLoader]:
    """
    Load data and create sample loader.
    :param NN_dir: str or Path, save directory.
    :param batch_size: int, batch size.
    :return: Tuple[SpatailOmicsDataset, DenseDataLoader], dataset and sample loader.
    """

    info('Loading dataset.')

    dataset = load_dataset(NN_dir=NN_dir)
    batch_size = batch_size if batch_size > 0 else len(dataset)
    sample_loader = DenseDataLoader(dataset, batch_size=batch_size)

    return dataset, sample_loader


def preprocessing_gnn(NN_dir: Union[str, Path],
                      batch_size: int = 0) -> Tuple[SpatailOmicsDataset, DenseDataLoader, pd.DataFrame]:
    """
    Preprocessing for GNN.
    :param NN_dir: str or Path, save directory.
    :param batch_size: int, batch size.
    :return: Tuple[SpatailOmicsDataset, DenseDataLoader, pd.DataFrame], dataset, sample loader, and meta data.
    """

    # meta data
    meta_data_df = pd.read_csv(get_meta_data_file(NN_dir), header=0)
    meta_data_df['Sample'] = meta_data_df['Sample'].astype(str).astype('category')
    meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype(str).astype('category')

    # dataset and sample loader
    dataset, sample_loader = load_data(NN_dir=NN_dir, batch_size=batch_size)

    return dataset, sample_loader, meta_data_df


def preprocessing_nt(NN_dir: Union[str, Path], GNN_dir: Union[str, Path]) -> Tuple[DataFrame, DataFrame, ndarray]:
    """
    Preprocessing for niche trajectory.
    :param NN_dir: str or Path, save directory.
    :param GNN_dir: str or Path, save directory.
    :return: Tuple[DataFrame, DataFrame, ndarray], meta data, niche-level niche cluster assign, and consolidate out_adj.
    """

    # params
    niche_level_niche_cluster_file = Path(f'{GNN_dir}/niche_level_niche_cluster.csv.gz')
    consolidate_out_adj_file = Path(f'{GNN_dir}/consolidate_out_adj.csv.gz')

    # load meta data
    meta_data_df = pd.read_csv(get_meta_data_file(NN_dir), header=0)
    meta_data_df['Sample'] = meta_data_df['Sample'].astype(str).astype('category')
    meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype(str).astype('category')

    # load niche-level niche cluster assign
    if not os.path.exists(niche_level_niche_cluster_file) or not os.path.exists(consolidate_out_adj_file):
        error(f'niche_level_niche_cluster.csv.gz or consolidate_out_adj.csv.gz does not exist in {GNN_dir} directory.')

    niche_level_niche_cluster_assign_df = pd.read_csv(niche_level_niche_cluster_file, header=0, index_col=0)

    # load consolidate out_adj
    consolidate_out_adj_array = np.loadtxt(consolidate_out_adj_file, delimiter=',')

    return meta_data_df, niche_level_niche_cluster_assign_df, consolidate_out_adj_array
