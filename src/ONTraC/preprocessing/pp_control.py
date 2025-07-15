import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from torch_geometric.loader import DenseDataLoader

from ..data import SpatailOmicsDataset, load_dataset
from ..external.deconvolution import apply_STdeconvolve
from ..log import error, info
from ..utils import get_meta_data_file
from .data import load_meta_data, save_cell_type_code, save_meta_data
from .expression import define_neighbors, perform_harmony, perform_leiden, perform_pca, perform_umap


def load_input_data(
    meta_input: Union[str, Path],
    NN_dir: Union[str, Path],
    exp_input: Optional[Union[str, Path]] = None,
    embedding_input: Optional[Union[str, Path]] = None,
    low_res_exp_input: Optional[Union[str, Path]] = None,
    deconvoluted_ct_composition: Optional[Union[str, Path]] = None,
    deconvoluted_exp_input: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load data from original inputs.
    :param meta_input: str or Path, meta data file.
    :param NN_dir: str or Path, save directory.
    :return: Dict[str, pd.DataFrame], loaded data.
    """

    output = {}

    # default values
    output['embedding_data'] = None

    # ----- load meta_data -----
    meta_data_df = load_meta_data(save_dir=NN_dir, meta_data_file=meta_input)
    ids = meta_data_df.iloc[:, 0].values.tolist()
    output['meta_data'] = meta_data_df  # N x X

    # cell or spot level data?
    id_name = meta_data_df.columns[0]
    if id_name == 'Cell_ID':
        if embedding_input is not None:
            embedding_df = pd.read_csv(embedding_input, index_col=0)
            # TODO: error message module
            if embedding_df.index.tolist() != ids:
                raise ValueError('The first column of embedding input should be same as the first column of meta_data.')
            output['embedding_data'] = embedding_df  # N x #embedding
        elif exp_input is not None:
            exp_df = pd.read_csv(exp_input, index_col=0)
            # TODO: error message module
            if exp_df.index.tolist() != ids:
                raise ValueError('The first column of exp input should be same as the first column of meta_data.')
            output['exp_data'] = exp_df  # N x #gene
    else:
        if low_res_exp_input is not None:
            low_res_exp_df = pd.read_csv(low_res_exp_input, index_col=0)
            # TODO: error message module
            if low_res_exp_df.columns.tolist() != ids:
                raise ValueError('The first row in low_res_exp_data should be same as the first column of meta_data.')
            output['low_res_exp'] = low_res_exp_df  # #gene x #N
        if deconvoluted_ct_composition is not None:
            deconvoluted_ct_composition_df = pd.read_csv(deconvoluted_ct_composition, index_col=0)
            deconvoluted_ct_composition_df.index = deconvoluted_ct_composition_df.index.astype(str)
            deconvoluted_ct_composition_df.columns = deconvoluted_ct_composition_df.columns.astype(str)
            # TODO: error message module
            if deconvoluted_ct_composition_df.shape[0] != meta_data_df.shape[0]:
                raise ValueError('The number of rows in meta_data and deconvoluted_ct_composition should be the same.')
            output['deconvoluted_ct_composition'] = deconvoluted_ct_composition_df  # N x #cell_type
        if deconvoluted_exp_input is not None and deconvoluted_ct_composition is not None:  # if deconvoluted_exp_input is provided, deconvoluted_ct_composition should also be provided
            deconvoluted_exp_df = pd.read_csv(deconvoluted_exp_input, index_col=0)
            deconvoluted_exp_df.index = deconvoluted_exp_df.index.astype(str)
            deconvoluted_exp_df.columns = deconvoluted_exp_df.columns.astype(str)
            # TODO: error message module
            if deconvoluted_exp_df.index.tolist() != deconvoluted_ct_composition_df.columns.tolist():
                raise ValueError(
                    'The index of deconvoluted_exp_data should be same as the columns of deconvoluted_ct_composition.')
            output['deconvoluted_exp_data'] = deconvoluted_exp_df  # #cell_type x #gene

    return output


def perform_deconvolution(NN_dir: Union[str, Path],
                          dc_method: str,
                          exp_df: pd.DataFrame,
                          dc_ct_num: int,
                          gen_ct_embedding: bool = False) -> pd.DataFrame:
    """
    Perform deconvolution.
    :param NN_dir: str or Path, save directory.
    :param dc_method: str, deconvolution method.
    :param exp_df: pd.DataFrame, expression matrix.  #gene x #spot
    :param dc_ct_num: int, number of cell types.
    :param gen_ct_embedding: Generate cell type embedding or not.
    :return: np.ndarray, deconvoluted cell type matrix.  #spot x #cell_type
    """

    info(message='            -------- deconvolution -------           ')

    if dc_method == 'STdeconvolve':
        info(message='Apply STdeconvolve to low resolution data.')
        ct_coding_df = apply_STdeconvolve(NN_dir=NN_dir,
                                          exp_df=exp_df,
                                          ct_num=dc_ct_num,
                                          gen_ct_embedding=gen_ct_embedding)

    return ct_coding_df


def cal_cell_type_coding(
    input_data: Dict[str, pd.DataFrame],
    NN_dir: Union[str, Path],
    resolution: Optional[float] = None,
    dc_method: Optional[str] = None,
    dc_ct_num: Optional[int] = None,
    gen_ct_embedding: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    """

    meta_data_df = input_data['meta_data']

    id_name = meta_data_df.columns[0]
    if id_name == 'Cell_ID':
        if 'Cell_Type' in meta_data_df.columns:  # option 1: cell-level data with Cell_Type info in meta_data
            meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype('category')
            if gen_ct_embedding and input_data.get('embedding_data', None) is None:  # generate cell type embedding
                if input_data.get('exp_data', None) is None:
                    raise ValueError(
                        'exp_data or embedding_data should be provided for cell-level data when generating cell type embedding.'
                    )
                else:
                    pca_embedding = perform_pca(input_data['exp_data'])
                    if 'Batch' in meta_data_df.columns and meta_data_df['Batch'].nunique() > 1:
                        pca_embedding = perform_harmony(pca_embedding, meta_data_df, 'Batch')
                # save PCA embedding
                input_data['embedding_data'] = pd.DataFrame(data=pca_embedding,
                                                            columns=[f'PC{i+1}' for i in range(pca_embedding.shape[1])],
                                                            index=meta_data_df[id_name])
                np.savetxt(Path(NN_dir).joinpath('PCA_embedding.csv'), pca_embedding, delimiter=',')

        else:
            if input_data.get('embedding_data', None) is not None:  # option 2: cell-level data with embedding info
                pca_embedding = input_data['embedding_data'].values
            elif input_data.get('exp_data', None) is not None:  # option 3: cell-level data with gene expression data
                pca_embedding = perform_pca(input_data['exp_data'])
                if 'Batch' in meta_data_df.columns and meta_data_df['Batch'].nunique() > 1:
                    pca_embedding = perform_harmony(pca_embedding, meta_data_df, 'Batch')
            else:
                raise ValueError(
                    'Cell_Type column, exp_data, or embedding_data should be provided for cell-level data.')

            # save PCA embedding
            input_data['embedding_data'] = pd.DataFrame(data=pca_embedding,
                                                        columns=[f'PC{i+1}' for i in range(pca_embedding.shape[1])],
                                                        index=meta_data_df[id_name])
            np.savetxt(Path(NN_dir).joinpath('PCA_embedding.csv'), pca_embedding, delimiter=',')

            if resolution is None:
                raise ValueError(
                    'resolution should be provided for cell-level data with gene expression or embedding data as input when Cell_Type is not provided in meta data input.'
                )

            connectivities = define_neighbors(pca_embedding)
            leiden_result = perform_leiden(connectivities, resolution=resolution)
            umap_embedding = perform_umap(pca_embedding)
            np.savetxt(Path(NN_dir).joinpath('UMAP_embedding.csv'), umap_embedding, delimiter=',')
            meta_data_df['Cell_Type'] = pd.Categorical(leiden_result)
            input_data['meta_data'] = meta_data_df
            save_meta_data(save_dir=NN_dir, meta_data_df=meta_data_df)

            # save meta data
            meta_data_df.to_csv(Path(NN_dir).joinpath('meta_data.csv'), index=False)

        # generate cell type coding matrix
        ct_coding_matrix = np.zeros(shape=(meta_data_df.shape[0],
                                           meta_data_df['Cell_Type'].cat.categories.shape[0]))  # N x #cell_type
        ct_coding_matrix[np.arange(meta_data_df.shape[0]), meta_data_df.Cell_Type.cat.codes.values] = 1
        ct_coding_df = pd.DataFrame(data=ct_coding_matrix,
                                    columns=meta_data_df['Cell_Type'].cat.categories,
                                    index=meta_data_df[id_name])
        # save cell type code
        save_cell_type_code(save_dir=NN_dir, cell_types=meta_data_df['Cell_Type'])

    else:  # id_name == 'Spot_ID', low resolution data
        if 'deconvoluted_ct_composition' in input_data:  # option 4: low resolution data with deconvoluted cell type composition
            ct_coding_df = input_data['deconvoluted_ct_composition']  # N x #cell_type
        elif 'low_res_exp' in input_data:  # option 5: low resolution data with original expression data
            if dc_method is None or dc_ct_num is None:
                raise ValueError('dc_method and dc_ct_num are required when you provide low_res_exp_data as input.')

            ct_coding_df = perform_deconvolution(NN_dir=NN_dir,
                                                 dc_method=dc_method,
                                                 exp_df=input_data['low_res_exp'],
                                                 dc_ct_num=dc_ct_num,
                                                 gen_ct_embedding=gen_ct_embedding)
        else:
            raise ValueError(
                'deconvoluted_ct_composition or low_res_exp_data should be provided for low resolution data.')

        # save cell type code
        save_cell_type_code(save_dir=NN_dir, cell_types=pd.Series(ct_coding_df.columns))

    input_data['ct_coding'] = ct_coding_df
    ct_coding_df.to_csv(Path(NN_dir).joinpath('ct_coding.csv'), index=True, index_label=id_name)

    return input_data


def preprocessing_nn(meta_input: Union[str, Path],
                     NN_dir: Union[str, Path],
                     exp_input: Optional[Union[str, Path]] = None,
                     embedding_input: Optional[Union[str, Path]] = None,
                     low_res_exp_input: Optional[Union[str, Path]] = None,
                     gen_ct_embedding: bool = False,
                     deconvoluted_ct_composition: Optional[Union[str, Path]] = None,
                     deconvoluted_exp_input: Optional[Union[str, Path]] = None,
                     resolution: Optional[float] = None,
                     dc_method: Optional[str] = None,
                     dc_ct_num: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocessing for niche network.

    Possible input parameters combinations:
    # class I: without embedding ajustment
    1. meta_input, NN_dir (cell-level data with Cell_Type info in meta_data)
    2. meta_input, NN_dir, exp_input, resolution (cell-level data with gene expression data)
    3. meta_input, NN_dir, embedding_input (cell-level data with embedding info)
    4. meta_input, NN_dir, low_res_exp_input, dc_method, dc_ct_num (spot-level data with original expression data)
    5. meta_input, NN_dir, low_res_exp_input, deconvoluted_ct_composition (spot-level data with deconvoluted cell type composition)
    # class II: with embedding ajustment
    1. embedding_adjust, sigma, meta_input, NN_dir, exp_input, resolution  (cell-level data with gene expression data)
    2. embedding_adjust, sigma, meta_input, NN_dir, embedding_input (cell-level data with embedding info)
    3. meta_input, NN_dir, low_res_exp_input, dc_method, dc_ct_num (spot-level data with original expression data)
    4. meta_input, NN_dir, low_res_exp_input, deconvoluted_ct_composition, deconvoluted_exp_input (spot-level data with deconvoluted cell type composition)

    Steps:
    1. define cell type
    2. cell type coding matrix
    3. embedding adjustment (optional)


    :param meta_input: str or Path, meta data file.
    :param NN_dir: str or Path, save directory.
    :param exp_input: str or Path, expression data file.
    :param embedding_input: str or Path, embedding data file.
    :param low_res_exp_input: str or Path, low resolution expression data file.
    :param gen_ct_embedding: bool, generate cell type embedding.
    :param deconvoluted_ct_composition: str or Path, deconvoluted cell type composition file.
    :param deconvoluted_exp_input: str or Path, deconvoluted expression data file.
    :param resolution: float, resolution.
    :param dc_method: str, deconvolution method.
    :param dc_ct_num: int, deconvoluted cell type number.
    :param embedding_adjust: bool, adjust embedding.
    :param sigma: float, sigma.
    :return: Tuple[pd.DataFrame, pd.DataFrame], meta data and cell type coding matrix.
    """

    input_data = load_input_data(meta_input=meta_input,
                                 NN_dir=NN_dir,
                                 exp_input=exp_input,
                                 embedding_input=embedding_input,
                                 low_res_exp_input=low_res_exp_input,
                                 deconvoluted_ct_composition=deconvoluted_ct_composition,
                                 deconvoluted_exp_input=deconvoluted_exp_input)

    input_data = cal_cell_type_coding(input_data=input_data,
                                      NN_dir=NN_dir,
                                      resolution=resolution,
                                      dc_method=dc_method,
                                      dc_ct_num=dc_ct_num,
                                      gen_ct_embedding=gen_ct_embedding)

    return input_data['meta_data'], input_data['embedding_data'], input_data['ct_coding']


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
    if 'Cell_Type' in meta_data_df.columns:
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
    if 'Cell_Type' in meta_data_df.columns:  # cell-level data
        meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype(str).astype('category')

    # load niche-level niche cluster assign
    if not os.path.exists(niche_level_niche_cluster_file) or not os.path.exists(consolidate_out_adj_file):
        error(f'niche_level_niche_cluster.csv.gz or consolidate_out_adj.csv.gz does not exist in {GNN_dir} directory.')

    niche_level_niche_cluster_assign_df = pd.read_csv(niche_level_niche_cluster_file, header=0, index_col=0)

    # load consolidate out_adj
    consolidate_out_adj_array = np.loadtxt(consolidate_out_adj_file, delimiter=',')

    return meta_data_df, niche_level_niche_cluster_assign_df, consolidate_out_adj_array
