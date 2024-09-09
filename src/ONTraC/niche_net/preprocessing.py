import sys
from optparse import Values
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..log import *
from ..utils import load_meta_data, save_cell_type_code


def perform_pca(expression_data: pd.DataFrame, n_components: int = 50) -> np.ndarray:
    """
    Perform PCA on the expression data.
    :param expression_data: Expression data.
    :param n_components: Number of components.
    :return: PCA result.
    """

    from sklearn.decomposition import PCA

    maximum_components = min(expression_data.shape[0], expression_data.shape[1])
    if n_components > maximum_components:
        n_components = maximum_components - 1

    model = PCA(n_components=n_components, svd_solver='arpack', random_state=0)
    info(f'Performing PCA with {n_components} components...')

    return model.fit_transform(X=expression_data.values)


def perform_harmony(embedding, meta_df: pd.DataFrame, batch_key: str) -> np.ndarray:
    """
    Perform Harmony on the embedding.
    :param embedding: Embedding.
    :param meta_df: Meta data.
    :param batch_key: Batch key.
    :return: Harmony result.
    """

    from harmonypy import run_harmony

    info(f'Performing Harmony using "{batch_key}" as batch ...')

    return run_harmony(data_mat=embedding, meta_data=meta_df, vars_use=batch_key).Z_corr.T


def define_neighbors(embedding: np.ndarray, n_neighbors: int = 20) -> csr_matrix:
    """
    Define neighbors.
    :param embedding: Embedding.
    :param n_neighbors: Number of neighbors.
    :return: Neighbors.
    """

    #TODO: n_neighbors should be less than the number of cells in each sample.

    import pynndescent
    from scipy.sparse import coo_matrix
    from umap.umap_ import fuzzy_simplicial_set

    info(f'Defining neighbors with {n_neighbors} neighbors...')

    index = pynndescent.NNDescent(embedding, n_neighbors=n_neighbors, metric='euclidean', n_jobs=4)
    knn_indices, knn_dists = index.neighbor_graph
    connectivities, _, _, _ = fuzzy_simplicial_set(X=coo_matrix(([], ([], [])), shape=(embedding.shape[0], 1)),
                                                   n_neighbors=n_neighbors,
                                                   random_state=0,
                                                   metric="euclidean",
                                                   knn_indices=knn_indices,
                                                   knn_dists=knn_dists,
                                                   return_dists=False)

    return connectivities


def perform_umap(embedding: np.ndarray,
                 n_neighbors: int = 20,
                 min_dist: float = 0.5,
                 n_components: int = 2) -> np.ndarray:
    """
    Perform UMAP on the embedding.
    :param embedding: Embedding.
    :param n_neighbors: Number of neighbors.
    :param min_dist: Minimum distance.
    :param n_components: Number of components.
    :return: UMAP result.
    """

    import umap

    info(f'Performing UMAP with {n_neighbors} neighbors, {min_dist} minimum distance, and {n_components} components...')

    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components).fit_transform(embedding)


def perform_leiden(connectivities: csr_matrix, resolution: float = 1.0) -> List[int]:
    """
    Perform Leiden algorithm.
    :param connectivities: Connectivities.
    :param resolution: Resolution.
    :return: Leiden result.
    """

    import igraph as ig
    import leidenalg

    sources, targets = connectivities.nonzero()
    g = ig.Graph(directed=None)
    g.add_vertices(connectivities.shape[0])
    g.add_edges(es=list(zip(sources, targets)), attributes={'weight': connectivities[sources, targets].tolist()[0]})

    info(f'Performing Leiden algorithm with resolution {resolution}...')

    partition = leidenalg.find_partition(graph=g,
                                         partition_type=leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=resolution)

    return partition.membership


def gen_original_data(options: Values) -> pd.DataFrame:
    """
    Load original data.
    :param options: Options.
    :return: meta data.

    option 1) there are gene expression data (csv format) and meta data (csv format)
    option 2) there are embedding data (csv format) and meta data (csv format)
    option 3) there is meta data (csv format) only
    option 4) there are decomposition input data (csv format) and meta data (csv format)
    """

    meta_data_df = load_meta_data(options=options)

    if options.exp_input is None and options.embedding_input is None:
        if options.decomposition_cell_type_composition_input is None and options.decomposition_expression_input is None:  # option 3
            # check if Cell_Type column in the meta data
            if 'Cell_Type' not in meta_data_df.columns:
                raise ValueError(
                    'There are no expression data or embedding data. Please provide Cell_Type in the meta data.')
            meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype('category')
            # there should more than 1 cell type
            if len(meta_data_df['Cell_Type'].cat.categories) < 2:
                raise ValueError(
                    'There are no expression data or embedding data. Please provide at least two cell types in the meta data.'
                )
            # save mappings of the categorical data
            save_cell_type_code(options=options, meta_data_df=meta_data_df)

            meta_data_df.to_csv(options.preprocessing_dir + '/meta_data.csv', index=False)
            return meta_data_df
        else:  # option 4
            if 'Cell_Type' in meta_data_df.columns:
                meta_data_df = meta_data_df.drop(columns=['Cell_Type'])
            decomposition_cell_type_composition_df = pd.read_csv(options.decomposition_cell_type_composition_input,
                                                                 header=0,
                                                                 index_col=0)
            decomposition_expression_df = pd.read_csv(options.decomposition_expression_input, header=0, index_col=0)
            if decomposition_cell_type_composition_df.shape[1] != decomposition_expression_df.shape[0]:
                error(
                    f'The number of cell type in the decomposition input data ({options.decomposition_cell_type_composition_input}) and ({options.decomposition_expression_input}) is not consistent. Please check it again.'
                )
                sys.exit(1)
            if decomposition_cell_type_composition_df.shape[0] != meta_data_df.shape[0]:
                warning(
                    f'The number of spots in the decomposition input data ({options.decomposition_cell_type_composition_input}) is not consistent with the meta data. We will use the intersection of them.'
                )
                common_spots = set(decomposition_cell_type_composition_df.index).intersection(
                    set(meta_data_df['Cell_ID']))
                decomposition_cell_type_composition_df = decomposition_cell_type_composition_df.loc[common_spots]
                meta_data_df = meta_data_df.loc[meta_data_df['Cell_ID'].isin(common_spots)]
            pca_embedding = perform_pca(decomposition_expression_df)
            if meta_data_df['Sample'].nunique() > 1:
                pca_embedding = perform_harmony(pca_embedding, meta_data_df, 'Sample')
            np.savetxt(options.preprocessing_dir + '/PCA_embedding.csv', pca_embedding, delimiter=',')
            for i in range(decomposition_cell_type_composition_df.shape[1]):
                meta_data_df[f'Cell_Type_{i}'] = decomposition_cell_type_composition_df.iloc[:, i]
            meta_data_df.to_csv(options.preprocessing_dir + '/meta_data.csv', index=False)
            cell_type_code = pd.DataFrame({
                'Code':
                range(decomposition_cell_type_composition_df.shape[1]),
                'Cell_Type': [f'Cell_Type_{i}' for i in range(decomposition_cell_type_composition_df.shape[1])]
            })
            cell_type_code.to_csv(f'{options.preprocessing_dir}/cell_type_code.csv', index=False)
            return meta_data_df

    if options.exp_input is None and options.embedding_input is not None:  # option 2
        # there should more than 1 cell type
        if len(meta_data_df['Cell_Type'].cat.categories) < 2:
            raise ValueError(
                'There are no expression data or embedding data. Please provide at least two cell types in the meta data.'
            )
        # save mappings of the categorical data
        save_cell_type_code(options=options, meta_data_df=meta_data_df)

        embedding_df = pd.read_csv(options.embedding_input, header=0, index_col=0, sep=',')
        if embedding_df.shape[0] != meta_data_df.shape[0]:
            warning(
                f'The number of cells in the embedding data ({options.embedding_input}) is not consistent with the meta data. We will use the intersection of them.'
            )
            common_cells = set(embedding_df.index).intersection(set(meta_data_df['Cell_ID']))
            embedding_df = embedding_df.loc[common_cells]
            meta_data_df = meta_data_df.loc[meta_data_df['Cell_ID'].isin(common_spots)]
        for i in range(embedding_df.shape[1]):
            meta_data_df[f'Embedding_{i}'] = embedding_df.iloc[:, i].values
        meta_data_df.to_csv(options.preprocessing_dir + '/meta_data.csv', index=False)
        return meta_data_df

    if options.exp_input is not None:  # option 1
        expression_data_df = pd.read_csv(options.exp_input, header=0, index_col=0, sep=',')
        if expression_data_df.shape[0] != meta_data_df.shape[0]:
            warning(
                f'The number of cells in the expression data ({options.exp_input}) is not consistent with the meta data. We will use the intersection of them.'
            )
            common_cells = set(expression_data_df.index).intersection(set(meta_data_df['Cell_ID']))
            expression_data_df = expression_data_df.loc[common_cells]
            meta_data_df = meta_data_df.loc[meta_data_df['Cell_ID'].isin(common_spots)]
        pca_embedding = perform_pca(expression_data_df)
        if 'Batch' in meta_data_df.columns:
            if meta_data_df['Batch'].nunique() > 1:
                pca_embedding = perform_harmony(pca_embedding, meta_data_df, 'Batch')
        elif meta_data_df['Sample'].nunique() > 1:
            pca_embedding = perform_harmony(pca_embedding, meta_data_df, 'Sample')
        np.savetxt(options.preprocessing_dir + '/PCA_embedding.csv', pca_embedding, delimiter=',')
        connectivities = define_neighbors(pca_embedding)
        leiden_result = perform_leiden(connectivities, resolution=options.resolution)
        umap_embedding = perform_umap(pca_embedding)
        np.savetxt(options.preprocessing_dir + '/UMAP_embedding.csv', umap_embedding, delimiter=',')
        meta_data_df['Cell_Type'] = pd.Categorical(leiden_result)
        for i in range(pca_embedding.shape[1]):
            meta_data_df[f'Embedding_{i}'] = pca_embedding[:, i]
        # save mappings of the categorical data
        save_cell_type_code(options=options, meta_data_df=meta_data_df)
        meta_data_df.to_csv(options.preprocessing_dir + '/meta_data.csv', index=False)
        return meta_data_df
