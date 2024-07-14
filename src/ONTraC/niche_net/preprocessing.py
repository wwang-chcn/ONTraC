from typing import List
from optparse import Values

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


from ..log import *


def perform_pca(expression_data: pd.DataFrame, n_components: int = 50) -> np.ndarray:
    """
    Perform PCA on the expression data.
    :param expression_data: Expression data.
    :param n_components: Number of components.
    :return: PCA result.
    """

    from sklearn.decomposition import PCA

    model = PCA(n_components=n_components, random_state=0)

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

    return run_harmony(data_mat=embedding, meta_data=meta_df, vars_use=batch_key)


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


def perform_leiden(connectivities: csr_matrix, resolution: float = 10.0) -> List[int]:
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

    partition = leidenalg.find_partition(graph=g,
                                         partition_type=leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=resolution)

    return partition.membership


def gen_original_data(options: Values) -> pd.DataFrame:
    """
    Load original data.

    option 1) there is gene expression data (csv format) and meta data (csv format)
    option 2) there is embedding data (csv format) and meta data (csv format)
    option 3) there is meta data (csv format) only
    """

    meta_data_df = pd.read_csv(options.meta_input, header=0, index_col=False, sep=',')

    if options.exp_input is None and options.embedding_input is None:
        # check if Cell_Type column in the meta data
        if 'Cell_Type' not in meta_data_df.columns:
            raise ValueError('There are no expression data or embedding data. Please provide Cell_Type in the meta data.')
        meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype('category')
        # there should more than 1 cell type
        if len(meta_data_df['Cell_Type'].cat.categories) < 2:
            raise ValueError('There are no expression data or embedding data. Please provide at least two cell types in the meta data.')
        # save mappings of the categorical data
        cell_type_code = pd.DataFrame(enumerate(meta_data_df['Cell_Type'].cat.categories), columns=['Code', 'Cell_Type'])
        cell_type_code.to_csv(f'{options.preprocessing_dir}/cell_type_code.csv', index=False)
        meta_data_df.to_csv(options.preprocessing_dir + '/meta_data.csv', index=False)
        return meta_data_df
    
    if options.exp_input is None and options.embedding_input is not None:
        meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype('category')
        # there should more than 1 cell type
        if len(meta_data_df['Cell_Type'].cat.categories) < 2:
            raise ValueError('There are no expression data or embedding data. Please provide at least two cell types in the meta data.')
        # save mappings of the categorical data
        cell_type_code = pd.DataFrame(enumerate(meta_data_df['Cell_Type'].cat.categories), columns=['Code', 'Cell_Type'])
        cell_type_code.to_csv(f'{options.preprocessing_dir}/cell_type_code.csv', index=False)

        embedding_df = pd.read_csv(options.embedding_input, header=0, index_col=0, sep=',')
        for i in range(embedding_df.shape[1]):
            meta_data_df[f'Embedding_{i}'] = embedding_df.iloc[:, i]
        meta_data_df.to_csv(options.preprocessing_dir + '/meta_data.csv', index=False)
        return meta_data_df
    
    if options.exp_input is not None:
        expression_data_df = pd.read_csv(options.exp_input, header=0, index_col=0, sep=',')
        pca_embedding = perform_pca(expression_data_df)
        if meta_data_df['Sample'].nunique() > 1:
            pca_embedding = perform_harmony(pca_embedding, meta_data_df, 'Sample')
        np.savetxt(options.preprocessing_dir + '/PCA_embedding.csv', pca_embedding, delimiter=',')
        connectivities = define_neighbors(pca_embedding)
        leiden_result = perform_leiden(connectivities)
        meta_data_df['Cell_Type'] = pd.Categorical(leiden_result)
        for i in range(pca_embedding.shape[1]):
            meta_data_df[f'Embedding_{i}'] = pca_embedding[:, i]
        # save mappings of the categorical data
        cell_type_code = pd.DataFrame(enumerate(meta_data_df['Cell_Type'].cat.categories), columns=['Code', 'Cell_Type'])
        cell_type_code.to_csv(f'{options.preprocessing_dir}/cell_type_code.csv', index=False)
        meta_data_df.to_csv(options.preprocessing_dir + '/meta_data.csv', index=False)
        return meta_data_df


    