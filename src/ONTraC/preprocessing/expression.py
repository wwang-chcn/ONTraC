from typing import List

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