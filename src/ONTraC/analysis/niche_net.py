from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.spatial import distance
from seaborn.matrix import ClusterGrid

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from ..log import info, warning
from .data import AnaData


def clustering_visualization(
        data_df: pd.DataFrame,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Visualization of clustering results.
    :param data_df: pd.DataFrame, data for visualization.
    :param output_file_path: str or Path, output file path.
    :return: None.
    """

    with sns.axes_style('white', rc={
            'xtick.bottom': True,
            'ytick.left': True
    }), sns.plotting_context('paper',
                             rc={
                                 'axes.titlesize': 8,
                                 'axes.labelsize': 8,
                                 'xtick.labelsize': 6,
                                 'ytick.labelsize': 6,
                                 'legend.fontsize': 6
                             }):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        sns.scatterplot(data=data_df, x='Embedding_1', y='Embedding_2', hue='Cell_Type', s=2, ax=ax)
        ax.set_xlabel('UMAP_1')
        ax.set_ylabel('UMAP_2')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=3, markerscale=4)
        fig.tight_layout()
        if output_file_path:
            fig.savefig(f'{output_file_path}/clustering.pdf', transparent=True)
            return None
        else:
            return fig


def clustering_visualization_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Visualization of clustering results.
    
    Args:
        ana_data: AnaData object.
    """

    # check if the embedding is available
    if ana_data.umap_embedding is None:
        warning('UMAP embedding is not available. Skip the clustering visualization.')
        return None

    data_df = pd.DataFrame(ana_data.umap_embedding, columns=['Embedding_1', 'Embedding_2'])
    data_df.index = ana_data.meta_data_df.index
    data_df['Cell_Type'] = ana_data.meta_data_df['Cell_Type']
    data_df['Cell_Type'] = data_df['Cell_Type'].astype('category')

    return clustering_visualization(data_df=data_df, output_file_path=ana_data.options.output)


def embedding_adjust_visualization(dis_df: pd.DataFrame,
                                   output_file_name: str,
                                   output_file_path: Optional[Union[str, Path]] = None) -> Optional[ClusterGrid]:
    with sns.axes_style('white', rc={
            'xtick.bottom': True,
            'ytick.left': True
    }), sns.plotting_context('paper',
                             rc={
                                 'axes.titlesize': 8,
                                 'axes.labelsize': 8,
                                 'xtick.labelsize': 6,
                                 'ytick.labelsize': 6,
                                 'legend.fontsize': 6
                             }):
        cluster_grid: ClusterGrid = sns.clustermap(dis_df, figsize=(dis_df.shape[0] / 6, dis_df.shape[0] / 6))
        if output_file_path is not None:
            cluster_grid.savefig(f'{output_file_path}/{output_file_name}', transparent=True)
            return None
        else:
            return cluster_grid


def embedding_adjust_visualization_from_anadata(ana_data: AnaData) -> Optional[List[ClusterGrid]]:
    """Visualization of embedding adjust.

    Args:
        ana_data: AnaData object.
    """

    if not ana_data.options.embedding_adjust:
        return None

    if ana_data.ct_embedding is None:
        warning('Cell type embedding is not available. Skip the embedding adjustment visualization.')
        return None

    cell_types = ana_data.cell_type_codes['Cell_Type'].tolist()
    raw_distance = distance.cdist(ana_data.ct_embedding.values, ana_data.ct_embedding.values, 'euclidean')

    # calculate distance between each cell type
    median_distance = np.median(raw_distance[np.triu_indices(raw_distance.shape[0], k=1)])
    info(f'Median distance between cell types: {median_distance}')
    raw_distance_df = pd.DataFrame(raw_distance, index=cell_types, columns=cell_types)

    if ana_data.options.sigma is None:
        ana_data.options.sigma = np.median(raw_distance[np.triu_indices(raw_distance.shape[0], k=1)])
        info(
            f'Sigma is not provided. Use the median ({ana_data.options.sigma}) of the distances between the cell type pairs.'
        )

    # calculate the M
    M = np.exp(-(raw_distance / (ana_data.options.sigma * median_distance))**2)
    M_df = pd.DataFrame(M, index=cell_types, columns=cell_types)

    output = []
    output.append(
        embedding_adjust_visualization(dis_df=raw_distance_df,
                                       output_file_name='raw_distance.pdf',
                                       output_file_path=ana_data.options.output))
    output.append(
        embedding_adjust_visualization(dis_df=M_df,
                                       output_file_name='adjusted_distance.pdf',
                                       output_file_path=ana_data.options.output))
    return output
