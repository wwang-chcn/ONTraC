from typing import Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.spatial import distance

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from ..log import info, warning
from ..niche_net import get_embedding_columns
from .data import AnaData


def clustering_visualization(ana_data: AnaData) -> Optional[Union[Tuple, None]]:
    """Visualization of clustering results.
    
    Args:
        ana_data: AnaData object.
    """

    if not ana_data.options.decomposition_expression_input:
        return None

    # load meta data
    meta_data = pd.read_csv(f'{ana_data.options.preprocessing_dir}/meta_data.csv', index_col=0)
    umap_embedding = np.loadtxt(f'{ana_data.options.preprocessing_dir}/PCA_embedding.csv', delimiter=',')
    data_df = meta_data['Cell_Type']
    data_df['Embedding_1'] = umap_embedding[:, 0]
    data_df['Embedding_2'] = umap_embedding[:, 1]
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
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        sns.scatterplot(data=data_df, x='Embedding_1', y='Embedding_2', hue='Cell_Type', s=2, ax=ax)
        ax.set_xlabel('UMAP_1')
        ax.set_ylabel('UMAP_2')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=3, markerscale=4)
        if ana_data.options.output:
            fig.savefig(f'{ana_data.options.output}/clustering.pdf', transparent=True)
            return None
        else:
            return fig


def embedding_adjust_visualization(ana_data: AnaData) -> Optional[Union[Tuple, None]]:
    """Visualization of embedding adjust.

    Args:
        ana_data: AnaData object.
    """

    if not ana_data.options.embedding_adjust:
        return None

    # load original data
    ori_data_df = pd.read_csv(f'{ana_data.options.preprocessing_dir}/meta_data.csv', index_col=0)
    ct_coding = pd.read_csv(f'{ana_data.options.preprocessing_dir}/cell_type_code.csv', index_col=0)
    if ana_data.options.decomposition_expression_input:
        ct_embedding = np.loadtxt(f'{ana_data.options.preprocessing_dir}/PCA_embedding.csv', delimiter=',')
        raw_distance = distance.cdist(ct_embedding, ct_embedding, 'euclidean')
    else:
        embedding_columns = get_embedding_columns(ori_data_df)
        if len(embedding_columns) < 2:
            warning('At least two (Embedding_1 and Embedding_2) should be in the original data. Skip the adjustment.')
            return None

        # calculate embedding postion for each cell type
        ct_embedding = ori_data_df[embedding_columns + ['Cell_Type']].groupby('Cell_Type').mean()
        raw_distance = distance.cdist(ct_embedding[embedding_columns].values, ct_embedding[embedding_columns].values,
                                      'euclidean')

    # calculate distance between each cell type
    median_distance = np.median(raw_distance[np.triu_indices(raw_distance.shape[0], k=1)])
    info(f'Median distance between cell types: {median_distance}')
    raw_distance_df = pd.DataFrame(raw_distance, index=ct_coding['Cell_Type'], columns=ct_coding['Cell_Type'])

    if ana_data.options.sigma is None:
        ana_data.options.sigma = np.median(raw_distance[np.triu_indices(raw_distance.shape[0], k=1)])
        info(
            f'Sigma is not provided. Use the median ({ana_data.options.sigma}) of the distances between the cell type pairs.'
        )

    # calculate the M
    M = np.exp(-(raw_distance / (ana_data.options.sigma * median_distance))**2)
    M_df = pd.DataFrame(M, index=ct_coding['Cell_Type'], columns=ct_coding['Cell_Type'])

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
        dis_cluster_grid = sns.clustermap(raw_distance_df,
                                          figsize=(raw_distance_df.shape[0] / 6, raw_distance_df.shape[0] / 6))
        M_cluster_grid = sns.clustermap(M_df, figsize=(M_df.shape[0] / 6, M_df.shape[0] / 6), vmin=0, vmax=1)
        if ana_data.options.output:
            dis_cluster_grid.savefig(f'{ana_data.options.output}/raw_distance.pdf', transparent=True)
            M_cluster_grid.savefig(f'{ana_data.options.output}/M.pdf', transparent=True)
            return None
        else:
            return dis_cluster_grid, M_cluster_grid
