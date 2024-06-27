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
from ..utils.niche_net_constr import get_embedding_columns, load_original_data
from .data import AnaData


def embedding_adjust_visualization(ana_data: AnaData) -> Optional[Union[Tuple, None]]:
    """Visualization of embedding adjust.

    Args:
        ana_data: AnaData object.
    """

    if not ana_data.options.embedding_adjust:
        return None

    # load original data
    ori_data_df = load_original_data(ana_data.options)
    embedding_columns = get_embedding_columns(ori_data_df)
    if len(embedding_columns) < 2:
        warning('At least two (Embedding_1 and Embedding_2) should be in the original data. Skip the adjustment.')
        return None

    # calculate embedding postion for each cell type
    ct_embedding = ori_data_df[embedding_columns + ['Cell_Type']].groupby('Cell_Type').mean()

    # calculate distance between each cell type
    raw_distance = distance.cdist(ct_embedding[embedding_columns].values, ct_embedding[embedding_columns].values,
                                  'euclidean')
    raw_distance_df = pd.DataFrame(raw_distance, index=ct_embedding.index, columns=ct_embedding.index)

    if ana_data.options.sigma is None:
        ana_data.options.sigma = np.median(raw_distance[np.triu_indices(raw_distance.shape[0], k=1)])
        info(
            f'Sigma is not provided. Use the median ({ana_data.options.sigma}) of the distances between the cell type pairs.'
        )

    # calculate the M
    M = np.exp(-raw_distance**2 / ana_data.options.sigma**2)
    M_df = pd.DataFrame(M, index=ct_embedding.index, columns=ct_embedding.index)

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
        M_cluster_grid = sns.clustermap(M_df, figsize=(M_df.shape[0] / 6, M_df.shape[0] / 6))
        if ana_data.options.output:
            dis_cluster_grid.savefig(f'{ana_data.options.output}/raw_distance.pdf', transparent=True)
            M_cluster_grid.savefig(f'{ana_data.options.output}/M.pdf', transparent=True)
            return None
        else:
            return dis_cluster_grid, M_cluster_grid
