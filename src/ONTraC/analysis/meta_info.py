import os
from optparse import Values
from typing import List, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from .constants import NT_SCORE_FEATS


def load_meta_info(options: Values) -> pd.DataFrame:
    """
    load data after ONTraC processing
    :param options, Values. dataset, preprocessing_dir and NTScore_dif needed.
    :return data_df, pd.DataFrame
    """

    raw_data_df = pd.read_csv(options.dataset, index_col=0)
    NTScore_df = pd.read_csv(f'{options.NTScore_dir}/NTScore.csv.gz', index_col=0)
    data_df = raw_data_df.join(NTScore_df[NT_SCORE_FEATS])

    # soft assignment info
    for sample in data_df['Sample'].unique():
        soft_assign_file = f'{options.GNN_dir}/{sample}_s.csv'
        if not os.path.exists(soft_assign_file):
            soft_assign_file = f'{options.GNN_dir}/{sample}_s.csv.gz'
            if not os.path.exists(soft_assign_file):
                raise FileNotFoundError(f"Cannot find soft assignment file: {soft_assign_file}.")
        soft_assign_arr = np.loadtxt(soft_assign_file,
                                     delimiter=',').argmax(axis=1)[:data_df[data_df['Sample'] == sample].shape[0]]
        data_df['graph_pooling'] = soft_assign_arr.astype(str)

    return data_df


def spatial_based_plot(options: Values, meta_info: pd.DataFrame) -> None:
    """
    plot spatial based analysis
    :param meta_info: pd.DataFrame, meta_info.
        'x', 'y', 'graph_pooling' should be included.
    :return: None
    """
    with sns.axes_style('white', rc={
            'xtick.bottom': True,
            'ytick.left': True
    }), sns.plotting_context('paper',
                             rc={
                                 'axes.titlesize': 14,
                                 'axes.labelsize': 12,
                                 'xtick.labelsize': 10,
                                 'ytick.labelsize': 10,
                                 'legend.fontsize': 10
                             }):
        features = ['graph_pooling'] + NT_SCORE_FEATS
        for feat in features:
            for sample in meta_info['Sample'].unique():
                sample_df = meta_info[meta_info['Sample'] == sample]
                fig, ax = plt.subplots()
                scatter = ax.scatter(sample_df['x'], sample_df['y'], c=sample_df[feat])
                ax.set_title(feat)
                ax.legend()
                fig.colorbar(scatter, ax=ax)
                fig.tight_layout()
                fig.savefig(f'{options.output}/{feat}_{sample}.pdf')