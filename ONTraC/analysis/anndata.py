import os
import re
from optparse import Values
from typing import Dict, Optional, Tuple

import anndata as ad
import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
from anndata import AnnData
from matplotlib import cm
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from torch_geometric.data import Data

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from ONTraC.log import *
from ONTraC.train.loss_funs import moran_I_features
from .utils import percentage_summary_along_continous_feat, move_legend


def create_adata(options: Values) -> Tuple[Dict[str, AnnData], AnnData]:
    """Load dataset from options.
    """

    # create AnnData with fake expression data
    adata_dict = {}
    for sample in options.data['Data']:
        name = sample.get('Name')
        coordinate_file = f"{sample['Coordinates']}"
        info(f'Read samples: {name} with coordinate file: {coordinate_file}.')
        # TODO: support 3D spatial data
        coordinate_arr = pd.read_csv(coordinate_file)[['x', 'y']].values
        adata_dict[name] = AnnData(X=csr_matrix(np.random.poisson(1, (coordinate_arr.shape[0], 100)), dtype=np.float32))

        adata_dict[name].obs.index = [f'{name}_{i+1:05d}' for i in range(adata_dict[name].obs.shape[0])
                                      ]  # temp solution for duplicate cell names
        # load spatial data
        adata_dict[name].obsm['spatial'] = coordinate_arr

    adata_combined = ad.concat(list(adata_dict.values()))

    return adata_dict, adata_combined


def load_embedding_data(options: Values, data: Data, adata_dict: Dict[str, AnnData], adata_combined: AnnData):
    """
    loadindg embedding data
    """
    for index, name in enumerate(adata_dict.keys()):
        embedding_file = f"{options.output}/{name}_z.csv"
        if not os.path.exists(embedding_file):
            embedding_file = f"{options.output}/{name}_z.csv.gz"
            if not os.path.exists(embedding_file):
                raise FileNotFoundError(f"Cannot find embedding file: {embedding_file}.")
        embedding_arr = np.loadtxt(embedding_file, delimiter=',')[data.mask[index].detach().cpu().numpy()]
        adata_dict[name].obsm['trained_embedding'] = embedding_arr

    adata_combined.obsm['trained_embedding'] = np.concatenate(
        [adata_dict[name].obsm['trained_embedding'] for name in adata_dict.keys()])  # type: ignore


def load_graph_pooling_results(options: Values, data: Data, adata_dict: Dict[str, AnnData], adata_combined: AnnData):
    """
    loading graph pooling results
    """
    for index, name in enumerate(adata_dict.keys()):
        soft_assign_file = f'{options.output}/{name}_s.csv'
        if not os.path.exists(soft_assign_file):
            soft_assign_file = f'{options.output}/{name}_s.csv.gz'
            if not os.path.exists(soft_assign_file):
                raise FileNotFoundError(f"Cannot find soft assignment file: {soft_assign_file}.")
        soft_assign_arr = np.loadtxt(soft_assign_file,
                                     delimiter=',').argmax(axis=1)[data.mask[index].detach().cpu().numpy()]
        adata_dict[name].obs['graph_pooling'] = soft_assign_arr.astype(str)

    adata_combined.obs['graph_pooling'] = np.concatenate(
        [adata_dict[name].obs['graph_pooling'] for name in adata_dict.keys()])  # type: ignore


def load_cell_NTScore(options: Values, data: Data, adata_dict: Dict[str, AnnData], adata_combined: AnnData):
    """
    loading cell-level NTScore
    """
    pseudo_time_file = f'{options.output}/cell_NTScore.csv'
    if not os.path.exists(pseudo_time_file):
        pseudo_time_file = f'{options.output}/cell_NTScore.csv.gz'
        if not os.path.exists(pseudo_time_file):
            raise FileNotFoundError(f"Cannot find cell-level NTScore file: {pseudo_time_file}.")
    pseudo_time_arr = np.loadtxt(pseudo_time_file, delimiter=',')[data.mask.flatten().detach().cpu().numpy()]
    adata_combined.obs['NTScore'] = pseudo_time_arr

    # copy to each sample adata
    for name in adata_dict.keys():
        adata_dict[name].obs['NTScore'] = adata_combined.obs['NTScore'].loc[adata_dict[name].obs.index]


def load_annotation_data(options: Values, data: Data, adata_dict: Dict[str, AnnData], adata_combined: AnnData):
    # --- load embedding data ---
    try:
        load_embedding_data(options, data, adata_dict, adata_combined)
    except FileNotFoundError as e:
        warning(str(e))

    # --- load graph pooling results ---
    try:
        load_graph_pooling_results(options, data, adata_dict, adata_combined)
    except FileNotFoundError as e:
        warning(str(e))

    # --- load cell-level NTScore ---
    try:
        load_cell_NTScore(options, data, adata_dict, adata_combined)
    except FileNotFoundError as e:
        warning(str(e))


def plot_NTScore(options: Values, data: Data, adata_dict: Dict[str, AnnData]) -> None:
    for index, name in enumerate(data.name):
        NTScore_df = pd.DataFrame({
            'NTScore': adata_dict[name].obs['NTScore'],
            'x': adata_dict[name].obsm['spatial'][:, 0],  # type: ignore
            'y': adata_dict[name].obsm['spatial'][:, 1],  # type: ignore
        })
        moran_I_value = moran_I_features(
            torch.cat([
                torch.FloatTensor(adata_dict[name].obs['NTScore'].values),
                torch.zeros(data.x[index].shape[0] - adata_dict[name].shape[0])
            ]), data.adj[index], data.mask[index]).detach().cpu().numpy().reshape(-1)[0]
        fig, ax = plt.subplots()
        cax = ax.scatter(data=NTScore_df, x='x', y='y', s=6, linewidth=0, c='NTScore', cmap='rainbow')
        ax.set_title(name)
        ax.annotate(f'Moran\'s I: {moran_I_value: .3f}',
                    xy=(0.02, 0.02),
                    xycoords='axes fraction',
                    ha='left',
                    va='bottom',
                    fontsize=12)
        fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
        fig.tight_layout()
        fig.savefig(f'{options.output}/{name}_NTScore.pdf')
        plt.close(fig)


def plot_feat_along_pseudo_time(results_dict, title, options) -> None:
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
        for discrete_column in results_dict.keys():
            fig, ax = plt.subplots()
            sns.lineplot(data=results_dict[discrete_column], x='percentile', y='value', hue=discrete_column, ax=ax)
            ax.set_title(title)
            xticks = np.array([0, 24, 49, 74, 99])
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'{x+1:d}' for x in xticks])
            ax.set_xlabel('Percentile of PseudoTime')
            ax.set_ylabel('Percentage')
            move_legend(ax, new_loc='upper left', bbox_to_anchor=(1, 1))
            fig.tight_layout()
            fig.savefig(f'{options.output}/{title}_combined_{discrete_column}_along_pseudo_time.pdf')
            plt.close(fig)


def feat_along_pseudo_time(options: Values, adata_dict: Dict[str, AnnData], adata_combined: AnnData) -> None:
    # all samples
    results_dict = percentage_summary_along_continous_feat(df=adata_combined.obs,
                                                           continous_column='pseudo_time',
                                                           discrete_columns=['annot', 'annot_niche_max'])
    plot_feat_along_pseudo_time(results_dict, 'all_samples', options)

    for name in adata_dict:
        results_dict = percentage_summary_along_continous_feat(df=adata_dict[name].obs,
                                                               continous_column='pseudo_time',
                                                               discrete_columns=['annot', 'annot_niche_max'])
        plot_feat_along_pseudo_time(results_dict, name, options)


def anndata_based_analysis(
    options: Values,
    data: Data,
) -> Optional[pd.DataFrame]:
    # ----- prepare data -----
    adata_dict, adata_combined = create_adata(options)
    load_annotation_data(options, data, adata_dict, adata_combined)

    # ----- analysis -----
    if 'trained_embedding' in adata_combined.obsm.keys():
        pass
        # 1. umap plots
        # umap_plots(options, 'trained_embedding', adata_dict, adata_combined)
        # 2. spatial plots
        # spatial_plots(options, 'trained_embedding', adata_dict)

    # 3. plot pseudo time
    plot_NTScore(options, data, adata_dict)

    # 4. feat_along_pseudo_time
    feat_along_pseudo_time(options, adata_dict, adata_combined)

    return adata_combined.obs
