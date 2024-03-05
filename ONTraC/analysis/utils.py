import os
import sys
from optparse import OptionParser, Values
from typing import Dict, Generator, List, Optional, Tuple

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DenseDataLoader

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from utils import (cluster_order_by_pseudotime, gini, move_legend,
                   percentage_summary_along_continous_feat, to_one_hot)

from ONTraC.data import SpatailOmicsDataset, create_torch_dataset
from ONTraC.log import *
from ONTraC.train.loss_funs import moran_I_features
from ONTraC.utils import get_rel_params, read_yaml_file


def plot_max_pro_cluster(options: Values, data: Data) -> None:
    if not os.path.exists(f'{options.output}/consolidate_s.csv.gz'):
        warning(f'File not found: {options.output}/consolidate_s.csv.gz')
        return None

    soft_assign_df = {}

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

        for index, name in enumerate(data.name):
            soft_assign_file = f'{options.output}/{name}_s.csv'
            soft_assign_file = soft_assign_file if os.path.isfile(soft_assign_file) else f'{soft_assign_file}.gz'
            if not os.path.isfile(soft_assign_file):  # skip if soft_assign_file not exist
                continue
            soft_assign_df[name] = {
                'x':
                data.pos[index][data.mask[index], 0].detach().cpu().numpy(),
                'y':
                data.pos[index][data.mask[index], 1].detach().cpu().numpy(),
                'Max Probability Cluster':
                np.loadtxt(soft_assign_file,
                           delimiter=',').argmax(axis=1)[data.mask[index].detach().cpu().numpy()].astype('str')
            }

            fig, ax = plt.subplots()
            sns.scatterplot(data=soft_assign_df[name],
                            x='x',
                            y='y',
                            hue='Max Probability Cluster',
                            hue_order=np.unique(soft_assign_df[data.name[0]]['Max Probability Cluster']),
                            ax=ax,
                            s=6)
            ax.set_title(name)
            fig.tight_layout()
            fig.savefig(f'{options.output}/{name}_pooling_results.pdf')
            plt.close(fig)


def cluster_order_by_pseudotime(options: Values, soft_assign: np.ndarray):
    pseudotime_cluster_file = f'{options.output}/pseudo_time_cluster.csv.gz'
    if not os.path.exists(pseudotime_cluster_file):
        pseudotime_cluster_file = f'{options.output}/pseudo_time_cluster.csv'
        if not os.path.exists(pseudotime_cluster_file):
            warning(f'File not found: {pseudotime_cluster_file}')
            return np.arange(soft_assign.shape[1])
    pseudotime_cluster = np.loadtxt(fname=pseudotime_cluster_file, delimiter=',')
    # get order according to pseudotime cluster
    order = np.argsort(pseudotime_cluster)
    return order


def plot_each_cluster_proportion(options: Values, data: Data):
    if not os.path.exists(f'{options.output}/consolidate_s.csv.gz'):
        warning(f'File not found: {options.output}/consolidate_s.csv.gz')
        return {}

    soft_assign_df = {}

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

        for index, name in enumerate(data.name):
            soft_assign_file = f'{options.output}/{name}_s.csv'
            soft_assign_file = soft_assign_file if os.path.isfile(path=soft_assign_file) else f'{soft_assign_file}.gz'
            if not os.path.isfile(soft_assign_file):  # skip if soft_assign_file not exist
                continue
            soft_assign = np.loadtxt(fname=soft_assign_file, delimiter=',')[data.mask[index].detach().cpu().numpy()]
            soft_assign_df[name] = {
                'x': data.pos[index][data.mask[index], 0].detach().cpu().numpy(),
                'y': data.pos[index][data.mask[index], 1].detach().cpu().numpy()
            }
            soft_assign_df[name].update({f'{i}': soft_assign[:, i] for i in range(soft_assign.shape[1])})

            # fig & ax setting
            k = soft_assign.shape[1]
            column_n = k // 2
            fig = plt.figure(figsize=(column_n * 4, 8))
            gs = gridspec.GridSpec(1, 10, figure=fig)
            if k == 1:
                gs_left = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs[0, :9])
                ax1 = fig.add_subplot(gs_left[0, 0])
                axes: List = [ax1]
            else:
                ncols: int = (k + 1) // 2
                gs_left = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=ncols, subplot_spec=gs[0, :9])
                axes = [[], []]
                for i in range(k):
                    row_index, col_index = i // ncols, i % ncols
                    axes[row_index].append(fig.add_subplot(gs_left[row_index, col_index]))
            ax_right = fig.add_subplot(gs[0, 9])

            cluster_order = cluster_order_by_pseudotime(options=options, soft_assign=soft_assign)

            images = []
            vmin = 0
            vmax = 1
            for index, i in enumerate(cluster_order):
                row_index, col_index = index // column_n, index % column_n
                ax = axes[row_index][col_index] if k > 2 else axes[index]
                # sns.scatterplot(data=soft_assign_df[name], x='x', y='y', hue=f'{i}', ax=ax, s=6)
                images.append(
                    ax.scatter(data=soft_assign_df[name], x='x', y='y', s=6, c=f'{i}', cmap='Oranges', linewidth=0))
                ax.set_title(f'{name} cluster {i}')  # type: ignore
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            fig.colorbar(mappable=images[0], cax=ax_right, orientation='vertical', fraction=.1)
            fig.tight_layout()
            fig.savefig(fname=f'{options.output}/{name}_cluster_prob.pdf')
            plt.close(fig=fig)

    return soft_assign_df


def cluster_connectivity(options: Values) -> None:
    adj_file = f'{options.output}/consolidate_out_adj.csv.gz'
    if not os.path.isfile(adj_file):
        adj_file = f'{options.output}/consolidate_out_adj.csv'
    if not os.path.isfile(adj_file):  # skip if adj_file not exist
        warning(f'File not found: {adj_file}')
        return

    adj_matrix = np.loadtxt(f'{options.output}/consolidate_out_adj.csv.gz', delimiter=',')
    G = nx.Graph(adj_matrix)

    # Drawing the graph
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

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
        fig, ax = plt.subplots()

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='skyblue',
            node_size=1500,
            edge_color=weights,
            width=3.0,
            edge_cmap=plt.cm.Reds,  # type: ignore
            connectionstyle='arc3,rad=0.1',
            ax=ax)
        ax.set_title('Network Graph of Connectivity with Weighted Edges')
        fig.tight_layout()
        fig.savefig(f'{options.output}/cluster_connectivity.pdf')
        plt.close(fig)


def cluster_spatial_continuity_gen(s: Tensor, data: Data) -> Generator:
    for i in range(s.shape[0]):
        moran = moran_I_features(X=s[i], W=data.adj[i], mask=data.mask[i]).detach().cpu().numpy().flatten()
        for j, m in enumerate(moran):
            yield m, f'{i}', f'{j}'


def cluster_spatial_continuity(options: Values, data: Data):
    s_arr = np.loadtxt(f'{options.output}/consolidate_s.csv.gz', delimiter=',')
    s_tensor = torch.Tensor(s_arr.reshape((data.x.shape[0], data.x.shape[1], -1)))
    cluster_moran = pd.DataFrame(cluster_spatial_continuity_gen(s_tensor, data),
                                 columns=['moran_I', 'sample', 'cluster'])
    cluster_moran.to_csv(f'{options.output}/cluster_moran.csv.gz', index=False)
