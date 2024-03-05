from optparse import Values

import matplotlib as mpl
import numpy as np
import pandas as pd
from torch_geometric.data import Data

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import gini, to_one_hot


def cell_type_dis_in_cluster(options: Values, data: Data, meta_df: pd.DataFrame) -> None:
    """
    Plot cell type distribution in each cluster.
    """

    soft_assign_file = f'{options.output}/consolidate_s.csv.gz'
    mask = data.mask.flatten().detach().cpu().numpy()
    soft_assign = np.loadtxt(soft_assign_file, delimiter=',')[mask]  # N x n_clusters
    meta_df['annot'] = meta_df['annot'].astype('category')
    cell_type = meta_df['annot']
    cell_type_cat = cell_type.cat.categories
    cell_type_one_hot = to_one_hot(cell_type.cat.codes, num_classes=len(cell_type_cat))  # N x n_cell_types

    cell_type_dis = np.matmul(soft_assign.T, cell_type_one_hot)  # n_clusters x n_cell_types
    cell_type_dis_df = pd.DataFrame(cell_type_dis, columns=cell_type_cat)

    # ----- gini index for intra-cluster cell type distribution -----
    intra_cluster_gini = cell_type_dis_df.apply(gini, axis=1).values
    intra_cluster_gini_df = pd.DataFrame(data={'gini': intra_cluster_gini, 'cluster': cell_type_dis_df.index})
    intra_cluster_gini_df.to_csv(f'{options.output}/intra_cluster_gini_cell_type.csv', index=False)

    # ----- summrized loadings for each cluster -----
    loadings_cluster = cell_type_dis_df.sum(axis=1)

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

        # ----- summrized loadings for each cluster -----
        loadings_cluster = soft_assign.sum(axis=0)
        fig, ax = plt.subplots()
        ax.pie(loadings_cluster,
               labels=[f'Cluster {i}' for i in range(cell_type_dis_df.shape[0])],
               autopct='%1.1f%%',
               pctdistance=1.25,
               labeldistance=.6)
        ax.set_title(f'Nodes number in each cluster')
        fig.tight_layout()
        fig.savefig(f'{options.output}/Clusters_loadings_piechart.pdf')
        plt.close(fig)

        # --- gini index ---
        gini_index = gini(loadings_cluster)
        np.savetxt(f'{options.output}/cluster_gini_index.csv', X=np.array([gini_index]), delimiter=',')

        # ----- heatmap for cluster × cell type -----
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=1), ax=ax, cmap='Blues')
        ax.set_title('Cell Type Distribution in Each Cluster')
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Cluster')
        fig.tight_layout()
        fig.savefig(f'{options.output}/cell_type_dis_in_clusters.pdf')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=0), ax=ax, cmap='Blues')
        ax.set_title('Cell Type Distribution across Clusters')
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Cluster')
        fig.tight_layout()
        fig.savefig(f'{options.output}/cell_type_dis_across_clusters.pdf')
        plt.close(fig)

        # ----- bar plot for cell type in each cluster -----
        cell_type_dis_df['cluster'] = cell_type_dis_df.index
        cell_type_dis_melt_df = pd.melt(
            cell_type_dis_df,
            id_vars='cluster',  # type: ignore
            var_name='annot',
            value_vars=cell_type_cat,  # type: ignore
            value_name='Number')
        g = sns.catplot(cell_type_dis_melt_df, kind="bar", x="Number", y="annot", col="cluster", height=4,
                        aspect=.5)  # type: ignore
        g.add_legend()
        g.figure.figsize = (6.4, 20)
        g.tight_layout()
        g.set_xticklabels(rotation='vertical')
        g.savefig(f'{options.output}/cell_type_number_within_clusters.pdf')


def niche_max_dis_in_cluster(options: Values, data: Data, meta_df: pd.DataFrame) -> None:
    """
    Plot cell type distribution in each cluster.
    """

    soft_assign_file = f'{options.output}/consolidate_s.csv.gz'

    mask = data.mask.flatten().detach().cpu().numpy()
    soft_assign = np.loadtxt(soft_assign_file, delimiter=',')[mask]  # N x n_clusters
    meta_df['annot_niche_max'] = meta_df['annot_niche_max'].astype('category')
    cell_type = meta_df['annot_niche_max']
    cell_type_cat = cell_type.cat.categories
    cell_type_one_hot = to_one_hot(cell_type.cat.codes, num_classes=len(cell_type_cat))  # N x n_cell_types

    cell_type_dis = np.matmul(soft_assign.T, cell_type_one_hot)  # n_clusters x n_cell_types
    cell_type_dis_df = pd.DataFrame(cell_type_dis, columns=cell_type_cat)

    # ----- gini index for intra-cluster niche type distribution -----
    intra_cluster_gini = cell_type_dis_df.apply(gini, axis=1).values
    intra_cluster_gini_df = pd.DataFrame(data={'gini': intra_cluster_gini, 'cluster': cell_type_dis_df.index})
    intra_cluster_gini_df.to_csv(f'{options.output}/intra_cluster_gini_niche_type.csv', index=False)

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
        # ----- heatmap for cluster × niche type -----
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=1), ax=ax, cmap='Blues')
        ax.set_title('Cell Type Distribution in Each Cluster')
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Cluster')
        fig.tight_layout()
        fig.savefig(f'{options.output}/niche_max_dis_in_clusters.pdf')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=0), ax=ax, cmap='Blues')
        ax.set_title('Cell Type Distribution across Clusters')
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Cluster')
        fig.tight_layout()
        fig.savefig(f'{options.output}/niche_max_dis_across_clusters.pdf')
        plt.close(fig)

        # ----- bar plot for niche type in each cluster -----
        cell_type_dis_df['cluster'] = cell_type_dis_df.index
        cell_type_dis_melt_df = pd.melt(
            cell_type_dis_df,
            id_vars='cluster',  # type: ignore
            var_name='annot',
            value_vars=cell_type_cat,  # type: ignore
            value_name='Number')
        g = sns.catplot(cell_type_dis_melt_df, kind="bar", x="annot", y="Number", col="cluster", height=4,
                        aspect=.5)  # type: ignore
        g.add_legend()
        g.figure.figsize = (6.4, 20)
        g.tight_layout()
        g.set_xticklabels(rotation='vertical')
        g.savefig(f'{options.output}/niche_max_number_within_clusters.pdf')


def pseudotime_in_each_cell_type(options: Values, meta_df: pd.DataFrame) -> None:
    """
    Plot pseudo time in each cell type.
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
        fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        sns.violinplot(data=meta_df, x='annot', y='pseudo_time', ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation='vertical')
        sns.violinplot(data=meta_df, x='annot_niche_max', y='pseudo_time', ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation='vertical')
        fig.tight_layout()
        fig.savefig(f'{options.output}/PseudoTime_in_each_cell_type.pdf')
        plt.close(fig)
