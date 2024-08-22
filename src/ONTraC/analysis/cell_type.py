from copy import deepcopy
from typing import Optional, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from ..log import warning
from .data import AnaData


def plot_violin_cell_type_along_NT_score(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot violinplot cell type composition along NT score.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes]
    """
    try:
        if 'Cell_NTScore' not in ana_data.NT_score.columns:
            warning("No NT score data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    data_df = ana_data.cell_id.join(ana_data.NT_score['Cell_NTScore'])
    if ana_data.options.reverse: data_df['Cell_NTScore'] = 1 - data_df['Cell_NTScore']

    if data_df['Cell_Type'].unique().shape[0] > 100:
        warning("There are more than 100 cell types, skip violin plot to avoid long runtime. You could manually plot it according to our tutorial.")
        return None

    fig, ax = plt.subplots(figsize=(6, ana_data.cell_type_codes.shape[0] / 2))
    sns.violinplot(data=data_df,
                   x='Cell_NTScore',
                   y='Cell_Type',
                   hue='Cell_Type',
                   cut=0,
                   fill=False,
                   common_norm=True,
                   legend=False,
                   ax=ax)
    ax.set_xlabel('Cell-level NT score')
    ax.set_ylabel('Cell Type')
    fig.tight_layout()
    if ana_data.options.output is not None:
        fig.savefig(f'{ana_data.options.output}/cell_type_along_NT_score_violin.pdf', transparent=True)
        return None
    else:
        return fig, ax


def plot_kde_cell_type_along_NT_score(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot kdeplot cell type composition along NT score.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes]
    """
    try:
        if 'Cell_NTScore' not in ana_data.NT_score.columns:
            warning("No NT score data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    data_df = ana_data.cell_id.join(ana_data.NT_score['Cell_NTScore'])
    if ana_data.options.reverse: data_df['Cell_NTScore'] = 1 - data_df['Cell_NTScore']

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=data_df, x='Cell_NTScore', hue='Cell_Type', multiple="fill", ax=ax)
    ax.set_ylabel('Fraction of cells')
    fig.tight_layout()
    if ana_data.options.output is not None:
        fig.savefig(f'{ana_data.options.output}/cell_type_along_NT_score_kde.pdf', transparent=True)
        return None
    else:
        return fig, ax


def plot_hist_cell_type_along_NT_score(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot histogram of cell type composition along NT score.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes]
    """
    try:
        if 'Cell_NTScore' not in ana_data.NT_score.columns:
            warning("No NT score data found.")
            return None
        if ana_data.cell_type_codes is None:
            warning("No cell type data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    data_df = ana_data.cell_id.join(ana_data.NT_score['Cell_NTScore'])
    if ana_data.options.reverse: data_df['Cell_NTScore'] = 1 - data_df['Cell_NTScore']

    fig, ax = plt.subplots(figsize=(len(data_df['Cell_Type'].unique()), 4))
    sns.histplot(data=data_df,
                 x='Cell_NTScore',
                 hue='Cell_Type',
                 hue_order=ana_data.cell_type_codes['Cell_Type'],
                 multiple="dodge",
                 ax=ax)
    ax.set_ylabel('Fraction of cells')
    fig.tight_layout()
    if ana_data.options.output is not None:
        fig.savefig(f'{ana_data.options.output}/cell_type_along_NT_score_hist.pdf', transparent=True)
        return None
    else:
        return fig, ax


def plot_cell_type_along_NT_score(ana_data: AnaData) -> None:
    """
    Plot all visualization of cell type along NT score.
    :param ana_data: AnaData, the data for analysis.
    """
    plot_violin_cell_type_along_NT_score(ana_data=ana_data)
    plot_kde_cell_type_along_NT_score(ana_data=ana_data)
    plot_hist_cell_type_along_NT_score(ana_data=ana_data)


def plot_cell_type_loading_in_niche_clusters(ana_data: AnaData,
                                             cell_type_dis_df: pd.DataFrame) -> Optional[sns.FacetGrid]:  #,
    #  nc_order: List[str]) -> Optional[sns.FacetGrid]:
    """
    Plot cell type loading in each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :param cell_type_dis_df: pd.DataFrame, the cell type distribution in each niche cluster.
    :param nc_order: List[str], the order of niche cluster.
    :return: None or Tuple[plt.Figure, plt.Axes]
    """

    data_df = deepcopy(cell_type_dis_df)
    cell_type = data_df.columns
    data_df['cluster'] = data_df.index
    cell_type_dis_melt_df = pd.melt(
        data_df,
        id_vars='cluster',  # type: ignore
        var_name='Cell type',
        value_vars=cell_type,  # type: ignore
        value_name='Number')
    # g = sns.catplot(cell_type_dis_melt_df, kind="bar", x="Number", y="Cell type", col="cluster", col_order= nc_order, height=4,
    g = sns.catplot(cell_type_dis_melt_df, kind="bar", x="Number", y="Cell type", col="cluster", height=2 + len(cell_type) / 6,
                    aspect=.5)  # type: ignore
    g.add_legend()
    g.tight_layout()
    g.set_xticklabels(rotation='vertical')
    if ana_data.options.output is not None:
        g.savefig(f'{ana_data.options.output}/cell_type_loading_in_niche_clusters.pdf', transparent=True)
        return None
    else:
        return g


def plot_cell_type_dis_in_niche_clusters(ana_data: AnaData,
                                         cell_type_dis_df: pd.DataFrame) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type distribution in each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :param cell_type_dis_df: pd.DataFrame, the cell type distribution in each niche cluster.
    :return: None or Tuple[plt.Figure, plt.Axes]
    """
    fig, ax = plt.subplots(figsize=(2 + cell_type_dis_df.shape[1] / 3, 4))
    sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=1), ax=ax)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Niche Cluster')
    fig.tight_layout()
    if ana_data.options.output is not None:
        fig.savefig(f'{ana_data.options.output}/cell_type_dis_in_niche_clusters.pdf', transparent=True)
        return None
    else:
        return fig, ax


def plot_cell_type_across_niche_cluster(ana_data: AnaData,
                                        cell_type_dis_df: pd.DataFrame) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type distribution across niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :param cell_type_dis_df: pd.DataFrame, the cell type distribution in each niche cluster.
    :return: None or Tuple[plt.Figure, plt.Axes]
    """
    fig, ax = plt.subplots(figsize=(2 + cell_type_dis_df.shape[1] / 3, 4))
    sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=0), ax=ax)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Niche Cluster')
    fig.tight_layout()
    if ana_data.options.output is not None:
        fig.savefig(f'{ana_data.options.output}/cell_type_dis_across_niche_cluster.pdf', transparent=True)
        return None
    else:
        return fig, ax


def plot_cell_type_with_niche_cluster(ana_data: AnaData) -> None:
    """
    Plot all visualization of cell type in each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    """

    try:
        if ana_data.cell_level_niche_cluster_assign is None:
            warning("No niche cluster assign data found.")
            return None
        if ana_data.cell_type_codes is None:
            warning("No cell type data found.")
            return None
        if ana_data.niche_cluster_score is None:
            warning("No niche cluster scores data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    # calculate cell type distribution in each niche cluster
    data_df = ana_data.cell_id.join(ana_data.cell_level_niche_cluster_assign)
    t = pd.CategoricalDtype(categories=ana_data.cell_type_codes['Cell_Type'], ordered=True)
    cell_type_one_hot = np.zeros(shape=(data_df.shape[0], ana_data.cell_type_codes.shape[0]))
    cell_type = data_df['Cell_Type'].astype(t)
    cell_type_one_hot[np.arange(data_df.shape[0]), cell_type.cat.codes] = 1  # N x n_cell_type
    cell_type_dis = np.matmul(data_df[ana_data.cell_level_niche_cluster_assign.columns].T,
                              cell_type_one_hot)  # n_clusters x n_cell_types
    cell_type_dis_df = pd.DataFrame(cell_type_dis)
    cell_type_dis_df.columns = ana_data.cell_type_codes['Cell_Type']
    if ana_data.options.output is not None:
        cell_type_dis_df.to_csv(f'{ana_data.options.output}/cell_type_dis_in_niche_clusters.csv', index=False)
    # nc_order
    nc_scores = 1 - ana_data.niche_cluster_score if ana_data.options.reverse else ana_data.niche_cluster_score
    nc_order = [f'NicheCluster_{x}' for x in nc_scores.argsort()]
    cell_type_dis_df = cell_type_dis_df.loc[nc_order]

    # plot_cell_type_loading_in_niche_clusters(ana_data=ana_data, cell_type_dis_df=cell_type_dis_df, nc_order=nc_order)
    plot_cell_type_loading_in_niche_clusters(ana_data=ana_data, cell_type_dis_df=cell_type_dis_df)
    plot_cell_type_dis_in_niche_clusters(ana_data=ana_data, cell_type_dis_df=cell_type_dis_df)
    plot_cell_type_across_niche_cluster(ana_data=ana_data, cell_type_dis_df=cell_type_dis_df)


def cell_type_visualization(ana_data: AnaData) -> None:
    """
    Visualize cell type based output.
    :param ana_data: AnaData, the data for analysis.
    """

    # 1. cell type along NT score
    plot_cell_type_along_NT_score(ana_data=ana_data)

    # 2. cell type X niche cluster
    plot_cell_type_with_niche_cluster(ana_data=ana_data)
