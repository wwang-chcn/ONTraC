from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
from .utils import saptial_figsize


def plot_spatial_cell_type_distribution_dataset(data_df: pd.DataFrame,
                                                output_file_path: Optional[Union[str, Path]] = None,
                                                **kwargs) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """
    Plot spatial cell type distribution.
    :param data_df: pd.DataFrame, the data for visualization.
    :param output_file_path: Optional[Union[str, Path]], the output directory.
    :param kwargs: Optional, the additional arguments for visualization using seaborn.scatterplot.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    samples = data_df['Sample'].unique()
    N = len(samples)
    fig, axes = plt.subplots(1, N, figsize=(4 * N, 3))
    for i, sample in enumerate(samples):
        sample_df = data_df.loc[data_df['Sample'] == sample]
        ax: plt.Axes = axes[i] if N > 1 else axes  # type: ignore
        sns.scatterplot(data=sample_df, x='x', y='y', hue='Cell_Type', s=8, ax=ax, **kwargs)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/spatial_cell_type_distribution.pdf', transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_spatial_cell_type_distribution_dataset_from_anadata(ana_data: AnaData,
                                                             **kwargs) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """
    Plot spatial cell type distribution.
    :param ana_data: AnaData, the data for analysis.
    :param kwargs: Optional, the additional arguments for visualization using seaborn.scatterplot.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    if 'Cell_Type' not in ana_data.meta_data_df.columns:
        warning("No cell type data found. Skip spatial cell type distribution visualization.")
        return None

    return plot_spatial_cell_type_distribution_dataset(data_df=ana_data.meta_data_df,
                                                       output_file_path=ana_data.options.output,
                                                       **kwargs)


def plot_spatial_cell_type_distribution_sample(data_df: pd.DataFrame,
                                               output_file_path: Optional[Union[str, Path]] = None,
                                               **kwargs) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot spatial cell type distribution.
    :param data_df: pd.DataFrame, the data for visualization.
    :param output_file_path: Optional[Union[str, Path]], the output directory.
    :param kwargs: Optional, the additional arguments for visualization using seaborn.scatterplot.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    samples = data_df['Sample'].unique()
    output = []
    for sample in samples:
        sample_df = data_df.loc[data_df['Sample'] == sample]
        fig, ax = plt.subplots(figsize=saptial_figsize(sample_df))
        sns.scatterplot(data=sample_df, x='x', y='y', hue='Cell_Type', s=8, ax=ax, **kwargs)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()
        output.append((fig, ax))
        if output_file_path is not None:
            fig.savefig(f'{output_file_path}/spatial_cell_type_distribution_{sample}.pdf', transparent=True)
            plt.close(fig)
    return output


def plot_spatial_cell_type_distribution_sample_from_anadata(ana_data: AnaData,
                                                            **kwargs) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot spatial cell type distribution.
    :param ana_data: AnaData, the data for analysis.
    :param kwargs: Optional, the additional arguments for visualization using seaborn.scatterplot.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    if 'Cell_Type' not in ana_data.meta_data_df.columns:
        warning("No cell type data found. Skip spatial cell type distribution visualization.")
        return None

    return plot_spatial_cell_type_distribution_sample(data_df=ana_data.meta_data_df,
                                                      output_file_path=ana_data.options.output,
                                                      **kwargs)


def plot_violin_cell_type_along_NT_score(data_df: pd.DataFrame,
                                         cell_types: List[str],
                                         output_file_path: Optional[Union[str, Path]] = None,
                                         **kwargs) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot violinplot cell type composition along NT score.
    :param data_df: pd.DataFrame, the data for visualization.
    :param cell_types: List[str], the cell types.
    :param output_file_path: Optional[Union[str, Path]], the output directory.
    :param kwargs: Optional, the additional arguments for visualization using seaborn.violinplot.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    if (n_cell_type := len(cell_types)) > 100:
        warning(
            "There are more than 100 cell types, skip violin plot to avoid long runtime. You could manually plot it according to our tutorial."
        )
        return None

    fig, ax = plt.subplots(figsize=(6, n_cell_type / 2))
    sns.violinplot(data=data_df,
                   x='Cell_NTScore',
                   y='Cell_Type',
                   hue='Cell_Type',
                   cut=0,
                   fill=False,
                   common_norm=True,
                   legend=False,
                   ax=ax,
                   **kwargs)
    ax.set_xlabel('Cell-level NT score')
    ax.set_ylabel('Cell Type')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/cell_type_along_NT_score_violin.pdf', transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_violin_cell_type_along_NT_score_from_anadata(ana_data: AnaData,
                                                      **kwargs) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot violinplot cell type composition along NT score.
    :param ana_data: AnaData, the data for analysis.
    :param kwargs: Optional, the additional arguments for visualization using seaborn.violinplot.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    try:
        if 'Cell_NTScore' not in ana_data.NT_score.columns:
            warning("No NT score data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    data_df = ana_data.meta_data_df.join(ana_data.NT_score['Cell_NTScore'])
    if ana_data.options.reverse: data_df['Cell_NTScore'] = 1 - data_df['Cell_NTScore']

    cell_types = ana_data.cell_type_codes['Cell_Type'].to_list()

    return plot_violin_cell_type_along_NT_score(data_df=data_df,
                                                cell_types=cell_types,
                                                output_file_path=ana_data.options.output,
                                                **kwargs)


def plot_kde_cell_type_along_NT_score(
        data_df: pd.DataFrame,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot kdeplot cell type composition along NT score.
    :param data_df: pd.DataFrame, the data for visualization.
    :param output_file_path: Optional[Union[str, Path]], the output directory.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=data_df, x='Cell_NTScore', hue='Cell_Type', multiple="fill", ax=ax)
    ax.set_ylabel('Fraction of cells')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/cell_type_along_NT_score_kde.pdf', transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_kde_cell_type_along_NT_score_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot kdeplot cell type composition along NT score.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    try:
        if 'Cell_NTScore' not in ana_data.NT_score.columns:
            warning("No NT score data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    data_df = ana_data.meta_data_df.join(ana_data.NT_score['Cell_NTScore'])
    if ana_data.options.reverse: data_df['Cell_NTScore'] = 1 - data_df['Cell_NTScore']

    return plot_kde_cell_type_along_NT_score(data_df=data_df, output_file_path=ana_data.options.output)


def plot_hist_cell_type_along_NT_score(
        data_df: pd.DataFrame,
        cell_types: List[str],
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot histogram of cell type composition along NT score.
    :param data_df: pd.DataFrame, the data for visualization.
    :param cell_types: List[str], the cell types, used for hue order of histogram.
    :param output_file_path: Optional[Union[str, Path]], the output directory.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    fig, ax = plt.subplots(figsize=(len(cell_types), 4))
    sns.histplot(data=data_df, x='Cell_NTScore', hue='Cell_Type', hue_order=cell_types, multiple="dodge", ax=ax)
    ax.set_ylabel('Fraction of cells')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/cell_type_along_NT_score_hist.pdf', transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_hist_cell_type_along_NT_score_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot histogram of cell type composition along NT score.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    try:
        if 'Cell_NTScore' not in ana_data.NT_score.columns:
            warning("No NT score data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    data_df = ana_data.meta_data_df.join(ana_data.NT_score['Cell_NTScore'])
    if ana_data.options.reverse: data_df['Cell_NTScore'] = 1 - data_df['Cell_NTScore']

    cell_types = ana_data.cell_type_codes['Cell_Type'].to_list()

    return plot_hist_cell_type_along_NT_score(data_df=data_df,
                                              cell_types=cell_types,
                                              output_file_path=ana_data.options.output)


def plot_cell_type_along_NT_score(ana_data: AnaData) -> None:
    """
    Plot all visualization of cell type along NT score.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
    """

    try:
        if 'Cell_NTScore' not in ana_data.NT_score.columns:
            warning("No NT score data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    data_df = ana_data.meta_data_df.join(ana_data.NT_score['Cell_NTScore'])
    if ana_data.options.reverse: data_df['Cell_NTScore'] = 1 - data_df['Cell_NTScore']

    cell_types = ana_data.cell_type_codes['Cell_Type'].to_list()

    plot_violin_cell_type_along_NT_score(data_df=data_df,
                                         cell_types=cell_types,
                                         output_file_path=ana_data.options.output)
    plot_kde_cell_type_along_NT_score(data_df=data_df, output_file_path=ana_data.options.output)
    plot_hist_cell_type_along_NT_score(data_df=data_df, cell_types=cell_types, output_file_path=ana_data.options.output)


def plot_cell_type_loading_in_niche_clusters(cell_type_dis_df: pd.DataFrame,
                                             output_file_path: Optional[Union[str,
                                                                              Path]] = None) -> Optional[sns.FacetGrid]:
    """
    Plot cell type loading in each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :param cell_type_dis_df: pd.DataFrame, the cell type distribution in each niche cluster.
    :return: None or Tuple[plt.Figure, plt.Axes].
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
    g = sns.catplot(cell_type_dis_melt_df,
                    kind="bar",
                    x="Number",
                    y="Cell type",
                    col="cluster",
                    height=2 + len(cell_type) / 6,
                    aspect=.5)  # type: ignore
    g.add_legend()
    g.tight_layout()
    g.set_xticklabels(rotation='vertical')
    if output_file_path is not None:
        g.savefig(f'{output_file_path}/cell_type_loading_in_niche_clusters.pdf', transparent=True)
        return None
    else:
        return g


def plot_cell_type_loading_in_niche_clusters_from_anadata(ana_data: AnaData) -> Optional[sns.FacetGrid]:
    """
    Plot cell type loading in each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    try:
        if ana_data.cell_level_niche_cluster_assign is None:
            warning("No niche cluster assign data found.")
            return None
        if ana_data.cell_type_codes is None:
            warning("No cell type data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    # calculate cell type distribution in each niche cluster
    data_df = ana_data.meta_data_df.join(ana_data.cell_level_niche_cluster_assign)
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

    return plot_cell_type_loading_in_niche_clusters(cell_type_dis_df=cell_type_dis_df,
                                                    output_file_path=ana_data.options.output)


def plot_cell_type_dis_in_niche_clusters(
        cell_type_dis_df: pd.DataFrame,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type distribution in each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :param cell_type_dis_df: pd.DataFrame, the cell type distribution in each niche cluster.
    :return: None or Tuple[plt.Figure, plt.Axes]
    """
    fig, ax = plt.subplots(figsize=(2 + cell_type_dis_df.shape[1] / 3, 1 + cell_type_dis_df.shape[0] / 5))
    sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=1), ax=ax)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Niche Cluster')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/cell_type_dis_in_niche_clusters.pdf', transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_cell_type_dis_in_niche_clusters_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type distribution in each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    try:
        if ana_data.cell_level_niche_cluster_assign is None:
            warning("No niche cluster assign data found.")
            return None
        if ana_data.cell_type_codes is None:
            warning("No cell type data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    # calculate cell type distribution in each niche cluster
    data_df = ana_data.meta_data_df.join(ana_data.cell_level_niche_cluster_assign)
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

    return plot_cell_type_dis_in_niche_clusters(cell_type_dis_df=cell_type_dis_df,
                                                output_file_path=ana_data.options.output)


def plot_cell_type_across_niche_cluster(
        cell_type_dis_df: pd.DataFrame,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type distribution across niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :param cell_type_dis_df: pd.DataFrame, the cell type distribution in each niche cluster.
    :return: None or Tuple[plt.Figure, plt.Axes]
    """
    fig, ax = plt.subplots(figsize=(2 + cell_type_dis_df.shape[1] / 3, 1 + cell_type_dis_df.shape[0] / 5))
    sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=0), ax=ax)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Niche Cluster')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/cell_type_dis_across_niche_cluster.pdf', transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_cell_type_across_niche_cluster_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type distribution across niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    try:
        if ana_data.cell_level_niche_cluster_assign is None:
            warning("No niche cluster assign data found.")
            return None
        if ana_data.cell_type_codes is None:
            warning("No cell type data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    # calculate cell type distribution in each niche cluster
    data_df = ana_data.meta_data_df.join(ana_data.cell_level_niche_cluster_assign)
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

    return plot_cell_type_across_niche_cluster(cell_type_dis_df=cell_type_dis_df,
                                               output_file_path=ana_data.options.output)


def plot_cell_type_with_niche_cluster(ana_data: AnaData) -> None:
    """
    Plot all visualization of cell type in each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
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
    data_df = ana_data.meta_data_df.join(ana_data.cell_level_niche_cluster_assign)
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
    plot_cell_type_loading_in_niche_clusters(cell_type_dis_df=cell_type_dis_df,
                                             output_file_path=ana_data.options.output)
    plot_cell_type_dis_in_niche_clusters(cell_type_dis_df=cell_type_dis_df, output_file_path=ana_data.options.output)
    plot_cell_type_across_niche_cluster(cell_type_dis_df=cell_type_dis_df, output_file_path=ana_data.options.output)


def cell_type_visualization(ana_data: AnaData) -> None:
    """
    Visualize cell type based output.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
    """

    # 1. cell type along NT score
    if not hasattr(ana_data.options, 'suppress_niche_trajectory') or not ana_data.options.suppress_niche_trajectory:
        plot_cell_type_along_NT_score(ana_data=ana_data)

    # 2. cell type X niche cluster
    plot_cell_type_with_niche_cluster(ana_data=ana_data)
