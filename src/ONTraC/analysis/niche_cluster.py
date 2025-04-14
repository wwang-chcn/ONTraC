from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import distance

from ..log import info, warning
from .data import AnaData
from .utils import gini, saptial_figsize


# ------------------------------------
# utility functions
# ------------------------------------
def cal_nc_scores(cell_level_niche_cluster_assign: pd.DataFrame,
                  reverse: bool = False,
                  niche_cluster_score: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the score of each niche cluster.
    :param cell_level_niche_cluster_assign: np.ndarray, the niche cluster assign data.
    :param reverse: bool, whether to reverse the color.
    :param niche_cluster_score: Optional[np.ndarray], the score of each niche cluster.
    :return: np.ndarray.
    """

    if niche_cluster_score is not None:
        return niche_cluster_score if not reverse else 1 - niche_cluster_score
    else:
        return np.linspace(0, 1, cell_level_niche_cluster_assign.shape[1])


def cal_nc_colors(nc_scores: np.ndarray) -> List:
    """
    Calculate the color of each niche cluster.
    :param nc_scores: np.ndarray, the score of each niche cluster.
    :return: List.
    """

    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)  # type: ignore
    nc_colors = [sm.to_rgba(nc_scores[n]) for n in range(nc_scores.shape[0])]

    return nc_colors


def cal_nc_palette(nc_colors: List) -> dict:
    """
    Calculate the color palette of each niche cluster.
    :param nc_colors: List, the color of each niche cluster.
    :return: dict.
    """

    nc_palette = {f'niche cluster {i}': nc_colors[i] for i in range(len(nc_colors))}
    return nc_palette


def cal_nc_order_index(nc_scores: np.ndarray) -> np.ndarray:
    """
    Calculate the order index of each niche cluster.
    :param nc_scores: np.ndarray, the score of each niche cluster.
    :return: np.ndarray.
    """

    return nc_scores.argsort()


def cal_nc_order(nc_order_index: np.ndarray) -> List:
    """
    Calculate the order of each niche cluster.
    :param nc_order_index: np.ndarray, the order index of each niche cluster.
    :return: List.
    """

    return [f'niche cluster {i}' for i in nc_order_index]


def cal_nc_feat(nc_assign: pd.DataFrame, ctc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the niche cluster feature (cell type composition) matrix.
    :param nc_assign: pd.DataFrame, the niche level niche cluster assign data. # N (#niche) x C (#niche cluster)
    :param ctc: pd.DataFrame, the niche level cell type composition data, aka X_0, the GNN input. # N (#niche) x F (#cell type)
    :return: pd.DataFrame.
    """

    return nc_assign.T @ ctc


def cal_nc_feat_similarity(nc_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the niche cluster feature similarity.
    :param nc_feat: pd.DataFrame, the niche cluster feature matrix.
    :return: pd.DataFrame.
    """

    return pd.DataFrame(data=1 - distance.cdist(XA=nc_feat, XB=nc_feat, metric='cosine'),
                        index=nc_feat.index,
                        columns=nc_feat.index)


# ------------------------------------
# plot functions
# ------------------------------------
def plot_niche_cluster_connectivity(
        niche_cluster_connectivity: np.ndarray,
        niche_cluster_score: Optional[np.ndarray] = None,
        niche_cluster_size: Optional[np.ndarray] = None,
        reverse: bool = False,
        figsize_scale: Optional[float] = None,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """
    Plot niche cluster connectivity.
    :param niche_cluster_connectivity: np.ndarray, the connectivity matrix.
    :param niche_cluster_score: Optional[np.ndarray], the score of each niche cluster.
    :param niche_cluster_size: Optional[np.ndarray], the size of each niche cluster.
    :param reverse: bool, whether to reverse the color.
    :param figsize_scale: float, the scale factor for the figure size. Default is None.
    If None, the scale factor is calculated based on the number of niche cluster in the data.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    # preprocessing
    ## set diagonal to 0
    np.fill_diagonal(niche_cluster_connectivity, 0)
    ## set the figure size
    if figsize_scale is None:
        figsize_scale = 1.0 if niche_cluster_connectivity.shape[0] < 10 else 1.5 * np.log10(
            niche_cluster_connectivity.shape[0])

    # Create a graph
    G = nx.Graph(niche_cluster_connectivity / niche_cluster_connectivity.max())

    # position
    pos = nx.spring_layout(G, seed=42)
    # edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    # node color
    if niche_cluster_score is not None:
        nc_scores = 1 - niche_cluster_score if reverse else niche_cluster_score
        nc_colors = cal_nc_colors(nc_scores)
    else:
        nc_colors = ["#1f78b4"] * niche_cluster_connectivity.shape[0]
    # node size
    if niche_cluster_size is None:
        node_size = 500
    else:
        # rescale the node with size 0 to 200, and the maximum size is 1000
        node_size = (niche_cluster_size / niche_cluster_size.max() * 800 + 200).tolist()

    # Create a figure
    ## figwidth
    figwidths = [6]
    if niche_cluster_size is not None:
        figwidths.append(1.5)  # type: ignore
    if niche_cluster_score is not None:
        figwidths.append(.5)  # type: ignore
    figwidths.append(.5)  # type: ignore
    fig = plt.figure(figsize=(sum(figwidths) * figsize_scale, 6 * figsize_scale))  # type: ignore

    # Create a gridspec with 1 row and 3 columns (most).
    axes = []
    gs = gridspec.GridSpec(nrows=1, ncols=len(figwidths), width_ratios=figwidths)
    ax_index = 0
    graph_ax = fig.add_subplot(gs[ax_index])
    axes.append(graph_ax)
    if niche_cluster_size is not None:
        ax_index += 1
        node_size_ax = fig.add_subplot(gs[ax_index])
        axes.append(node_size_ax)
    if niche_cluster_score is not None:
        ax_index += 1
        node_colorbar_ax = fig.add_subplot(gs[ax_index])
        axes.append(node_colorbar_ax)
    ax_index += 1
    edge_colorbar_ax = fig.add_subplot(gs[ax_index])
    axes.append(edge_colorbar_ax)

    # Draw the graph
    nx.draw_networkx_nodes(
        G=G,
        pos=pos,
        node_color=nc_colors,  # type: ignore
        node_size=node_size,
        ax=graph_ax)
    nx.draw_networkx_edges(
        G=G,
        pos=pos,
        edge_color=weights,  # type: ignore
        alpha=weights,
        width=3.0,
        edge_cmap=plt.cm.Reds,  # type: ignore
        ax=graph_ax,
        node_size=node_size)
    nx.draw_networkx_labels(G, pos, ax=graph_ax)
    graph_ax.axis('off')

    # Draw legend for node size
    if niche_cluster_size is not None:
        ## prepare
        node_size_ax.axis('off')  # hide the
        node_size_legend_num = 5  # TODO: make it become a parameter
        sizes = np.linspace(start=200, stop=1000, num=node_size_legend_num)
        max_niche_cluster_size = niche_cluster_size.max()  # type: ignore
        magnitude = 10**int(np.floor(np.log10(abs(max_niche_cluster_size))))
        max_label = round(max_niche_cluster_size / magnitude) * magnitude
        labels = [f'{int(x):d}' for x in np.linspace(0, max_label, node_size_legend_num)]
        ## draw legend
        handles = [
            plt.Line2D(  # type: ignore
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor='gray',
                markersize=np.sqrt(size),
                label=label) for size, label in zip(sizes, labels)
        ]
        legend = node_size_ax.legend(handles=handles,
                                     title='Niche cluster size',
                                     loc='center',
                                     frameon=False,
                                     bbox_to_anchor=(0.6, 0.6))
        ## Rotate the title and adjust the position
        title = legend.get_title()
        title.set_rotation(90)
        title.set_verticalalignment('center')  # Align title vertically
        title.set_horizontalalignment('right')  # Align title horizontally
        title.set_position((-70, -60))  # Shift the title slightly to the right

    # Draw the colorbar for nodes
    if niche_cluster_score is not None:
        gradient = np.linspace(0, 1, 1000).reshape(-1, 1)
        node_colorbar_ax.imshow(gradient, aspect='auto', cmap=plt.cm.rainbow)  # type: ignore
        node_colorbar_ax.set_xticks([])
        node_colorbar_ax.set_yticks(np.linspace(0, 1000, 5))
        node_colorbar_ax.set_yticklabels(f'{x:.2f}' for x in np.linspace(0, 1, 5))
        node_colorbar_ax.set_ylabel('NT score (nodes)')

    # Draw the colorbar for edges
    colors = [(1, 1, 1, 0), (1, 0, 0, 1)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    gradient = np.linspace(1, 0, 1000).reshape(-1, 1)
    edge_colorbar_ax.imshow(gradient, aspect='auto', cmap=custom_cmap)
    edge_colorbar_ax.set_xticks([])
    edge_colorbar_ax.set_yticks(np.linspace(1000, 0, 5))
    edge_colorbar_ax.set_yticklabels(f'{x:.2f}' for x in np.linspace(0, niche_cluster_connectivity.max(), 5))
    edge_colorbar_ax.set_ylabel('Connectivity (edges)')

    fig.suptitle('Niche cluster connectivity')
    fig.tight_layout()

    if output_file_path is not None:
        fig.savefig(output_file_path, transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_niche_cluster_connectivity_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """
    Plot niche cluster connectivity.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    if ana_data.niche_cluster_connectivity is None:
        warning("No niche cluster connectivity data found. Skip niche cluster connectivity plot.")
        return None

    if ana_data.cell_level_niche_cluster_assign is None:
        niche_cluster_size = np.ones(ana_data.niche_cluster_connectivity.shape[0])
    else:
        niche_cluster_size = ana_data.cell_level_niche_cluster_assign.sum(axis=0)

    output_file_path = f'{ana_data.options.output}/niche_cluster_connectivity.pdf' if ana_data.options.output is not None else None

    return plot_niche_cluster_connectivity(
        niche_cluster_connectivity=ana_data.niche_cluster_connectivity,
        niche_cluster_score=ana_data.niche_cluster_score,
        niche_cluster_size=niche_cluster_size,  # type: ignore
        reverse=ana_data.options.reverse,
        output_file_path=output_file_path)


def plot_niche_cluster_connectivity_bysample_from_anadata(ana_data: AnaData) -> None:
    """
    Plot niche cluster connectivity by sample.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
    """

    if ana_data.niche_cluster_connectivity is None:
        warning("No niche cluster connectivity data found. Skip niche cluster connectivity plot.")
        return None

    for sample in ana_data.meta_data_df['Sample'].unique():

        niche_cluster_connectivity = np.loadtxt(f'{ana_data.options.GNN_dir}/{sample}_out_adj.csv.gz', delimiter=',')

        if ana_data.cell_level_niche_cluster_assign is None:
            niche_cluster_size = np.ones(ana_data.niche_cluster_connectivity.shape[0])
        else:
            niche_cluster_size = ana_data.cell_level_niche_cluster_assign.loc[ana_data.meta_data_df[
                ana_data.meta_data_df['Sample'] == sample].index].sum(axis=0)

        output_file_path = f'{ana_data.options.output}/{sample}_cluster_connectivity.pdf'

        plot_niche_cluster_connectivity(niche_cluster_connectivity=niche_cluster_connectivity,
                                        niche_cluster_score=ana_data.niche_cluster_score,
                                        niche_cluster_size=niche_cluster_size,
                                        reverse=ana_data.options.reverse,
                                        output_file_path=output_file_path)


def plot_graph_niche_cluster_feat_similarity(
        niche_cluster_feat_similarity: np.ndarray,
        niche_cluster_score: Optional[np.ndarray] = None,
        niche_cluster_size: Optional[np.ndarray] = None,
        reverse: bool = False,
        figsize_scale: Optional[float] = None,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """
    Plot niche cluster feature similarity.
    :param niche_cluster_feat_similarity: np.ndarray, the feature similarity matrix. # N (#niche) x N (#niche)
    :param niche_cluster_score: Optional[np.ndarray], the score of each niche cluster. # N (#niche)
    :param niche_cluster_size: Optional[np.ndarray], the size of each niche cluster. # N (#niche)
    :param reverse: bool, whether to reverse the color.
    :param figsize_scale: float, the scale factor for the figure size. Default is None.
    If None, the scale factor is calculated based on the number of niche cluster in the data.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    # preprocessing
    ## set diagonal to 0
    np.fill_diagonal(niche_cluster_feat_similarity, 0)
    ## set the figure size
    if figsize_scale is None:
        figsize_scale = 1.0 if niche_cluster_feat_similarity.shape[0] < 10 else 1.5 * np.log10(
            niche_cluster_feat_similarity.shape[0])

    # Create a graph
    G = nx.Graph(niche_cluster_feat_similarity)

    # position
    pos = nx.spring_layout(G, seed=42)
    # edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    # node color
    if niche_cluster_score is not None:
        nc_scores = 1 - niche_cluster_score if reverse else niche_cluster_score
        nc_colors = cal_nc_colors(nc_scores)
    else:
        nc_colors = ["#1f78b4"] * niche_cluster_feat_similarity.shape[0]
    # node size
    if niche_cluster_size is None:
        node_size = 500
    else:
        # rescale the node with size 0 to 200, and the maximum size is 1000
        node_size = (niche_cluster_size / niche_cluster_size.max() * 800 + 200).tolist()

    # Create a figure
    ## figwidth
    figwidths = [6]
    if niche_cluster_size is not None:
        figwidths.append(1.5)  # type: ignore
    if niche_cluster_score is not None:
        figwidths.append(.5)  # type: ignore
    figwidths.append(.5)  # type: ignore
    fig = plt.figure(figsize=(sum(figwidths) * figsize_scale, 6 * figsize_scale))  # type: ignore

    # Create a gridspec with 1 row and 3 columns (most).
    axes = []
    gs = gridspec.GridSpec(nrows=1, ncols=len(figwidths), width_ratios=figwidths)
    ax_index = 0
    graph_ax = fig.add_subplot(gs[ax_index])
    axes.append(graph_ax)
    if niche_cluster_size is not None:
        ax_index += 1
        node_size_ax = fig.add_subplot(gs[ax_index])
        axes.append(node_size_ax)
    if niche_cluster_score is not None:
        ax_index += 1
        node_colorbar_ax = fig.add_subplot(gs[ax_index])
        axes.append(node_colorbar_ax)
    ax_index += 1
    edge_colorbar_ax = fig.add_subplot(gs[ax_index])
    axes.append(edge_colorbar_ax)

    # Draw the graph
    nx.draw_networkx_nodes(
        G=G,
        pos=pos,
        node_color=nc_colors,  # type: ignore
        node_size=node_size,
        ax=graph_ax)
    nx.draw_networkx_edges(
        G=G,
        pos=pos,
        edge_color=weights,  # type: ignore
        alpha=weights,
        width=3.0,
        edge_cmap=plt.cm.Reds,  # type: ignore
        ax=graph_ax,
        node_size=node_size)
    nx.draw_networkx_labels(G, pos, ax=graph_ax)
    graph_ax.axis('off')

    # Draw legend for node size
    if niche_cluster_size is not None:
        ## prepare
        node_size_ax.axis('off')  # hide the
        node_size_legend_num = 5  # TODO: make it become a parameter
        sizes = np.linspace(start=200, stop=1000, num=node_size_legend_num)
        max_niche_cluster_size = niche_cluster_size.max()  # type: ignore
        magnitude = 10**int(np.floor(np.log10(abs(max_niche_cluster_size))))
        max_label = round(max_niche_cluster_size / magnitude) * magnitude
        labels = [f'{int(x):d}' for x in np.linspace(0, max_label, node_size_legend_num)]
        ## draw legend
        handles = [
            plt.Line2D(  # type: ignore
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor='gray',
                markersize=np.sqrt(size),
                label=label) for size, label in zip(sizes, labels)
        ]
        legend = node_size_ax.legend(handles=handles,
                                     title='Niche cluster size',
                                     loc='center',
                                     frameon=False,
                                     bbox_to_anchor=(0.6, 0.6))
        ## Rotate the title and adjust the position
        title = legend.get_title()
        title.set_rotation(90)
        title.set_verticalalignment('center')  # Align title vertically
        title.set_horizontalalignment('right')  # Align title horizontally
        title.set_position((-70, -60))  # Shift the title slightly to the right

    # Draw the colorbar for nodes
    if niche_cluster_score is not None:
        gradient = np.linspace(0, 1, 1000).reshape(-1, 1)
        node_colorbar_ax.imshow(gradient, aspect='auto', cmap=plt.cm.rainbow)  # type: ignore
        node_colorbar_ax.set_xticks([])
        node_colorbar_ax.set_yticks(np.linspace(0, 1000, 5))
        node_colorbar_ax.set_yticklabels(f'{x:.2f}' for x in np.linspace(0, 1, 5))
        node_colorbar_ax.set_ylabel('NT score (nodes)')

    # Draw the colorbar for edges
    colors = [(1, 1, 1, 0), (1, 0, 0, 1)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    gradient = np.linspace(1, 0, 1000).reshape(-1, 1)
    edge_colorbar_ax.imshow(gradient, aspect='auto', cmap=custom_cmap)
    edge_colorbar_ax.set_xticks([])
    edge_colorbar_ax.set_yticks(np.linspace(1000, 0, 5))
    edge_colorbar_ax.set_yticklabels(f'{x:.2f}' for x in np.linspace(0, niche_cluster_feat_similarity.max(), 5))
    edge_colorbar_ax.set_ylabel('Feature similarity (edges)')

    fig.suptitle('Niche cluster feature similarity')
    fig.tight_layout()

    if output_file_path is not None:
        fig.savefig(output_file_path)  # type: ignore
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_graph_niche_cluster_feat_similarity_from_anadata(
        ana_data: AnaData) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """
    Plot graph niche cluster feature similarity from anadata.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, List[plt.Axes]].
    """

    if ana_data.cell_type_composition is None:
        warning("No cell type composition data found. Skip niche cluster feature similarity plot.")
        return None

    if ana_data.niche_level_niche_cluster_assign is None:
        warning("No cluster assignment data found. Skip niche cluster feature similarity plot.")
        return None

    niche_cluster_feat = cal_nc_feat(nc_assign=ana_data.niche_level_niche_cluster_assign,
                                     ctc=ana_data.cell_type_composition[ana_data.cell_type_codes['Cell_Type'].tolist()])
    niche_cluster_feat_similarity = cal_nc_feat_similarity(niche_cluster_feat)

    if ana_data.cell_level_niche_cluster_assign is None:
        niche_cluster_size = np.ones(niche_cluster_feat_similarity.shape[0])
    else:
        niche_cluster_size = ana_data.cell_level_niche_cluster_assign.sum(axis=0)

    output_file_path = f'{ana_data.options.output}/niche_cluster_feature_similarity.pdf' if ana_data.options.output is not None else None

    return plot_graph_niche_cluster_feat_similarity(
        niche_cluster_feat_similarity=niche_cluster_feat_similarity.values,
        niche_cluster_score=ana_data.niche_cluster_score,
        niche_cluster_size=niche_cluster_size,  # type: ignore
        reverse=ana_data.options.reverse,
        output_file_path=output_file_path)


def plot_graph_niche_cluster_feat_similarity_bysample_from_anadata(ana_data: AnaData) -> None:
    """
    Plot graph niche cluster feature similarity by sample.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
    """

    if ana_data.cell_type_composition is None:
        warning("No cell type composition data found.")
        return None

    if ana_data.niche_level_niche_cluster_assign is None:
        warning("No cluster assignment data found.")
        return None

    for sample in ana_data.meta_data_df['Sample'].unique():

        niche_cluster_feat = cal_nc_feat(
            nc_assign=ana_data.niche_level_niche_cluster_assign.loc[ana_data.meta_data_df[
                ana_data.meta_data_df['Sample'] == sample].index],
            ctc=ana_data.cell_type_composition.loc[ana_data.meta_data_df[
                ana_data.meta_data_df['Sample'] == sample].index][ana_data.cell_type_codes['Cell_Type'].tolist()])
        niche_cluster_feat_similarity = cal_nc_feat_similarity(niche_cluster_feat)

        if ana_data.cell_level_niche_cluster_assign is None:
            niche_cluster_size = np.ones(niche_cluster_feat_similarity.shape[0])
        else:
            niche_cluster_size = ana_data.cell_level_niche_cluster_assign.loc[ana_data.meta_data_df[
                ana_data.meta_data_df['Sample'] == sample].index].sum(axis=0)

        output_file_path = f'{ana_data.options.output}/{sample}_niche_cluster_feature_similarity.pdf'

        plot_graph_niche_cluster_feat_similarity(niche_cluster_feat_similarity=niche_cluster_feat_similarity.values,
                                                 niche_cluster_score=ana_data.niche_cluster_score,
                                                 niche_cluster_size=niche_cluster_size,
                                                 reverse=ana_data.options.reverse,
                                                 output_file_path=output_file_path)


def plot_heatmap_niche_cluster_feat_similarity(
        niche_cluster_feat_similarity: pd.DataFrame,
        niche_cluster_order: List,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot heatmap of niche cluster feature similarity.
    :param niche_cluster_feat_similarity: pd.DataFrame, the feature similarity matrix.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    fig, ax = plt.subplots(figsize=(niche_cluster_feat_similarity.shape[0], niche_cluster_feat_similarity.shape[1]))
    sns.heatmap(niche_cluster_feat_similarity.loc[niche_cluster_order, niche_cluster_order],
                vmin=0,
                vmax=1,
                cmap='Reds',
                annot=True,
                fmt='.2f',
                ax=ax)
    ax.set_title('Niche cluster feature similarity')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(fname=str(output_file_path))
        plt.close(fig=fig)
        return None
    else:
        return fig, ax


def plot_heatmap_niche_cluster_feat_similarity_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot heatmap of niche cluster feature similarity from anadata.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    if ana_data.cell_type_composition is None:
        warning("No cell type composition data found. Skip niche cluster feature similarity plot.")
        return None

    if ana_data.niche_level_niche_cluster_assign is None or ana_data.cell_level_niche_cluster_assign is None:
        warning("No cluster assignment data found. Skip niche cluster feature similarity plot.")
        return None

    niche_cluster_feat = cal_nc_feat(nc_assign=ana_data.niche_level_niche_cluster_assign,
                                     ctc=ana_data.cell_type_composition[ana_data.cell_type_codes['Cell_Type'].tolist()])
    niche_cluster_feat_similarity = cal_nc_feat_similarity(niche_cluster_feat)

    nc_scores = cal_nc_scores(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                              reverse=ana_data.options.reverse,
                              niche_cluster_score=ana_data.niche_cluster_score)
    nc_order = [f'NicheCluster_{x}' for x in nc_scores.argsort()]

    output_file_path = f'{ana_data.options.output}/Heatmap_niche_cluster_feature_similarity.pdf' if ana_data.options.output is not None else None

    return plot_heatmap_niche_cluster_feat_similarity(niche_cluster_feat_similarity=niche_cluster_feat_similarity,
                                                      niche_cluster_order=nc_order,
                                                      output_file_path=output_file_path)


def plot_heatmap_niche_cluster_feat_similarity_by_sample_from_anadata(ana_data: AnaData) -> None:
    """
    Plot heatmap of niche cluster feature similarity by sample.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
    """

    if ana_data.cell_type_composition is None:
        warning("No cell type composition data found. Skip niche cluster feature similarity plot.")
        return None

    if ana_data.niche_level_niche_cluster_assign is None or ana_data.cell_level_niche_cluster_assign is None:
        warning("No cluster assignment data found. Skip niche cluster feature similarity plot.")
        return None

    nc_scores = cal_nc_scores(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                              reverse=ana_data.options.reverse,
                              niche_cluster_score=ana_data.niche_cluster_score)
    nc_order = [f'NicheCluster_{x}' for x in nc_scores.argsort()]

    for sample in ana_data.meta_data_df['Sample'].unique():

        niche_cluster_feat = cal_nc_feat(
            nc_assign=ana_data.niche_level_niche_cluster_assign.loc[ana_data.meta_data_df[
                ana_data.meta_data_df['Sample'] == sample].index],
            ctc=ana_data.cell_type_composition.loc[ana_data.meta_data_df[
                ana_data.meta_data_df['Sample'] == sample].index][ana_data.cell_type_codes['Cell_Type'].tolist()])
        niche_cluster_feat_similarity = cal_nc_feat_similarity(niche_cluster_feat)

        output_file_path = f'{ana_data.options.output}/{sample}_heatmap_niche_cluster_feature_similarity.pdf'

        plot_heatmap_niche_cluster_feat_similarity(niche_cluster_feat_similarity=niche_cluster_feat_similarity,
                                                   niche_cluster_order=nc_order,
                                                   output_file_path=output_file_path)


def plot_cluster_proportion(
        niche_cluster_loading: pd.Series,
        niche_cluster_colors: List,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the proportion of each cluster.
    :param niche_cluster_loading: pd.Series, the loading of each niche cluster.
    :param niche_cluster_colors: List, the color of each niche cluster.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(niche_cluster_loading,
           labels=[f'Niche cluster {i}' for i in range(niche_cluster_loading.shape[0])],
           colors=niche_cluster_colors,
           autopct='%1.1f%%',
           pctdistance=1.25,
           labeldistance=.6)
    ax.set_title(f'Niche proportions for each niche cluster')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/niche_cluster_proportion.pdf')
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_cluster_proportion_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the proportion of each cluster.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """
    if ana_data.niche_level_niche_cluster_assign is None or ana_data.cell_level_niche_cluster_assign is None:
        warning("No cluster assignment data found. Skip niche cluster proportion plot.")
        return None

    nc_scores = cal_nc_scores(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                              reverse=ana_data.options.reverse,
                              niche_cluster_score=ana_data.niche_cluster_score)
    nc_colors = cal_nc_colors(nc_scores)

    # loadings
    niche_cluster_loading = ana_data.niche_level_niche_cluster_assign.sum(axis=0)

    return plot_cluster_proportion(niche_cluster_loading=niche_cluster_loading,
                                   niche_cluster_colors=nc_colors,
                                   output_file_path=ana_data.options.output)


def plot_niche_cluster_loadings_dataset(
    cell_level_niche_cluster_assign: pd.DataFrame,
    meta_data_df: pd.DataFrame,
    nc_scores: np.ndarray,
    output_file_path: Optional[Union[str,
                                     Path]] = None) -> Optional[Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]]:
    """
    Plot niche cluster loadings for each cell.
    :param cell_level_niche_cluster_assign: pd.DataFrame, the niche cluster assign data.
    :param meta_data_df: pd.DataFrame, the meta data.
    :param nc_scores: np.ndarray, the score of each niche cluster, default is None.
    :param output_file_path: Optional[Union[str, Path]], the output file path, default is None.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    samples = meta_data_df['Sample'].unique()
    n_sample = len(samples)
    n_niche_cluster = cell_level_niche_cluster_assign.shape[1]
    nc_order_index = cal_nc_order_index(nc_scores)

    fig, axes = plt.subplots(n_sample, n_niche_cluster, figsize=(3.3 * n_niche_cluster, 3 * n_sample))
    for i, sample in enumerate(samples):
        sample_df = cell_level_niche_cluster_assign.loc[meta_data_df[meta_data_df['Sample'] == sample].index]
        sample_df = sample_df.join(meta_data_df[['x', 'y']])
        for j, c_index in enumerate(nc_order_index):
            ax = axes[i, j] if n_sample > 1 else axes[j]
            scatter = ax.scatter(sample_df['x'],
                                 sample_df['y'],
                                 c=sample_df[f'NicheCluster_{c_index}'],
                                 cmap='Reds',
                                 vmin=0,
                                 vmax=1,
                                 s=4)
            ax.set_title(f'{sample}: niche cluster {c_index}')
            plt.colorbar(scatter)
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/niche_cluster_loadings.pdf')
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_niche_cluster_loadings_dataset_from_anadata(
        ana_data: AnaData) -> Optional[Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]]:
    """
    Plot niche cluster loadings for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]].
    """

    if ana_data.cell_level_niche_cluster_assign is None:
        warning("No cluster assignment data found. Skip niche cluster loadings plot.")
        return None

    nc_scores = cal_nc_scores(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                              reverse=ana_data.options.reverse,
                              niche_cluster_score=ana_data.niche_cluster_score)

    return plot_niche_cluster_loadings_dataset(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                                               meta_data_df=ana_data.meta_data_df,
                                               nc_scores=nc_scores,
                                               output_file_path=ana_data.options.output)


def plot_niche_cluster_loadings_sample(
        cell_level_niche_cluster_assign: pd.DataFrame,
        meta_data_df: pd.DataFrame,
        nc_scores: np.ndarray,
        spatial_scaling_factor: float = 1.0,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot niche cluster loadings for each cell.
    :param cell_level_niche_cluster_assign: pd.DataFrame, the niche cluster assign data.
    :param meta_data_df: pd.DataFrame, the meta data.
    :param nc_scores: np.ndarray, the score of each niche cluster, default is None.
    :param spatial_scaling_factor: float, the scale factor control the size of spatial-based plots.
    :param output_file_path: Optional[Union[str, Path]], the output file path, default is None.
    :return: None or List[Tuple[plt.Figure, plt.Axes]].
    """

    samples = meta_data_df['Sample'].unique()
    n_niche_cluster = cell_level_niche_cluster_assign.shape[1]
    nc_order = cal_nc_order_index(nc_scores)

    output = []
    for sample in samples:
        sample_df = cell_level_niche_cluster_assign.loc[meta_data_df[meta_data_df['Sample'] == sample].index]
        sample_df = sample_df.join(meta_data_df[['x', 'y']])
        fig_width, fig_height = saptial_figsize(sample_df, scaling_factor=spatial_scaling_factor)
        fig, axes = plt.subplots(nrows=1, ncols=n_niche_cluster, figsize=(fig_width * n_niche_cluster, fig_height))
        for j, c_index in enumerate(nc_order):
            ax = axes[j]  #  there should more than one niche cluster
            scatter = ax.scatter(sample_df['x'],
                                 sample_df['y'],
                                 c=sample_df[f'NicheCluster_{c_index}'],
                                 cmap='Reds',
                                 vmin=0,
                                 vmax=1,
                                 s=4)
            ax.set_title(f'{sample}: niche cluster {c_index}')
            plt.colorbar(scatter)
        fig.tight_layout()
        output.append((fig, axes))
        if output_file_path is not None:
            fig.savefig(f'{output_file_path}/niche_cluster_loadings_{sample}.pdf')
            plt.close(fig)
    return output if len(output) > 0 else None


def plot_niche_cluster_loadings_sample_from_anadata(ana_data: AnaData) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot niche cluster loadings for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or List[Tuple[plt.Figure, plt.Axes]].
    """

    if ana_data.cell_level_niche_cluster_assign is None:
        warning("No cluster assign data found. Skip niche cluster loadings plot.")
        return None

    nc_scores = cal_nc_scores(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                              reverse=ana_data.options.reverse,
                              niche_cluster_score=ana_data.niche_cluster_score)

    return plot_niche_cluster_loadings_sample(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                                              meta_data_df=ana_data.meta_data_df,
                                              nc_scores=nc_scores,
                                              spatial_scaling_factor=ana_data.options.scale_factor,
                                              output_file_path=ana_data.options.output)


def plot_niche_cluster_loadings(
    ana_data: AnaData
) -> Optional[Union[List[Tuple[plt.Figure, plt.Axes]], Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]]]:
    """
    Plot niche cluster loadings for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Union[List[Tuple[plt.Figure, plt.Axes]], Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]].
    """
    if hasattr(ana_data.options, 'sample') and ana_data.options.sample:
        return plot_niche_cluster_loadings_sample_from_anadata(ana_data=ana_data)
    else:
        return plot_niche_cluster_loadings_dataset_from_anadata(ana_data=ana_data)


def plot_max_niche_cluster_dataset(
        cell_level_max_niche_cluster: pd.DataFrame,
        meta_data_df: pd.DataFrame,
        nc_scores: np.ndarray,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the maximum niche cluster for each cell.
    :param cell_level_max_niche_cluster: pd.DataFrame, the maximum niche cluster data.
    :param meta_data_df: pd.DataFrame, the meta data.
    :param nc_scores: Optional[np.ndarray], the score of each niche cluster, default is None.
    :param output_file_path: Optional[Union[str, Path]], the output file path, default is None.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    samples = meta_data_df['Sample'].unique()
    n_sample = len(samples)

    # colors
    nc_colors = cal_nc_colors(nc_scores)
    nc_palette = cal_nc_palette(nc_colors)

    fig, axes = plt.subplots(1, n_sample, figsize=(5 * n_sample, 3))
    for i, sample in enumerate(samples):
        ax: Axes = axes[i] if n_sample > 1 else axes  # type: ignore
        sample_df = cell_level_max_niche_cluster.loc[meta_data_df[meta_data_df['Sample'] == sample].index]
        sample_df = sample_df.join(meta_data_df[['x', 'y']])
        sample_df['Niche_Cluster'] = 'niche cluster ' + sample_df['Niche_Cluster'].astype(str)
        sns.scatterplot(data=sample_df,
                        x='x',
                        y='y',
                        hue='Niche_Cluster',
                        hue_order=[f'niche cluster {j}' for j in cal_nc_order_index(nc_scores)],
                        palette=nc_palette,
                        s=10,
                        ax=ax)
        ax.set_title(f'{sample}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/max_niche_cluster.pdf')
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_max_niche_cluster_dataset_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the maximum niche cluster for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    if ana_data.cell_level_max_niche_cluster is None or ana_data.cell_level_niche_cluster_assign is None:
        warning("No cluster assignment data found. Skip max niche cluster plot.")
        return None

    nc_scores = cal_nc_scores(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                              reverse=ana_data.options.reverse,
                              niche_cluster_score=ana_data.niche_cluster_score)

    return plot_max_niche_cluster_dataset(cell_level_max_niche_cluster=ana_data.cell_level_max_niche_cluster,
                                          meta_data_df=ana_data.meta_data_df,
                                          nc_scores=nc_scores,
                                          output_file_path=ana_data.options.output)


def plot_max_niche_cluster_sample(
        cell_level_max_niche_cluster: pd.DataFrame,
        meta_data_df: pd.DataFrame,
        nc_scores: np.ndarray,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot the maximum niche cluster for each cell.
    :param cell_level_max_niche_cluster: pd.DataFrame, the maximum niche cluster data.
    :param meta_data_df: pd.DataFrame, the meta data.
    :param nc_scores: np.ndarray, the score of each niche cluster.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or List[Tuple[plt.Figure, plt.Axes]].
    """

    samples = meta_data_df['Sample'].unique()
    n_niche_cluster = len(nc_scores)

    # colors
    nc_colors = cal_nc_colors(nc_scores)
    nc_palette = cal_nc_palette(nc_colors)

    output = []
    for sample in samples:
        sample_df = cell_level_max_niche_cluster.loc[meta_data_df[meta_data_df['Sample'] == sample].index]
        sample_df = sample_df.join(meta_data_df[['x', 'y']])
        sample_df['Niche_Cluster'] = 'niche cluster ' + sample_df['Niche_Cluster'].astype(str)
        fig_width, fig_height = saptial_figsize(sample_df, scaling_factor=1)
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        sns.scatterplot(data=sample_df,
                        x='x',
                        y='y',
                        hue='Niche_Cluster',
                        hue_order=[f'niche cluster {j}' for j in cal_nc_order_index(nc_scores)],
                        palette=nc_palette,
                        s=10,
                        ax=ax)
        ax.set_title(f'{sample}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()
        if output_file_path is not None:
            fig.savefig(f'{output_file_path}/max_niche_cluster_{sample}.pdf')
            plt.close(fig)
        else:
            output.append((fig, ax))
    return output if len(output) > 0 else None


def plot_max_niche_cluster_sample_from_anadata(ana_data: AnaData) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot the maximum niche cluster for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or List[Tuple[plt.Figure, plt.Axes]].
    """

    if ana_data.cell_level_max_niche_cluster is None or ana_data.cell_level_niche_cluster_assign is None:
        warning("No cluster assignment data found. Skip max niche cluster plot.")
        return None

    nc_scores = cal_nc_scores(cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
                              reverse=ana_data.options.reverse,
                              niche_cluster_score=ana_data.niche_cluster_score)

    return plot_max_niche_cluster_sample(cell_level_max_niche_cluster=ana_data.cell_level_max_niche_cluster,
                                         meta_data_df=ana_data.meta_data_df,
                                         nc_scores=nc_scores,
                                         output_file_path=ana_data.options.output)


def plot_max_niche_cluster(
        ana_data: AnaData) -> Optional[Union[List[Tuple[plt.Figure, plt.Axes]], Tuple[plt.Figure, plt.Axes]]]:
    """
    Plot the maximum niche cluster for each cell.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """
    if hasattr(ana_data.options, 'sample') and ana_data.options.sample:
        return plot_max_niche_cluster_sample_from_anadata(ana_data=ana_data)
    else:
        return plot_max_niche_cluster_dataset_from_anadata(ana_data=ana_data)


def plot_niche_cluster_gini(
        intra_cluster_gini_df: pd.DataFrame,
        output_file_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the Gini coefficient of each niche cluster.
    :param intra_cluster_gini_df: pd.DataFrame, the Gini coefficient of each niche cluster.
    :param output_file_path: Optional[Union[str, Path]], the output file path.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=intra_cluster_gini_df, x='cluster', y='gini', ax=ax)
    ax.set_xlabel('Niche Cluster')
    ax.set_ylabel('Gini coefficient across each cell')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f'{output_file_path}/niche_cluster_gini.pdf')
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_niche_cluster_gini_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot the Gini coefficient of each niche cluster.
    :param ana_data: AnaData, the data for analysis.
    :return: None or Tuple[plt.Figure, plt.Axes].
    """
    if ana_data.cell_level_niche_cluster_assign is None:
        warning("No niche cluster assign data found. Skip niche cluster Gini plot.")
        return None

    intra_cluster_gini = ana_data.cell_level_niche_cluster_assign.apply(gini, axis=0).values
    intra_cluster_gini_df = pd.DataFrame(data={
        'gini': intra_cluster_gini,
        'cluster': ana_data.cell_level_niche_cluster_assign.columns
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=intra_cluster_gini_df, x='cluster', y='gini', ax=ax)
    ax.set_xlabel('Niche Cluster')
    ax.set_ylabel('Gini coefficient across each cell')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.tight_layout()
    if ana_data.options.output is not None:
        fig.savefig(f'{ana_data.options.output}/niche_cluster_gini.pdf')
        plt.close(fig)
        return None
    else:
        return fig, ax


def niche_cluster_visualization(ana_data: AnaData) -> None:
    """
    All spatial visualization will include here.
    :param ana_data: AnaData, the data for analysis.
    :return: None.
    """

    # 1. plot niche cluster connectivity
    plot_niche_cluster_connectivity_from_anadata(ana_data=ana_data)

    if hasattr(ana_data.options, 'sample') and ana_data.options.sample:
        plot_niche_cluster_connectivity_bysample_from_anadata(ana_data=ana_data)

    # 2. plot niche cluster feature similarity
    plot_graph_niche_cluster_feat_similarity_from_anadata(ana_data=ana_data)

    if hasattr(ana_data.options, 'sample') and ana_data.options.sample:
        plot_graph_niche_cluster_feat_similarity_bysample_from_anadata(ana_data=ana_data)

    plot_heatmap_niche_cluster_feat_similarity_from_anadata(ana_data=ana_data)

    if hasattr(ana_data.options, 'sample') and ana_data.options.sample:
        plot_heatmap_niche_cluster_feat_similarity_by_sample_from_anadata(ana_data=ana_data)

    # 3. share of each cluster
    plot_cluster_proportion_from_anadata(ana_data=ana_data)

    # 4. niche cluster loadings for each cell
    if not hasattr(ana_data.options,
                   'suppress_niche_cluster_loadings') or not ana_data.options.suppress_niche_cluster_loadings:
        plot_niche_cluster_loadings(ana_data=ana_data)
    else:
        info("Skip niche cluster loadings plot according to `suppress_niche_cluster_loadings` option.")

    # 5. maximum niche cluster for each cell
    plot_max_niche_cluster(ana_data=ana_data)

    # 6. gini coefficient of each niche cluster
    plot_niche_cluster_gini_from_anadata(ana_data=ana_data)
