from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd

from ..log import info, warning
from .data import AnaData
from .utils import get_n_colors

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def construct_meta_cell_along_trajectory(meta_data_df: pd.DataFrame,
                                         trajectory: str,
                                         n_cells: int = 10) -> pd.DataFrame:
    """
    Construct features for meta-cells by binning cells along the trajectory.

    Parameters
    ----------
    meta_data_df : pd.DataFrame
        DataFrame containing features for each cell. Rows are cells and columns are features. Features should be continuous values.
    trajectory : str
        Column name in feat_df that contains the trajectory values.
    n_cells : int
        Number of cells to bin in each meta-cell.
    
    Returns
    -------
    metacell_df : pd.DataFrame
        DataFrame containing features for meta-cells. Rows are meta-cells and columns are features.
    """

    # ensure trajectory in meta_data_df
    assert trajectory in meta_data_df.columns, f"Trajectory column {trajectory} not found in meta_data_df."

    # cell number should be at least 2*n_cells
    assert len(meta_data_df) >= 2 * n_cells, f"Number of cells in meta_data_df should be at least 2*n_cells."

    # sort meta_data_df by trajectory
    meta_data_df = meta_data_df.sort_values(by=trajectory)

    # bin cells into meta-cells
    rolled_mean = meta_data_df.rolling(window=n_cells).mean()
    mask = np.arange(len(rolled_mean)) % n_cells == 0
    mask = np.repeat(mask.reshape(-1, 1), rolled_mean.shape[1], axis=1)
    rolling_mean = rolled_mean.where(mask)
    rolling_mean = rolling_mean.dropna()

    return rolling_mean


def cal_features_correlation_along_trajectory(data_df: pd.DataFrame,
                                              trajectory: str,
                                              features: Optional[List[str]] = None,
                                              top_n: Optional[int] = None,
                                              rho_threshold: Optional[float] = None,
                                              p_val_threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Calculate the correlation between features and a trajectory in a DataFrame.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing features for each (meta-)cell.
        Rows are meta-cells and columns are features and trajectory.
        All columns should be continuous values.
        All columns except the trajectory column should be features.
    trajectory : str
        Column name in data_df that contains the trajectory values.
    features : Optional[List[str]]
        Optional; if provided, only consider these features for correlation.
        If None, consider all features except the trajectory column.
    top_n : Optional[int]
        Optional; if provided, return only the top N features with highest absolute correlation to the trajectory.
        If None, return all features that meet the correlation and p-value thresholds.
    rho_threshold : Optional[float]
        Optional; minimum absolute correlation coefficient to consider a feature. If None, no threshold is applied.
    p_val_threshold : Optional[float]
        Optional; maximum p-value to consider a feature. If None, no threshold is applied.

    Returns
    -------
    pd.DataFrame
        DataFrame containing top correlated features.
    """

    if trajectory not in data_df.columns:
        raise ValueError(f"Trajectory column '{trajectory}' not found in metacell_data_df.")

    feature_list: List[str] = data_df.columns.difference(
        [trajectory]).tolist() if features is None else features
    correlations = []
    for feat in feature_list:
        rho, p_val = pearsonr(data_df[trajectory], data_df[feat])
        correlations.append((feat, rho, p_val))
    correlations_df = pd.DataFrame(correlations, columns=['Feature', 'PCC', 'P_Value'])
    correlations_df = correlations_df.dropna()
    correlations_df = correlations_df.sort_values(by='PCC', ascending=False)
    if rho_threshold is not None:
        correlations_df = correlations_df[abs(correlations_df['PCC']) >= rho_threshold]
    if p_val_threshold is not None:
        correlations_df = correlations_df[correlations_df['P_Value'] <= p_val_threshold]
    if top_n is not None:
        top_n_df = correlations_df.head(top_n)
        top_n_df = top_n_df[top_n_df['PCC']>0]
        bottom_n_df = correlations_df.tail(top_n)
        bottom_n_df = bottom_n_df[bottom_n_df['PCC']<0]
        correlations_df = pd.concat([correlations_df.head(top_n), correlations_df.tail(top_n)], ignore_index=True)
    return correlations_df.set_index('Feature')


def plot_scatter_feat_along_trajectory(data_df: pd.DataFrame,
                                       trajectory: str,
                                       feature: str,
                                       fit_reg: bool = True,
                                       annotate_pos: Union[str, None] = 'upper left',
                                       figszie: Tuple[int, int] = (5, 3),
                                       ylabel: Optional[str] = None,
                                       scatter_kws: Optional[Dict] = None,
                                       line_kws: Optional[Dict] = None,
                                       **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot scatter plot of feature along trajectory.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing features for each cell. Rows are cells and columns are features. Features should be continuous values.
    trajectory : str
        Column name in feat_df that contains the trajectory values.
    feature : str
        Column name in feat_df that contains the feature values.
    fit_reg : bool
        Whether to plot regression line.
    annotate_pos : str
        Position to annotate PCC and p-value. Should be one of 'upper left', 'upper right', 'lower left', 'lower right', None.
        None means no annotation.
    figszie : Tuple[int, int]
        Figure size. Default is (5, 3).
    ylabel : str
        Label for y-axis.
    scatter_kws : dict
        Additional arguments for sns.regplot scatter_kws.
    line_kws : dict
        Additional arguments for sns.regplot line_kws.
    **kwargs
        Additional arguments for sns.regplot.

    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax : plt.Axes
        Axes object.
    """

    # default parameters
    default_scatter_kws = {'edgecolor': None}
    default_line_kws = {'color': 'red'}
    default_kwargs = {'ci': None}

    # update default kws
    scatter_kws = {**default_scatter_kws, **(scatter_kws or {})}
    line_kws = {**default_line_kws, **(line_kws or {})}
    kwargs = {**default_kwargs, **kwargs}

    # ensure trajectory in data_df
    assert trajectory in data_df.columns, f"Trajectory column {trajectory} not found in data_df."

    # ensure feature in data_df
    assert feature in data_df.columns, f"Feature column {feature} not found in data_df."

    # sort data_df by trajectory
    data_df = data_df.sort_values(by=trajectory)

    # Visualization
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
        fig, ax = plt.subplots(figsize=figszie)

        sns.regplot(data=data_df,
                    x=trajectory,
                    y=feature,
                    fit_reg=fit_reg,
                    scatter_kws=scatter_kws,
                    line_kws=line_kws,
                    **kwargs,
                    ax=ax)

        # descriptive annotation
        ax.set_title(feature)
        ax.set_xlabel(trajectory)
        ax.set_ylabel(ylabel if ylabel else feature)

        # correlation annotation
        rho, p_val = pearsonr(data_df[trajectory], data_df[feature])
        if annotate_pos == "upper left":
            ax.annotate(f"PCC = {rho:.2f}\nP = {p_val:.6f}",
                        xy=(0.02, 0.98),
                        xycoords="axes fraction",
                        ha="left",
                        va="top")
        elif annotate_pos == "upper right":
            ax.annotate(f"PCC = {rho:.2f}\nP = {p_val:.6f}",
                        xy=(0.98, 0.98),
                        xycoords="axes fraction",
                        ha="right",
                        va="top")
        elif annotate_pos == "lower left":
            ax.annotate(f"PCC = {rho:.2f}\nP = {p_val:.6f}",
                        xy=(0.02, 0.02),
                        xycoords="axes fraction",
                        ha="left",
                        va="bottom")
        elif annotate_pos == "lower right":
            ax.annotate(f"PCC = {rho:.2f}\nP = {p_val:.6f}",
                        xy=(0.98, 0.02),
                        xycoords="axes fraction",
                        ha="right",
                        va="bottom")
        elif annotate_pos is None:
            pass
        else:
            raise ValueError(f"annotate_pos should be one of 'upper left', 'upper right', 'lower left', 'lower right'")

    return fig, ax


def plot_cell_type_composition_along_trajectory(
        data_df: pd.DataFrame,
        trajectory: str,
        cell_types: Union[str, List[str]],
        agg_cell_num: int = 10,
        figsize: Tuple[int, int] = (6, 2),
        palette: Optional[Dict[str, str]] = None,
        output_file_path: Optional[Path] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type composition along trajectory.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing cell type information for each cell. Rows are cells and columns are cell types and trajectory.
    trajectory : str
        Column name in data_df that contains the trajectory values.
    cell_types : str or list of str
        Column name(s) in data_df that contains the cell type information.
    agg_cell_num : int
        Number of cells to aggregate in each bin along the trajectory. Default is 10. 1 means no aggregation.
    figsize : Optional[Tuple[int, int]]
        Figure size. Default is (6, 2).
    palette : Optional[Dict[str, str]]
        Color palette for cell types. If None, use default color palette. Keys are cell types and values are colors.
    output_file_path : Optional[Path]
        If provided, save the figure to this path. If None, do not save the figure and return the figure, and axes objects.
    
    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax : plt.Axes
        Axes object.
    """

    # ensure trajectory in data_df
    assert trajectory in data_df.columns, f"Trajectory column {trajectory} not found in data_df."

    # ensure cell_types in data_df
    if isinstance(cell_types, str):
        cell_types = [cell_types]
    for cell_type in cell_types:
        assert cell_type in data_df.columns, f"Cell type column {cell_type} not found in data_df."

    # if palette is provided, all cell types should be in palette
    if palette is not None:
        for cell_type in cell_types:
            assert cell_type in palette, f"Cell type {cell_type} not found in palette."
    else:
        palette = {cell_type: color for cell_type, color in zip(cell_types, get_n_colors(len(cell_types)))}

    # sort data_df by trajectory
    data_df = data_df.sort_values(by=trajectory)

    # bin cells into meta-cells
    if agg_cell_num > 1:
        rolling_mean = construct_meta_cell_along_trajectory(meta_data_df=data_df.loc[:, cell_types + [trajectory]],
                                                            trajectory=trajectory,
                                                            n_cells=agg_cell_num)
    else:
        rolling_mean = data_df

    # plot cell type composition along trajectory
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
        fig, ax = plt.subplots(figsize=figsize)

        for cell_type in cell_types:
            ax.plot(rolling_mean[trajectory], rolling_mean[cell_type], label=cell_type, color=palette[cell_type])

        ax.set_ylim(0, 1)
        ax.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.set_xlabel(trajectory)
        ax.set_ylabel('Normalized Cell Type Composition')
        ax.set_title('Cell Type Composition Along Niche Trajectory')
        fig.tight_layout()

    if output_file_path is not None:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file_path, bbox_inches='tight', dpi=300)
        info(f"Saved cell type composition plot to {output_file_path}")
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_cell_type_composition_along_trajectory_from_anadata(
        ana_data: AnaData,
        cell_types: Optional[Union[str, List[str]]] = None,
        agg_cell_num: int = 10,
        figsize: Tuple[int, int] = (6, 2),
        palette: Optional[Dict[str, str]] = None,
        output_file_path: Optional[Union[Path, str]] = None) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type composition (niche features) along trajectory from AnaData object.

    Parameters
    ----------
    ana_data : AnaData
        AnaData object.
    cell_types : str or list of str
        Column name(s) in AnaData.meta_data_df that contains the cell type information.
        Default is None, which means all cell types in AnaData.cell_type_codes will be used.
    agg_cell_num : int
        Number of cells to aggregate in each bin along the trajectory. Default is 10. 1 means no aggregation.
    figsize : Tuple[int, int]
        Figure size. Default is (6, 2).
    palette : Optional[Dict[str, str]]
        Color palette for cell types. If None, use default color palette. Keys are cell types and values are colors.
    output_file_path : Optional[Union[Path, str]]
        Path to save the figure. If None, the default path
        {ana_data.options.output}/lineplot_raw_cell_type_composition_along_trajectory.pdf is used. 
        If ana_data.options.output is also None, the figure will not be saved and the function 
        will return the figure and axes objects instead.
    
    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax : plt.Axes
        Axes object.
    """

    # ensure ana_data has trajectory and cell types
    if ana_data.NT_score is None:
        warning("NT score is not set in AnaData object. Skipping cell type composition along trajectory plot.")
    if ana_data.cell_type_codes is None:
        warning("Cell type codes are not set in AnaData object. Skipping cell type composition along trajectory plot.")
    if ana_data.cell_type_composition is None:
        warning(
            "Cell type composition is not set in AnaData object. Skipping cell type composition along trajectory plot.")

    # data_df
    data_df = ana_data.meta_data_df.copy()
    data_df = data_df.join(1 - ana_data.NT_score['Cell_NTScore'] if hasattr(ana_data.options, 'reverse')
                           and ana_data.options.reverse else ana_data.NT_score['Cell_NTScore'])  # type: ignore
    data_df = data_df.join(ana_data.cell_type_composition)

    # cell types
    if cell_types is None:
        cell_types = ana_data.cell_type_codes['Cell_Type'].values.tolist()
    elif isinstance(cell_types, str):
        cell_types = [cell_types]

    # output file path
    if output_file_path is None:
        if not hasattr(ana_data.options, 'output') or ana_data.options.output is None:
            output_file_path = None
        else:
            output_file_path = Path(ana_data.options.output) / 'lineplot_raw_cell_type_composition_along_trajectory.pdf'
    else:
        output_file_path = Path(output_file_path)
        if not output_file_path.suffix:
            output_file_path = output_file_path.with_suffix('.pdf')

    return plot_cell_type_composition_along_trajectory(
        data_df=data_df,
        trajectory='Cell_NTScore',
        cell_types=cell_types,  # type: ignore
        agg_cell_num=agg_cell_num,
        figsize=figsize,
        palette=palette,
        output_file_path=output_file_path)
