"""This module contains functions for cell type-based analysis."""

from .utils import saptial_figsize, validate_cell_type_palette
from .niche_cluster import cal_nc_order, cal_nc_order_index, cal_nc_scores
from .data import AnaData
from ..log import info, warning
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.stats import gaussian_kde
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"


def plot_spatial_cell_type_distribution_dataset(
    data_df: pd.DataFrame,
    output_file_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """Plot spatial cell type distribution.

    Parameters
    ----------
    data_df :
        pd.DataFrame, the data for visualization.
    output_file_path :
        Optional[Union[str, Path]], the output directory.
    kwargs :
        Additional keyword arguments passed to ``matplotlib.axes.Axes.scatter``.
        For example, use ``s`` to control marker area. Defaults to ``s=8``.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    # data_df should have 'x', 'y', 'Cell_Type', and 'Sample' columns
    if "Cell_Type" not in data_df.columns:
        warning("No `Cell_Type` column found. Skip spatial cell type distribution visualization.")
        return None
    if "Sample" not in data_df.columns:
        warning("No `Sample` column found. Skip spatial cell type distribution visualization.")
        return None
    if "x" not in data_df.columns:
        warning("No `x` column found. Skip spatial cell type distribution visualization.")
        return None
    if "y" not in data_df.columns:
        warning("No `y` column found. Skip spatial cell type distribution visualization.")
        return None

    # Cell_Type column should be categorical
    data_df["Cell_Type"] = data_df["Cell_Type"].astype("category")

    # Check parameters for palette
    cell_types = data_df["Cell_Type"].cat.categories.tolist()
    palette = kwargs.get("palette", None)
    kwargs["palette"] = validate_cell_type_palette(cell_types=cell_types, palette=palette)
    kwargs.setdefault("s", 8)

    samples = data_df["Sample"].unique()
    N = len(samples)

    fig, axes = plt.subplots(1, N, figsize=(4 * N, 3))
    for i, sample in enumerate(samples):
        sample_df = data_df.loc[data_df["Sample"] == sample]
        ax: plt.Axes = axes[i] if N > 1 else axes  # type: ignore
        sns.scatterplot(data=sample_df, x="x", y="y", hue="Cell_Type", ax=ax, **kwargs)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f"{output_file_path}/spatial_cell_type_distribution.pdf", transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_spatial_cell_type_distribution_dataset_from_anadata(
    ana_data: AnaData,
    **kwargs,
) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """Plot spatial cell type distribution.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.
    kwargs :
        Additional keyword arguments passed to ``matplotlib.axes.Axes.scatter``.
        For example, use ``s`` to control marker area. Defaults to ``s=8``.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if "Cell_Type" not in ana_data.meta_data_df.columns:
        warning("No cell type data found. Skip spatial cell type distribution visualization.")
        return None

    return plot_spatial_cell_type_distribution_dataset(
        data_df=ana_data.meta_data_df,
        output_file_path=ana_data.options.output,
        **kwargs,
    )


def plot_spatial_cell_type_distribution_sample(
    data_df: pd.DataFrame,
    output_file_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """Plot spatial cell type distribution.

    Parameters
    ----------
    data_df :
        pd.DataFrame, the data for visualization.
    output_file_path :
        Optional[Union[str, Path]], the output directory.
    kwargs :
        Additional keyword arguments passed to ``matplotlib.axes.Axes.scatter``.
        For example, use ``s`` to control marker area. Defaults to ``s=8``.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    # data_df should have 'x', 'y', 'Cell_Type', and 'Sample' columns
    if "Cell_Type" not in data_df.columns:
        warning("No `Cell_Type` column found. Skip spatial cell type distribution visualization.")
        return None
    if "Sample" not in data_df.columns:
        warning("No `Sample` column found. Skip spatial cell type distribution visualization.")
        return None
    if "x" not in data_df.columns:
        warning("No `x` column found. Skip spatial cell type distribution visualization.")
        return None
    if "y" not in data_df.columns:
        warning("No `y` column found. Skip spatial cell type distribution visualization.")
        return None

    # Cell_Type column should be categorical
    data_df["Cell_Type"] = data_df["Cell_Type"].astype("category")

    # Check parameters for palette
    cell_types = data_df["Cell_Type"].cat.categories.tolist()
    palette = kwargs.get("palette", None)
    kwargs["palette"] = validate_cell_type_palette(cell_types=cell_types, palette=palette)
    kwargs.setdefault("s", 8)

    samples = data_df["Sample"].unique()
    output = []

    for sample in samples:
        sample_df = data_df.loc[data_df["Sample"] == sample]
        fig, ax = plt.subplots(figsize=saptial_figsize(sample_df))
        sns.scatterplot(data=sample_df, x="x", y="y", hue="Cell_Type", ax=ax, **kwargs)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        output.append((fig, ax))
        if output_file_path is not None:
            fig.savefig(f"{output_file_path}/spatial_cell_type_distribution_{sample}.pdf", transparent=True)
            plt.close(fig)
    return output


def plot_spatial_cell_type_distribution_sample_from_anadata(
    ana_data: AnaData,
    **kwargs,
) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """Plot spatial cell type distribution.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.
    kwargs :
        Additional keyword arguments passed to ``matplotlib.axes.Axes.scatter``.
        For example, use ``s`` to control marker area. Defaults to ``s=8``.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if "Cell_Type" not in ana_data.meta_data_df.columns:
        warning("No cell type data found. Skip spatial cell type distribution visualization.")
        return None

    return plot_spatial_cell_type_distribution_sample(
        data_df=ana_data.meta_data_df,
        output_file_path=ana_data.options.output,
        **kwargs,
    )


def _count_weighted_kde(
    values: np.ndarray,
    x_grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bw_method: Any = None,
    fallback_sigma: Optional[float] = None,
) -> np.ndarray:
    """Return a count-weighted density on ``x_grid``."""

    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    valid_mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid_mask]
    weights = weights[valid_mask]
    total_weight = float(weights.sum())

    if len(values) == 0 or total_weight <= 0:
        return np.zeros_like(x_grid, dtype=float)

    if len(values) >= 2 and np.unique(values).size >= 2:
        kde = gaussian_kde(values, bw_method=bw_method, weights=weights)
        return kde(x_grid) * total_weight

    if fallback_sigma is None:
        fallback_sigma = (x_grid.max() - x_grid.min()) / 100
    fallback_sigma = max(fallback_sigma, 1e-8)

    density = np.zeros_like(x_grid, dtype=float)
    for value, weight in zip(values, weights):
        density += weight * np.exp(-0.5 * ((x_grid - value) / fallback_sigma) ** 2)
    return density / (fallback_sigma * np.sqrt(2 * np.pi))


def _prepare_cell_type_composition_plot_data(
    data_df: pd.DataFrame,
    value_col: str = "Cell_NTScore",
    cell_types: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    show_categories: Optional[List[str]] = None,
    default_order_by: str = "abundance",
    max_cell_types: int = 100,
    plot_name: str = "cell type composition plot",
) -> Optional[Tuple[pd.DataFrame, List[str], List[str]]]:
    """Validate wide cell-type composition data and choose plotted cell types."""

    if value_col not in data_df.columns:
        warning(f"No `{value_col}` column found. Skip {plot_name}.")
        return None

    candidate_cell_types = (
        [cell_type for cell_type in cell_types if cell_type != value_col]
        if cell_types is not None
        else [column for column in data_df.columns if column != value_col]
    )
    candidate_cell_types = [cell_type for cell_type in candidate_cell_types if cell_type in data_df.columns]
    if len(candidate_cell_types) == 0:
        warning(f"No cell type composition columns found. Skip {plot_name}.")
        return None

    composition_df = data_df[candidate_cell_types].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    data = composition_df.copy()
    data[value_col] = pd.to_numeric(data_df[value_col], errors="coerce")
    data = data.dropna(subset=[value_col])
    if data.empty:
        warning(f"No valid NT score data found. Skip {plot_name}.")
        return None

    valid_cell_types = [cell_type for cell_type in candidate_cell_types if data[cell_type].sum() > 0]
    if len(valid_cell_types) == 0:
        warning(f"No positive cell type composition values found. Skip {plot_name}.")
        return None

    if show_categories is None:
        shown_cell_types = valid_cell_types
    else:
        shown_cell_types = [cell_type for cell_type in show_categories if cell_type in valid_cell_types]

    if order is None and default_order_by == "mean_nt_score":
        shown_cell_types = sorted(
            shown_cell_types,
            key=lambda cell_type: np.average(data[value_col], weights=data[cell_type]),
        )
    elif order is None:
        shown_cell_types = sorted(shown_cell_types, key=lambda cell_type: data[cell_type].sum(), reverse=True)
    else:
        shown_cell_types = [cell_type for cell_type in order if cell_type in shown_cell_types]

    if len(shown_cell_types) == 0:
        warning(f"No cell types to show. Skip {plot_name}.")
        return None
    if len(shown_cell_types) > max_cell_types:
        warning(f"More than {max_cell_types} cell types detected; skipping {plot_name} to avoid long runtime.")
        return None

    return data[[value_col] + valid_cell_types], valid_cell_types, shown_cell_types


def _prepare_cell_type_composition_nt_score_data(
    ana_data: AnaData,
    cell_types: Optional[List[str]] = None,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """Build wide cell-type composition plus ``Niche_NTScore`` data from ``AnaData``."""

    if ana_data.NT_score is None:
        warning("No NT score data found. Skip cell type composition along NT score visualization.")
        return None
    if "Cell_NTScore" not in ana_data.NT_score.columns:
        warning("No `Cell_NTScore` column found. Skip cell type composition along NT score visualization.")
        return None

    try:
        raw_cell_type_composition = ana_data.raw_cell_type_composition
        cell_type_codes = ana_data.cell_type_codes
    except FileNotFoundError as e:
        warning(str(e))
        return None

    if raw_cell_type_composition is None:
        warning("No raw cell type composition data found. Skip cell type composition along NT score visualization.")
        return None
    if cell_type_codes is None or "Cell_Type" not in cell_type_codes.columns:
        warning("No cell type codes found. Skip cell type composition along NT score visualization.")
        return None

    default_cell_types = cell_type_codes["Cell_Type"].tolist()
    selected_cell_types = default_cell_types if cell_types is None else cell_types
    selected_cell_types = [cell_type for cell_type in selected_cell_types if cell_type in raw_cell_type_composition]
    if len(selected_cell_types) == 0:
        warning("No valid cell type composition columns found. Skip cell type composition along NT score visualization.")
        return None

    data_df = raw_cell_type_composition[selected_cell_types].copy()
    cell_nt_score = ana_data.NT_score["Cell_NTScore"]
    if getattr(ana_data.options, "reverse", False):
        cell_nt_score = 1 - cell_nt_score
    cell_nt_score = cell_nt_score.rename("Niche_NTScore")
    data_df = data_df.join(cell_nt_score)
    return data_df, selected_cell_types


def _calculate_cell_type_composition_density(
    data_df: pd.DataFrame,
    value_col: str,
    all_cell_types: List[str],
    shown_cell_types: List[str],
    bw_method: Any,
    grid_size: int,
    renormalize_shown: bool,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate weighted NT-score density and cell-type composition density."""

    values = data_df[value_col].to_numpy(dtype=float)
    x_min = data_df[value_col].min()
    x_max = data_df[value_col].max()
    x_padding = 0.03 * (x_max - x_min) if x_max > x_min else 1
    x_grid = np.linspace(x_min - x_padding, x_max + x_padding, grid_size)
    fallback_sigma = (x_grid.max() - x_grid.min()) / 100

    category_density_all = {}
    for cell_type in all_cell_types:
        category_density_all[cell_type] = _count_weighted_kde(
            values=values,
            weights=data_df[cell_type].to_numpy(dtype=float),
            x_grid=x_grid,
            bw_method=bw_method,
            fallback_sigma=fallback_sigma,
        )

    shown_density_matrix = np.vstack([category_density_all[cell_type] for cell_type in shown_cell_types])
    if renormalize_shown:
        denominator = shown_density_matrix.sum(axis=0)
    else:
        all_density_matrix = np.vstack([category_density_all[cell_type] for cell_type in all_cell_types])
        denominator = all_density_matrix.sum(axis=0)

    composition_matrix = shown_density_matrix / (denominator[None, :] + eps)
    all_weights = data_df[all_cell_types].sum(axis=1).to_numpy(dtype=float)
    density_all = _count_weighted_kde(
        values=values,
        weights=all_weights,
        x_grid=x_grid,
        bw_method=bw_method,
        fallback_sigma=fallback_sigma,
    )
    return x_grid, density_all, composition_matrix


def _cell_type_composition_legend_layout(
    n_cell_types: int,
    base_figsize: Tuple[float, float],
    figsize: Optional[Tuple[float, float]] = None,
    max_legend_rows: int = 12,
) -> Tuple[int, Tuple[float, float]]:
    """Return legend columns and a figure size large enough for the legend."""

    legend_ncol = max(1, int(np.ceil(n_cell_types / max_legend_rows)))
    legend_rows = max(1, int(np.ceil(n_cell_types / legend_ncol)))
    min_figsize = (
        base_figsize[0] + max(0, legend_ncol - 1) * 1.6,
        max(base_figsize[1], 1.0 + legend_rows * 0.25),
    )
    if figsize is None:
        return legend_ncol, min_figsize
    return legend_ncol, (max(figsize[0], min_figsize[0]), max(figsize[1], min_figsize[1]))


def _cell_type_composition_entity_labels(ana_data: AnaData) -> Tuple[str, str]:
    """Return top-density and count-axis labels for AnaData resolution."""

    if ana_data.meta_data_df.index.name == 'Cell_ID':
        return "All cells", "Number of cells"
    elif ana_data.meta_data_df.index.name == 'Spot_ID':
        return "All spots", "Number of spots"
    else:
        warning("Unrecognized metadata index name. Defaulting to 'All cells' and 'Number of cells' for cell type composition plot labels.")
        return "All cells", "Number of cells"


def plot_violin_cell_type_composition_along_NT_score(
    data_df: pd.DataFrame,
    value_col: str = "Niche_NTScore",
    cell_types: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    show_categories: Optional[List[str]] = None,
    renormalize_shown: bool = True,
    palette: Optional[Union[List[str], Dict[str, str]]] = None,
    category_name: str = "Cell Type",
    value_name: str = "Niche-level NT score",
    count_name: str = "Number of cells",
    top_density_name: str = "All cells",
    bw_method: Any = None,
    grid_size: int = 400,
    max_violin_width: float = 0.35,
    show_fill: bool = False,
    fill_alpha: float = 0.15,
    linewidth: float = 1.2,
    top_color: str = "0.3",
    figsize: Optional[Tuple[float, float]] = None,
    show_grid: bool = False,
    eps: float = 1e-12,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """Plot cell type composition along NT score as count-weighted composition violins.

    Parameters
    ----------
    data_df :
        pd.DataFrame, wide cell type composition data with an NT-score column.
    value_col :
        str, column storing spatial trajectory values (Niche_NTScore).
    cell_types :
        Optional[List[str]], cell type composition columns to use.
    order :
        Optional[List[str]], cell type order to show. Defaults to ascending
        weighted mean NT score.
    show_categories :
        Optional[List[str]], subset of cell types to show in the composition and count panels.
    renormalize_shown :
        bool, whether to normalize composition among shown cell types only.
    palette :
        Optional[List[str] or Dict[str, str]], cell type color palette.
    category_name :
        str, y-axis label for cell type categories.
    value_name :
        str, x-axis label for NT score values.
    count_name :
        str, x-axis label for the count panel.
    top_density_name :
        str, y-axis label for the all-cell density panel.
    bw_method :
        Optional, bandwidth method passed to ``scipy.stats.gaussian_kde``.
    grid_size :
        int, number of NT-score grid points used for KDE evaluation.
    max_violin_width :
        float, maximum half-width of each composition violin.
    show_fill :
        bool, whether to fill each composition violin.
    fill_alpha :
        float, fill transparency when ``show_fill`` is true.
    linewidth :
        float, line width for composition violin outlines.
    top_color :
        str, color for the all-cell density panel.
    figsize :
        Optional[Tuple[float, float]], figure size.
    show_grid :
        bool, whether to show vertical guide grids.
    eps :
        float, small denominator offset for stable composition calculation.
    output_file_path :
        Optional[Union[str, Path]], the output directory.

    Returns
    -------
    None or Tuple[plt.Figure, List[plt.Axes]]."""

    prepared_data = _prepare_cell_type_composition_plot_data(
        data_df=data_df,
        value_col=value_col,
        cell_types=cell_types,
        order=order,
        show_categories=show_categories,
        default_order_by="mean_nt_score",
        plot_name="cell type composition violin plot",
    )
    if prepared_data is None:
        return None
    data, all_cell_types, shown_cell_types = prepared_data

    n_categories = len(shown_cell_types)
    y_positions = np.arange(n_categories)
    x_grid, density_all, composition_matrix = _calculate_cell_type_composition_density(
        data_df=data,
        value_col=value_col,
        all_cell_types=all_cell_types,
        shown_cell_types=shown_cell_types,
        bw_method=bw_method,
        grid_size=grid_size,
        renormalize_shown=renormalize_shown,
        eps=eps,
    )
    validated_palette = validate_cell_type_palette(cell_types=shown_cell_types, palette=palette)

    if figsize is None:
        figsize = (10, max(5, 0.42 * n_categories + 1.5))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[5, 1.3],
        height_ratios=[1.2, max(3, n_categories * 0.45)],
        wspace=0.06,
        hspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_violin = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_bar = fig.add_subplot(gs[1, 1], sharey=ax_violin)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    ax_top.fill_between(x_grid, 0, density_all, color=top_color, alpha=0.2, linewidth=0)
    ax_top.plot(x_grid, density_all, color=top_color, linewidth=1.5)
    ax_top.set_ylabel(top_density_name)
    ax_top.tick_params(axis="x", labelbottom=False)
    if show_grid:
        ax_top.grid(axis="x", linestyle=":", color="0.7", linewidth=0.8)
    else:
        ax_top.grid(False)

    for i, cell_type in enumerate(shown_cell_types):
        color = validated_palette[cell_type]
        width = composition_matrix[i, :] * max_violin_width
        y_lower = i - width
        y_upper = i + width
        ax_violin.hlines(
            y=i,
            xmin=x_grid.min(),
            xmax=x_grid.max(),
            color=color,
            linewidth=0.8,
            alpha=0.8,
        )
        if show_fill:
            ax_violin.fill_between(x_grid, y_lower, y_upper, color=color, alpha=fill_alpha, linewidth=0)
        ax_violin.plot(x_grid, y_upper, color=color, linewidth=linewidth)
        ax_violin.plot(x_grid, y_lower, color=color, linewidth=linewidth)

    ax_violin.set_yticks(y_positions)
    ax_violin.set_yticklabels(shown_cell_types)
    ax_violin.tick_params(axis="y", labelleft=True, left=True)
    ax_violin.set_xlabel(value_name)
    ax_violin.set_ylabel(category_name)
    ax_violin.set_ylim(n_categories - 0.5, -0.5)
    if show_grid:
        ax_violin.grid(axis="x", linestyle=":", color="0.7", linewidth=0.8)
    else:
        ax_violin.grid(False)

    count_values = data[shown_cell_types].sum(axis=0).to_numpy()
    ax_bar.barh(
        y_positions,
        count_values,
        color=[validated_palette[cell_type] for cell_type in shown_cell_types],
        height=0.6,
    )
    ax_bar.set_xlabel(count_name)
    ax_bar.tick_params(axis="y", labelleft=False, left=False)
    ax_bar.set_ylim(n_categories - 0.5, -0.5)

    sns.despine(ax=ax_top)
    sns.despine(ax=ax_violin)
    sns.despine(ax=ax_bar, left=True)

    if output_file_path is not None:
        fig.savefig(f"{output_file_path}/cell_type_composition_along_NT_score_violin.pdf", transparent=True)
        plt.close(fig)
        return None
    return fig, [ax_top, ax_violin, ax_bar]


def plot_violin_cell_type_composition_along_NT_score_from_anadata(
    ana_data: AnaData,
    cell_types: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    show_categories: Optional[List[str]] = None,
    renormalize_shown: bool = True,
    palette: Optional[Union[List[str], Dict[str, str]]] = None,
    category_name: str = "Cell Type",
    value_name: str = "Niche-level NT score",
    count_name: Optional[str] = None,
    top_density_name: Optional[str] = None,
    bw_method: Any = None,
    grid_size: int = 400,
    max_violin_width: float = 0.35,
    show_fill: bool = False,
    fill_alpha: float = 0.15,
    linewidth: float = 1.2,
    top_color: str = "0.3",
    figsize: Optional[Tuple[float, float]] = None,
    show_grid: bool = False,
    eps: float = 1e-12,
) -> Optional[Tuple[plt.Figure, List[plt.Axes]]]:
    """Plot cell type composition along NT score as count-weighted composition violins.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis. Cell type composition is read from
        ``raw_cell_type_composition`` and NT scores are read from ``Cell_NTScore``.
    cell_types :
        Optional[List[str]], cell type composition columns to use.
    order :
        Optional[List[str]], cell type order to show. Defaults to ascending
        weighted mean NT score.
    show_categories :
        Optional[List[str]], subset of cell types to show in the composition and count panels.
    renormalize_shown :
        bool, whether to normalize composition among shown cell types only.
    palette :
        Optional[List[str] or Dict[str, str]], cell type color palette.
    category_name :
        str, y-axis label for cell type categories.
    value_name :
        str, x-axis label for NT score values.
    count_name :
        Optional[str], x-axis label for the count panel. If None, defaults to
        ``Number of spots`` for spot-level data and ``Number of cells`` otherwise.
    top_density_name :
        Optional[str], y-axis label for the aggregate density panel. If None,
        defaults to ``All spots`` for spot-level data and ``All cells`` otherwise.
    bw_method :
        Optional, bandwidth method passed to ``scipy.stats.gaussian_kde``.
    grid_size :
        int, number of NT-score grid points used for KDE evaluation.
    max_violin_width :
        float, maximum half-width of each composition violin.
    show_fill :
        bool, whether to fill each composition violin.
    fill_alpha :
        float, fill transparency when ``show_fill`` is true.
    linewidth :
        float, line width for composition violin outlines.
    top_color :
        str, color for the all-cell density panel.
    figsize :
        Optional[Tuple[float, float]], figure size.
    show_grid :
        bool, whether to show vertical guide grids.
    eps :
        float, small denominator offset for stable composition calculation.

    Returns
    -------
    None or Tuple[plt.Figure, List[plt.Axes]]."""

    prepared_data = _prepare_cell_type_composition_nt_score_data(ana_data=ana_data, cell_types=cell_types)
    if prepared_data is None:
        return None
    data_df, cell_types = prepared_data
    default_top_density_name, default_count_name = _cell_type_composition_entity_labels(ana_data=ana_data)
    if count_name is None:
        count_name = default_count_name
    if top_density_name is None:
        top_density_name = default_top_density_name

    return plot_violin_cell_type_composition_along_NT_score(
        data_df=data_df,
        value_col="Niche_NTScore",
        cell_types=cell_types,
        order=order,
        show_categories=show_categories,
        renormalize_shown=renormalize_shown,
        palette=palette,
        category_name=category_name,
        value_name=value_name,
        count_name=count_name,
        top_density_name=top_density_name,
        bw_method=bw_method,
        grid_size=grid_size,
        max_violin_width=max_violin_width,
        show_fill=show_fill,
        fill_alpha=fill_alpha,
        linewidth=linewidth,
        top_color=top_color,
        figsize=figsize,
        show_grid=show_grid,
        eps=eps,
        output_file_path=ana_data.options.output,
    )


def plot_kde_cell_type_composition_along_NT_score(
    data_df: pd.DataFrame,
    value_col: str = "Niche_NTScore",
    cell_types: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    show_categories: Optional[List[str]] = None,
    renormalize_shown: bool = True,
    palette: Optional[Union[List[str], Dict[str, str]]] = None,
    bw_method: Any = None,
    grid_size: int = 400,
    figsize: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
    output_file_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot weighted cell type composition KDE along NT score.

    Parameters
    ----------
    data_df :
        pd.DataFrame, wide cell type composition data with an NT-score column.
    value_col :
        str, column storing spatial trajectory values (Niche_NTScore).
    cell_types :
        Optional[List[str]], cell type composition columns to use.
    order :
        Optional[List[str]], cell type order to show. Defaults to descending abundance.
    show_categories :
        Optional[List[str]], subset of cell types to show.
    renormalize_shown :
        bool, whether to normalize composition among shown cell types only.
    palette :
        Optional[List[str] or Dict[str, str]], cell type color palette.
    bw_method :
        Optional, bandwidth method passed to ``scipy.stats.gaussian_kde``.
    grid_size :
        int, number of NT-score grid points used for KDE evaluation.
    figsize :
        Optional[Tuple[float, float]], figure size. If None, automatically sized
        to fit the legend.
    eps :
        float, small denominator offset for stable composition calculation.
    output_file_path :
        Optional[Union[str, Path]], the output directory.
    kwargs :
        Optional, additional arguments passed to ``matplotlib.axes.Axes.stackplot``.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    prepared_data = _prepare_cell_type_composition_plot_data(
        data_df=data_df,
        value_col=value_col,
        cell_types=cell_types,
        order=order,
        show_categories=show_categories,
        plot_name="cell type composition KDE plot",
    )
    if prepared_data is None:
        return None
    data, all_cell_types, shown_cell_types = prepared_data

    x_grid, _, composition_matrix = _calculate_cell_type_composition_density(
        data_df=data,
        value_col=value_col,
        all_cell_types=all_cell_types,
        shown_cell_types=shown_cell_types,
        bw_method=bw_method,
        grid_size=grid_size,
        renormalize_shown=renormalize_shown,
        eps=eps,
    )
    validated_palette = validate_cell_type_palette(cell_types=shown_cell_types, palette=palette)

    legend_ncol, figsize = _cell_type_composition_legend_layout(
        n_cell_types=len(shown_cell_types),
        base_figsize=(8, 4),
        figsize=figsize,
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(
        x_grid,
        composition_matrix,
        colors=[validated_palette[cell_type] for cell_type in shown_cell_types],
        labels=shown_cell_types,
        **kwargs,
    )
    ax.set_xlabel("Cell-level NT score")
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=legend_ncol)
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(
            f"{output_file_path}/cell_type_composition_along_NT_score_kde.pdf",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_kde_cell_type_composition_along_NT_score_from_anadata(
    ana_data: AnaData,
    cell_types: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    show_categories: Optional[List[str]] = None,
    renormalize_shown: bool = True,
    palette: Optional[Union[List[str], Dict[str, str]]] = None,
    bw_method: Any = None,
    grid_size: int = 400,
    figsize: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
    **kwargs,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot weighted cell type composition KDE along NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis. Cell type composition is read from
        ``raw_cell_type_composition`` and NT scores are read from ``Cell_NTScore``.
    cell_types :
        Optional[List[str]], cell type composition columns to use.
    order :
        Optional[List[str]], cell type order to show. Defaults to descending abundance.
    show_categories :
        Optional[List[str]], subset of cell types to show.
    renormalize_shown :
        bool, whether to normalize composition among shown cell types only.
    palette :
        Optional[List[str] or Dict[str, str]], cell type color palette.
    bw_method :
        Optional, bandwidth method passed to ``scipy.stats.gaussian_kde``.
    grid_size :
        int, number of NT-score grid points used for KDE evaluation.
    figsize :
        Optional[Tuple[float, float]], figure size. If None, automatically sized
        to fit the legend.
    eps :
        float, small denominator offset for stable composition calculation.
    kwargs :
        Optional, additional arguments passed to ``matplotlib.axes.Axes.stackplot``.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    prepared_data = _prepare_cell_type_composition_nt_score_data(ana_data=ana_data, cell_types=cell_types)
    if prepared_data is None:
        return None
    data_df, cell_types = prepared_data

    return plot_kde_cell_type_composition_along_NT_score(
        data_df=data_df,
        value_col="Niche_NTScore",
        cell_types=cell_types,
        order=order,
        show_categories=show_categories,
        renormalize_shown=renormalize_shown,
        palette=palette,
        bw_method=bw_method,
        grid_size=grid_size,
        figsize=figsize,
        eps=eps,
        output_file_path=ana_data.options.output,
        **kwargs,
    )


def plot_hist_cell_type_composition_along_NT_score(
    data_df: pd.DataFrame,
    value_col: str = "Niche_NTScore",
    cell_types: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    show_categories: Optional[List[str]] = None,
    renormalize_shown: bool = True,
    palette: Optional[Union[List[str], Dict[str, str]]] = None,
    bins: Any = 20,
    figsize: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
    output_file_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot weighted cell type composition histogram along NT score.

    Parameters
    ----------
    data_df :
        pd.DataFrame, wide cell type composition data with an NT-score column.
    value_col :
        str, column storing spatial trajectories for each niche (Niche_NTScore).
    cell_types :
        Optional[List[str]], cell type composition columns to use.
    order :
        Optional[List[str]], cell type order to show. Defaults to descending abundance.
    show_categories :
        Optional[List[str]], subset of cell types to show.
    renormalize_shown :
        bool, whether to normalize composition among shown cell types only.
    palette :
        Optional[List[str] or Dict[str, str]], cell type color palette.
    bins :
        Histogram bins passed to ``numpy.histogram_bin_edges``.
    figsize :
        Optional[Tuple[float, float]], figure size. If None, automatically sized
        to fit the legend.
    eps :
        float, small denominator offset for stable composition calculation.
    output_file_path :
        Optional[Union[str, Path]], the output directory.
    kwargs :
        Optional, additional arguments passed to ``matplotlib.axes.Axes.bar``.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    prepared_data = _prepare_cell_type_composition_plot_data(
        data_df=data_df,
        value_col=value_col,
        cell_types=cell_types,
        order=order,
        show_categories=show_categories,
        plot_name="cell type composition histogram",
    )
    if prepared_data is None:
        return None
    data, all_cell_types, shown_cell_types = prepared_data

    values = data[value_col].to_numpy(dtype=float)
    bin_edges = np.histogram_bin_edges(values, bins=bins)
    if len(bin_edges) < 2:
        warning("Cannot build NT-score histogram bins. Skip cell type composition histogram.")
        return None
    n_bins = len(bin_edges) - 1
    bin_indices = np.digitize(values, bin_edges[1:-1], right=False)

    shown_weight_matrix = np.vstack(
        [
            np.bincount(bin_indices, weights=data[cell_type].to_numpy(dtype=float), minlength=n_bins)
            for cell_type in shown_cell_types
        ]
    )
    if renormalize_shown:
        denominator = shown_weight_matrix.sum(axis=0)
    else:
        all_weight_matrix = np.vstack(
            [
                np.bincount(bin_indices, weights=data[cell_type].to_numpy(dtype=float), minlength=n_bins)
                for cell_type in all_cell_types
            ]
        )
        denominator = all_weight_matrix.sum(axis=0)
    hist_matrix = shown_weight_matrix / (denominator[None, :] + eps)

    validated_palette = validate_cell_type_palette(cell_types=shown_cell_types, palette=palette)
    base_figsize = (max(8, len(shown_cell_types) / 2), 4)
    legend_ncol, figsize = _cell_type_composition_legend_layout(
        n_cell_types=len(shown_cell_types),
        base_figsize=base_figsize,
        figsize=figsize,
    )

    fig, ax = plt.subplots(figsize=figsize)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2
    bottom = np.zeros(n_bins)
    for i, cell_type in enumerate(shown_cell_types):
        ax.bar(
            bin_centers,
            hist_matrix[i],
            width=bin_widths * 0.95,
            bottom=bottom,
            color=validated_palette[cell_type],
            align="center",
            label=cell_type,
            **kwargs,
        )
        bottom += hist_matrix[i]
    ax.set_xlabel("Cell-level NT score")
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=legend_ncol)
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(
            f"{output_file_path}/cell_type_composition_along_NT_score_hist.pdf",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_hist_cell_type_composition_along_NT_score_from_anadata(
    ana_data: AnaData,
    cell_types: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    show_categories: Optional[List[str]] = None,
    renormalize_shown: bool = True,
    palette: Optional[Union[List[str], Dict[str, str]]] = None,
    bins: Any = 20,
    figsize: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
    **kwargs,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot weighted cell type composition histogram along NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis. Cell type composition is read from
        ``raw_cell_type_composition`` and NT scores are read from ``Cell_NTScore``.
    cell_types :
        Optional[List[str]], cell type composition columns to use.
    order :
        Optional[List[str]], cell type order to show. Defaults to descending abundance.
    show_categories :
        Optional[List[str]], subset of cell types to show.
    renormalize_shown :
        bool, whether to normalize composition among shown cell types only.
    palette :
        Optional[List[str] or Dict[str, str]], cell type color palette.
    bins :
        Histogram bins passed to ``numpy.histogram_bin_edges``.
    figsize :
        Optional[Tuple[float, float]], figure size. If None, automatically sized
        to fit the legend.
    eps :
        float, small denominator offset for stable composition calculation.
    kwargs :
        Optional, additional arguments passed to ``matplotlib.axes.Axes.bar``.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    prepared_data = _prepare_cell_type_composition_nt_score_data(ana_data=ana_data, cell_types=cell_types)
    if prepared_data is None:
        return None
    data_df, cell_types = prepared_data

    return plot_hist_cell_type_composition_along_NT_score(
        data_df=data_df,
        value_col="Niche_NTScore",
        cell_types=cell_types,
        order=order,
        show_categories=show_categories,
        renormalize_shown=renormalize_shown,
        palette=palette,
        bins=bins,
        figsize=figsize,
        eps=eps,
        output_file_path=ana_data.options.output,
        **kwargs,
    )


def plot_cell_type_along_NT_score(ana_data: AnaData) -> None:
    """Plot all visualization of cell type along NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None."""

    prepared_data = _prepare_cell_type_composition_nt_score_data(ana_data=ana_data)
    if prepared_data is None:
        return None
    data_df, cell_types = prepared_data
    top_density_name, count_name = _cell_type_composition_entity_labels(ana_data=ana_data)

    plot_violin_cell_type_composition_along_NT_score(
        data_df=data_df,
        cell_types=cell_types,
        count_name=count_name,
        top_density_name=top_density_name,
        output_file_path=ana_data.options.output,
    )
    plot_kde_cell_type_composition_along_NT_score(
        data_df=data_df,
        cell_types=cell_types,
        output_file_path=ana_data.options.output,
    )
    plot_hist_cell_type_composition_along_NT_score(
        data_df=data_df,
        cell_types=cell_types,
        output_file_path=ana_data.options.output,
    )


def plot_cell_type_loading_in_niche_clusters(
    cell_type_dis_df: pd.DataFrame, output_file_path: Optional[Union[str, Path]] = None
) -> Optional[sns.FacetGrid]:
    """Plot cell type loading in each niche cluster.

    Parameters
    ----------
    cell_type_dis_df :
        pd.DataFrame, the cell type distribution in each niche cluster.
    output_file_path :
        Optional[Union[str, Path]], the output directory.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if cell_type_dis_df.shape[0] > 50:
        warning(
            "More than 50 cell types detected; skipping niche-cluster loading plot."
        )
        return None

    data_df = deepcopy(cell_type_dis_df)
    cell_type = data_df.columns
    data_df["cluster"] = data_df.index
    cell_type_dis_melt_df = pd.melt(
        data_df,
        id_vars="cluster",  # type: ignore
        var_name="Cell type",
        value_vars=cell_type,  # type: ignore
        value_name="Number",
    )
    # g = sns.catplot(..., kind="bar", x="Number", y="Cell type", col="cluster", ...)
    g = sns.catplot(
        cell_type_dis_melt_df,
        kind="bar",
        x="Number",
        y="Cell type",
        col="cluster",
        height=2 + len(cell_type) / 6,
        aspect=0.5,
    )  # type: ignore
    g.add_legend()
    g.tight_layout()
    g.set_xticklabels(rotation="vertical")
    if output_file_path is not None:
        g.savefig(f"{output_file_path}/cell_type_loading_in_niche_clusters.pdf", transparent=True)
        return None
    else:
        return g


def plot_cell_type_loading_in_niche_clusters_from_anadata(ana_data: AnaData) -> Optional[sns.FacetGrid]:
    """Plot cell type loading in each niche cluster.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if ana_data.cell_level_niche_cluster_assign is None:
        warning("No niche cluster assign data found. Skip cell type loading in niche clusters.")
        return None
    if ana_data.cell_type_codes is None:
        warning("No cell type codes found. Skip cell type loading in niche clusters.")
        return None

    # calculate cell type distribution in each niche cluster
    data_df = ana_data.meta_data_df.join(ana_data.cell_level_niche_cluster_assign)
    t = pd.CategoricalDtype(categories=ana_data.cell_type_codes["Cell_Type"], ordered=True)
    cell_type_one_hot = np.zeros(shape=(data_df.shape[0], ana_data.cell_type_codes.shape[0]))
    cell_type = data_df["Cell_Type"].astype(t)
    cell_type_one_hot[np.arange(data_df.shape[0]), cell_type.cat.codes] = 1  # N x n_cell_type
    cell_type_dis = np.matmul(
        data_df[ana_data.cell_level_niche_cluster_assign.columns].T, cell_type_one_hot
    )  # n_clusters x n_cell_types
    cell_type_dis_df = pd.DataFrame(cell_type_dis)
    cell_type_dis_df.columns = ana_data.cell_type_codes["Cell_Type"]
    if ana_data.options.output is not None:
        cell_type_dis_df.to_csv(f"{ana_data.options.output}/cell_type_dis_in_niche_clusters.csv", index=False)

    # nc_order
    nc_scores = cal_nc_scores(
        cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
        reverse=ana_data.options.reverse,
        niche_cluster_score=ana_data.niche_cluster_score,
    )
    nc_order = [f"NicheCluster_{x}" for x in nc_scores.argsort()]
    cell_type_dis_df = cell_type_dis_df.loc[nc_order]

    return plot_cell_type_loading_in_niche_clusters(
        cell_type_dis_df=cell_type_dis_df, output_file_path=ana_data.options.output
    )


def plot_cell_type_com_in_niche_clusters(
    cell_type_dis_df: pd.DataFrame, output_file_path: Optional[Union[str, Path]] = None
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot cell type composition in each niche cluster.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.
    cell_type_dis_df :
        pd.DataFrame, the cell type distribution in each niche cluster.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]"""

    fig, ax = plt.subplots(figsize=(2 + cell_type_dis_df.shape[1] / 3, 1 + cell_type_dis_df.shape[0] / 5))
    sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=1), ax=ax)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Niche Cluster")
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f"{output_file_path}/cell_type_composition_in_niche_clusters.pdf", transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_cell_type_com_in_niche_clusters_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot cell type composition in each niche cluster.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if ana_data.cell_level_niche_cluster_assign is None:
        warning("No niche cluster assign data found. Skip cell type composition in niche clusters visualization.")
        return None
    if ana_data.cell_type_codes is None:
        warning("No cell type codes found. Skip cell type composition in niche clusters visualization.")
        return None

    # calculate cell type distribution in each niche cluster
    data_df = ana_data.meta_data_df.join(ana_data.cell_level_niche_cluster_assign)
    t = pd.CategoricalDtype(categories=ana_data.cell_type_codes["Cell_Type"], ordered=True)
    cell_type_one_hot = np.zeros(shape=(data_df.shape[0], ana_data.cell_type_codes.shape[0]))
    cell_type = data_df["Cell_Type"].astype(t)
    cell_type_one_hot[np.arange(data_df.shape[0]), cell_type.cat.codes] = 1  # N x n_cell_type
    cell_type_dis = np.matmul(
        data_df[ana_data.cell_level_niche_cluster_assign.columns].T, cell_type_one_hot
    )  # n_clusters x n_cell_types
    cell_type_dis_df = pd.DataFrame(cell_type_dis)
    cell_type_dis_df.columns = ana_data.cell_type_codes["Cell_Type"]
    if ana_data.options.output is not None:
        cell_type_dis_df.to_csv(f"{ana_data.options.output}/cell_type_dis_in_niche_clusters.csv", index=False)

    # nc_order
    nc_scores = cal_nc_scores(
        cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
        reverse=ana_data.options.reverse,
        niche_cluster_score=ana_data.niche_cluster_score,
    )
    nc_order = [f"NicheCluster_{x.split()[-1]}" for x in cal_nc_order(cal_nc_order_index(nc_scores))]
    # cell type distribution dataframe
    cell_type_dis_df = cell_type_dis_df.loc[nc_order]

    return plot_cell_type_com_in_niche_clusters(
        cell_type_dis_df=cell_type_dis_df, output_file_path=ana_data.options.output
    )


def plot_cell_type_dis_across_niche_cluster(
    cell_type_dis_df: pd.DataFrame, output_file_path: Optional[Union[str, Path]] = None
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot cell type distribution across niche cluster.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.
    cell_type_dis_df :
        pd.DataFrame, the cell type distribution in each niche cluster.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]"""

    fig, ax = plt.subplots(figsize=(2 + cell_type_dis_df.shape[1] / 3, 1 + cell_type_dis_df.shape[0] / 5))
    sns.heatmap(cell_type_dis_df.apply(lambda x: x / x.sum(), axis=0), ax=ax)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Niche Cluster")
    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f"{output_file_path}/cell_type_dis_across_niche_cluster.pdf", transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_cell_type_dis_across_niche_cluster_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot cell type distribution across niche cluster.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if ana_data.cell_level_niche_cluster_assign is None:
        warning("No niche cluster assign data found. Skip cell type distribution across niche cluster.")
        return None
    if ana_data.cell_type_codes is None:
        warning("No cell type codes found. Skip cell type distribution across niche cluster.")
        return None

    # calculate cell type distribution in each niche cluster
    data_df = ana_data.meta_data_df.join(ana_data.cell_level_niche_cluster_assign)
    t = pd.CategoricalDtype(categories=ana_data.cell_type_codes["Cell_Type"], ordered=True)
    cell_type_one_hot = np.zeros(shape=(data_df.shape[0], ana_data.cell_type_codes.shape[0]))
    cell_type = data_df["Cell_Type"].astype(t)
    cell_type_one_hot[np.arange(data_df.shape[0]), cell_type.cat.codes] = 1  # N x n_cell_type
    cell_type_dis = np.matmul(
        data_df[ana_data.cell_level_niche_cluster_assign.columns].T, cell_type_one_hot
    )  # n_clusters x n_cell_types
    cell_type_dis_df = pd.DataFrame(cell_type_dis)
    cell_type_dis_df.columns = ana_data.cell_type_codes["Cell_Type"]
    if ana_data.options.output is not None:
        cell_type_dis_df.to_csv(f"{ana_data.options.output}/cell_type_dis_in_niche_clusters.csv", index=False)

    # nc_order
    nc_scores = cal_nc_scores(
        cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
        reverse=ana_data.options.reverse,
        niche_cluster_score=ana_data.niche_cluster_score,
    )
    nc_order = [f"NicheCluster_{x.split()[-1]}" for x in cal_nc_order(cal_nc_order_index(nc_scores))]
    # cell type distribution dataframe
    cell_type_dis_df = cell_type_dis_df.loc[nc_order]

    return plot_cell_type_dis_across_niche_cluster(
        cell_type_dis_df=cell_type_dis_df, output_file_path=ana_data.options.output
    )


def plot_cell_type_with_niche_cluster(ana_data: AnaData) -> None:
    """Plot all visualization of cell type in each niche cluster.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None."""

    if ana_data.cell_level_niche_cluster_assign is None:
        warning("No niche cluster assign data found. Skip cell type with niche cluster visualization.")
        return None
    if ana_data.cell_type_codes is None:
        warning("No cell type codes found. Skip cell type with niche cluster visualization.")
        return None

    # calculate cell type distribution in each niche cluster
    data_df = ana_data.meta_data_df.join(ana_data.cell_level_niche_cluster_assign)
    cell_type_dis = np.matmul(
        data_df[ana_data.cell_level_niche_cluster_assign.columns].T, ana_data.cell_type_coding
    )  # n_clusters x n_cell_types
    cell_type_dis_df = pd.DataFrame(cell_type_dis)
    cell_type_dis_df.columns = ana_data.cell_type_codes["Cell_Type"]
    if ana_data.options.output is not None:
        cell_type_dis_df.to_csv(f"{ana_data.options.output}/cell_type_dis_in_niche_clusters.csv", index=False)

    # nc_order
    nc_scores = cal_nc_scores(
        cell_level_niche_cluster_assign=ana_data.cell_level_niche_cluster_assign,
        reverse=ana_data.options.reverse,
        niche_cluster_score=ana_data.niche_cluster_score,
    )
    nc_order = [f"NicheCluster_{x.split()[-1]}" for x in cal_nc_order(cal_nc_order_index(nc_scores))]
    # cell type distribution dataframe
    cell_type_dis_df = cell_type_dis_df.loc[nc_order]

    plot_cell_type_loading_in_niche_clusters(
        cell_type_dis_df=cell_type_dis_df, output_file_path=ana_data.options.output
    )
    plot_cell_type_com_in_niche_clusters(cell_type_dis_df=cell_type_dis_df, output_file_path=ana_data.options.output)
    plot_cell_type_dis_across_niche_cluster(cell_type_dis_df=cell_type_dis_df, output_file_path=ana_data.options.output)


def cell_type_visualization(ana_data: AnaData) -> None:
    """Visualize cell type based output.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None."""

    # 1. cell type along NT score
    if getattr(ana_data.options, "suppress_niche_trajectory", False):
        warning("Skip cell type along NT score visualization due to `suppress_niche_trajectory` option.")
    else:
        plot_cell_type_along_NT_score(ana_data=ana_data)

    # 2. cell type X niche cluster
    plot_cell_type_with_niche_cluster(ana_data=ana_data)
