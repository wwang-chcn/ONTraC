"""This module contains functions for spatial-based analysis and visualization."""

from .utils import saptial_figsize
from .data import AnaData
from ..log import info, warning
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"


def _prepare_nt_score_sample_df(
    NT_score: pd.DataFrame,
    meta_data_df: pd.DataFrame,
    sample: str,
    score_column: str,
) -> pd.DataFrame:
    """Collect coordinates and a single NT-score column for one sample."""

    sample_index = meta_data_df.index[meta_data_df["Sample"] == sample]
    sample_df = meta_data_df.loc[sample_index, ["x", "y"]].join(NT_score.loc[sample_index, [score_column]])
    return sample_df.dropna(subset=["x", "y", score_column]).copy()


def _mask_sparse_triangles(triangulation: Triangulation, x: np.ndarray, y: np.ndarray) -> None:
    """Mask triangles that bridge large spatial gaps."""

    if triangulation.triangles.size == 0 or x.size < 4:
        return

    distances, _ = cKDTree(np.column_stack((x, y))).query(np.column_stack((x, y)), k=2)
    typical_spacing = float(np.nanmedian(distances[:, 1]))
    if not np.isfinite(typical_spacing) or typical_spacing <= 0:
        return

    triangles = triangulation.triangles
    x_tri = x[triangles]
    y_tri = y[triangles]
    edge_lengths = np.stack(
        (
            np.hypot(x_tri[:, 0] - x_tri[:, 1], y_tri[:, 0] - y_tri[:, 1]),
            np.hypot(x_tri[:, 1] - x_tri[:, 2], y_tri[:, 1] - y_tri[:, 2]),
            np.hypot(x_tri[:, 2] - x_tri[:, 0], y_tri[:, 2] - y_tri[:, 0]),
        ),
        axis=1,
    )
    triangle_mask = np.max(edge_lengths, axis=1) > typical_spacing * 6
    if np.any(triangle_mask):
        triangulation.set_mask(triangle_mask)


def _get_typical_spatial_spacing(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the local sampling spacing for an irregular spatial cloud."""

    if x.size < 2:
        return 1.0

    distances, _ = cKDTree(np.column_stack((x, y))).query(np.column_stack((x, y)), k=2)
    typical_spacing = float(np.nanmedian(distances[:, 1]))
    if not np.isfinite(typical_spacing) or typical_spacing <= 0:
        x_span = np.nanmax(x) - np.nanmin(x) if x.size > 0 else 1.0
        y_span = np.nanmax(y) - np.nanmin(y) if y.size > 0 else 1.0
        typical_spacing = max(float(max(x_span, y_span)) / 25.0, 1.0)
    return typical_spacing


def _estimate_nt_grid_shape(
    x: np.ndarray,
    y: np.ndarray,
    min_points: int = 40,
    max_points: int = 140,
) -> Tuple[int, int]:
    """Choose a regular grid resolution that follows the tissue aspect ratio."""

    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1.0)
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1.0)
    aspect = x_span / y_span
    base = int(np.clip(np.sqrt(max(x.size, 9)) * 3.0, min_points, max_points))
    if aspect >= 1:
        nx = base
        ny = int(np.clip(round(base / aspect), min_points // 2, max_points))
    else:
        ny = base
        nx = int(np.clip(round(base * aspect), min_points // 2, max_points))
    return max(nx, 20), max(ny, 20)


def _build_nt_score_vector_field(
    sample_df: pd.DataFrame,
    score_column: str,
    reverse: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Interpolate NT score to a regular grid and compute its spatial gradient."""

    if sample_df.empty:
        return None

    x = sample_df["x"].to_numpy(dtype=float)
    y = sample_df["y"].to_numpy(dtype=float)
    scores = sample_df[score_column].to_numpy(dtype=float)
    if reverse:
        scores = 1 - scores
    scores = np.clip(scores, 0, 1)

    if x.size < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None

    nx, ny = _estimate_nt_grid_shape(x=x, y=y)
    xi = np.linspace(np.nanmin(x), np.nanmax(x), nx)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), ny)
    grid_x, grid_y = np.meshgrid(xi, yi)
    points = np.column_stack((x, y))

    score_grid = griddata(points, scores, (grid_x, grid_y), method="linear")
    nearest_grid = griddata(points, scores, (grid_x, grid_y), method="nearest")
    if score_grid is None or nearest_grid is None:
        return None

    score_grid = np.where(np.isnan(score_grid), nearest_grid, score_grid)
    typical_spacing = _get_typical_spatial_spacing(x=x, y=y)
    grid_distances, _ = cKDTree(points).query(np.column_stack((grid_x.ravel(), grid_y.ravel())), k=1)
    valid_mask = grid_distances.reshape(grid_x.shape) <= typical_spacing * 3.5
    score_grid = np.ma.masked_where(~valid_mask, score_grid)
    if score_grid.count() == 0:
        return None

    filled_scores = score_grid.filled(np.nan)
    grad_y, grad_x = np.gradient(filled_scores, yi, xi)
    grad_x = np.ma.masked_invalid(np.ma.array(grad_x, mask=~valid_mask))
    grad_y = np.ma.masked_invalid(np.ma.array(grad_y, mask=~valid_mask))
    score_grid = np.ma.masked_invalid(score_grid)
    return grid_x, grid_y, score_grid, grad_x, grad_y, valid_mask


def _style_nt_score_axis(ax: Axes, score_title: str) -> None:
    """Apply shared styling for NT-score field plots."""

    ax.set_title(score_title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f7f7f7")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _normalize_foreground_layer(foreground_layer: Optional[str]) -> Optional[str]:
    """Normalize the front-layer selector."""

    if foreground_layer in (None, "none"):
        return None
    return foreground_layer


def _validate_nt_layers(background_layer: str, foreground_layer: Optional[str]) -> Optional[str]:
    """Validate supported NT-score layer combinations."""

    valid_background_layers = {"scatter", "fluid"}
    valid_foreground_layers = {None, "quiver", "stream"}
    foreground_layer = _normalize_foreground_layer(foreground_layer)
    if background_layer not in valid_background_layers:
        raise ValueError(
            f"Unsupported background layer `{background_layer}`. Use one of {sorted(valid_background_layers)}."
        )
    if foreground_layer not in valid_foreground_layers:
        raise ValueError(
            f"Unsupported foreground layer `{foreground_layer}`. Use one of `None`, `quiver`, or `stream`."
        )
    return foreground_layer


def _get_nt_score_arrays(
    sample_df: pd.DataFrame,
    score_column: str,
    reverse: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract x/y coordinates and NT scores for one sample."""

    if sample_df.empty:
        return None

    x = sample_df["x"].to_numpy(dtype=float)
    y = sample_df["y"].to_numpy(dtype=float)
    scores = sample_df[score_column].to_numpy(dtype=float)
    if reverse:
        scores = 1 - scores
    return x, y, np.clip(scores, 0, 1)


def _plot_nt_score_scatter_background(
    ax: Axes,
    sample_df: pd.DataFrame,
    score_column: str,
    reverse: bool = False,
):
    """Plot the original scatter-style NT-score background."""

    sample_arrays = _get_nt_score_arrays(sample_df=sample_df, score_column=score_column, reverse=reverse)
    if sample_arrays is None:
        return None
    x, y, scores = sample_arrays
    point_size = 1 if x.size > 200 else 2
    return ax.scatter(x, y, c=scores, cmap="rainbow", vmin=0, vmax=1, s=point_size, linewidths=0)


def _plot_nt_score_fluid_background(
    ax: Axes,
    sample_df: pd.DataFrame,
    score_column: str,
    reverse: bool = False,
):
    """Plot the smooth fluid-style NT-score background."""

    sample_arrays = _get_nt_score_arrays(sample_df=sample_df, score_column=score_column, reverse=reverse)
    if sample_arrays is None:
        return None
    x, y, scores = sample_arrays

    cmap = mpl.colormaps.get_cmap("turbo")
    if x.size >= 3 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
        triangulation = Triangulation(x, y)
        _mask_sparse_triangles(triangulation=triangulation, x=x, y=y)
        contour_levels = np.linspace(0, 1, 21)
        line_levels = np.linspace(0, 1, 11)
        background = ax.tricontourf(triangulation, scores, levels=contour_levels, cmap=cmap, vmin=0, vmax=1)
        ax.tricontour(triangulation, scores, levels=line_levels, colors="black", linewidths=0.3, alpha=0.35)
        ax.scatter(x, y, c=scores, cmap=cmap, vmin=0, vmax=1, s=1, alpha=0.12, linewidths=0)
        return background
    return ax.scatter(x, y, c=scores, cmap=cmap, vmin=0, vmax=1, s=2, linewidths=0)


def _plot_nt_score_quiver_overlay(
    ax: Axes,
    sample_df: pd.DataFrame,
    score_column: str,
    reverse: bool = False,
):
    """Overlay a quiver plot of the NT-score gradient."""

    field = _build_nt_score_vector_field(sample_df=sample_df, score_column=score_column, reverse=reverse)
    if field is None:
        return None

    grid_x, grid_y, _, grad_x, grad_y, valid_mask = field
    stride_x = max(1, grid_x.shape[1] // 24)
    stride_y = max(1, grid_x.shape[0] // 24)
    grid_slice = (slice(None, None, stride_y), slice(None, None, stride_x))
    qx = np.ma.array(grad_x[grid_slice], mask=~valid_mask[grid_slice])
    qy = np.ma.array(grad_y[grid_slice], mask=~valid_mask[grid_slice])
    mask = np.ma.getmaskarray(qx) | np.ma.getmaskarray(qy)
    if np.all(mask):
        return None

    qx_data = qx.filled(np.nan)[~mask]
    qy_data = qy.filled(np.nan)[~mask]
    magnitude = np.hypot(qx_data, qy_data)
    return ax.quiver(
        grid_x[grid_slice][~mask],
        grid_y[grid_slice][~mask],
        qx_data,
        qy_data,
        magnitude,
        cmap="Greys",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.003,
        headwidth=3.4,
        headlength=4.4,
        headaxislength=3.8,
        alpha=0.85,
    )


def _plot_nt_score_stream_overlay(
    ax: Axes,
    sample_df: pd.DataFrame,
    score_column: str,
    reverse: bool = False,
):
    """Overlay streamlines of the NT-score gradient."""

    field = _build_nt_score_vector_field(sample_df=sample_df, score_column=score_column, reverse=reverse)
    if field is None:
        return None

    grid_x, grid_y, _, grad_x, grad_y, valid_mask = field
    speed = np.ma.sqrt(grad_x**2 + grad_y**2)
    speed_values = speed.filled(0.0)
    speed_scale = float(np.nanpercentile(speed_values, 95)) if np.any(np.isfinite(speed_values)) else 0.0
    if not np.isfinite(speed_scale) or speed_scale <= 0:
        speed_scale = 1.0
    linewidth = 0.4 + 1.6 * np.clip(speed_values / speed_scale, 0, 1)

    u = np.ma.array(grad_x, mask=~valid_mask).filled(0.0)
    v = np.ma.array(grad_y, mask=~valid_mask).filled(0.0)
    return ax.streamplot(
        grid_x[0],
        grid_y[:, 0],
        u,
        v,
        density=1.1,
        color=speed_values,
        cmap="Greys",
        linewidth=linewidth,
        arrowsize=0.8,
        minlength=0.1,
        maxlength=4.0,
        broken_streamlines=True,
    )


def _compose_nt_score_title(
    sample: str,
    score_label: str,
    background_layer: str,
    foreground_layer: Optional[str],
) -> str:
    """Build a descriptive title for the selected NT-score layers."""

    layer_suffix = background_layer if foreground_layer is None else f"{background_layer} + {foreground_layer}"
    return f"{sample} {score_label} ({layer_suffix})"


def _compose_nt_output_stem(
    score_prefix: str,
    background_layer: str,
    foreground_layer: Optional[str],
) -> str:
    """Build the output filename stem from selected layers."""

    foreground_layer = _normalize_foreground_layer(foreground_layer)
    if background_layer == "scatter" and foreground_layer is None:
        return score_prefix
    if foreground_layer is None:
        return f"{score_prefix}_{background_layer}"
    if background_layer == "scatter":
        return f"{score_prefix}_{foreground_layer}"
    return f"{score_prefix}_{background_layer}_{foreground_layer}"


def _plot_nt_score_layered_field(
    ax: Axes,
    sample_df: pd.DataFrame,
    score_column: str,
    score_title: str,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
    reverse: bool = False,
):
    """Plot an NT-score view using configurable background and foreground layers."""

    foreground_layer = _validate_nt_layers(background_layer=background_layer, foreground_layer=foreground_layer)
    if sample_df.empty:
        warning(f"No valid coordinates found for {score_title}.")
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        _style_nt_score_axis(ax=ax, score_title=score_title)
        return None

    if background_layer == "fluid":
        mappable = _plot_nt_score_fluid_background(
            ax=ax,
            sample_df=sample_df,
            score_column=score_column,
            reverse=reverse,
        )
    else:
        mappable = _plot_nt_score_scatter_background(
            ax=ax,
            sample_df=sample_df,
            score_column=score_column,
            reverse=reverse,
        )

    if foreground_layer == "quiver":
        if (
            _plot_nt_score_quiver_overlay(
                ax=ax,
                sample_df=sample_df,
                score_column=score_column,
                reverse=reverse,
            )
            is None
        ):
            warning(f"Cannot build a quiver overlay for {score_title}.")
    elif foreground_layer == "stream":
        if (
            _plot_nt_score_stream_overlay(
                ax=ax,
                sample_df=sample_df,
                score_column=score_column,
                reverse=reverse,
            )
            is None
        ):
            warning(f"Cannot build a stream overlay for {score_title}.")

    _style_nt_score_axis(ax=ax, score_title=score_title)
    return mappable


def _plot_nt_score_dataset(
    NT_score: pd.DataFrame,
    meta_data_df: pd.DataFrame,
    score_column: str,
    score_label: str,
    score_prefix: str,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
    reverse: bool = False,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[Figure, Union[Axes, np.ndarray]]]:
    """Plot a layered NT-score view for all samples in one figure."""

    foreground_layer = _validate_nt_layers(background_layer=background_layer, foreground_layer=foreground_layer)
    samples: List[str] = meta_data_df["Sample"].unique().tolist()

    fig, axes = plt.subplots(1, len(samples), figsize=(4.2 * len(samples), 3.4), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    mappable = None
    for i, sample in enumerate(samples):
        sample_df = _prepare_nt_score_sample_df(
            NT_score=NT_score,
            meta_data_df=meta_data_df,
            sample=sample,
            score_column=score_column,
        )
        rendered = _plot_nt_score_layered_field(
            ax=axes_flat[i],
            sample_df=sample_df,
            score_column=score_column,
            score_title=_compose_nt_score_title(
                sample=sample,
                score_label=score_label,
                background_layer=background_layer,
                foreground_layer=foreground_layer,
            ),
            background_layer=background_layer,
            foreground_layer=foreground_layer,
            reverse=reverse,
        )
        if rendered is not None:
            mappable = rendered

    if mappable is not None:
        colorbar = fig.colorbar(mappable, ax=axes_flat.tolist(), fraction=0.02, pad=0.02)
        colorbar.set_label("NT score")
    if output_file_path is not None:
        output_stem = _compose_nt_output_stem(
            score_prefix=score_prefix,
            background_layer=background_layer,
            foreground_layer=foreground_layer,
        )
        fig.savefig(f"{output_file_path}/{output_stem}.pdf", transparent=True)
        plt.close(fig)
        return None
    return fig, axes_flat[0] if len(samples) == 1 else axes_flat


def _plot_nt_score_sample(
    NT_score: pd.DataFrame,
    meta_data_df: pd.DataFrame,
    score_column: str,
    score_label: str,
    score_prefix: str,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
    reverse: bool = False,
    spatial_scaling_factor: float = 1.0,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[List[Tuple[Figure, Axes]]]:
    """Plot one layered NT-score view per sample."""

    foreground_layer = _validate_nt_layers(background_layer=background_layer, foreground_layer=foreground_layer)
    samples: List[str] = meta_data_df["Sample"].unique().tolist()
    output: List[Tuple[Figure, Axes]] = []
    output_stem = _compose_nt_output_stem(
        score_prefix=score_prefix,
        background_layer=background_layer,
        foreground_layer=foreground_layer,
    )
    for sample in samples:
        sample_df = _prepare_nt_score_sample_df(
            NT_score=NT_score,
            meta_data_df=meta_data_df,
            sample=sample,
            score_column=score_column,
        )
        if sample_df.empty:
            warning(f"No valid coordinates found for sample {sample}. Skip {score_label} plot.")
            continue

        fig_width, fig_height = saptial_figsize(sample_df, scaling_factor=spatial_scaling_factor)
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(max(fig_width + 0.8, 3.5), max(fig_height + 0.4, 3.0)),
            constrained_layout=True,
        )
        rendered = _plot_nt_score_layered_field(
            ax=ax,
            sample_df=sample_df,
            score_column=score_column,
            score_title=_compose_nt_score_title(
                sample=sample,
                score_label=score_label,
                background_layer=background_layer,
                foreground_layer=foreground_layer,
            ),
            background_layer=background_layer,
            foreground_layer=foreground_layer,
            reverse=reverse,
        )
        if rendered is not None:
            colorbar = fig.colorbar(rendered, ax=ax, fraction=0.04, pad=0.02)
            colorbar.set_label("NT score")
        if output_file_path is not None:
            fig.savefig(f"{output_file_path}/{sample}_{output_stem}.pdf", transparent=True)
            plt.close(fig)
        else:
            output.append((fig, ax))
    return output if len(output) > 0 else None


def plot_cell_type_composition_dataset(
    meta_data_df: pd.DataFrame,
    cell_type_codes: pd.DataFrame,
    cell_type_composition: pd.DataFrame,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot spatial distribution of cell type composition.

    Parameters
    ----------
    meta_data_df :
        pd.DataFrame, the meta data.
    cell_type_codes :
        pd.DataFrame, the cell type codes.
    cell_type_composition :
        pd.DataFrame, the cell type composition data.
    output_file_path :
        Optional[Union[str, Path]], the output file path.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    samples: List[str] = meta_data_df["Sample"].unique().tolist()
    cell_types: List[str] = cell_type_codes["Cell_Type"].tolist()

    M, N = len(samples), len(cell_types)
    fig, axes = plt.subplots(M, N, figsize=(3.5 * N, 3 * M))
    for i, sample in enumerate(samples):
        sample_df = cell_type_composition.loc[meta_data_df["Sample"] == sample]
        sample_df = sample_df.join(meta_data_df.loc[sample_df.index][["x", "y"]])
        for j, cell_type in enumerate(cell_types):
            ax = axes[i, j] if M > 1 else axes[j]
            scatter = ax.scatter(
                sample_df["x"], sample_df["y"], c=sample_df[cell_type], cmap="Reds", vmin=0, vmax=1, s=1
            )
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(scatter)
            ax.set_title(f"{sample} {cell_type}")

    fig.tight_layout()
    if output_file_path is not None:
        fig.savefig(f"{output_file_path}/cell_type_composition.pdf", transparent=True)
        plt.close(fig)
        return None
    else:
        return fig, axes


def plot_cell_type_composition_dataset_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot spatial distribution of cell type composition.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    try:
        if ana_data.cell_type_composition is None:
            warning("No cell type composition data found. Skip spatial cell type composition visualization.")
            return None
        if ana_data.cell_type_codes is None:
            warning("No cell type codes found. Skip spatial cell type composition visualization.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    return plot_cell_type_composition_dataset(
        meta_data_df=ana_data.meta_data_df,
        cell_type_codes=ana_data.cell_type_codes,
        cell_type_composition=ana_data.cell_type_composition,
        output_file_path=ana_data.options.output,
    )


def plot_cell_type_composition_sample(
    meta_data_df: pd.DataFrame,
    cell_type_codes: pd.DataFrame,
    cell_type_composition: pd.DataFrame,
    spatial_scaling_factor: float = 1.0,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """Plot spatial distribution of cell type composition.

    Parameters
    ----------
    meta_data_df :
        pd.DataFrame, the meta data.
    cell_type_codes :
        pd.DataFrame, the cell type codes.
    cell_type_composition :
        pd.DataFrame, the cell type composition data.
    spatial_scaling_factor :
        float, the scale factor control the size of spatial-based plots.
    output_file_path :
        Optional[Union[str, Path]], the output file path.

    Returns
    -------
    None."""

    samples: List[str] = meta_data_df["Sample"].unique().tolist()
    cell_types: List[str] = cell_type_codes["Cell_Type"].tolist()

    output = []
    N = len(cell_types)
    for sample in samples:
        sample_df = cell_type_composition.loc[meta_data_df["Sample"] == sample]
        sample_df = sample_df.join(meta_data_df[["x", "y"]])
        fig_width, fig_height = saptial_figsize(sample_df, scaling_factor=spatial_scaling_factor)
        fig, axes = plt.subplots(1, N, figsize=(fig_width * N, fig_height))
        for j, cell_type in enumerate(cell_types):
            ax = axes[j]  # At least two cell types are required, checked at original data loading.
            scatter = ax.scatter(
                sample_df["x"], sample_df["y"], c=sample_df[cell_type], cmap="Reds", vmin=0, vmax=1, s=1
            )
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(scatter)
            ax.set_title(f"{sample} {cell_type}")
        fig.tight_layout()
        if output_file_path is not None:
            fig.savefig(f"{output_file_path}/{sample}_cell_type_composition.pdf", transparent=True)
            plt.close(fig)
        else:
            output.append((fig, axes))
    return output if len(output) > 0 else None


def plot_cell_type_composition_sample_from_anadata(ana_data: AnaData) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """Plot spatial distribution of cell type composition.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    try:
        if ana_data.cell_type_composition is None:
            warning("No cell type composition data found. Skip spatial cell type composition visualization.")
            return None
        if ana_data.cell_type_codes is None:
            warning("No cell type codes found. Skip spatial cell type composition visualization.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    return plot_cell_type_composition_sample(
        meta_data_df=ana_data.meta_data_df,
        cell_type_codes=ana_data.cell_type_codes,
        cell_type_composition=ana_data.cell_type_composition,
        spatial_scaling_factor=ana_data.options.scale_factor,
        output_file_path=ana_data.options.output,
    )


def plot_cell_type_composition(
    ana_data: AnaData,
) -> Optional[Union[List[Tuple[plt.Figure, plt.Axes]], Tuple[plt.Figure, plt.Axes]]]:
    """Plot spatial distribution of cell type composition.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if getattr(ana_data.options, "sample", False):
        return plot_cell_type_composition_sample_from_anadata(ana_data=ana_data)
    else:
        return plot_cell_type_composition_dataset_from_anadata(ana_data=ana_data)


def plot_adjust_cell_type_composition_sample_from_anadata(
    ana_data: AnaData,
) -> Optional[List[Tuple[plt.Figure, plt.Axes]]]:
    """Plot spatial distribution of adjusted cell type composition.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    try:
        if ana_data.adjust_cell_type_composition is None:
            warning("No adjusted cell type composition data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    return plot_cell_type_composition_sample(
        meta_data_df=ana_data.meta_data_df,
        cell_type_codes=ana_data.cell_type_codes,
        cell_type_composition=ana_data.adjust_cell_type_composition,
        spatial_scaling_factor=ana_data.options.scale_factor,
        output_file_path=ana_data.options.output,
    )


def plot_adjust_cell_type_composition_dataset_from_anadata(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Plot spatial distribution of adjusted cell type composition.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    try:
        if ana_data.adjust_cell_type_composition is None:
            warning("No adjusted cell type composition data found.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    return plot_cell_type_composition_dataset(
        meta_data_df=ana_data.meta_data_df,
        cell_type_codes=ana_data.cell_type_codes,
        cell_type_composition=ana_data.adjust_cell_type_composition,
        output_file_path=ana_data.options.output,
    )


def plot_adjust_cell_type_composition(
    ana_data: AnaData,
) -> Optional[Union[List[Tuple[plt.Figure, plt.Axes]], Tuple[plt.Figure, plt.Axes]]]:
    """Plot spatial distribution of adjusted cell type composition.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if getattr(ana_data.options, "sample", False):
        return plot_adjust_cell_type_composition_sample_from_anadata(ana_data=ana_data)
    else:
        return plot_adjust_cell_type_composition_dataset_from_anadata(ana_data=ana_data)


def plot_niche_NT_score_dataset(
    NT_score: pd.DataFrame,
    meta_data_df: pd.DataFrame,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
    reverse: bool = False,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[Figure, Union[Axes, np.ndarray]]]:
    """Plot spatial distribution of niche NT score.

    Parameters
    ----------
    NT_score :
        pd.DataFrame, the NT score data.
    meta_data_df :
        pd.DataFrame, the meta data.
    reverse :
        bool, reverse the NT score or not.
    output_file_path :
        Optional[Union[str, Path]], the output file path.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    return _plot_nt_score_dataset(
        NT_score=NT_score,
        meta_data_df=meta_data_df,
        score_column="Niche_NTScore",
        score_label="Niche-level NT Score",
        score_prefix="niche_NT_score",
        background_layer=background_layer,
        foreground_layer=foreground_layer,
        reverse=reverse,
        output_file_path=output_file_path,
    )


def plot_niche_NT_score_dataset_from_anadata(
    ana_data: AnaData,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
) -> Optional[Tuple[Figure, Union[Axes, np.ndarray]]]:
    """Plot spatial distribution of niche NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    try:
        if ana_data.NT_score is None:
            warning("No NT score data found. Skip spatial niche-level NT score visualization.")
            return None
        if "Niche_NTScore" not in ana_data.NT_score.columns:
            warning("Niche_NTScore not found in the NT score data. Skip spatial niche-level NT score visualization.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    return plot_niche_NT_score_dataset(
        NT_score=ana_data.NT_score,
        meta_data_df=ana_data.meta_data_df,
        background_layer=background_layer,
        foreground_layer=foreground_layer,
        reverse=ana_data.options.reverse,
        output_file_path=ana_data.options.output,
    )


def plot_niche_NT_score_sample(
    NT_score: pd.DataFrame,
    meta_data_df: pd.DataFrame,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
    reverse: bool = False,
    spatial_scaling_factor: float = 1.0,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[List[Tuple[Figure, Axes]]]:
    """Plot spatial distribution of niche NT score.

    Parameters
    ----------
    NT_score :
        pd.DataFrame, the NT score data.
    meta_data_df :
        pd.DataFrame, the meta data.
    reverse :
        bool, reverse the NT score or not.
    spatial_scaling_factor :
        float, the scale factor control the size of spatial-based plots.
    output_file_path :
        Optional[Union[str, Path]], the output file path.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    return _plot_nt_score_sample(
        NT_score=NT_score,
        meta_data_df=meta_data_df,
        score_column="Niche_NTScore",
        score_label="Niche-level NT Score",
        score_prefix="niche_NT_score",
        background_layer=background_layer,
        foreground_layer=foreground_layer,
        reverse=reverse,
        spatial_scaling_factor=spatial_scaling_factor,
        output_file_path=output_file_path,
    )


def plot_niche_NT_score_sample_from_anadata(
    ana_data: AnaData,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
) -> Optional[List[Tuple[Figure, Axes]]]:
    """Plot spatial distribution of niche NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    try:
        if ana_data.NT_score is None:
            warning("No NT score data found. Skip spatial niche-level NT score visualization.")
            return None
        if "Niche_NTScore" not in ana_data.NT_score.columns:
            warning("Niche_NTScore not found in the NT score data. Skip spatial niche-level NT score visualization.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    return plot_niche_NT_score_sample(
        NT_score=ana_data.NT_score,
        meta_data_df=ana_data.meta_data_df,
        background_layer=background_layer,
        foreground_layer=foreground_layer,
        reverse=ana_data.options.reverse,
        spatial_scaling_factor=ana_data.options.scale_factor,
        output_file_path=ana_data.options.output,
    )


def plot_niche_NT_score(
    ana_data: AnaData,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
) -> Optional[Union[List[Tuple[Figure, Axes]], Tuple[Figure, Union[Axes, np.ndarray]]]]:
    """Plot spatial distribution of niche NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if getattr(ana_data.options, "sample", False):
        return plot_niche_NT_score_sample_from_anadata(
            ana_data=ana_data,
            background_layer=background_layer,
            foreground_layer=foreground_layer,
        )
    else:
        return plot_niche_NT_score_dataset_from_anadata(
            ana_data=ana_data,
            background_layer=background_layer,
            foreground_layer=foreground_layer,
        )


def plot_cell_NT_score_dataset(
    NT_score: pd.DataFrame,
    meta_data_df: pd.DataFrame,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
    reverse: bool = False,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[Figure, Union[Axes, np.ndarray]]]:
    """Plot spatial distribution of cell NT score.

    Parameters
    ----------
    NT_score :
        pd.DataFrame, the NT score data.
    meta_data_df :
        pd.DataFrame, the meta data.
    reverse :
        bool, reverse the NT score or not.
    output_file_path :
        Optional[Union[str, Path]], the output file path.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    return _plot_nt_score_dataset(
        NT_score=NT_score,
        meta_data_df=meta_data_df,
        score_column="Cell_NTScore",
        score_label="Cell-level NT Score",
        score_prefix="cell_NT_score",
        background_layer=background_layer,
        foreground_layer=foreground_layer,
        reverse=reverse,
        output_file_path=output_file_path,
    )


def plot_cell_NT_score_dataset_from_anadata(
    ana_data: AnaData,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
) -> Optional[Tuple[Figure, Union[Axes, np.ndarray]]]:
    """Plot spatial distribution of cell NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    try:
        if ana_data.NT_score is None:
            warning("No NT score data found. Skip spatial cell-level NT score visualization.")
            return None
        if "Cell_NTScore" not in ana_data.NT_score.columns:
            warning("Cell_NTScore not found in the NT score data. Skip spatial cell-level NT score visualization.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    return plot_cell_NT_score_dataset(
        NT_score=ana_data.NT_score,
        meta_data_df=ana_data.meta_data_df,
        background_layer=background_layer,
        foreground_layer=foreground_layer,
        reverse=ana_data.options.reverse,
        output_file_path=ana_data.options.output,
    )


def plot_cell_NT_score_sample(
    NT_score: pd.DataFrame,
    meta_data_df: pd.DataFrame,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
    reverse: bool = False,
    spatial_scaling_factor: float = 1.0,
    output_file_path: Optional[Union[str, Path]] = None,
) -> Optional[List[Tuple[Figure, Axes]]]:
    """Plot spatial distribution of cell NT score.

    Parameters
    ----------
    NT_score :
        pd.DataFrame, the NT score data.
    meta_data_df :
        pd.DataFrame, the meta data.
    reverse :
        bool, reverse the NT score or not.
    spatial_scaling_factor :
        float, the scale factor control the size of spatial-based plots.
    output_file_path :
        Optional[Union[str, Path]], the output file path.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    return _plot_nt_score_sample(
        NT_score=NT_score,
        meta_data_df=meta_data_df,
        score_column="Cell_NTScore",
        score_label="Cell-level NT Score",
        score_prefix="cell_NT_score",
        background_layer=background_layer,
        foreground_layer=foreground_layer,
        reverse=reverse,
        spatial_scaling_factor=spatial_scaling_factor,
        output_file_path=output_file_path,
    )


def plot_cell_NT_score_sample_from_anadata(
    ana_data: AnaData,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
) -> Optional[List[Tuple[Figure, Axes]]]:
    """Plot spatial distribution of cell NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    try:
        if ana_data.NT_score is None:
            warning("No NT score data found. Skip spatial cell-level NT score visualization.")
            return None
        if "Cell_NTScore" not in ana_data.NT_score.columns:
            warning("Cell_NTScore not found in the NT score data. Skip spatial cell-level NT score visualization.")
            return None
    except FileNotFoundError as e:
        warning(str(e))
        return None

    return plot_cell_NT_score_sample(
        NT_score=ana_data.NT_score,
        meta_data_df=ana_data.meta_data_df,
        background_layer=background_layer,
        foreground_layer=foreground_layer,
        reverse=ana_data.options.reverse,
        spatial_scaling_factor=ana_data.options.scale_factor,
        output_file_path=ana_data.options.output,
    )


def plot_cell_NT_score(
    ana_data: AnaData,
    background_layer: str = "scatter",
    foreground_layer: Optional[str] = None,
) -> Optional[Union[List[Tuple[Figure, Axes]], Tuple[Figure, Union[Axes, np.ndarray]]]]:
    """Plot spatial distribution of cell NT score.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None or Tuple[plt.Figure, plt.Axes]."""

    if getattr(ana_data.options, "sample", None):
        return plot_cell_NT_score_sample_from_anadata(
            ana_data=ana_data,
            background_layer=background_layer,
            foreground_layer=foreground_layer,
        )
    else:
        return plot_cell_NT_score_dataset_from_anadata(
            ana_data=ana_data,
            background_layer=background_layer,
            foreground_layer=foreground_layer,
        )


def spatial_visualization(ana_data: AnaData) -> None:
    """All spatial visualization will include here.

    Parameters
    ----------
    ana_data :
        AnaData, the data for analysis.

    Returns
    -------
    None."""

    # 1. cell type composition
    if getattr(ana_data.options, "suppress_cell_type_composition", False):
        info("Skip spatial cell type composition visualization according to `suppress_cell_type_composition` option.")
    else:
        plot_cell_type_composition(ana_data=ana_data)
        if getattr(ana_data.options, "embedding_adjust", False):
            plot_adjust_cell_type_composition(ana_data=ana_data)
        else:
            info("Skip the adjusted cell type composition visualization due to no embedding adjust setting.")

    # 2. NT score
    if getattr(ana_data.options, "suppress_niche_trajectory", False):
        info("Skip spatial NT score visualization according to `suppress_niche_trajectory` option.")
    else:
        plot_niche_NT_score(ana_data=ana_data, background_layer="scatter", foreground_layer="quiver")
        plot_cell_NT_score(ana_data=ana_data, background_layer="scatter", foreground_layer="quiver")
