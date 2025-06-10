from typing import Dict, List, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd

from ..log import warning
from .data import AnaData


def saptial_figsize(sample_df, scaling_factor: Union[int, float] = 1) -> Tuple[int, int]:
    """
    Calculate the figure size for spatial-based plot according to the points and the span of x and y.
    :param sample_df: pd.DataFrame, the sample data.
    :param scale_factor: float, the scale factor control the size of spatial-based plots. The larger the scale factor,
    the larger the figure size.
    :return: tuple[int, int], the figure size.
    """

    n_points = sample_df[['x', 'y']].dropna().shape[0]
    # debug(f'n_points: {n_points}')

    x_span = sample_df['x'].dropna().max() - sample_df['x'].dropna().min()
    y_span = sample_df['y'].dropna().max() - sample_df['y'].dropna().min()
    # debug(f'x_span: {x_span}')
    # debug(f'y_span: {y_span}')

    points_density = n_points / x_span / y_span * 10_000

    # debug(f'points density: {points_density}')

    fig_width = x_span / 2_000 * scaling_factor * np.sqrt(points_density) + .5  # Adding 2 for colorbar space
    fig_height = y_span / 2_000 * scaling_factor * np.sqrt(points_density) + .2  # Adding 1.5 for title space

    # debug(f'scale_factor: {scale_factor}')
    # debug(f'fig_width: {fig_width}')
    # debug(f'fig_height: {fig_height}')

    return fig_width, fig_height


def gini(array: Union[np.ndarray, pd.Series]) -> float:
    """Calculate the Gini coefficient of a numpy array.
    :param array: np.ndarray or pd.Series, the array for calculating Gini coefficient.
    :return: float, the Gini coefficient.
    """
    #
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    if isinstance(array, pd.Series):
        array = np.array(array)
    array = array.flatten()  # type: ignore
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)  # type: ignore
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)  # type: ignore
    # Number of array elements:
    n = array.shape[0]  # type: ignore
    # Index per array element:
    index = np.arange(1, n + 1)  # type: ignore
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # type: ignore


def get_n_colors(n: int) -> List[str]:
    """Get n colors.
    
    Parameters
    ----------
    n : int
        The number of colors.

    Returns
    -------
    list
        The list of colors.

    Examples
    --------
    >>> get_n_colors(10)
    ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    >>> get_n_colors(20)
    ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    >>> get_n_colors(30)
    ['#30123b', '#392a73', '#4143a7', '#455ccf', '#4773eb', '#458afc', '#3d9efe', '#2eb4f2', '#1fc9dd', '#18dbc5',
    '#1fe9af', '#35f394', '#52fa7a', '#75fe5c', '#96fe44', '#affa37', '#c6f034', '#dbe236', '#ebd339', '#f7c13a',
    '#fdac34', '#fe932a', '#f9781e', '#f15d13', '#e7490c', '#d83706', '#c52603', '#af1801', '#950d01', '#7a0403']
    """

    if n <= 10:  # use default colormaps to generate colors
        return [f'C{i}' for i in range(n)]
    elif n <= 20:  # use tab20 colormaps to generate colors
        tab20_cmap = mpl.colormaps.get_cmap('tab20')  # type: ignore
        return [mpl.colors.to_hex(c=x) for x in tab20_cmap(np.arange(n))]
    else:  # use turbo colormaps to generate colors
        turbo_cmap = mpl.colormaps.get_cmap('turbo')  # type: ignore
        return [mpl.colors.to_hex(c=x) for x in turbo_cmap(np.linspace(start=0, stop=1, num=n))]


def get_palette_for_cell_types(cell_types: List[str]) -> Dict[str, str]:
    """
    Get the color palette for cell types.

    Parameters
    ----------
    cell_types : list
        The list of cell types.

    Returns
    -------
    list
        The list of colors.

    Examples
    --------
    >>> get_palette_for_cell_types(['A', 'B', 'C'])
    {'A': 'C0', 'B': 'C1', 'C': 'C2'}
    >>> get_palette_for_cell_types(['ct1', 'ct2', 'ct3', 'ct4', 'ct5', 'ct6', 'ct7', 'ct8', 'ct9', 'ct10', 'ct11',
    'ct12', 'ct13', 'ct14', 'ct15', 'ct16', 'ct17', 'ct18', 'ct19', 'ct20'])
    {'ct1': '#1f77b4', 'ct2': '#aec7e8', 'ct3': '#ff7f0e', 'ct4': '#ffbb78', 'ct5': '#2ca02c', 'ct6': '#98df8a',
    'ct7': '#d62728', 'ct8': '#ff9896', 'ct9': '#9467bd', 'ct10': '#c5b0d5', 'ct11': '#8c564b', 'ct12': '#c49c94',
    'ct13': '#e377c2', 'ct14': '#f7b6d2', 'ct15': '#7f7f7f', 'ct16': '#c7c7c7', 'ct17': '#bcbd22', 'ct18': '#dbdb8d',
    'ct19': '#17becf', 'ct20': '#9edae5'}
    """

    return dict(zip(cell_types, get_n_colors(len(cell_types))))


def validate_cell_type_palette(cell_types: List[str],
                               palette: Union[List[str], Dict[str, str], None] = None) -> Dict[str, str]:
    """
    Validate given cell type palette.

    Parameters
    ----------
    cell_types : list
        The list of cell types.
    palette : list or dict or None
        The given palette. Should cover all cell types.

    Returns
    -------
    dict
        The validated palette.

    Examples
    --------
    >>> validate_cell_type_palette(['A', 'B', 'C'], ['C0', 'C1', 'C2'])
    {'A': 'C0', 'B': 'C1', 'C': 'C2'}
    >>> validate_cell_type_palette(['ct1', 'ct2', 'ct3'], {'ct1': '#1f77b4', 'ct2': '#aec7e8', 'ct3': '#ff7f0e'})
    {'ct1': '#1f77b4', 'ct2': '#aec7e8', 'ct3': '#ff7f0e'}
    >>> validate_cell_type_palette(['ct1', 'ct2', 'ct3'], None)
    {'ct1': 'C0', 'ct2': 'C1', 'ct3': 'C2'}
    """

    if palette is None:
        return get_palette_for_cell_types(cell_types)
    elif isinstance(palette, list):
        if len(palette) < len(cell_types):
            warning('The given palette is not enough for all cell types. Use default palette instead.')
            return get_palette_for_cell_types(cell_types)
        return dict(zip(cell_types, palette))
    elif isinstance(palette, dict):
        for cell_type in cell_types:
            if cell_type not in palette:
                warning(f'There are no colors for cell type: {cell_type}. Use default palette instead.')
                return get_palette_for_cell_types(cell_types)
        else:
            return palette
    else:
        warning("The given palette's type is not supported. Use default palette instead.")
        return get_palette_for_cell_types(cell_types)


def cal_niche_level_raw_cell_type_counts_from_anadata(ana_data: AnaData) -> pd.DataFrame:
    """
    Calculate the raw cell type counts for each niche level from AnaData.

    Parameters
    ----------
    ana_data : AnaData
        The AnaData object containing the data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with niche levels as index and cell types as columns, containing raw cell type counts.
    """

    # if cell type codes not None, they Cell_Type column in meta_data_df should already be converted to Categorical
    if ana_data.cell_type_codes is None:
        raise AttributeError('Cell type codes are not found in AnaData.')

    if 'Cell_Type' not in ana_data.meta_data_df.columns:
        raise AttributeError('Cell_Type column is not found in meta_data_df of AnaData.')
    if not hasattr(ana_data.options, 'NN_dir') or ana_data.options.NN_dir is None:
        raise AttributeError('NN_dir is not found in AnaData options.')

    cell_type = ana_data.meta_data_df['Cell_Type'].cat.codes
    cell_label_one_hot = np.zeros(shape=(ana_data.meta_data_df.shape[0],
                                         len(ana_data.meta_data_df['Cell_Type'].cat.categories)))
    cell_label_one_hot[np.arange(ana_data.meta_data_df.shape[0]), cell_type] = 1  # N(#cells) x F(#cell_types)
    cell_label_one_hot_df = pd.DataFrame(data=cell_label_one_hot,
                                         index=ana_data.meta_data_df.index,
                                         columns=ana_data.meta_data_df['Cell_Type'].cat.categories)

    sample_niche_ctc_df_list = []
    for sample in ana_data.meta_data_df['Sample'].unique():
        sample_data_df = ana_data.meta_data_df[ana_data.meta_data_df['Sample'] == sample]
        neighborIndices_matrix = np.loadtxt(f'{ana_data.options.NN_dir}/{sample}_NeighborIndicesMatrix.csv.gz',
                                            delimiter=',').astype(int)
        adj = np.zeros((sample_data_df.shape[0], sample_data_df.shape[0]))
        for i in range(neighborIndices_matrix.shape[0]):
            adj[i, neighborIndices_matrix[i]] = 1

        sample_niche_ctc_df_list.append(
            pd.DataFrame(adj, index=sample_data_df.index, columns=sample_data_df.index)
            @ cell_label_one_hot_df.loc[sample_data_df.index])

    return pd.concat(sample_niche_ctc_df_list).loc[ana_data.meta_data_df.index]
