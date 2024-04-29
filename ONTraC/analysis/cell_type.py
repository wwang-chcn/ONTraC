from typing import Optional, Tuple

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from ..log import warning
from .data import AnaData


def plot_cell_type_along_NT_score(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot cell type composition along NT score.
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
        fig.savefig(f'{ana_data.options.output}/cell_type_along_NT_score.pdf', transparent=True)
        return None
    else:
        return fig, ax


def cell_type_visualization(ana_data: AnaData) -> None:
    """
    Visualize cell type based output.
    :param ana_data: AnaData, the data for analysis.
    """

    # 1. cell type composition along NT score
    plot_cell_type_along_NT_score(ana_data=ana_data)
