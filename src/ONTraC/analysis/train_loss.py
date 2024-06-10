from typing import Optional, Tuple

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns

from .data import AnaData


def train_loss_visualiztion(ana_data: AnaData) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    if ana_data.train_loss is None:
        return None
    train_loss_data = ana_data.train_loss
    loss_df, final_loss_dict = train_loss_data['loss_df'], train_loss_data['loss_dict']

    fig, axes = plt.subplots(loss_df.shape[1], 1, figsize=(6.4, 2 * loss_df.shape[1]))
    for index, loss_name in enumerate(loss_df.columns):
        sns.lineplot(data=loss_df, x='Epoch', y=loss_name, ax=axes[index])
        if loss_name in final_loss_dict:
            axes[index].annotate(f'Final: {final_loss_dict[loss_name]:.4f}',
                                 xy=(0.98, 0.98),
                                 xycoords='axes fraction',
                                 ha='right',
                                 va='top',
                                 fontsize=12)
            axes[index].set_ylabel(loss_name.replace('_', ' ').capitalize())
    fig.tight_layout()
    if ana_data.options.output is not None:
        fig.savefig(f'{ana_data.options.output}/train_loss.pdf', dpi=300)
        plt.close(fig)
        return None
    else:
        return fig, axes
