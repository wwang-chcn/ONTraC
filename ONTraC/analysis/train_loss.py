import os
import sys
from optparse import OptionParser, Values
from typing import Dict, List, Tuple
from warnings import warn

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DenseDataLoader

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns


def loss_record_df(output_dir: str, log_file: str) -> Tuple[pd.DataFrame, Dict]:
    loss, loss_name = [], []
    batch_flag = False
    final_loss_dict = {}
    with open(log_file, 'r') as fhd:
        for line in fhd:
            # train loss record
            if 'INFO' in line and 'epoch' in line and 'loss' in line:
                loss_ = []
                loss_name = []
                line = line.strip().split()
                init_index = line.index('loss:')  # get the first loss index
                # get loss name & losss
                for i in range(init_index, len(line), 2):
                    loss_name.append(line[i].strip(':'))
                for i in range(init_index + 1, len(line), 2):
                    loss_.append(float(line[i].strip(',')))
                # insert epoch and batch information
                if not batch_flag and line[init_index - 2] == 'batch:':
                    batch_flag = True
                loss_.insert(0, int(line[init_index - 1].strip(',')))
                if batch_flag:
                    loss_.insert(0, int(line[init_index - 3].strip(',')))
                loss.append(loss_)
            # eval loss record
            elif 'INFO' in line and 'Evaluate loss' in line:
                line = line.strip().split(', ', 1)
                final_loss_dict: Dict[str, float] = eval(line[1])
    loss_name[0] = 'total_loss'
    if batch_flag:
        loss_name.insert(0, 'Batch')
    loss_name.insert(0, 'Epoch')
    loss_df = pd.DataFrame(loss, columns=loss_name)
    loss_df.to_csv(f'{output_dir}/train_loss.csv', index=False)
    epoch_loss_columns = [col for col in loss_df.columns if 'loss' in col]
    epoch_loss_columns.insert(0, 'Epoch')
    epoch_loss_df = loss_df[epoch_loss_columns].groupby('Epoch').mean()
    epoch_loss_df.to_csv(f'{output_dir}/train_loss_epoch.csv')
    return epoch_loss_df, final_loss_dict


def plot_train_loss(options: Values) -> None:
    """Plot training loss.

    Args:
        options: Options from optparse.
    """
    # load training loss
    log_file = f'log/{options.name}.log'
    if not os.path.exists(log_file):
        warn(f'Log file not found: {log_file}')
        return
    loss_df, final_loss_dict = loss_record_df(options.output, log_file)

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
        fig.tight_layout()
        fig.savefig(f'{options.output}/train_loss.pdf', dpi=300)
        plt.close(fig)
