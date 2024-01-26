import os
import re
import shutil
import sys
from optparse import OptionGroup, OptionParser, Values
from random import randint

from ..log import *


def add_basic_options_group(optparser: OptionParser) -> OptionGroup:
    """
    Add basic options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """

    # basic options group
    group_basic = OptionGroup(optparser, "Basic options")
    optparser.add_option_group(group_basic)
    group_basic.add_option(
        '-i',
        '--input',
        dest='input',
        type='string',
        help='Directory contains input dataset. This directory should be the output directory of createDataSet.py.')
    group_basic.add_option(
        '-o',
        '--output',
        dest='output',
        type='string',
        help=
        'Directory to output the result. Won\'t be overwritten if target directory exists. If -o is not specified, -oc must be specified.'
    )
    group_basic.add_option(
        '--oc',
        dest='oc',
        type='string',
        help=
        'Directory to output the result. Will be overwritten if target directory exists. If -o is specified, --oc will be ignored.'
    )
    return group_basic


def add_train_options_group(optparser: OptionParser) -> OptionGroup:
    """
    Add train options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """

    # overall train options group
    group_train = OptionGroup(optparser, "Options for training")
    optparser.add_option_group(group_train)
    group_train.add_option('--device',
                           dest='device',
                           type='string',
                           default='cuda',
                           help='Device for training. Default is cuda.')
    group_train.add_option('--epochs',
                           dest='epochs',
                           type='int',
                           default=100,
                           help='Number of maximum epochs for training. Default is 100.')
    group_train.add_option('--patience',
                           dest='patience',
                           type='int',
                           default=50,
                           help='Number of epochs wait for better result. Default is 50.')
    group_train.add_option('--min-delta',
                           dest='min_delta',
                           type='float',
                           default=0.001,
                           help='Minimum delta for better result. Default is 0.001')
    group_train.add_option('--min-epochs',
                           dest='min_epochs',
                           type='int',
                           default=100,
                           help='Minimum number of epochs for training. Default is 100. Set to 0 to disable.')
    group_train.add_option('--batch-size',
                           dest='batch_size',
                           type='int',
                           default=0,
                           help='Batch size for training. Default is 0 for whole dataset.')
    group_train.add_option('-s', '--seed', dest='seed', type='int', help='Random seed for training. Default is random.')
    group_train.add_option('--lr',
                           dest='lr',
                           type='float',
                           default=1e-3,
                           help='Learning rate for training. Default is 1e-3.')
    return group_train


def add_GNN_options_group(group_train: OptionGroup) -> None:
    """
    Add GNN options group to optparser.
    :param optparser: OptionParser object.
    :return: None.
    """

    # GNN options group
    group_train.add_option('--hidden-feats',
                           dest='hidden_feats',
                           type='int',
                           default=32,
                           help='Number of hidden features. Default is 32.')


def add_GSAE_options_group(group_train: OptionGroup) -> None:
    """
    Add Graph Smooth Autoencoder options group to optparser.
    :param optparser: OptionParser object.
    :return: None.
    """

    # GSAE options group
    group_train.add_option('--recon-loss-weight',
                           dest='recon_loss_weight',
                           type='float',
                           default=1,
                           help='Weight for reconstruction loss. Default is 1.')
    group_train.add_option('--graph-smooth-loss-weight',
                           dest='graph_smooth_loss_weight',
                           type='float',
                           default=1,
                           help='Weight for graph smooth loss. Default is 1.')


def add_NP_options_group(group_train: OptionGroup) -> None:
    """
    Add Node Pooling options group to optparser.
    :param optparser: OptionParser object.
    :return: None.
    """

    # NP options group
    group_train.add_option('-k',
                           '--k-cluster',
                           dest='k',
                           type='int',
                           default=8,
                           help='Number of spatial clusters. Default is 8.')
    group_train.add_option('--dropout', dest='dropout', type='float', default=0, help='Dropout rate. Default is 0.')
    group_train.add_option('--spectral-loss-weight',
                           dest='spectral_loss_weight',
                           type='float',
                           default=1,
                           help='Weight for spectral loss. Default is 1.')
    group_train.add_option('--ortho-loss-weight',
                           dest='ortho_loss_weight',
                           type='float',
                           default=0,
                           help='Weight for ortho loss. Default is 0.')
    group_train.add_option('--cluster-loss-weight',
                           dest='cluster_loss_weight',
                           type='float',
                           default=1,
                           help='Weight for cluster loss. Default is 1.')
    group_train.add_option('--bin-spectral-loss-weight',
                           dest='bin_spectral_loss_weight',
                           type='float',
                           default=0,
                           help='Weight for binary spectral loss. Default is 0.')
    group_train.add_option('--bin-ortho-loss-weight',
                           dest='bin_ortho_loss_weight',
                           type='float',
                           default=0,
                           help='Weight for binary ortho loss. Default is 0.')
    group_train.add_option('--bin-cluster-loss-weight',
                           dest='bin_cluster_loss_weight',
                           type='float',
                           default=0,
                           help='Weight for binary cluster loss. Default is 0.')
    group_train.add_option('--feat-similarity-loss-weight',
                           dest='feat_similarity_loss_weight',
                           type='float',
                           default=0,
                           help='Weight for feature similarity loss. Default is 0.')
    group_train.add_option('--assign-exponent',
                           dest='assign_exponent',
                           type='float',
                           default=1,
                           help='Exponent for assignment. Default is 1.')


def add_pseudotime_options_group(optparser: OptionParser) -> OptionGroup:
    """
    Add pseudotime options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """

    # basic options group
    group_pseudotime = OptionGroup(optparser, "Pseudotime options")
    optparser.add_option_group(group_pseudotime)
    group_pseudotime.add_option('--init-node-label', dest='init_node_label', type='int', help='Initial node label.')
    return group_pseudotime


def validate_basic_options(optparser: OptionParser, options: Values, output_dir_exist_OK: bool = False) -> Values:
    """
    Validate input and output options.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: Validated options object.
    """

    # check input directory
    if getattr(options, 'input') is None:
        error(f'Input directory is not specified, exit!\n')
        optparser.print_help()
        sys.exit(1)
    elif not os.path.isdir(options.input):
        error(f'Input directory not exist, exit: {options.input}')
        sys.exit(1)

    # check output directory
    if getattr(options, 'output') is None and getattr(options, 'oc') is None:
        error(f'Output directory is not specified, exit!\n')
        optparser.print_help()
        sys.exit(1)
    elif getattr(options, 'output') is None and getattr(options, 'oc') is not None:
        options.output = getattr(options, 'oc')
        if os.path.isdir(options.output):
            info(f'Output directory ({options.output}) already exist, overwrite it.')
            shutil.rmtree(options.output)
    elif getattr(options, 'output') is not None:
        if os.path.isdir(options.output):
            if output_dir_exist_OK is False:
                error(f'Output directory ({options.output}) already exist, exit!')
                sys.exit(1)
            else:
                info(f'Output directory ({options.output}) already exist.')

    return options


def validate_train_options(optparser: OptionParser, options: Values) -> Values:
    """
    Validate train options.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: Validated options object.
    """

    # determin random seed
    if getattr(options, 'seed') is None:
        options.seed = randint(0, 10000)

    return options


def validate_NP_options(optparser: OptionParser, options: Values) -> Values:
    """
    Validate Node Pooling options.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: Validated options object.
    """

    # check k
    if getattr(options, 'k') < 2:
        error(f'k must be greater than 1, exit!')
        sys.exit(1)

    # droupout
    if getattr(options, 'dropout') < 0 or getattr(options, 'dropout') > 1:
        warning(f'droupout must be between 0 and 1, using default value!')
        options.dropout = 0

    # check init_node_label
    if getattr(options, 'init_node_label') is None:
        error(f'init_node_label is not specified, exit!')
        sys.exit(1)

    # check assign_exponent
    if getattr(options, 'assign_exponent') < 1:
        warning(f'assign_exponent must be greater than 1, using default value!')
        options.assign_exponent = 1

    return options


def validate_pseudotime_options(optparser: OptionParser, options: Values) -> Values:
    """
    Validate pseudotime options.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: Validated options object.
    """

    return options


def write_basic_options_memo(options: Values) -> None:
    """
    Write basic options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info('           -------- basic options -------            ')
    info(f'input:   {options.input}')
    info(f'output:  {options.output}')


def write_train_options_memo(options: Values) -> None:
    """
    Write train options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info('           -------- train options -------            ')
    info(f'device:  {options.device}')
    info(f'epochs:  {options.epochs}')
    info(f'batch_size:  {options.batch_size}')
    info(f'patience:  {options.patience}')
    info(f'min_delta:  {options.min_delta}')
    info(f'min_epochs:  {options.min_epochs}')
    info(f'seed:  {options.seed}')
    info(f'lr:  {options.lr}')


def write_GNN_options_memo(options: Values) -> None:
    """
    Write GNN options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info(f'hidden_feats:  {options.hidden_feats}')


def write_GSAE_options_memo(options: Values) -> None:
    """
    Write GSAE options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info(f'recon_loss_weight:  {options.recon_loss_weight}')
    info(f'graph_smooth_loss_weight:  {options.graph_smooth_loss_weight}')


def write_NP_options_memo(options: Values) -> None:
    """
    Write Node Pooling options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info(f'k:  {options.k}')
    info(f'droupout:  {options.dropout}')
    info(f'spectral_loss_weight:  {options.spectral_loss_weight}')
    info(f'ortho_loss_weight:  {options.ortho_loss_weight}')
    info(f'cluster_loss_weight:  {options.cluster_loss_weight}')
    info(f'bin_spectral_loss_weight:  {options.bin_spectral_loss_weight}')
    info(f'bin_ortho_loss_weight:  {options.bin_ortho_loss_weight}')
    info(f'bin_cluster_loss_weight:  {options.bin_cluster_loss_weight}')
    info(f'feat_similarity_loss_weight:  {options.feat_similarity_loss_weight}')
    info(f'assign_exponent:  {options.assign_exponent}')


def write_pseudotime_options_memo(options: Values) -> None:
    """
    Write pseudotime options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info('           -------- pseudotime options -------            ')
    info(f'init_node_label:  {options.init_node_label}')


__all__ = [
    'add_basic_options_group', 'add_train_options_group', 'add_GNN_options_group', 'add_GSAE_options_group',
    'add_NP_options_group', 'add_pseudotime_options_group', 'validate_basic_options', 'validate_train_options',
    'validate_NP_options', 'validate_pseudotime_options', 'write_basic_options_memo', 'write_train_options_memo',
    'write_GNN_options_memo', 'write_GSAE_options_memo', 'write_NP_options_memo', 'write_pseudotime_options_memo'
]
