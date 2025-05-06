import sys
from optparse import OptionGroup, OptionParser, Values
from random import randint
from typing import Optional

import torch

from ..log import *


def add_train_options_group(optparser: OptionParser) -> OptionGroup:
    """
    Add train options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """
    # overall train options group
    group_train = OptionGroup(optparser, "Options for GNN training")
    optparser.add_option_group(group_train)
    group_train.add_option('--device',
                           dest='device',
                           type='string',
                           help='Device for training. We support cpu and cuda now. Auto select if not specified.')
    group_train.add_option('--epochs',
                           dest='epochs',
                           type='int',
                           default=1000,
                           help='Number of maximum epochs for training. Default is 1000.')
    group_train.add_option('--patience',
                           dest='patience',
                           type='int',
                           default=100,
                           help='Number of epochs wait for better result. Default is 100.')
    group_train.add_option('--min-delta',
                           dest='min_delta',
                           type='float',
                           default=0.001,
                           help='Minimum delta for better result. Should be in (0, 1). Default is 0.001.')
    group_train.add_option('--min-epochs',
                           dest='min_epochs',
                           type='int',
                           default=50,
                           help='Minimum number of epochs for training. Default is 50. Set to 0 to disable.')
    group_train.add_option('--batch-size',
                           dest='batch_size',
                           type='int',
                           default=0,
                           help='Batch size for training. Default is 0 for whole dataset.')
    group_train.add_option('-s', '--seed', dest='seed', type='int', help='Random seed for training. Default is random.')
    group_train.add_option('--lr',
                           dest='lr',
                           type='float',
                           default=0.03,
                           help='Learning rate for training. Default is 0.03.')
    return group_train


def add_GCN_options_group(group_train: OptionGroup) -> None:
    """
    Add GCN options group to optparser.
    :param optparser: OptionParser object.
    :return: None.
    """

    # GCN options group
    group_train.add_option('--hidden-feats',
                           dest='hidden_feats',
                           type='int',
                           default=4,
                           help='Number of hidden features. Default is 4.')
    group_train.add_option('--n-gcn-layers',
                           dest='n_gcn_layers',
                           type='int',
                           default=2,
                           help='Number of GCN layers. Default is 2.')


def add_GP_options_group(group_train: OptionGroup) -> None:
    """
    Add Graph Pooling options group to optparser.
    :param optparser: OptionParser object.
    :return: None.
    """

    # NP options group
    group_train.add_option('-k',
                           '--k-clusters',
                           dest='k',
                           type='int',
                           default=6,
                           help='Number of niche clusters. Default is 6. No more than 8.')
    group_train.add_option('--modularity-loss-weight',
                           dest='modularity_loss_weight',
                           type='float',
                           default=1,
                           help='Weight for modularity loss. Default is 1.')
    group_train.add_option('--purity-loss-weight',
                           dest='purity_loss_weight',
                           type='float',
                           default=30,
                           help='Weight for purity loss. Default is 30.')
    group_train.add_option('--regularization-loss-weight',
                           dest='regularization_loss_weight',
                           type='float',
                           default=0.1,
                           help='Weight for regularization loss. Default is 0.1.')
    group_train.add_option('--beta',
                           dest='beta',
                           type='float',
                           default=0.3,
                           help='Beta value control niche cluster assignment matrix. Default is 0.3.')


def validate_train_options(options: Values, optparser: Optional[OptionParser] = None) -> None:
    """
    Validate train options.
    
    Parameters
    ----------
    options : Values
        Options object.
    optparser : Optional[OptionParser], optional
        OptionParser object. The default is None.

    Returns
    -------
    None
    """

    # device
    if getattr(options, 'device', None) is None:
        info('Device not specified, choose automatically.')
    elif options.device.startswith(('cuda', 'cpu')):
        if options.device.startswith('cuda') and not torch.cuda.is_available():
            warning('CUDA is not available, use CPU instead.')
            options.device = 'cpu'
    else:
        warning(f'Invalid device {options.device}! Choose automatically.')
        options.device = None
    if getattr(options, 'device', None) is None:
        if torch.cuda.is_available():
            options.device = 'cuda'
        # elif torch.backends.mps.is_available():  # TODO: torch_geometric do not support MPS now
        #     options.device = 'mps'
        else:
            options.device = 'cpu'

    # epochs
    if getattr(options, 'epochs', None) is None:
        info('epochs is not set. Using default value 1000.')
        options.epochs = 1000
    elif not isinstance(options.epochs, int):
        error(f'epochs must be an integer, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.epochs < 1:
        error(f'epochs must be greater than 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # patience
    if getattr(options, 'patience', None) is None:
        info('patience is not set. Using default value 100.')
        options.patience = 100
    elif not isinstance(options.patience, int):
        error(f'patience must be an integer, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.patience < 1:
        error(f'patience must be greater than 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    
    # min_delta
    if getattr(options, 'min_delta', None) is None:
        info('min_delta is not set. Using default value 0.001.')
        options.min_delta = 0.001
    elif not isinstance(options.min_delta, float):
        error(f'min_delta must be a float, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif not 0 < options.min_delta < 1:
        error(f'min_delta must be in (0, 1), exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # min_epochs
    if getattr(options, 'min_epochs', None) is None:
        info('min_epochs is not set. Using default value 50.')
        options.min_epochs = 50
    elif not isinstance(options.min_epochs, int):
        error(f'min_epochs must be an integer, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.min_epochs < 0:
        error(f'min_epochs must be greater than or equal to 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # batch_size
    if getattr(options, 'batch_size', None) is None:
        info('batch_size is not set. Using default value 0.')
        options.batch_size = 0
    elif not isinstance(options.batch_size, int):
        error(f'batch_size must be an integer, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.batch_size < 0:
        error(f'batch_size must be greater than or equal to 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # seed
    if getattr(options, 'seed', None) is None:
        options.seed = randint(0, 10000)
    
    # lr
    if getattr(options, 'lr', None) is None:
        info('lr is not set. Using default value 0.03.')
        options.lr = 0.03
    elif not isinstance(options.lr, float):
        error(f'lr must be a float, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.lr <= 0:
        error(f'lr must be greater than 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)


def validate_GCN_options(options: Values, optparser: Optional[OptionParser] = None) -> None:
    """
    Validate GCN options.
    
    Parameters
    ----------
    options : Values
        Options object.
    optparser : Optional[OptionParser], optional
        Option

    Returns
    -------
    None
    """

    # check hidden_feats
    if getattr(options, 'hidden_feats', None) is None:
        info('hidden_feats is not set. Using default value 4.')
        options.hidden_feats = 4
    elif getattr(options, 'hidden_feats') < 2:
        error(f'hidden_feats must be greater than 1, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    
    # check n_gcn_layers
    if getattr(options, 'n_gcn_layers', None) is None:
        info('n_gcn_layers is not set. Using default value 2.')
        options.n_gcn_layers = 2
    elif getattr(options, 'n_gcn_layers') < 1:
        error(f'n_gcn_layers must be greater than 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)


def validate_GP_options(options: Values, optparser: Optional[OptionParser] = None) -> None:
    """
    Validate Graph Pooling options.
    
    
    Parameters
    ----------
    options : Values
        Options object.
    optparser : Optional[OptionParser], optional
        OptionParser object. The default is None.

    Returns
    -------
    None
    """

    # check k
    if getattr(options, 'k', None) is None:
        info('k is not set. Using default value 6.')
        options.k = 6
    elif not isinstance(options.k, int):
        error(f'k must be an integer, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif getattr(options, 'k') < 2:
        error(f'k must be greater than 1, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif getattr(options, 'k') > 8:
        warning(f'We recommend k to be no more than 8. You can set k to a larger value, but it may cause wired results and extra time cost in niche trajectory construction.')
        if getattr(options, 'k') > 10:
            error(f'k must be no more than 10, exit!')
            if optparser is not None: optparser.print_help()
            sys.exit(1)

    # check modularity_loss_weight
    if getattr(options, 'modularity_loss_weight', None) is None:
        info('modularity_loss_weight is not set. Using default value 0.3.')
        options.modularity_loss_weight = 0.3
    elif not isinstance(options.modularity_loss_weight, (float, int)):
        error(f'modularity_loss_weight must be a number, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.modularity_loss_weight < 0:
        error(f'modularity_loss_weight must be greater than or equal to 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # check purity_loss_weight
    if getattr(options, 'purity_loss_weight', None) is None:
        info('purity_loss_weight is not set. Using default value 300.')
        options.purity_loss_weight = 300
    elif not isinstance(options.purity_loss_weight, (float, int)):
        error(f'purity_loss_weight must be a number, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.purity_loss_weight < 0:
        error(f'purity_loss_weight must be greater than or equal to 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # check regularization_loss_weight
    if getattr(options, 'regularization_loss_weight', None) is None:
        info('regularization_loss_weight is not set. Using default value 0.1.')
        options.regularization_loss_weight = 0.1
    elif not isinstance(options.regularization_loss_weight, (float, int)):
        error(f'regularization_loss_weight must be a number, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.regularization_loss_weight < 0:
        error(f'regularization_loss_weight must be greater than or equal to 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # check beta
    if getattr(options, 'beta', None) is None:
        info('beta is not set. Using default value 0.03.')
        options.beta = 0.03
    elif not isinstance(options.beta, (float, int)):
        error(f'beta must be a number, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    elif options.beta < 0:
        error(f'beta must be greater than or equal to 0, exit!')
        if optparser is not None: optparser.print_help()
        sys.exit(1)


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


def write_GCN_options_memo(options: Values) -> None:
    """
    Write GCN options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info(f'hidden_feats:  {options.hidden_feats}')
    info(f'n_gcn_layers:  {options.n_gcn_layers}')


def write_GP_options_memo(options: Values) -> None:
    """
    Write Graph Pooling options memo to stdout.
    :param options: Options object.
    :return: None.
    """

    info(f'k:  {options.k}')
    info(f'modularity_loss_weight:  {options.modularity_loss_weight}')
    info(f'purity_loss_weight:  {options.purity_loss_weight}')
    info(f'regularization_loss_weight:  {options.regularization_loss_weight}')
    info(f'beta:  {options.beta}')


__all__ = [
    'add_train_options_group',
    'add_GCN_options_group',
    'add_GP_options_group',
    'validate_train_options',
    'validate_GCN_options',
    'validate_GP_options',
    'write_train_options_memo',
    'write_GCN_options_memo',
    'write_GP_options_memo',
]
