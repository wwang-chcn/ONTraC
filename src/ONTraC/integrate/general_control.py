import os
import shutil
from optparse import Values
from typing import List, Optional

import pandas as pd

from ..log import *
from ..optparser._IO import write_io_options_memo
from ..optparser._NN import write_niche_net_constr_memo
from ..optparser._NT import write_NT_options_memo
from ..optparser._train import write_GCN_options_memo, write_GP_options_memo, write_train_options_memo
from ..run.processes import niche_trajectory_construct, gnn, niche_network_construct


def io_opt_valid(options: Values, process='ontrac', io_options: Optional[List[str]] = None) -> Values:
    """
    Validate I/O options
    :param options: options
    :param process: str, process name
    :param io_options: List of I/O options.
    :return: options
    """

    # TODO: valida based on process

    if io_options is None:
        return options

    # input files
    if 'input' in io_options:

        # meta_input
        if hasattr(options, 'dataset') and not hasattr(options, 'meta_input'):
            warning('The dataset option will be deprecated from v3.0. Please use meta_input instead.')
            options.meta_input = options.dataset
        if not hasattr(options, 'meta_input'):
            raise AttributeError('Please provide a meta_input in options.')
        if not os.path.isfile(options.meta_input):
            raise ValueError(f'The input file ({options.meta_input}) you given does not exist.')
        if not options.meta_input.endswith(('csv', 'csv.gz')):
            raise ValueError(f'The input file ({options.meta_input}) should be in csv format.')

        # low resolution expression data
        if hasattr(options, 'deconvoluted_ct_composition'):
            if not os.path.isfile(options.deconvoluted_ct_composition):
                raise ValueError(f'The The deconvoluted cell type composition file ({options.deconvoluted_ct_composition}) you given does not exist.')
            if not options.deconvoluted_ct_composition.endswith(('csv', 'csv.gz')):
                raise ValueError(f'The deconvoluted cell type composition file ({options.deconvoluted_ct_composition}) should be in csv format.')

    # NN_dir
    if 'NN_dir' in io_options:
        if hasattr(options, 'preprocessing_dir') and not hasattr(options, 'NN_dir'):
            warning('The preprocessing_dir option will be deprecated from v3.0. Please use NN_dir instead.')
            options.NN_dir = options.preprocessing_dir
        if not hasattr(options, 'NN_dir'):
            raise AttributeError('Please provide a directory for niche network outputs.')
        if os.path.isdir(options.NN_dir):
            warning(f'The directory ({options.NN_dir}) you given already exists. It will be overwritten.')
            shutil.rmtree(options.NN_dir)
        else:
            info(f'Create directory: {options.NN_dir}')
        os.makedirs(options.NN_dir)

    # GNN_dir
    if 'GNN_dir' in io_options:
        if not hasattr(options, 'GNN_dir'):
            raise AttributeError('Please provide a directory for the GNN output.')
        if os.path.isdir(options.GNN_dir):
            warning(f'The directory ({options.GNN_dir}) you given already exists. It will be overwritten.')
            shutil.rmtree(options.GNN_dir)
        else:
            info(f'Create directory: {options.GNN_dir}')
        os.makedirs(options.GNN_dir)

    # NT_dir
    if 'NT_dir' in io_options:
        if hasattr(options, 'NTScore_dir') and not hasattr(options, 'NT_dir'):
            warning('The NTScore_dir option will be deprecated from v3.0. Please use NT_dir instead.')
            options.NT_dir = options.NTScore_dir
        if not hasattr(options, 'NT_dir'):
            raise AttributeError('Please provide a directory for the niche trajectory output.')
        if os.path.isdir(options.NT_dir):
            warning(f'The directory ({options.NT_dir}) you given already exists. It will be overwritten.')
            shutil.rmtree(options.NT_dir)
        else:
            info(f'Create directory: {options.NT_dir}')
        os.makedirs(options.NT_dir)

    return options


def preprocessing_opt_valid(options: Values, process='ontrac') -> Values:
    """
    Validate preprocessing options
    :param options: options
    :param process: str, process name
    :return: options
    """

    return options


def niche_net_opt_valid(options: Values, process='ontrac') -> Values:
    """
    Validate niche network construction options
    :param options: options
    :param process: str, process name
    :return: options
    """

    # TODO: valida based on process

    # n_cpu
    if not hasattr(options, 'n_cpu'):
        options.n_cpu = 4
    elif not isinstance(options.n_cpu, int):
        raise ValueError(f'n_cpu should be an integer. You provided {options.n_cpu}.')
    else:
        options.n_cpu = int(options.n_cpu)
    if options.n_cpu < 1:
        raise ValueError(f'n_cpu should be greater than 0. You provided {options.n_cpu}.')

    # n_neighbors
    if not hasattr(options, 'n_neighbors'):
        options.n_neighbors = 50
    elif not isinstance(options.n_neighbors, int):
        raise ValueError(f'n_neighbors should be an integer. You provided {options.n_neighbors}.')
    else:
        options.n_neighbors = int(options.n_neighbors)
    if options.n_neighbors < 1:
        raise ValueError(f'n_neighbors should be greater than 0. You provided {options.n_neighbors}.')

    # n_local
    if not hasattr(options, 'n_local'):
        options.n_local = 20
    elif not isinstance(options.n_local, int):
        raise ValueError(f'n_local should be an integer. You provided {options.n_local}.')
    else:
        options.n_local = int(options.n_local)
    if options.n_local < 2:
        raise ValueError(f'n_local should be greater than 1. You provided {options.n_local}.')

    return options


def gnn_opt_valid(options: Values, process='ontrac') -> Values:
    """
    Validate GNN options
    :param options: options
    :param process: str, process name
    :return: options
    """

    # TODO: valida based on process

    # device
    if not hasattr(options, 'device'):
        options.device = 'cpu'
    elif not isinstance(options.device, str):
        raise ValueError(f'device should be a string. You provided {options.device}.')
    else:
        options.device = str(options.device)
    if options.device not in ['cpu', 'cuda']:
        raise ValueError(f'device should be either "cpu" or "cuda". You provided {options.device}.')

    # epochs
    if not hasattr(options, 'epochs'):
        options.epochs = 1000
    elif not isinstance(options.epochs, int):
        raise ValueError(f'epochs should be an integer. You provided {options.epochs}.')
    else:
        options.epochs = int(options.epochs)
    if options.epochs < 1:
        raise ValueError(f'epochs should be greater than 0. You provided {options.epochs}.')

    # patience
    if not hasattr(options, 'patience'):
        options.patience = 100
    elif not isinstance(options.patience, int):
        raise ValueError(f'patience should be an integer. You provided {options.patience}.')
    else:
        options.patience = int(options.patience)
    if options.patience < 1:
        raise ValueError(f'patience should be greater than 0. You provided {options.patience}.')

    # min_delta
    if not hasattr(options, 'min_delta'):
        options.min_delta = 0.001
    elif not isinstance(options.min_delta, float):
        raise ValueError(f'min_delta should be a float. You provided {options.min_delta}.')
    else:
        options.min_delta = float(options.min_delta)
    if not 0 < options.min_delta < 1:
        raise ValueError(f'min_delta should be in the range of (0, 1). You provided {options.min_delta}.')

    # min_epochs
    if not hasattr(options, 'min_epochs'):
        options.min_epochs = 50
    elif not isinstance(options.min_epochs, int):
        raise ValueError(f'min_epochs should be an integer. You provided {options.min_epochs}.')
    else:
        options.min_epochs = int(options.min_epochs)
    if options.min_epochs < 1:
        raise ValueError(f'min_epochs should be greater than 0. You provided {options.min_epochs}.')

    # batch_size
    if not hasattr(options, 'batch_size'):
        options.batch_size = 0
    elif not isinstance(options.batch_size, int):
        raise ValueError(f'batch_size should be an integer. You provided {options.batch_size}.')
    else:
        options.batch_size = int(options.batch_size)
    if options.batch_size < 0:
        raise ValueError(f'batch_size should be greater than or equal to 0. You provided {options.batch_size}.')

    # seed
    if not hasattr(options, 'seed'):
        options.seed = 42
    elif not isinstance(options.seed, int):
        raise ValueError(f'seed should be an integer. You provided {options.seed}.')
    else:
        options.seed = int(options.seed)

    # lr
    if not hasattr(options, 'lr'):
        options.lr = 0.03
    elif not isinstance(options.lr, float):
        raise ValueError(f'Learning rate should be a float. You provided {options.lr}.')
    else:
        options.lr = float(options.lr)
    if options.lr < 0:
        raise ValueError(f'Learning rate should be greater than 0. You provided {options.lr}.')

    # hidden_feats
    if not hasattr(options, 'hidden_feats'):
        options.hidden_feats = 4
    elif not isinstance(options.hidden_feats, int):
        raise ValueError(f'hidden_feats should be an integer. You provided {options.hidden_feats}.')
    else:
        options.hidden_feats = int(options.hidden_feats)
    if options.hidden_feats < 1:
        raise ValueError(f'hidden_feats should be greater than 0. You provided {options.hidden_feats}.')

    # n_gcn_layers
    if not hasattr(options, 'n_gcn_layers'):
        options.n_gcn_layers = 2
    elif not isinstance(options.n_gcn_layers, int):
        raise ValueError(f'n_gcn_layers should be an integer. You provided {options.n_gcn_layers}.')
    else:
        options.n_gcn_layers = int(options.n_gcn_layers)

    # k
    if not hasattr(options, 'k'):
        options.k = 6
    elif not isinstance(options.k, int):
        raise ValueError(f'k-cluster should be an integer. You provided {options.k}.')
    else:
        options.k = int(options.k)
    if options.k < 2:
        raise ValueError(f'k-cluster should be greater than 1. You provided {options.k}.')

    # modularity_loss_weight
    if not hasattr(options, 'modularity_loss_weight'):
        options.modularity_loss_weight = 0.3
    elif not isinstance(options.modularity_loss_weight, float):
        raise ValueError(f'modularity_loss_weight should be a float. You provided {options.modularity_loss_weight}.')
    else:
        options.modularity_loss_weight = float(options.modularity_loss_weight)
    if options.modularity_loss_weight < 0:
        raise ValueError(
            f'modularity_loss_weight should be greater than or equal to 0. You provided {options.modularity_loss_weight}.'
        )

    # purity_loss_weight
    if not hasattr(options, 'purity_loss_weight'):
        options.purity_loss_weight = 300.0
    elif not isinstance(options.purity_loss_weight, float):
        raise ValueError(f'purity_loss_weight should be a float. You provided {options.purity_loss_weight}.')
    else:
        options.purity_loss_weight = float(options.purity_loss_weight)
    if options.purity_loss_weight < 0:
        raise ValueError(
            f'purity_loss_weight should be greater than or equal to 0. You provided {options.purity_loss_weight}.')

    # regularization_loss_weight
    if not hasattr(options, 'regularization_loss_weight'):
        options.regularization_loss_weight = 0.1
    elif not isinstance(options.regularization_loss_weight, float):
        raise ValueError(
            f'regularization_loss_weight should be a float. You provided {options.regularization_loss_weight}.')
    else:
        options.regularization_loss_weight = float(options.regularization_loss_weight)
    if options.regularization_loss_weight < 0:
        raise ValueError(
            f'regularization_loss_weight should be greater than or equal to 0. You provided {options.regularization_loss_weight}.'
        )

    # beta
    if not hasattr(options, 'beta'):
        options.beta = 0.03
    elif not isinstance(options.beta, float):
        raise ValueError(f'beta should be a float. You provided {options.beta}.')
    else:
        options.beta = float(options.beta)
    if options.beta < 0:
        raise ValueError(f'beta should be greater than or equal to 0. You provided {options.beta}.')

    return options


def nt_opt_valid(options: Values, process='ontrac') -> Values:
    """
    Validate niche trajectory options.
    Placehold and do nothing.

    :param options: options
    :param process: str, process name
    :return: options
    """

    return options


def options_valid(options: Values, process='ontrac') -> Values:
    """
    Validate options
    :param options: options
    :param process: str, process name
    :return: options
    """

    # TODO: valida based on process
    # TODO: integrated into optparser module

    io_options = ['input', 'NN_dir', 'GNN_dir', 'NT_dir']

    # I/O options
    options = io_opt_valid(options=options, process=process, io_options=io_options)
    # preprocessing
    options = preprocessing_opt_valid(options=options, process=process)
    # niche network construction options
    options = niche_net_opt_valid(options=options, process=process)
    # GNN options
    options = gnn_opt_valid(options=options, process=process)
    # niche trajectory
    options = nt_opt_valid(options=options, process=process)

    info('------------------ RUN params memo ------------------ ')
    write_io_options_memo(options=options, io_options=io_options)
    write_niche_net_constr_memo(options=options)
    write_train_options_memo(options=options)
    write_GCN_options_memo(options=options)
    write_GP_options_memo(options=options)
    write_NT_options_memo(options=options)
    info('--------------- RUN params memo end ----------------- ')

    return options


def run_ontrac(options: Values) -> None:
    """
    Run ONTraC
    :param options: options
    :return: None
    """

    # ----- options validation -----
    options = options_valid(options=options)

    # ----- Niche Network Construct -----
    niche_network_construct(options=options)

    # ----- GNN -----
    gnn(options=options)

    # ----- NT score -----
    niche_trajectory_construct(options=options)
