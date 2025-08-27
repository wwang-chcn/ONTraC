from optparse import Values

from ..constants import IO_OPTIONS
from ..log import *
from ..optparser.command import (_opt_analysis_validate, _opt_gnn_validate, _opt_gt_validate, _opt_nn_validate,
                                 _opt_nt_validate, _opt_ontrac_validate)
from ..run.processes import gnn, niche_network_construct, niche_trajectory_construct


def options_validate(options: Values, process='ONTraC') -> Values:
    """
    Validate options
    :param options: options
    :param process: str, process name
    :return: options
    """

    # process should be a key in IO_OPTIONS
    if process not in IO_OPTIONS:
        raise ValueError(f'Invalid process name: {process}. Please choose one from {list(IO_OPTIONS.keys())}.')
    # get I/O options
    io_options = IO_OPTIONS[process]

    if process == 'ONTraC':
        options = _opt_ontrac_validate(options=options, io_options=io_options)
    elif process == 'ONTraC_NN':
        options = _opt_nn_validate(options=options, io_options=io_options)
    elif process == 'ONTraC_GNN':
        options = _opt_gnn_validate(options=options, io_options=io_options)
    elif process == 'ONTraC_NT':
        options = _opt_nt_validate(options=options, io_options=io_options)
    elif process == 'ONTraC_GT':
        options = _opt_gt_validate(options=options, io_options=io_options)
    elif process == 'ONTraC_analysis':
        options = _opt_analysis_validate(options=options, io_options=io_options)
    else:
        raise ValueError(f'Invalid process name: {process}. Please choose one from {list(IO_OPTIONS.keys())}.')

    return options


def run_ontrac(options: Values) -> None:
    """
    Run ONTraC
    :param options: options
    :return: None
    """

    # ----- options validation -----
    options = options_validate(options=options)

    # ----- Niche Network Construct -----
    niche_network_construct(options=options)

    # ----- GNN -----
    gnn(options=options)

    # ----- NT score -----
    niche_trajectory_construct(options=options)
