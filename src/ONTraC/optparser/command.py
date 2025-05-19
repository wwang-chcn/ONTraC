from optparse import OptionGroup, OptionParser, Values
from typing import Dict, List

from ..constants import IO_OPTIONS  # type: ignore
from ..log import *
from ..optparser._analysis import *
from ..version import __version__
from ._IO import *
from ._NN import *
from ._NT import *
from ._preprocessing import *
from ._train import *


# ------------------------------------
# ONTraC functions
# ------------------------------------
def prepare_ontrac_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in this function first.
    :return: OptionParser object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC']  # type: ignore

    # usage and description
    usage = f'''USAGE: %prog <--NN-dir NN_DIR> <--GNN-dir GNN_DIR> <--NT-dir NT_DIR> <--meta-input META_INPUT>
    [--exp-input EXP_INPUT] [--embedding-input EMBEDDING_INPUT] [--low-res-exp-input LOW_RES_EXP_INPUT] [--deconvoluted-ct-composition DECONVOLUTED_CT_COMPOSITION]
    [--deconvoluted-exp-input DECONVOLUTED_EXP_INPUT] [--resolution RESOLUTION] [--deconvolution-method DC_METHOD] [--deconvolution-ct-num DC_CT_NUM]
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL] [--embedding-adjust] [--sigma SIGMA] [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE]
    [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS]
    [--n-gcn-layers N_GCN_LAYERS] [-k K] [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT]
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA] [--trajectory-construct TRAJECTORY_CONSTRUCT] [--equal-space]'''
    description = 'All steps of ONTraC including niche network construction, GNN, and niche construction.'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=io_options)

    # preprocessing options group
    add_preprocessing_options_group(optparser=optparser)

    # Niche net construction options group
    add_niche_net_constr_options_group(optparser=optparser)

    # train and model options
    group_train: OptionGroup = add_train_options_group(optparser=optparser)
    add_GCN_options_group(group_train=group_train)
    add_GP_options_group(group_train=group_train)

    # Niche trajectory options group
    add_NT_options_group(optparser=optparser)

    return optparser


def _opt_ontrac_validate(options: Values, io_options: Dict[str, List[str]], optparser: Optional[OptionParser] = None) -> Values:
    """Validate options from a OptParser object.

    :param options: Options object.
    :param io_options: I/O options.
    :param optparser: OptionParser object.
    :return: Values object.
    """

    # IO
    validate_io_options(options=options, io_options=io_options, optparser=optparser)
    # preprocessing
    validate_preprocessing_options(optparser=optparser, options=options)
    # niche network construction
    validate_niche_net_constr_options(options=options, optparser=optparser)
    # training
    validate_train_options(options=options, optparser=optparser)
    validate_GCN_options(options=options, optparser=optparser)
    validate_GP_options(options=options, optparser=optparser)
    # niche trajectory
    validate_NT_options(options=options, optparser=optparser)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options=options, io_options=io_options)
    write_preprocessing_memo(options=options)
    write_niche_net_constr_memo(options=options)
    write_train_options_memo(options=options)
    write_GCN_options_memo(options=options)
    write_GP_options_memo(options=options)
    write_NT_options_memo(options=options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


def opt_ontrac_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC']  # type: ignore

    (options, args) = optparser.parse_args()

    options = _opt_ontrac_validate(options=options, io_options=io_options, optparser=optparser)

    return options


# ------------------------------------
# ONTraC_NN functions
# ------------------------------------
def prepare_nn_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_NN']  # type: ignore

    # usage and description
    usage = f'''USAGE: %prog <--NN-dir NN_DIR> <--meta-input META_INPUT> [--exp-input EXP_INPUT]
    [--embedding-input EMBEDDING_INPUT] [--low-res-exp-input LOW_RES_EXP_INPUT]
    [--deconvoluted-ct-composition DECONVOLUTED_CT_COMPOSITION] [--deconvoluted-exp-input DECONVOLUTED_EXP_INPUT]
    [--resolution RESOLUTION] [--deconvolution-method DC_METHOD] [--deconvolution-ct-num DC_CT_NUM]
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL] [--embedding-adjust] [--sigma SIGMA]'''
    description = 'Create niche network and calculate features (normalized cell type composition). (Step 1 of ONTraC)'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=io_options)

    # preprocessing options group
    add_preprocessing_options_group(optparser=optparser)

    # niche network construction options group
    add_niche_net_constr_options_group(optparser=optparser)

    return optparser


def _opt_nn_validate(options: Values,
                     io_options: Dict[str, List[str]],
                     optparser: Optional[OptionParser] = None) -> Values:
    """Validate options from a OptParser object.

    Parameters
    ----------
    options : Values
        Options object.
    io_options : Dict[str, List[str]]
        I/O options.
    optparser : Optional[OptionParser], optional
        OptionParser object, by default None

    Returns
    -------
    Values
        Values object.
    """

    # IO
    validate_io_options(options=options, io_options=io_options, optparser=optparser)
    # niche network construction
    validate_niche_net_constr_options(options=options, optparser=optparser)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options=options, io_options=io_options)
    write_niche_net_constr_memo(options=options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


def opt_nn_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_NN']  # type: ignore

    (options, args) = optparser.parse_args()

    options = _opt_nn_validate(options=options, io_options=io_options, optparser=optparser)

    return options


# ------------------------------------
# ONTraC_GNN functions
# ------------------------------------
def prepare_gnn_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_GNN']  # type: ignore

    # usage and description
    usage = f'''USAGE: %prog <--NN-dir NN_DIR> <--GNN-dir GNN_DIR> [--device DEVICE]
    [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [--n-gcn-layers N_GCN_LAYERS] [-k K]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]'''
    description = 'Graph Neural Network (GNN, GCN + GP). The core algorithm of ONTraC. (Step 2/3 of ONTraC)'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=io_options)

    # train and model options
    group_train: OptionGroup = add_train_options_group(optparser=optparser)
    add_GCN_options_group(group_train=group_train)
    add_GP_options_group(group_train=group_train)

    return optparser


def _opt_gnn_validate(options: Values,
                      io_options: Dict[str, List[str]],
                      optparser: Optional[OptionParser] = None) -> Values:
    """Validate options from a OptParser object.

    Parameters
    ----------
    options : Values
        Options object.
    io_options : Dict[str, List[str]]
        I/O options.
    optparser : Optional[OptionParser], optional
        OptionParser object, by default None

    Returns
    -------
    Values
        Values object.
    """

    # IO
    validate_io_options(options=options, io_options=io_options, optparser=optparser)
    # training
    validate_train_options(options=options, optparser=optparser)
    validate_GCN_options(options=options, optparser=optparser)
    validate_GP_options(options=options, optparser=optparser)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options=options, io_options=io_options)
    write_train_options_memo(options=options)
    write_GCN_options_memo(options=options)
    write_GP_options_memo(options=options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


def opt_gnn_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_GNN']  # type: ignore

    (options, args) = optparser.parse_args()

    options = _opt_gnn_validate(options=options, io_options=io_options, optparser=optparser)

    return options


# ------------------------------------
# ONTraC_NT functions
# ------------------------------------
def prepare_nt_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_NT']  # type: ignore

    # usage and description
    usage = f'''USAGE: %prog <--NN-dir NN_DIR> <--GNN-dir GNN_DIR> <--NT-dir NT_DIR> 
            [--trajectory-construct TRAJECTORY_CONSTRUCT] [--equal-space]'''
    description = 'ONTraC_NT: construct niche trajectory for niche cluster and project the NT score to each cell. (Step 4 of ONTraC)'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    add_IO_options_group(optparser=optparser, io_options=io_options)
    add_NT_options_group(optparser=optparser)

    return optparser


def _opt_nt_validate(options: Values,
                     io_options: Dict[str, List[str]],
                     optparser: Optional[OptionParser] = None) -> Values:
    """Validate options from a OptParser object.

    Parameters
    ----------
    options : Values
        Options object.
    io_options : Dict[str, List[str]]
        I/O options.
    optparser : Optional[OptionParser], optional
        OptionParser object, by default None

    Returns
    -------
    Values
        Values object.
    """

    # IO
    validate_io_options(options=options, io_options=io_options, optparser=optparser)
    # niche trajectory
    validate_NT_options(options=options, optparser=optparser)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options=options, io_options=io_options)
    write_NT_options_memo(options=options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


def opt_nt_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_NT']  # type: ignore

    (options, args) = optparser.parse_args()

    options = _opt_nt_validate(options=options, io_options=io_options, optparser=optparser)

    return options


# ------------------------------------
# ONTraC_GT functions
# ------------------------------------
def prepare_gt_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_GT']  # type: ignore

    usage = f'''USAGE: %prog <--NN-dir NN_DIR> <--GNN-dir GNN_DIR> <--NT-dir NT_DIR> [--device DEVICE]
    [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [--n-gcn-layers N_GCN_LAYERS] [-k K]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA] [--trajectory-construct TRAJECTORY_CONSTRUCT]
    [--equal-space]'''
    description = 'ONTraC_GT: GNN and Niche Trajectory'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=io_options)

    # train and model options
    group_train: OptionGroup = add_train_options_group(optparser=optparser)
    add_GCN_options_group(group_train=group_train)
    add_GP_options_group(group_train=group_train)

    # Niche trajectory
    add_NT_options_group(optparser=optparser)

    return optparser


def _opt_gt_validate(options: Values,
                     io_options: Dict[str, List[str]],
                     optparser: Optional[OptionParser] = None) -> Values:
    """Validate options from a OptParser object.

    Parameters
    ----------
    options : Values
        Options object.
    io_options : Dict[str, List[str]]
        I/O options.
    optparser : Optional[OptionParser], optional
        OptionParser object, by default None

    Returns
    -------
    Values
        Values object.
    """

    # IO
    validate_io_options(options=options, io_options=io_options, optparser=optparser)
    # training
    validate_train_options(options=options, optparser=optparser)
    validate_GCN_options(options=options, optparser=optparser)
    validate_GP_options(options=options, optparser=optparser)
    # niche trajectory
    validate_NT_options(options=options, optparser=optparser)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, io_options)
    write_train_options_memo(options)
    write_GCN_options_memo(options)
    write_GP_options_memo(options)
    write_NT_options_memo(options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


def opt_gt_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_GT']  # type: ignore

    (options, args) = optparser.parse_args()

    options = _opt_gt_validate(options=options, io_options=io_options, optparser=optparser)

    return options


# ------------------------------------
# ONTraC_analysis functions
# ------------------------------------
def prepare_analysis_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_analysis']  # type: ignore

    usage = f'''USAGE: %prog [--NN-dir NN_DIR] [--GNN-dir GNN_DIR] [--NT-dir NT_DIR] [-o OUTPUT]
    [--meta-input META_INPUT] [-l LOG] [--embedding-adjust] [--sigma SIGMA] [-r REVERSE] [-s SAMPLE]
    [--scale-factor SCALE_FACTOR] [--suppress-cell-type-composition] [--suppress-niche-cluster-loadings]
    [--suppress-niche-trajectory]
    '''
    description = 'ONTraC_analysis: analysis of ONTraC results'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    add_IO_options_group(optparser=optparser, io_options=io_options)
    add_visualization_group(optparser)
    add_suppress_group(optparser)

    return optparser


def _opt_analysis_validate(options: Values,
                           io_options: Dict[str, List[str]],
                           optparser: Optional[OptionParser] = None) -> Values:
    """Validate options from a OptParser object.

    Parameters
    ----------
    options : Values
        Options object.
    io_options : Dict[str, List[str]]
        I/O options.
    optparser : Optional[OptionParser], optional
        OptionParser object, by default None

    Returns
    -------
    Values
        Values object.
    """

    # IO
    validate_io_options(options=options, io_options=io_options, optparser=optparser)
    # visualization
    validate_visualization_options(options, optparser=optparser)
    validate_suppress_options(options, optparser=optparser)

    # print parameters to stdout
    info(message='------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, io_options)
    write_visualization_options_memo(options)
    write_suppress_options_memo(options)
    info(message='--------------- RUN params memo end ----------------- ')

    return options


def opt_analysis_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    # args
    io_options: Dict[str, List[str]] = IO_OPTIONS['ONTraC_analysis']  # type: ignore

    (options, args) = optparser.parse_args()

    options = _opt_analysis_validate(options=options, io_options=io_options, optparser=optparser)

    return options


# ------------------------------------
# functions to be exported
# ------------------------------------
__all__ = [
    'prepare_ontrac_optparser', 'opt_ontrac_validate', 'prepare_nn_optparser', 'opt_nn_validate',
    'prepare_gnn_optparser', 'opt_gnn_validate', 'prepare_nt_optparser', 'opt_nt_validate', 'prepare_gt_optparser',
    'opt_gt_validate', 'prepare_analysis_optparser', 'opt_analysis_validate'
]
