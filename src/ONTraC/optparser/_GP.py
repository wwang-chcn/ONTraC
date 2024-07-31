from optparse import OptionParser, Values

from ..log import *
from ..version import __version__
from ._IO import *
from ._train import *
from ._NT import add_NT_options_group, write_NT_options_memo

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['input', 'preprocessing_dir', 'GNN_dir', 'NTScore_dir']


# ------------------------------------
# Functions
# ------------------------------------
def prepare_GP_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    """
    usage = f'''USAGE: %prog <--meta-input META_INPUT> [--exp-input EXP_INPUT] [--embedding-input EMBEDDING_INPUT]
    [--decomposition-cell-type-composition-input DECOMPOSITION_CELL_TYPE_COMPOSITION_INPUT]
    [--decomposition-expression-input DECOMPOSITION_EXPRESSION_INPUT] <--preprocessing-dir PREPROCESSING_DIR>
    <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>  [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE]
    [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED] [--lr LR]
    [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS] [--modularity-loss-weight MODULARITY_LOSS_WEIGHT]
    [--purity-loss-weight PURITY_LOSS_WEIGHT] [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]'''
    description = 'GP (Graph Pooling): GNN & Node Pooling'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}',
                             description=description,
                             usage=usage,
                             add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)

    # train and model options
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_NP_options_group(group_train)

    # Niche trajectory
    add_NT_options_group(optparser)

    return optparser


def opt_GP_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """

    (options, args) = optparser.parse_args()

    validate_io_options(optparser, options, IO_OPTIONS)
    validate_train_options(optparser, options)

    info('------------------ RUN params memo ------------------ ')
    # print parameters to stdout
    write_io_options_memo(options, IO_OPTIONS)
    write_train_options_memo(options)
    write_GNN_options_memo(options)
    write_NP_options_memo(options)
    write_NT_options_memo(options)
    info('--------------- RUN params memo end ----------------- ')

    return options
