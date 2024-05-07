import os
import sys
from optparse import OptionGroup, OptionParser, Values

from ..log import *
from ._create_dataset import write_niche_net_constr_memo
from ._IO import *
from ._train import *

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['dataset', 'preprocessing_dir', 'GNN_dir', 'NTScore_dir']


# ------------------------------ß------
# Functions
# ------------------------------------
def prepare_ontrac_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in this function first.
    """
    program_name = os.path.basename(sys.argv[0])
    usage = f'''USAGE: {program_name} <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] 
    [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTERS]
    [--modularity-loss-weight MODULARITY_LOSS_WEIGHT] [--purity-loss-weight PURITY_LOSS_WEIGHT] 
    [--regularization-loss-weight REGULARIZATION_LOSS_WEIGHT] [--beta BETA]'''
    description = 'All steps of ONTraC including dataset creation, Graph Pooling, and NT score calculation.'

    # option processor
    optparser = OptionParser(version=f'{program_name} 0.1', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)

    # Niche net construction option group
    group_niche = OptionGroup(optparser, "Niche Network Construction")
    group_niche.add_option('--n-cpu',
                           dest='n_cpu',
                           type='int',
                           default=4,
                           help='Number of CPUs used for parallel computing. Default is 4.')
    group_niche.add_option('--n-neighbors',
                           dest='n_neighbors',
                           type='int',
                           default=50,
                           help='Number of neighbors used for kNN graph construction. Default is 50.')
    optparser.add_option_group(group_niche)

    # Graph Pooling
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_NP_options_group(group_train)

    return optparser


def opt_ontrac_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """
    (options, args) = optparser.parse_args()

    # IO
    validate_io_options(optparser=optparser, options=options, io_options=IO_OPTIONS)
    # training
    validate_train_options(optparser, options)
    validate_NP_options(optparser, options)

    # print parameters to stdout
    info('------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, IO_OPTIONS)
    write_niche_net_constr_memo(options)
    write_train_options_memo(options)
    write_GNN_options_memo(options)
    write_NP_options_memo(options)
    info('--------------- RUN params memo end ----------------- ')

    return options
