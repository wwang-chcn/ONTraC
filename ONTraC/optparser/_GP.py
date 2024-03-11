import os
import sys
from optparse import OptionParser, Values

from ..log import *
from ._train import *

def prepare_GP_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    """
    program_name = os.path.basename(sys.argv[0])
    usage = f'''USAGE: {program_name} <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> [--device DEVICE] 
    [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] 
    [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTER] [--k-cluster K_CLUSTER] 
    [--spectral-loss-weight SPECTRAL_LOSS_WEIGHT] [--cluster-loss-weight CLUSTER_LOSS_WEIGHT] 
    [--feat-similarity-loss-weight FEAT_SIMILARITY_LOSS_WEIGHT] [--assign-exponent ASSIGN_EXPONENT]'''
    description = 'GP (Graph Pooling): GNN & Node Pooling'

    # option processor
    optparser = OptionParser(version=f'{program_name} 0.1', description=description, usage=usage, add_help_option=True)
    group_basic = add_basic_options_group(optparser)
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_NP_options_group(group_train)

    return optparser


def opt_GP_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """

    (options, args) = optparser.parse_args()

    # check program name
    program_name = os.path.basename(sys.argv[0])

    if program_name == 'createDataSet':
        info('--------------------- RUN memo ---------------------')

    validate_basic_options(optparser, options)
    validate_train_options(optparser, options)

    # print parameters to stdout
    write_train_options_memo(options)
    write_GNN_options_memo(options)
    write_NP_options_memo(options)
    info('------------------- RUN memo end -------------------')

    return options
