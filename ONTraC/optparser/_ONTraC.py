import os
import shutil
import sys
from optparse import OptionGroup, OptionParser, Values

from ONTraC.optparser._create_dataset import *
from ONTraC.optparser._GP import *
from ONTraC.optparser._train import *

from ..log import *

def prepare_ontrac_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in this function first.
    """
    program_name = os.path.basename(sys.argv[0])
    usage = f'''USAGE: {program_name} <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--device DEVICE] [--epochs EPOCHS] [--patience PATIENCE] [--min-delta MIN_DELTA] 
    [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED] [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTER] 
    [--k-cluster K_CLUSTER] [--spectral-loss-weight SPECTRAL_LOSS_WEIGHT] [--cluster-loss-weight CLUSTER_LOSS_WEIGHT] 
    [--feat-similarity-loss-weight FEAT_SIMILARITY_LOSS_WEIGHT] [--assign-exponent ASSIGN_EXPONENT]'''
    description = 'All steps of ONTraC including dataset createion and GP (Graph Pooling - GNN & Node Pooling)'

    # option processor
    optparser = OptionParser(version=f'{program_name} 0.1', description=description, usage=usage, add_help_option=True)

    # IO option group
    group_io = OptionGroup(optparser, "IO")
    group_io.add_option('-d',
                           '--dataset',
                           dest='dataset',
                           type='string',
                           help='Original input dataset.')
    group_io.add_option('--preprocessing-dir',
                           dest='preprocessing_dir',
                           type='string',
                           help='Directory for preprocessing outputs.')
    group_io.add_option('--GNN-dir',
                           dest='GNN_dir',
                           type='string',
                           help='Directory for the GNN output.')
    group_io.add_option('--NTScore-dir',
                           dest='NTScore_dir',
                           type='string',
                           help='Directory for the NTScore output')
    optparser.add_option_group(group_io)

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

    # GP
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_NP_options_group(group_train)

    return optparser

# _ONTraC
def opt_ontrac_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """
    (options, args) = optparser.parse_args()

    # Create Dataset
    opt_create_ds_validate(prepare_create_ds_optparser())

    # GP
    validate_basic_options(optparser, options)
    validate_train_options(optparser, options)

    # print parameters to stdout
    info('--------------------- RUN memo ---------------------')
    info('         -------- IO options -------        ')
    info(f'preprocessing output directory:  {options.preprocessing_dir}')
    info(f'GNN output directory:  {options.GNN_dir}')
    info(f'NTScore output directory:  {options.NTScore_dir}')
    info(f'dataset: {options.dataset}')
    info('     ------ niche net constr options ------    ')
    info(f'n_cpu:   {options.n_cpu}')
    info(f'n_neighbors: {options.n_neighbors}')

    return options
