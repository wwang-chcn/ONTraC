import os
import shutil
import sys
from optparse import OptionGroup, OptionParser, Values

from ONTraC.optparser._GP import *
from ONTraC.optparser._train import *

from ..log import *
# ------------------------------------
# Helper Functions
# ------------------------------------
def create_ds_output_check(optparser, options):
    # check output directory
    if getattr(options, 'preprocessing_dir') is None:
        error(f'Output directory is not specified, exit!\n')
        optparser.print_help()
        sys.exit(1)
    elif getattr(options, 'preprocessing_dir') is not None:
        if os.path.isdir(options.preprocessing_dir):
            error(f'Output directory ({options.output}) already exist, exit!')
            sys.exit(1)
        os.makedirs(options.preprocessing_dir)

def create_ds_original_data_check(optparser, options):
    # check original data file
    example_original_data_file = os.path.join(os.path.dirname(__file__), '../example_files/example_original_data.csv')
    if getattr(options, 'dataset') is None:
        error(f'Original dataset is not specified, exit!')
        error(f'You can find example original data file in {example_original_data_file}')
        optparser.print_help()
        sys.exit(1)
    elif not os.path.isfile(options.dataset):
        error(f'Original dataset file does not exist, exit: {options.dataset}')
        error(f'You can find example original data file in {example_original_data_file}')
        sys.exit(1)
    elif not options.dataset.endswith('.csv'):
        error(f'Original data file must ends with .csv, exit: {options.dataset}')
        error(f'You can find example original data file in {example_original_data_file}')
        sys.exit(1)

# ------------------------------------
# ONTraC Functions
# ------------------------------------
def prepare_ontrac_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in this function first.
    """
    program_name = os.path.basename(sys.argv[0])
    usage = f'''USAGE: {program_name} <-d DATASET> [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--device DEVICE] [--epochs EPOCHS]
    [--patience PATIENCE] [--min-delta MIN_DELTA] [--min-epochs MIN_EPOCHS] [--batch-size BATCH_SIZE] [-s SEED] [--seed SEED]
    [--lr LR] [--hidden-feats HIDDEN_FEATS] [-k K_CLUSTER] [--k-cluster K_CLUSTER] [--spectral-loss-weight SPECTRAL_LOSS_WEIGHT]
    [--cluster-loss-weight CLUSTER_LOSS_WEIGHT] [--feat-similarity-loss-weight FEAT_SIMILARITY_LOSS_WEIGHT] [--assign-exponent ASSIGN_EXPONENT]
    [--preprocessing-dir PREPROCESSING_DIR] [--GNN-dir GNN_DIR] [--NTScore-dir NTSCORE_DIR]'''
    description = 'All steps of ONTraC including dataset createion and GP (Graph Pooling - GNN & Node Pooling)'

    # option processor
    optparser = OptionParser(version=f'{program_name} 0.1', description=description, usage=usage, add_help_option=True)

    # Create Dataset
    group_basic = OptionGroup(optparser, "Basic options for preprocessing")
    group_basic.add_option('-d',
                           '--dataset',
                           dest='dataset',
                           type='string',
                           help='Original input dataset.')
    group_basic.add_option('--n-cpu',
                           dest='n_cpu',
                           type='int',
                           default=4,
                           help='Number of CPUs used for parallel computing. Default is 4.')
    group_basic.add_option('--n-neighbors',
                           dest='n_neighbors',
                           type='int',
                           default=50,
                           help='Number of neighbors used for kNN graph construction. Default is 50.')
    optparser.add_option_group(group_basic)

    # GP
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_NP_options_group(group_train)

    # Output
    group_output = OptionGroup(optparser, "Options for output directories")
    group_output.add_option('--preprocessing-dir',
                           dest='preprocessing_dir',
                           type='string',
                           help='Directory for preprocessing outputs.')
    group_output.add_option('--GNN-dir',
                           dest='GNN_dir',
                           type='string',
                           help='Directory for the GNN output.')
    group_output.add_option('--NTScore-dir',
                           dest='NTScore_dir',
                           type='string',
                           help='Directory for the NTScore output')

    return optparser

# _ONTraC
def opt_ontrac_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """
    (options, args) = optparser.parse_args()

    # Create Dataset
    create_ds_output_check(optparser, options)
    create_ds_original_data_check(optparser, options)

    # GP
    validate_basic_options(optparser, options)
    validate_train_options(optparser, options)

    # print parameters to stdout
    info('--------------------- RUN memo ---------------------')
    info('       -------- create dataset options -------      ')
    info(f'output:  {options.preprocessing_dir}')
    info(f'dataset: {options.dataset}')
    info(f'n_cpu:   {options.n_cpu}')
    info(f'n_neighbors: {options.n_neighbors}')
    info('             -------- GP options -------            ')
    write_basic_options_memo(options)
    write_train_options_memo(options)
    write_GNN_options_memo(options)
    write_NP_options_memo(options)
    info('----------------------------------------------------')

    return options
