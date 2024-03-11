import os
import shutil
import sys
from optparse import OptionGroup, OptionParser, Values

from ..log import *


def prepare_create_ds_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    """

    prog_name = os.path.basename(sys.argv[0])
    usage = f'''USAGE: {prog_name} <-d DATASET> [--preprocessing-dir PREPROCESSING_DIR] [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS]'''
    description = 'Create dataset for follwoing analysis.'

    # option processor
    optparser = OptionParser(version=f'{prog_name} 0.1', description=description, usage=usage, add_help_option=True)

    # basic options group
    group_basic = OptionGroup(optparser, "Basic options for running")
    group_basic.add_option('--preprocessing-dir',
                        dest='preprocessing_dir',
                        type='string',
                        help='Directory for preprocessing outputs.')
    group_basic.add_option('-d', '--dataset', dest='dataset', type='string', help='Original input dataset.')
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

    return optparser


def opt_create_ds_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """

    (options, args) = optparser.parse_args()

    # check output directory
    if getattr(options, 'preprocessing_dir') is None:
        error(f'Output directory is not specified, exit!\n')
        optparser.print_help()
        sys.exit(1)

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

    # print parameters to stdout
    param_text = '--------------------- RUN memo --------------------- \n'
    param_text += '           -------- basic options -------            \n'
    param_text += f'output:  {options.preprocessing_dir}\n'
    param_text += f'dataset: {options.dataset}\n'
    param_text += f'n_cpu:   {options.n_cpu}\n'
    param_text += f'n_neighbors: {options.n_neighbors}\n'
    param_text += '---------------------------------------------------- \n'
    sys.stdout.write(param_text)
    sys.stdout.flush()

    return options
