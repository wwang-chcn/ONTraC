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
    usage = f'''USAGE: {prog_name} <-d DATASET> [-o OUTPUT] [--oc OUTPUT] [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS]'''
    description = 'Create dataset for follwoing analysis.'

    # option processor
    optparser = OptionParser(version=f'{prog_name} 0.1', description=description, usage=usage, add_help_option=True)

    # basic options group
    group_basic = OptionGroup(optparser, "Basic options for running")
    group_basic.add_option(
        '-o',
        '--output',
        dest='output',
        type='string',
        help=
        'Directory to output the result. Won\'t be overwritten if target directory exists. If -o is not specified, -oc must be specified.'
    )
    group_basic.add_option(
        '--oc',
        dest='oc',
        type='string',
        help=
        'Directory to output the result. Will be overwritten if target directory exists. If -o is specified, --oc will be ignored.'
    )
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
    if getattr(options, 'output') is None and getattr(options, 'oc') is None:
        error(f'Output directory is not specified, exit!\n')
        optparser.print_help()
        sys.exit(1)
    elif getattr(options, 'output') is None and getattr(options, 'oc') is not None:
        options.output = getattr(options, 'oc')
        if os.path.isdir(options.output):
            info(f'Output directory ({options.output}) already exist, overwrite it.')
        os.makedirs(options.output, exist_ok=True)
    elif getattr(options, 'output') is not None:
        if os.path.isdir(options.output):
            error(f'Output directory ({options.output}) already exist, exit!')
            sys.exit(1)
        os.makedirs(options.output)

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
    param_text += f'output:  {options.output}\n'
    param_text += f'dataset: {options.dataset}\n'
    param_text += f'n_cpu:   {options.n_cpu}\n'
    param_text += f'n_neighbors: {options.n_neighbors}\n'
    param_text += '---------------------------------------------------- \n'
    sys.stdout.write(param_text)
    sys.stdout.flush()

    return options
