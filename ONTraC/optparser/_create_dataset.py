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
    usage = f'''USAGE: {prog_name} <-y YAML> [-o OUTPUT] [--oc OUTPUT] [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS]'''
    description = 'Create dataset for follwoing analysis.'

    # option processor
    optparser = OptionParser(version=f'{prog_name} 0.1', description=description, usage=usage, add_help_option=True)

    # basic options group
    group_basic = OptionGroup(optparser, "Basci options for running")
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
    group_basic.add_option('-y',
                           '--yaml',
                           dest='yaml',
                           type='string',
                           help='Yaml file contains input dataset information.')
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
            shutil.rmtree(options.output)
    elif getattr(options, 'output') is not None:
        if os.path.isdir(options.output):
            error(f'Output directory ({options.output}) already exist, exit!')
            sys.exit(1)

    # check YAML file
    example_yaml_file = os.path.join(os.path.dirname(__file__), 'example.yaml')
    if getattr(options, 'yaml') is None:
        error(f'YAML file is not specified, exit!')
        error(f'You can find example YAML file in {example_yaml_file}')
        optparser.print_help()
        sys.exit(1)
    elif not os.path.isfile(options.yaml):
        error(f'YAML file not exist, exit: {options.yaml}')
        error(f'You can find example YAML file in {example_yaml_file}')
        sys.exit(1)
    elif not options.yaml.endswith('.yaml'):
        error(f'YAML file must ends with .yaml, exit: {options.yaml}')
        error(f'You can find example YAML file in {example_yaml_file}')
        sys.exit(1)

    # print parameters to stdout
    param_text = '--------------------- RUN memo --------------------- \n'
    param_text += '           -------- basic options -------            \n'
    param_text += f'output:  {options.output}\n'
    param_text += f'yaml:    {options.yaml}\n'
    param_text += f'n_cpu:   {options.n_cpu}\n'
    param_text += f'n_neighbors: {options.n_neighbors}\n'
    param_text += '---------------------------------------------------- \n'
    sys.stdout.write(param_text)
    sys.stdout.flush()

    return options
