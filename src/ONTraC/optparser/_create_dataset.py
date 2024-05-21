import os
import sys
from optparse import OptionGroup, OptionParser, Values

from ..log import *
from ..version import __version__
from ._IO import *

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['dataset', 'preprocessing_dir']


# ------------------------------------
# Functions
# ------------------------------------
def add_niche_net_constr_options_group(optparser: OptionParser) -> None:
    """
    Add niche network construction options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """
    # niche network construction options group
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


def write_niche_net_constr_memo(options: Values):
    """Write niche network construction memos to stdout.

    Args:
        options: Options object.
    """

    # print parameters to stdout
    info('      -------- niche net constr options -------      ')
    info(f'n_cpu:   {options.n_cpu}')
    info(f'n_neighbors: {options.n_neighbors}')


def prepare_create_ds_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    """

    usage = f'''USAGE: %prog <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR> [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS]'''
    description = 'Create dataset for follwoing analysis.'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}',
                             description=description,
                             usage=usage,
                             add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)

    # niche network construction options group
    add_niche_net_constr_options_group(optparser)

    return optparser


def opt_create_ds_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """

    (options, args) = optparser.parse_args()

    validate_io_options(optparser=optparser, options=options, io_options=IO_OPTIONS)

    # print parameters to stdout
    info('------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, IO_OPTIONS)
    write_niche_net_constr_memo(options)
    info('--------------- RUN params memo end ----------------- ')

    return options
