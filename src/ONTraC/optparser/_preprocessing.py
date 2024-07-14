import sys
from optparse import OptionGroup, OptionParser, Values

from ..log import *
from ..version import __version__
from ._IO import *

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['input', 'preprocessing_dir']


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
                           help='Number of CPUs used for parallel computing in dataset preprocessing. Default is 4.')
    group_niche.add_option(
        '--n-neighbors',
        dest='n_neighbors',
        type='int',
        default=50,
        help=
        'Number of neighbors used for kNN graph construction. It should be less than the number of cells in each sample. Default is 50.'
    )
    group_niche.add_option(
        '--n-local',
        dest='n_local',
        type='int',
        default=20,
        help=
        'Specifies the nth closest local neighbors used for gaussian distance normalization. It should be less than the number of cells in each sample. Default is 20.'
    )
    group_niche.add_option(
        '--embedding-adjust',
        dest='embedding_adjust',
        action='store_true',
        default=False,
        help=
        'Adjust the cell type coding according to embeddings. Default is False. At least two (Embedding_1 and Embedding_2) should be in the original data if embedding_adjust is True.'
    )
    group_niche.add_option(
        '--sigma',
        dest='sigma',
        type='float',
        default=1,
        help='Sigma for the exponential function to control the similarity between different cell types or clusters. Default is 1.')
    optparser.add_option_group(group_niche)


def validate_niche_net_constr_options(optparser: OptionParser, options: Values) -> None:
    """
    Validate niche network construction options.
    :param optparser: OptionParser object.
    :param options: Options object.
    """
    if options.n_cpu < 1:
        error('n_cpu must be greater than 0.')
        optparser.print_help()
        sys.exit(1)

    if options.n_neighbors < 1:
        error('n_neighbors must be greater than 0.')
        optparser.print_help()
        sys.exit(1)

    if options.n_local < 1:
        error('n_local must be greater than 0.')
        optparser.print_help()
        sys.exit(1)

    if options.embedding_adjust:
        if options.embedding_input is None and options.exp_input is None:
            error('Please provide an embedding file or expression data file in csv format.')
            optparser.print_help()
            sys.exit(1)


def write_niche_net_constr_memo(options: Values):
    """Write niche network construction memos to stdout.

    Args:
        options: Options object.
    """

    # print parameters to stdout
    info('      -------- niche net constr options -------      ')
    info(f'n_cpu:   {options.n_cpu}')
    info(f'n_neighbors: {options.n_neighbors}')
    info(f'n_local: {options.n_local}')
    info(f'embedding_adjust: {options.embedding_adjust}')
    info(f'sigma: {options.sigma}')


def prepare_preprocessing_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    """

    usage = f'''USAGE: %prog <-d DATASET> <--preprocessing-dir PREPROCESSING_DIR>
    [--n-cpu N_CPU] [--n-neighbors N_NEIGHBORS] [--n-local N_LOCAL] [--embedding-adjust] [--sigma SIGMA]'''
    description = 'Create dataset for follwoing analysis.'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    # I/O options group
    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)

    # niche network construction options group
    add_niche_net_constr_options_group(optparser)

    return optparser


def opt_preprocessing_validate(optparser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """

    (options, args) = optparser.parse_args()

    validate_io_options(optparser=optparser, options=options, io_options=IO_OPTIONS)
    validate_niche_net_constr_options(optparser, options)

    # print parameters to stdout
    info('------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, IO_OPTIONS)
    write_niche_net_constr_memo(options)
    info('--------------- RUN params memo end ----------------- ')

    return options
