from optparse import OptionParser, Values

from ..log import *
from ..version import __version__
from ._IO import *
from ._train import *

# ------------------------------------
# Constants
# ------------------------------------
IO_OPTIONS = ['preprocessing_dir', 'GNN_dir', 'NTScore_dir']


# ------------------------------------
# Functions
# ------------------------------------
def prepare_NT_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    :return: OptionParser object.
    """
    usage = f'''USAGE: %prog <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR> 
            [--trajectory-construct TRAJECTORY_CONSTRUCT] [--DM-embedding-index DM_EMBEDDING_INDEX] [--equal-space]'''
    description = 'Niche trajectory: construct niche trajectory for niche cluster and project the NT score to each cell'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}', description=description, usage=usage, add_help_option=True)

    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)
    add_NT_options_group(optparser=optparser)

    return optparser


def add_NT_options_group(optparser: OptionParser) -> None:
    """
    Add niche trajectory options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """

    group_NT = OptionGroup(optparser, "Options for niche trajectory")
    optparser.add_option_group(group_NT)
    group_NT.add_option(
        '--trajectory-construct',
        dest='trajectory_construct',
        default='BF',
        choices=['BF', 'TSP', 'DM'],
        help=
        "Method to construct the niche trajectory. Choices: BF (brute force), TSP (Travelling salesman problem), DM (diffusion map). Default is 'BF' (brute-force)."
    )
    group_NT.add_option(
        '--DM-embedding-index',
        dest='DM_embedding_index',
        default=1,
        type='int',
        help=
        'The index of the embedding in the diffusion map. Valid only when --trajectory-construct is set to DM. Default is 1 which means the first embedding.'
    )
    group_NT.add_option(
        '--equal-space',
        dest='equal_space',
        action='store_true',
        default=False,
        help=
        'Whether to assign equally spaced values to for each niche cluster. Default is False, based on total loadings of each niche cluster.'
    )


def validate_NT_options(optparser: OptionParser, options: Values) -> None:
    """
    Validate niche trajectory options from a OptParser object.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: None.
    """

    if options.trajectory_construct != 'DM' and options.DM_embedding_index is not None:
        options.DM_embedding_index = None

    if options.trajectory_construct == 'DM' and options.DM_embedding_index < 1:
        error('The embedding index should be greater than or equal to 1.')
        optparser.print_help()
        sys.exit(1)

    if options.trajectory_construct == 'DM' and not isinstance(options.DM_embedding_index, int):
        error('The embedding index should be an integer.')
        optparser.print_help()
        sys.exit(1)


def write_NT_options_memo(options: Values) -> None:
    """
    Write niche trajectory options memo.
    :param options: Values, options.
    :return: None.
    """

    info('---------------- Niche trajectory options ----------------')
    info(f'Equally spaced niche cluster scores: {options.equal_space}')
    info(f'Niche trajectory construction method: {options.trajectory_construct}')
    if options.trajectory_construct == 'DM':
        info(f'Diffusion map embedding index: {options.DM_embedding_index}')


def opt_NT_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    :param optparser: OptionParser object.
    :return: Values object.
    """

    (options, args) = optparser.parse_args()

    validate_io_options(optparser, options, IO_OPTIONS)
    validate_NT_options(optparser, options)

    # print parameters to stdout
    info('------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, IO_OPTIONS)
    write_NT_options_memo(options)
    info('--------------- RUN params memo end ----------------- ')

    return options
