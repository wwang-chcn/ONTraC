import sys
from optparse import OptionGroup, OptionParser, Values

from ..log import *
from ..version import __version__
from ._IO import *
from ._train import *


# ------------------------------------
# Functions
# ------------------------------------
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


def validate_NT_options(options: Values, optparser: OptionParser) -> None:
    """
    Validate niche trajectory options.

    Parameters
    ----------
    options : Values
        Options object.
    optparser : Optional[OptionParser], optional
        OptionParser object. The default is None.

    Returns
    -------
    None
    """

    # trajectory_construct
    if getattr(options, 'trajectory_construct', None) is None:
        info('trajectory_construct is not set. Using default value BF.')
        options.trajectory_construct = 'BF'
    elif options.trajectory_construct not in ['BF', 'TSP', 'DM']:
        error('trajectory_construct must be either BF, TSP or DM.')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # DM_embedding_index
    if options.trajectory_construct != 'DM' and getattr(options, 'DM_embedding_index', None) is not None:
        options.DM_embedding_index = None
    
    if options.trajectory_construct == 'DM':
        if getattr(options, 'DM_embedding_index', None) is None:
            info('DM_embedding_index is not set. Using default value 1.')
            options.DM_embedding_index = 1
        elif not isinstance(options.DM_embedding_index, int):
            error('The embedding index should be an integer.')
            optparser.print_help()
            sys.exit(1)
        elif options.DM_embedding_index < 1:
            error('The embedding index should be greater than or equal to 1.')
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
