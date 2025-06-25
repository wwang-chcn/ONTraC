import sys
from optparse import OptionGroup, OptionParser, Values
from typing import Optional

from matplotlib.artist import get

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
        choices=['BF'],
        help="Method to construct the niche trajectory. Default is 'BF' (brute-force)")
    group_NT.add_option(
        '--equal-space',
        dest='equal_space',
        default=True,
        action='store_true',
        help="This options will be deprecated from v3.0. Always set to True.")


def validate_NT_options(options: Values, optparser: Optional[OptionParser] = None) -> None:
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
    elif options.trajectory_construct not in ['BF']:
        error('trajectory_construct must be BF')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    # equal_space
    if getattr(options, 'equal_space', None) is None:
        info('equal_space is not set. Using default value True.')
        options.equal_space = True
    elif not isinstance(options.equal_space, bool):
        error('equal_space must be a boolean value')
        if optparser is not None: optparser.print_help()
        sys.exit(1)
    else:
        info(f'equal_space is set to {options.equal_space}. This option will be deprecated from v3.0. Always set to True.')
        options.equal_space = True


def write_NT_options_memo(options: Values) -> None:
    """
    Write niche trajectory options memo.
    :param options: Values, options.
    :return: None.
    """

    info('---------------- Niche trajectory options ----------------')
    info(f'Equally spaced niche cluster scores: {options.equal_space}')
    info(f'Niche trajectory construction method: {options.trajectory_construct}')
