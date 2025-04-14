import sys
from optparse import OptionGroup, OptionParser, Values
from typing import Optional

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


def write_NT_options_memo(options: Values) -> None:
    """
    Write niche trajectory options memo.
    :param options: Values, options.
    :return: None.
    """

    info('---------------- Niche trajectory options ----------------')
    info(f'Niche trajectory construction method: {options.trajectory_construct}')
