from optparse import OptionParser, Values

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
        choices=['BF', 'TSP'],
        help="Method to construct the niche trajectory. Default is 'BF' (brute-force). A faster alternative is 'TSP'.")


def validate_NT_options(optparser: OptionParser, options: Values) -> None:
    """
    Validate niche trajectory options.
    Placehold and do nothing.

    :param optparser: OptionParser object.
    :param options: Options object.
    :return: None.
    """

    pass


def write_NT_options_memo(options: Values) -> None:
    """
    Write niche trajectory options memo.
    :param options: Values, options.
    :return: None.
    """

    info('---------------- Niche trajectory options ----------------')
    info(f'Niche trajectory construction method: {options.trajectory_construct}')
