import os
import sys
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
    """
    usage = f'''USAGE: %prog <--preprocessing-dir PREPROCESSING_DIR> <--GNN-dir GNN_DIR> <--NTScore-dir NTSCORE_DIR>'''
    description = 'PseudoTime: Calculate PseudoTime for each node in a graph'

    # option processor
    optparser = OptionParser(version=f'%prog {__version__}',
                             description=description,
                             usage=usage,
                             add_help_option=True)

    add_IO_options_group(optparser=optparser, io_options=IO_OPTIONS)

    return optparser


def opt_NT_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """

    (options, args) = optparser.parse_args()

    validate_io_options(optparser, options, IO_OPTIONS)

    # print parameters to stdout
    info('------------------ RUN params memo ------------------ ')
    write_io_options_memo(options, IO_OPTIONS)
    info('--------------- RUN params memo end ----------------- ')

    return options
