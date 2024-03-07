import os
import sys
from optparse import OptionGroup, OptionParser, Values

from ..log import *
from ._train import *


def prepare_NT_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    """
    program_name = os.path.basename(sys.argv[0])
    usage = f'''USAGE: {program_name} <-i INPUT> [-o OUTPUT] [--oc OUTPUT]'''
    description = 'PseudoTime: Calculate PseudoTime for each node in a graph'

    # option processor
    optparser = OptionParser(version=f'{program_name} 0.1', description=description, usage=usage, add_help_option=True)

    group_basic = add_basic_options_group(optparser)

    return optparser


def opt_NT_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """

    (options, args) = optparser.parse_args()

    validate_basic_options(optparser, options, output_dir_exist_OK=True)

    # print parameters to stdout
    info('--------------------- RUN memo ---------------------')
    write_basic_options_memo(options)
    info('----------------------------------------------------')

    return options
