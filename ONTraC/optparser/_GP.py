import os
import sys
from optparse import OptionParser, Values

from ..log import *
from ._train import *


def prepare_GP_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    """
    program_name = os.path.basename(sys.argv[0])
    usage = f'''USAGE: {program_name} <-i INPUT> [-o OUTPUT] [--oc OUTPUT]'''
    description = 'GP (Graph Pooling): GNN & Node Pooling'

    # option processor
    optparser = OptionParser(version=f'{program_name} 0.1', description=description, usage=usage, add_help_option=True)

    group_basic = add_basic_options_group(optparser)
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_NP_options_group(group_train)

    return optparser


def opt_GP_validate(optparser: OptionParser) -> Values:
    """Validate options from a OptParser object.

    Ret: Validated options object.
    """

    (options, args) = optparser.parse_args()

    validate_basic_options(optparser, options)
    validate_train_options(optparser, options)

    # print parameters to stdout
    info('--------------------- RUN memo ---------------------')
    write_basic_options_memo(options)
    write_train_options_memo(options)
    write_GNN_options_memo(options)
    write_NP_options_memo(options)
    info('----------------------------------------------------')

    return options
