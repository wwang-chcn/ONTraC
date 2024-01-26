import os
import sys
from optparse import OptionGroup, OptionParser, Values

from ._train import *
from ..log import *


def prepare_GSAE_optparser() -> OptionParser:
    """
    Prepare optparser object. New options will be added in thisfunction first.
    """
    program_name = os.path.basename(sys.argv[0])
    usage = f'''USAGE: {program_name} <-i INPUT> [-o OUTPUT] [--oc OUTPUT]'''
    description = 'GSAE: Graph Smooth Autoencoder'

    # option processor
    optparser = OptionParser(version=f'{program_name} 0.1', description=description, usage=usage, add_help_option=True)

    group_basic = add_basic_options_group(optparser)
    group_train = add_train_options_group(optparser)
    add_GNN_options_group(group_train)
    add_GSAE_options_group(group_train)

    return optparser


def opt_GSAE_validate(optparser) -> Values:
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
    write_GSAE_options_memo(options)
    info('----------------------------------------------------')

    return options
