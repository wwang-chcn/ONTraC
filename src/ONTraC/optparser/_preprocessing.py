# pre-processing options

import sys
from optparse import OptionGroup, OptionParser, Values

from ..log import *


# ------------------------------------
# Functions
# ------------------------------------
def add_preprocessing_options_group(optparser: OptionParser) -> None:
    """
    Add preprocessing options group to optparser.
    :param optparser: OptionParser object.
    :return: OptionGroup object.
    """
    # preprocessing options group
    group_preprocessing = OptionGroup(optparser, "Preprocessing")
    group_preprocessing.add_option('--deconvolution-method',
                                   dest='dc_method',
                                   default='STdeconvolve',
                                   choices=['STdeconvolve'],
                                   help='Deconvolution method used for low resolution data. Default is STdeconvolve.')
    group_preprocessing.add_option('--deconvolution-cell-type-number',
                                   dest='dc_cell_type_num',
                                   type='int',
                                   default=10,
                                   help='Number of cell type that the deconvolution method will deconvolve.')
    optparser.add_option_group(group_preprocessing)


def validate_preprocessing_options(optparser: OptionParser, options: Values) -> None:
    """
    Validate preprocessing options.
    :param optparser: OptionParser object.
    :param options: Options object.
    :return: None.
    """

    if options.low_res_exp_input is None:  # low resolution data is not provided
        options.STdeconvolve = None
        options.dc_cell_type_num = None
    else:  # low resolution data is provided
        if options.dc_cell_type_num < 2:
            error('deconvolution_cell_type_number must be greater than 2.')
            optparser.print_help()
            sys.exit(1)
        if options.dc_cell_type_num < 4:
            warning('deconvolution_cell_type_number is less than 4. The result may not be reliable.')


def write_preprocessing_memo(options: Values) -> None:
    """Write preprocessing memos to stdout.

    :param options: Options object.
    :return: None.
    """
    # print parameters to stdout
    info(message='      -------- preprocessing options -------      ')
    if options.low_res_exp_input is not None:
        info(message=f'deconvolution_method: {options.dc_method}')
        info(message=f'deconvolution_cell_type_number: {options.dc_cell_type_num}')
