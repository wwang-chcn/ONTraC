# pre-processing options

import sys
from optparse import OptionGroup, OptionParser, Values
from typing import Optional

from ..log import *
from ..version import __version__
from ._IO import *


# ------------------------------------
# Functions
# ------------------------------------
def add_preprocessing_options_group(optparser: OptionParser) -> None:
    """
    Add preprocessing options group to optparser.
    :param optparser: OptionParser object.
    :param preprocessing_options: Set of preprocessing options.
    :return: OptionGroup object.
    """

    # preprocessing options group
    group_preprocessing = OptionGroup(optparser, "Preprocessing")
    group_preprocessing.add_option(
        '--resolution',
        dest='resolution',
        type=float,
        default=10.0,
        help=
        'Resolution for leiden clustering. Used for clustering cells into cell types when gene expression data is provided. Default is 10.0.'
    )
    group_preprocessing.add_option('--deconvolution-method',
                                   dest='dc_method',
                                   default='STdeconvolve',
                                   choices=['STdeconvolve'],
                                   help='Deconvolution method used for low resolution data. Default is STdeconvolve.')
    group_preprocessing.add_option('--deconvolution-ct-num',
                                   dest='dc_ct_num',
                                   type='int',
                                   default=10,
                                   help='Number of cell type that the deconvolution method will deconvolve.')
    optparser.add_option_group(group_preprocessing)


def validate_preprocessing_options(options: Values, optparser: Optional[OptionParser] = None) -> None:
    """
    Validate preprocessing options.
    :param options: Options object.
    :param optparser: OptionParser object.
    :return: None
    """

    if not hasattr(options, 'resolution'):
        options.resolution = 10.0
    if getattr(options, 'resolution', 10.0) < 0:
        error('resolution must be greater than 0.')
        if optparser is not None: optparser.print_help()
        sys.exit(1)

    if not hasattr(options, 'dc_method'):
        options.dc_method = 'STdeconvolve'

    if not hasattr(options, 'dc_ct_num'):
        options.dc_ct_num = 10
    else:
        if getattr(options, 'dc_ct_num', 10) < 2:
            error('deconvolution_cell_type_number must be greater than 2.')
            if optparser is not None: optparser.print_help()
            sys.exit(1)
        if getattr(options, 'dc_ct_num', 10) < 4:
            warning('deconvolution_cell_type_number is less than 4. The result may not be reliable.')


def write_preprocessing_memo(options: Values):
    """Write preprocessing memos to stdout.

    Args:
        options: Options object.
    """

    # print parameters to stdout
    info(message='      -------- preprocessing options -------      ')
    if hasattr(options, 'resolution'):
        info(message=f'resolution: {options.resolution}')
    if hasattr(options, 'dc_method'):
        info(message=f'deconvolution method: {options.dc_method}')
        info(message=f'deconvolution cell type number: {options.dc_ct_num}')
