import os
import sys
from optparse import OptionGroup, OptionParser, Values
from typing import Optional, Set

from ..log import *


def add_IO_options_group(optparser: OptionParser, io_options: Optional[Set[str]]) -> None:
    """
    Add I/O options group to optparser.
    :param optparser: OptionParser object.
    :param io_options: Set of I/O options.
    :return: OptionGroup object.
    """
    if io_options is None:
        return
    # I/O options group
    group_io = OptionGroup(optparser, "IO")

    # directories
    if 'NN_dir' in io_options:
        group_io.add_option('--NN-dir', dest='NN_dir', type='string', help='Directory for niche network outputs.')
    if 'GNN_dir' in io_options:
        group_io.add_option('--GNN-dir', dest='GNN_dir', type='string', help='Directory for the GNN output.')
    if 'NT_dir' in io_options:
        group_io.add_option('--NT-dir', dest='NT_dir', type='string', help='Directory for the niche trajectory output.')
    if 'output' in io_options:
        group_io.add_option('-o', '--output', dest='output', type='string', help='Directory for analysis output.')

    # input files
    if 'meta' in io_options:
        group_io.add_option(
            '--meta-input',
            dest='meta_input',
            type='string',
            help=
            'Meta data file in csv format. Each row is a cell. The first column should be the cell name with column name Cell_ID. Coordinates (x, y) and sample should be included. Cell type is required for cell-level data.'
        )
    if 'input' in io_options:
        group_io.add_option(
            '--exp-input',
            dest='exp_input',
            type='string',
            default=None,
            help=
            'Normalized expression file in csv format. Each row is a cell and each column is a gene. The first column should be the cell name with column name Cell_ID. If not provided, cell type should be included in the meta data file.'
        )
        group_io.add_option(
            '--embedding-input',
            dest='embedding_input',
            type='string',
            default=None,
            help='Embedding file in csv format. The first column should be the cell name with column name Cell_ID.')
        group_io.add_option('--low-res-exp-input',
                            dest='low_res_exp_input',
                            type='string',
                            default=None,
                            help='Gene X spot matrix in csv format for low-resolution dataset.')
        group_io.add_option(
            '--deconvoluted-ct-composition',
            dest='deconvoluted_ct_composition',
            type='string',
            default=None,
            help=
            'Deconvoluted cell type composition of each spot in csv format. The first column should be the spot name with column name Spot_ID.'
        )
        group_io.add_option(
            '--deconvoluted-exp-input',
            dest='deconvoluted_exp_input',
            type='string',
            default=None,
            help=
            'Deconvoluted expression of each cell type in csv format. The first column should be the cell type name corresponding to the columns name of decomposition outputed cell type composition.'
        )
    if 'log' in io_options:
        group_io.add_option('-l', '--log', dest='log', type='string', help='Log file.')

    # deprecated options
    if 'NN_dir' in io_options:
        group_io.add_option('--preprocessing-dir',
                            dest='preprocessing_dir',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --NN-dir instead.')
    if 'NT_dir' in io_options:
        group_io.add_option('--NTScore-dir',
                            dest='NTScore_dir',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --NT-dir instead.')
    if 'input' in io_options:
        group_io.add_option('-d',
                            '--dataset',
                            dest='dataset',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --meta-input instead.')

    optparser.add_option_group(group_io)


def validate_io_options(optparser: OptionParser,
                        options: Values,
                        io_options: Optional[Set[str]],
                        required: bool = True,
                        overwrite_validation: bool = True) -> None:
    """Validate IO options from a OptParser object.
    :param optparser: OptionParser object.
    :param options: Options object.
    :param io_options: Set of I/O options.
    :param required: Required flag.
    :param overwrite_validation: Overwrite validation flag.
    :return: None.
    """
    if io_options is None:
        return

    if 'NN_dir' in io_options:
        if hasattr(options,
                   'preprocessing_dir') and options.preprocessing_dir is not None and (not hasattr(options, 'NN_dir')
                                                                                       or options.NN_dir is None):
            warning('The --preprocessing-dir option will be deprecated from v3.0. Please use --NN-dir instead.')
            options.NN_dir = options.preprocessing_dir
        if required and (not hasattr(options, 'NN_dir') or options.NN_dir is None):
            error('Please provide a directory for niche network outputs.')
            optparser.print_help()
            sys.exit(1)
        if hasattr(options, 'NN_dir') and options.NN_dir is not None and os.path.isdir(options.NN_dir):
            if overwrite_validation:
                warning(f'The directory ({options.NN_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.NN_dir}')
            os.makedirs(options.NN_dir, exist_ok=True)

    if 'GNN_dir' in io_options:
        if required and (not hasattr(options, 'GNN_dir') or options.GNN_dir is None):
            error('Please provide a directory for the GNN output.')
            optparser.print_help()
            sys.exit(1)
        if hasattr(options, 'GNN_dir') and options.GNN_dir is not None and os.path.isdir(options.GNN_dir):
            if overwrite_validation:
                warning(f'The directory ({options.GNN_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.GNN_dir}')
            os.makedirs(options.GNN_dir, exist_ok=True)

    if 'NT_dir' in io_options:
        if hasattr(options, 'NTScore_dir') and options.NTScore_dir is not None and (not hasattr(options, 'NT_dir')
                                                                                    or options.NT_dir is None):
            warning('The --NTScore-dir option will be deprecated from v3.0. Please use --NT-dir instead.')
            options.NT_dir = options.NTScore_dir
        if required and (not hasattr(options, 'NT_dir') or options.NT_dir is None):
            error('Please provide a directory for the NTScore output.')
            optparser.print_help()
            sys.exit(1)
        if hasattr(options, 'NT_dir') and options.NT_dir is not None and os.path.isdir(options.NT_dir):
            if overwrite_validation:
                warning(f'The directory ({options.NT_dir}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.NT_dir}')
            os.makedirs(options.NT_dir, exist_ok=True)

    if 'input' in io_options:
        # meta data
        if not required:
            pass
        else:
            if options.dataset and not options.meta_input:
                warning('The --dataset option will be deprecated from v3.0. Please use --meta-input instead.')
                options.meta_input = options.dataset
            if not hasattr(options, 'meta_input') or options.meta_input is None:
                error('Please provide a meta data file in csv format.')
                optparser.print_help()
                sys.exit(1)
            if not os.path.isfile(options.meta_input):
                error(f'The input file ({options.meta_input}) you given does not exist.')
                optparser.print_help()
                sys.exit(1)
            if not options.meta_input.endswith(('csv', 'csv.gz')):
                error(f'The input file ({options.meta_input}) should be in csv format.')
                optparser.print_help()
                sys.exit(1)
        # embedding
        if options.embedding_input:
            if not os.path.isfile(options.embedding_input):
                error(f'The embedding file ({options.embedding_input}) you given does not exist.')
                optparser.print_help()
                sys.exit(1)
            if not options.embedding_input.endswith(('csv', 'csv.gz')):
                error(f'The embedding file ({options.embedding_input}) should be in csv format.')
                optparser.print_help()
                sys.exit(1)
            options.exp_input = None
        # expression data
        if options.exp_input:
            if not os.path.isfile(options.exp_input):
                error(f'The expression data file ({options.exp_input}) you given does not exist.')
                optparser.print_help()
                sys.exit(1)
            if not options.exp_input.endswith(('csv', 'csv.gz')):
                error(f'The expression data file ({options.exp_input}) should be in csv format.')
                optparser.print_help()
                sys.exit(1)
        # low-res expression data
        if options.low_res_exp_input:
            if not os.path.isfile(options.low_res_exp_input):
                error(
                    f'The low-resolution expression data file ({options.low_res_exp_input}) you given does not exist.')
                optparser.print_help()
                sys.exit(1)
            if not options.low_res_exp_input.endswith(('csv', 'csv.gz')):
                error(f'The low-resolution expression data file ({options.low_res_exp_input}) should be in csv format.')
                optparser.print_help()
                sys.exit(1)
        # check deconvoluted results files
        if options.deconvoluted_ct_composition:
            if not os.path.isfile(options.deconvoluted_ct_composition):
                error(
                    f'The deconvoluted outputed cell type composition file ({options.deconvoluted_ct_composition}) you given does not exist.'
                )
                optparser.print_help()
                sys.exit(1)
            if not options.deconvoluted_ct_composition.endswith(('csv', 'csv.gz')):
                error(
                    f'The deconvoluted outputed cell type composition file ({options.decomposition_cell_type_composition_input}) should be in csv format.'
                )
                optparser.print_help()
                sys.exit(1)
        if options.deconvoluted_exp_input:
            if not hasattr(options, 'deconvoluted_ct_composition') or options.deconvoluted_ct_composition is None:
                error(
                    message=
                    'If you want to provide deconvolution results as input. Deconvoluted cell type composition file is required.'
                )
                optparser.print_help()
                sys.exit(1)
            if not os.path.isfile(options.deconvoluted_exp_input):
                error(
                    f'The decomposition outputed expression file ({options.deconvoluted_exp_input}) you given does not exist.'
                )
                optparser.print_help()
                sys.exit(1)
            if not options.deconvoluted_exp_input.endswith(('csv', 'csv.gz')):
                error(
                    f'The decomposition outputed expression file ({options.deconvoluted_exp_input}) should be in csv format.'
                )
                optparser.print_help()

    if 'output' in io_options:  # this is a required option (ONTraC_analysis only)
        if not hasattr(options, 'output') or options.output is None:
            error('Please provide a directory for analysis output.')
            optparser.print_help()
            sys.exit(1)
        if os.path.isdir(options.output):
            if overwrite_validation:
                warning(f'The directory ({options.output}) you given already exists. It will be overwritten.')
            else:
                pass
        else:
            info(f'Creating directory: {options.output}')
            os.makedirs(options.output, exist_ok=True)

    if 'log' in io_options:  # this is a optional option (ONTraC_analysis only)
        if hasattr(options, 'log') and options.log is not None and not os.path.isfile(options.log):
            error(f'Log file: {options.log} you given does not exist.')
            sys.exit(1)


def write_io_options_memo(options: Values, io_options: Optional[Set[str]]) -> None:
    """Write IO options to stdout.
    :param options: Options object.
    :param io_options: Set of I/O options.
    :return: None.
    """
    if io_options is None:
        return
    info('            -------- I/O options -------             ')
    if 'NN_dir' in io_options and hasattr(options, 'NN_dir') and options.NN_dir is not None:
        info(f'Niche network output directory:  {options.NN_dir}')
    if 'GNN_dir' in io_options and hasattr(options, 'GNN_dir') and options.GNN_dir is not None:
        info(f'GNN output directory:  {options.GNN_dir}')
    if 'NT_dir' in io_options and hasattr(options, 'NT_dir') and options.NT_dir is not None:
        info(f'Niche trajectory output directory:  {options.NT_dir}')
    if 'output' in io_options:
        info(f'Output directory:  {options.output}')
    if 'input' in io_options and hasattr(options, 'meta_input') and options.meta_input is not None:
        info(f'Meta data file:  {options.meta_input}')
        # TODO: refine logic here
        if hasattr(options, 'exp_input') and options.exp_input is not None:
            info(f'expression data file:  {options.exp_input}')
        if hasattr(options, 'embedding_input') and options.embedding_input is not None:
            info(f'embedding file:  {options.embedding_input}')
        if hasattr(options, 'low_res_exp_input') and options.low_res_exp_input is not None:
            info(f'low-resolution expression data file:  {options.low_res_exp_input}')
        if hasattr(options, 'deconvoluted_ct_composition') and options.deconvoluted_ct_composition is not None:
            info(f'deconvoluted outputed cell type composition file:  {options.deconvoluted_ct_composition}')
        if hasattr(options, 'deconvoluted_exp_input') and options.deconvoluted_exp_input is not None:
            info(f'decomposition outputed expression file:  {options.deconvoluted_exp_input}')
    if 'log' in io_options:
        if options.log:
            info(f'Log file:  {options.log}')
