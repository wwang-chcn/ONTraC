import os
import sys
from optparse import OptionGroup, OptionParser, Values
from typing import Dict, List, Optional

from ..log import *


class IOOption:

    def __init__(self, name: str, attr: str):
        """
        I/O option class.
        
        Parameters
        ----------
        name: str
            Name of the I/O option.
        attr: str
            Attribute of the I/O option.

        Returns
        -------
        None
        """
        self.name = name
        self.attr = attr


class IOOptionsCollection:

    def __init__(self):
        """
        I/O options collection class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self._iooptions = {}  # Maps name to IOOption instance

    def add(self, io_option: IOOption) -> None:
        """
        Add an I/O option.
        
        Parameters
        ----------
        io_option: IOOption
            I/O option instance.

        Returns
        -------
        None
        """
        if io_option.name in self._iooptions:
            raise ValueError(f'{io_option.name} already exists.')
        self._iooptions[io_option.name] = io_option

    def has_io_option(self, name: str) -> bool:
        """
        Check if an I/O option exists.
        
        Parameters
        ----------
        name: str
            Name of the I/O option.

        Returns
        -------
        bool
            True if the I/O option exists, False otherwise.
        """
        return name in self._iooptions

    def get_io_option_attr(self, name: str) -> str:
        """
        Get the attribute of an I/O option.

        Parameters
        ----------
        name: str
            Name of the I/O option.

        Returns
        -------
        str
            Attribute of the I/O option.
        """
        if name not in self._iooptions:
            raise ValueError(f'{name} does not exist.')
        return self._iooptions[name].attr


def io_dicts_to_io_options_collection(io_dicts: Dict[str, List[str]]) -> IOOptionsCollection:
    """
    Convert a dictionary of I/O options to an IOOptionsCollection instance.

    Parameters
    ----------
    io_dicts: Dict[str, List[str]]
        Dictionary of I/O options.

    Returns
    -------
    IOOptionsCollection
        I/O options collection instance.
    """
    ioc = IOOptionsCollection()
    for module, io_options in io_dicts.items():
        for io_option in io_options:
            ioc.add(IOOption(name=io_option, attr=module))
    return ioc


def add_IO_options_group(optparser: OptionParser, io_options: Optional[Dict[str, List[str]]]) -> None:
    """
    Add I/O options group to optparser.

    Parameters
    ----------
    optparser: OptionParser
        OptionParser object.
    io_options: Dict[str, List[str]]
        List of I/O options.

    Returns
    -------
    None
    """
    if io_options is None:
        return
    else:  # only list of I/O options needed
        ioc = io_dicts_to_io_options_collection(io_options)
    # I/O options group
    group_io = OptionGroup(optparser, "IO")

    # directories
    if ioc.has_io_option('NN_dir'):
        group_io.add_option('--NN-dir', dest='NN_dir', type='string', help='Directory for niche network outputs.')
    if ioc.has_io_option('GNN_dir'):
        group_io.add_option('--GNN-dir', dest='GNN_dir', type='string', help='Directory for the GNN output.')
    if ioc.has_io_option('NT_dir'):
        group_io.add_option('--NT-dir', dest='NT_dir', type='string', help='Directory for the niche trajectory output.')
    if ioc.has_io_option('output'):
        group_io.add_option('-o', '--output', dest='output', type='string', help='Directory for analysis output.')

    # input files
    if ioc.has_io_option('input'):
        group_io.add_option(
            '--meta-input',
            dest='meta_input',
            type='string',
            help=
            'Meta data file in csv format. Each row is a cell. The first column should be the cell name with column name Cell_ID. Coordinates (x, y) and sample should be included. Cell type is required for cell-level data.'
        )
    if ioc.has_io_option('log'):
        group_io.add_option('-l', '--log', dest='log', type='string', help='Log file.')

    # deprecated options
    if ioc.has_io_option('NN_dir'):
        group_io.add_option('--preprocessing-dir',
                            dest='preprocessing_dir',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --NN-dir instead.')
    if ioc.has_io_option('NT_dir'):
        group_io.add_option('--NTScore-dir',
                            dest='NTScore_dir',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --NT-dir instead.')
    if ioc.has_io_option('input'):
        group_io.add_option('-d',
                            '--dataset',
                            dest='dataset',
                            type='string',
                            help='This options will be deprecated from v3.0. Please use --meta-input instead.')

    optparser.add_option_group(group_io)


def validate_io_options(options: Values,
                        io_options: Optional[Dict[str, List[str]]] = None,
                        optparser: Optional[OptionParser] = None) -> None:
    """Validate IO options from a OptParser object.

    Parameters
    ----------
    options: Values
        Options object.
    io_options: Dict[str, List[str]]
        List of I/O options.
    optparser: OptionParser
        OptionParser object.

    Returns
    -------
    None
    """
    if io_options is None:
        return
    else:
        ioc = io_dicts_to_io_options_collection(io_options)

    if ioc.has_io_option('NN_dir'):
        # deprecated `preprocessing_dir` check
        if hasattr(options,
                   'preprocessing_dir') and options.preprocessing_dir is not None and (not hasattr(options, 'NN_dir')
                                                                                       or options.NN_dir is None):
            warning('The --preprocessing-dir option will be deprecated from v3.0. Please use --NN-dir instead.')
            options.NN_dir = options.preprocessing_dir
        # required check
        if ioc.get_io_option_attr('NN_dir') in ['required', 'overwrite'] and (not hasattr(options, 'NN_dir')
                                                                              or options.NN_dir is None):
            error('Please provide a directory for niche network outputs.')
            if optparser is not None: optparser.print_help()
            sys.exit(1)
        # overwrite warning
        elif ioc.get_io_option_attr('NN_dir') == 'overwrite' and hasattr(
                options, 'NN_dir') and options.NN_dir is not None and os.path.isdir(options.NN_dir):
            warning(f'The directory ({options.NN_dir}) you given already exists. It will be overwritten.')
        # optional check
        elif ioc.get_io_option_attr('NN_dir') == 'optional' and hasattr(options,
                                                                        'NN_dir') and options.NN_dir is not None:
            if not os.path.isdir(options.NN_dir):  # directory does not exist
                warning(f'The directory ({options.NN_dir}) you given does not exist.')
                options.NN_dir = None
            elif not os.listdir(options.NN_dir):  # empty directory
                warning(f'The directory ({options.NN_dir}) you given is empty.')
                options.NN_dir = None
        # create directory
        else:
            os.makedirs(options.NN_dir, exist_ok=True)

    if ioc.has_io_option('GNN_dir'):
        # required check
        if ioc.get_io_option_attr('GNN_dir') in ['required', 'overwrite'] and (not hasattr(options, 'GNN_dir')
                                                                               or options.GNN_dir is None):
            error('Please provide a directory for the GNN output.')
            if optparser is not None: optparser.print_help()
            sys.exit(1)
        # overwrite warning
        elif ioc.get_io_option_attr('GNN_dir') == 'overwrite' and hasattr(
                options, 'GNN_dir') and options.GNN_dir is not None and os.path.isdir(options.GNN_dir):
            warning(f'The directory ({options.GNN_dir}) you given already exists. It will be overwritten.')
        # optional check
        elif ioc.get_io_option_attr('GNN_dir') == 'optional' and hasattr(options,
                                                                         'GNN_dir') and options.GNN_dir is not None:
            if not os.path.isdir(options.GNN_dir):  # directory does not exist
                warning(f'The directory ({options.GNN_dir}) you given does not exist.')
                options.GNN_dir = None
            elif not os.listdir(options.GNN_dir):  # empty directory
                warning(f'The directory ({options.GNN_dir}) you given is empty.')
                options.GNN_dir = None
        # create directory
        else:
            os.makedirs(options.GNN_dir, exist_ok=True)

    if ioc.has_io_option('NT_dir'):
        # deprecated `NTScore_dir` check
        if hasattr(options, 'NTScore_dir') and options.NTScore_dir is not None and (not hasattr(options, 'NT_dir')
                                                                                    or options.NT_dir is None):
            warning('The --NTScore-dir option will be deprecated from v3.0. Please use --NT-dir instead.')
            options.NT_dir = options.NTScore_dir
        # required check
        if ioc.get_io_option_attr('NT_dir') in ['required', 'overwrite'] and (not hasattr(options, 'NT_dir')
                                                                              or options.NT_dir is None):
            error('Please provide a directory for the NTScore output.')
            if optparser is not None: optparser.print_help()
            sys.exit(1)
        # overwrite warning
        elif ioc.get_io_option_attr('NT_dir') == 'overwrite' and hasattr(
                options, 'NT_dir') and options.NT_dir is not None and os.path.isdir(options.NT_dir):
            warning(f'The directory ({options.NT_dir}) you given already exists. It will be overwritten.')
        # optional check
        elif ioc.get_io_option_attr('NT_dir') == 'optional' and hasattr(options,
                                                                        'NT_dir') and options.NT_dir is not None:
            if not os.path.isdir(options.NT_dir):  # directory does not exist
                warning(f'The directory ({options.NT_dir}) you given does not exist.')
                options.NT_dir = None
            elif not os.listdir(options.NT_dir):  # empty directory
                warning(f'The directory ({options.NT_dir}) you given is empty.')
                options.NT_dir = None
        # create directory
        else:
            os.makedirs(options.NT_dir, exist_ok=True)

    if ioc.has_io_option('input'):
        if hasattr(options, 'dataset') and options.dataset is not None and (not hasattr(options, 'meta_input')
                                                                            or options.meta_input is None):
            warning('The --dataset option will be deprecated from v3.0. Please use --meta-input instead.')
            options.meta_input = options.dataset
        if not hasattr(options, 'meta_input') or options.meta_input is None:
            error('Please provide a meta data file in csv format.')
            if optparser is not None: optparser.print_help()
            sys.exit(1)
        if not os.path.isfile(options.meta_input):
            error(f'The input file ({options.meta_input}) you given does not exist.')
            if optparser is not None: optparser.print_help()
            sys.exit(1)
        if not options.meta_input.endswith(('csv', 'csv.gz')):
            error(f'The input file ({options.meta_input}) should be in csv format.')
            if optparser is not None: optparser.print_help()
            sys.exit(1)

    if ioc.has_io_option('output'):  # this is a optional option (ONTraC_analysis only)
        if not hasattr(options, 'output') or options.output is None:
            pass
        elif os.path.isdir(options.output):
            warning(f'The directory ({options.output}) you given already exists. It will be overwritten.')
        else:
            info(f'Creating directory: {options.output}')
            os.makedirs(options.output, exist_ok=True)

    if ioc.has_io_option('log'):  # this is a optional option (ONTraC_analysis only)
        if hasattr(options, 'log') and options.log is not None and not os.path.isfile(options.log):
            warning(f'Log file: {options.log} you given does not exist.')
            options.log = None


def write_io_options_memo(options: Values, io_options: Optional[Dict[str, List[str]]]) -> None:
    """Write IO options to stdout.

    Parameters
    ----------
    options: Values
        Options object.
    io_options: Dict[str, List[str]]
        List of I/O options.

    Returns
    -------
    None
    """
    if io_options is None:
        return
    else:
        ioc = io_dicts_to_io_options_collection(io_options)

    info('            -------- I/O options -------             ')
    if ioc.has_io_option('NN_dir') and hasattr(options, 'NN_dir') and options.NN_dir is not None:
        info(f'Niche network output directory:  {options.NN_dir}')
    if ioc.has_io_option('GNN_dir') and hasattr(options, 'GNN_dir') and options.GNN_dir is not None:
        info(f'GNN output directory:  {options.GNN_dir}')
    if ioc.has_io_option('NT_dir') and hasattr(options, 'NT_dir') and options.NT_dir is not None:
        info(f'Niche trajectory output directory:  {options.NT_dir}')
    if ioc.has_io_option('output'):  # this is a overwrite option (ONTraC_analysis only)
        info(f'Output directory:  {options.output}')
    if ioc.has_io_option(name='input'):  # this is a required option
        info(f'Meta data file:  {options.meta_input}')
    if ioc.has_io_option(name='log') and hasattr(
            options, 'log') and options.log is not None:  # this is a optional option (ONTraC_analysis only)
        info(f'Log file:  {options.log}')


__all__ = ['add_IO_options_group', 'validate_io_options', 'write_io_options_memo']
