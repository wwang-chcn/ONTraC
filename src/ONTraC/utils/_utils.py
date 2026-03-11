"""This module contains utility functions and classes for ONTraC, including general-purpose functions that are used across different modules of ONTraC."""

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import yaml


def write_version_info() -> None:
    """Write version information to stdout.
    
    Returns
    -------
    None."""
    from .. import __version__
    template = f'''##################################################################################

         ▄▄█▀▀██   ▀█▄   ▀█▀ █▀▀██▀▀█                   ▄▄█▀▀▀▄█
        ▄█▀    ██   █▀█   █     ██    ▄▄▄ ▄▄   ▄▄▄▄   ▄█▀     ▀
        ██      ██  █ ▀█▄ █     ██     ██▀ ▀▀ ▀▀ ▄██  ██
        ▀█▄     ██  █   ███     ██     ██     ▄█▀ ██  ▀█▄      ▄
         ▀▀█▄▄▄█▀  ▄█▄   ▀█    ▄██▄   ▄██▄    ▀█▄▄▀█▀  ▀▀█▄▄▄▄▀

                        version: {__version__}

##################################################################################
'''

    sys.stdout.write(template)
    sys.stdout.flush()


def get_rel_params(NN_dir: Union[str, Path], params: Dict) -> Dict:
    """Get relative paths for params.
    
    Parameters
    ----------
    NN_dir :
        str or Path, NN_dir.
    params :
        Dict, input samples.
    
    Returns
    -------
    Dict, relative paths."""
    rel_params = deepcopy(params)
    for data in rel_params['Data']:
        for k, v in data.items():
            if k == 'Name':
                continue
            data[k] = f'{NN_dir}/{v}'
    return rel_params


def read_yaml_file(yaml_file: str) -> dict:
    """Read yaml file.
    
    Parameters
    ----------
    yaml_file :
        str, yaml file.
    
    Returns
    -------
    dict, parameters."""
    with open(yaml_file, 'r') as fhd:
        params = yaml.load(fhd, Loader=yaml.FullLoader)
    return params


def count_lines(filename: str) -> int:
    """Count lines of a file.
    
    Parameters
    ----------
    filename :
        file name.
    
    Returns
    -------
    number of lines."""
    i = 0
    if filename.endswith('.gz'):
        import gzip
        fhd = gzip.open(filename, 'rt')
    else:
        fhd = open(filename, 'r')
    for _ in fhd:
        i += 1
    fhd.close()
    return i


def get_meta_data_file(NN_dir: Union[str, Path]) -> Union[str, Path]:
    """Get meta data file.
    
    Parameters
    ----------
    NN_dir :
        str or Path, NN_dir.
    
    Returns
    -------
    str, meta data file."""
    # meta data
    meta_data_file = f'{NN_dir}/meta_data.csv.gz'
    if not os.path.isfile(meta_data_file):
        meta_data_file = f'{NN_dir}/meta_data.csv'
    if not os.path.isfile(meta_data_file):
        raise ValueError(
            'meta_data.csv. is required for preprocessing. Copy the input meta_data.csv to the NN_dir may solve this problem.'
        )
    return Path(meta_data_file)


def round_epoch_filter(epoch: int) -> bool:
    """Round epoch filter.
        Only round epoch (1, 2, ..., 9, 10, 20, ..., 90, 100, ...) will be saved.
    
    Parameters
    ----------
    epoch :
        int.
    
    Returns
    -------
    bool."""

    def _is_power_of_10(n: int) -> bool:
        """Check if n is power of 10.
        
        Parameters
        ----------
        n :
            int.
        
        Returns
        -------
        bool."""
        num = len(str(n))
        return n % (10**(num - 1)) == 0

    return epoch < 10 or _is_power_of_10(epoch)
