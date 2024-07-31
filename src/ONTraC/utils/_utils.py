import sys
from copy import deepcopy
from optparse import Values
from typing import Dict

import pandas as pd
import yaml

from ..log import warning


def write_version_info() -> None:
    """
    Write version information to stdout
    """
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


def load_meta_data(options: Values) -> pd.DataFrame:
    """
    Load meta data
    :param options: Values, options
    :return: pd.DataFrame, meta data

    1) read meta data file (csv format)
    2) check if Cell_ID, Sample, Cell_Type (optional), x, and y columns in the meta data
    3) make the Cell_Type column categorical
    4) return
        1. meta data with Cell_ID, Sample, Cell_Type (optional), x, and y columns
        2. samples
    """

    # read meta data file
    meta_data_df = pd.read_csv(options.dataset, header=0, index_col=False, sep=',')

    # check if Cell_ID, Sample, Cell_Type, x, and y columns in the meta data
    if 'Cell_ID' not in meta_data_df.columns:
        raise ValueError('Cell_ID column is missing in the meta data.')
    if 'Sample' not in meta_data_df.columns:
        raise ValueError('Sample column is missing in the meta data.')
    if 'x' not in meta_data_df.columns:
        raise ValueError('x column is missing in the meta data.')
    if 'y' not in meta_data_df.columns:
        raise ValueError('y column is missing in the meta data.')

    # check if there any duplicated Cell_ID
    if meta_data_df['Cell_ID'].duplicated().any():
        warning(
            'There are duplicated Cell_ID in the meta data. Sample name will added to Cell_ID to distinguish them.')
        meta_data_df['Cell_ID'] = meta_data_df['Sample'] + '_' + meta_data_df['Cell_ID']
    if meta_data_df['Cell_ID'].isnull().any():
        raise ValueError(
            f'Duplicated Cell_ID within same sample found! Please check the meta data file: {options.data_file}.')

    meta_data_df = meta_data_df.dropna()

    # make the Sample column string
    meta_data_df['Sample'] = meta_data_df['Sample'].astype(str)

    return meta_data_df


def read_yaml_file(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as fhd:
        params = yaml.load(fhd, Loader=yaml.FullLoader)
    return params


def count_lines(filename: str) -> int:
    """
    Count lines of a file
    :param filename: file name
    :return: number of lines
    """
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


def get_rel_params(options: Values, params: Dict) -> Dict:
    """
    Get relative paths for params
    :param options: Values, options
    :param params: Dict, input samples
    :return: Dict, relative paths
    """
    rel_params = deepcopy(params)
    for data in rel_params['Data']:
        for k, v in data.items():
            if k == 'Name':
                continue
            data[k] = f'{options.preprocessing_dir}/{v}'
    return rel_params


def round_epoch_filter(epoch: int) -> bool:
    """
    Round epoch filter
    Only round epoch (1, 2, ..., 9, 10, 20, ..., 90, 100, ...) will be saved
    :param epoch: int
    :return: bool
    """

    def _is_power_of_10(n: int) -> bool:
        """
        Check if n is power of 10
        :param n: int
        :return: bool
        """
        num = len(str(n))
        return n % (10**(num - 1)) == 0

    return epoch < 10 or _is_power_of_10(epoch)
