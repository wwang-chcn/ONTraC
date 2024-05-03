import sys
from copy import deepcopy
from optparse import Values
from typing import Dict, Optional

import torch
import yaml

from ..log import warning


def write_version_info() -> None:
    """
    Write version information to stdout
    """
    from .. import __version__
    template = f'''########################################################

 ▄▄█▀▀██   ▀█▄   ▀█▀ █▀▀██▀▀█                   ▄▄█▀▀▀▄█
▄█▀    ██   █▀█   █     ██    ▄▄▄ ▄▄   ▄▄▄▄   ▄█▀     ▀
██      ██  █ ▀█▄ █     ██     ██▀ ▀▀ ▀▀ ▄██  ██
▀█▄     ██  █   ███     ██     ██     ▄█▀ ██  ▀█▄      ▄
 ▀▀█▄▄▄█▀  ▄█▄   ▀█    ▄██▄   ▄██▄    ▀█▄▄▀█▀  ▀▀█▄▄▄▄▀

                version: {__version__}

########################################################
'''

    sys.stdout.write(template)
    sys.stdout.flush()


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


def device_validate(device_name: Optional[str] = None) -> torch.device:
    if device_name is None:
        device_name = 'cpu'
    elif device_name.startswith('cuda') and not torch.cuda.is_available():
        warning('CUDA is not available, use CPU instead.')
        device_name = 'cpu'
    # elif device_name.startswith('mps') and not torch.backends.mps.is_available():
    #     warning('MPS is not available, use CPU instead.')
    #     device_name = 'cpu'
    else:
        device_name = 'cpu'

    device = torch.device(device=device_name)

    return device
