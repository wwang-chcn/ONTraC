from optparse import Values
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from .log import *
from .utils import count_lines, get_rel_params, read_yaml_file


# ------------------------------------
# Classes
# ------------------------------------
class SpatailOmicsDataset(InMemoryDataset):

    def __init__(self, root, params: Dict, transform=None, pre_transform=None):
        self.params = params
        super(SpatailOmicsDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):  # required by InMemoryDataset
        # return list(
        #     flatten([[sample for name, sample in data.items() if name != 'Name'] for data in self.params['Data']]))
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for index, sample in enumerate(self.params['Data']):
            info(f'Processing sample {index + 1} of {len(self.params["Data"])}: {sample["Name"]}')
            data = Data(
                x=torch.from_numpy(np.loadtxt(sample['Features'], dtype=np.float32, delimiter=',')),
                edge_index=torch.from_numpy(
                    np.loadtxt(
                        sample['EdgeIndex'],
                        dtype=np.int64,  # int32 should be enought
                        delimiter=',')).t().contiguous(),
                # TODO: support 3D coordinates
                pos=torch.from_numpy(pd.read_csv(sample['Coordinates'])[['x', 'y']].values),
                name=sample['Name'])
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)


# ------------------------------------
# Misc functions
# ------------------------------------
def max_nodes(samples: List[Dict[str, str]]) -> int:
    """
    Get the maximum number of nodes in a dataset.
    :param params: List[Dict[str, str], list of samples.
    :return: int, maximum number of nodes.
    """
    max_nodes = 0
    for sample in samples:
        max_nodes = max(max_nodes, count_lines(sample['Coordinates']))
    return max_nodes


def load_dataset(options: Values) -> SpatailOmicsDataset:
    """
    Load dataset.
    :param options: Values, input options.
    :return: SpatailOmicsDataset, torch dataset.
    """
    params = read_yaml_file(f'{options.preprocessing_dir}/samples.yaml')
    rel_params = get_rel_params(options, params)
    dataset = create_torch_dataset(options, rel_params)
    return dataset


# ------------------------------------
# Flow control functions
# ------------------------------------
def create_torch_dataset(options: Values, params: Dict) -> SpatailOmicsDataset:
    """
    Create torch dataset.
    :param params: Dict, input samples.
    :return: None.
    """

    dataset = SpatailOmicsDataset(root=options.preprocessing_dir, params=params)  # transform edge_index to adj matrix
    dataset.process()
    return dataset
