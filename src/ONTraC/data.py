"""Data loading and dataset helpers for ONTraC."""

from typing import Dict, List, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset

from .log import *
from .utils import count_lines, get_rel_params, read_yaml_file


# ------------------------------------
# Classes
# ------------------------------------
class SpatailOmicsDataset(InMemoryDataset):
    """In-memory PyG dataset for ONTraC sample-level graph inputs.

    The dataset expects preprocessed ONTraC outputs where each sample provides:

    - a node feature matrix (`Features`)
    - an edge list (`EdgeIndex`)
    - 2D coordinates (`Coordinates`)

    Each sample is converted into a :class:`torch_geometric.data.Data` object.
    """

    def __init__(self, root, params: Dict, transform=None, pre_transform=None):
        """Initialize the dataset container.

                Parameters
                ----------
        root :
            str or Path
                    Root directory used by :class:`~torch_geometric.data.InMemoryDataset`.
        params :
            Dict
                    Parsed ``samples.yaml`` content with relative file paths resolved.
        transform :
            callable, optional
                    Transform applied on access.
        pre_transform :
            callable, optional
                    Transform applied before serialization.
        """
        self.params = params
        super(SpatailOmicsDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):  # required by InMemoryDataset
        """List raw files expected by PyG.

        Returns
        -------
        list[str]
            Empty list because ONTraC manages preprocessing and file checks.
        """
        # return list(
        #     flatten([[sample for name, sample in data.items() if name != 'Name'] for data in self.params['Data']]))
        return []

    @property
    def processed_file_names(self):
        """Name of serialized processed file.

        Returns
        -------
        list[str]
            ``['data.pt']`` for compatibility with PyG conventions.
        """
        return ["data.pt"]

    def download(self):
        """No-op download hook.

        ONTraC expects local input files and does not download remote datasets.
        """
        pass

    def process(self):
        """Convert configured samples into :class:`torch_geometric.data.Data` objects."""
        data_list = []
        for index, sample in enumerate(self.params["Data"]):
            info(f'Processing sample {index + 1} of {len(self.params["Data"])}: {sample["Name"]}')
            data = Data(
                x=torch.from_numpy(np.loadtxt(sample["Features"], dtype=np.float32, delimiter=",")),
                edge_index=torch.from_numpy(np.loadtxt(sample["EdgeIndex"], dtype=np.int64, delimiter=","))
                .t()
                .contiguous(),
                # TODO: support 3D coordinates
                pos=torch.from_numpy(pd.read_csv(sample["Coordinates"])[["x", "y"]].values),
                name=sample["Name"],
            )
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)


# ------------------------------------
# Misc functions
# ------------------------------------
def max_nodes(samples: List[Dict[str, str]]) -> int:
    """Compute the maximum number of cells/spots among input samples.

        Parameters
        ----------
    samples :
        list[dict[str, str]]
            Sample records from ``samples.yaml``.

        Returns
        -------
        int
            Largest row count across all coordinate files.
    """
    max_nodes = 0
    for sample in samples:
        max_nodes = max(max_nodes, count_lines(sample["Coordinates"]))
    return max_nodes


def load_dataset(NN_dir: Union[str, Path]) -> SpatailOmicsDataset:
    """Load ONTraC graph dataset from a preprocessing directory.

        Parameters
        ----------
    NN_dir :
        str or Path
            Directory that contains ``samples.yaml`` and per-sample CSV artifacts.

        Returns
        -------
        SpatailOmicsDataset
            Processed dense-graph dataset ready for GNN training.
    """
    params = read_yaml_file(f"{NN_dir}/samples.yaml")
    rel_params = get_rel_params(NN_dir=NN_dir, params=params)
    dataset = create_torch_dataset(NN_dir=NN_dir, params=rel_params)
    return dataset


# ------------------------------------
# Flow control functions
# ------------------------------------
def create_torch_dataset(NN_dir: Union[str, Path], params: Dict) -> SpatailOmicsDataset:
    """Build a dense PyG dataset with a uniform node budget.

        Parameters
        ----------
    NN_dir :
        str or Path
            Output directory used for dataset artifacts.
    params :
        Dict
            Resolved sample configuration, typically from ``samples.yaml``.

        Returns
        -------
        SpatailOmicsDataset
    Dataset transformed with
        class:`torch_geometric.transforms.ToDense`.
    """

    # Step 1: Get the maximum number of nodes
    m_nodes = max_nodes(params["Data"])
    # upcelling m_nodes to the nearest 100
    m_nodes = int(np.ceil(m_nodes / 100.0)) * 100
    info(f"Maximum number of cell in one sample is: {m_nodes}.")

    # Step 2: Create torch dataset
    dataset = SpatailOmicsDataset(
        root=NN_dir, params=params, transform=T.ToDense(m_nodes)
    )  # transform edge_index to adj matrix
    dataset.process()
    return dataset
