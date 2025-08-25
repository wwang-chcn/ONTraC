import os
from optparse import Values
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from ..log import info, warning
from ..utils import get_meta_data_file, get_rel_params, read_yaml_file


# ----------------------------
# Misc Functions
# ----------------------------
def load_loss_record_data(options) -> Optional[Dict]:
    """
    Parse the log file and save the training loss to a csv file.
    Args:
        options: Values, the options from optparse
    Returns:
        Dict: {'loss_df': DataFrame, 'loss_dict': Dict} or None
        loss_df is the training loss DataFrame
        loss_dict is the final loss dictionary

    loss record example:
        16:31:47 --- INFO: epoch: 1, batch: 1, loss: 8.554190635681152, spectral_loss: -6.64739809508319e-06, cluster_loss: 0.06959901750087738, feat_similarity_loss: 8.484598159790039
    """
    loss, loss_name = [], []  # type: ignore
    if options.log is None:
        return None
    with open(options.log, 'r') as fhd:
        for line in fhd:  # type: ignore
            # train loss record
            if 'INFO' in line and 'epoch' in line and 'loss' in line:
                loss_ = []
                loss_name = []
                line: list[str] = line.strip().split()  # type: ignore
                init_index = line.index('loss:')  # get the first loss index
                # get loss name & losss
                for i in range(init_index, len(line), 2):  # loss name
                    loss_name.append(line[i].strip(':'))
                for i in range(init_index + 1, len(line), 2):  # loss value
                    loss_.append(float(line[i].strip(',')))
                # insert epoch and batch information
                loss_.insert(0, int(line[init_index - 1].strip(',')))  # batch index
                loss_.insert(0, int(line[init_index - 3].strip(',')))  # epoch index
                loss.append(loss_)
            # eval loss record
            elif 'INFO' in line and ('Evaluation loss' in line or 'Evaluate loss' in line):
                line = line.strip().split(', ', 1)  # type: ignore
                final_loss_dict: Dict[str, float] = eval(line[1])

    # loss name edit
    loss_name[0] = 'total_loss'
    loss_name = ['Epoch', 'Batch'] + loss_name[:]

    # save loss to csv
    loss_df = pd.DataFrame(loss, columns=loss_name)
    if options.output is not None:
        loss_df.to_csv(f'{options.output}/train_loss.csv', index=False)

    # calculate epoch loss
    epoch_loss_columns = [col for col in loss_df.columns if 'loss' in col]
    epoch_loss_columns.insert(0, 'Epoch')
    epoch_loss_df = loss_df[epoch_loss_columns].groupby('Epoch').mean()
    if options.output is not None:
        epoch_loss_df.to_csv(f'{options.output}/train_loss_epoch.csv')

    return {'loss_df': epoch_loss_df, 'loss_dict': final_loss_dict}


def load_niche_cluster_connectivity(options: Values) -> Optional[np.ndarray]:
    """
    Load the niche cluster connectivity from the output of GNN.
    Args:
        options: Values, the options from optparse
    Returns:
        Optional[np.ndarray]: the niche cluster connectivity
    """

    if options.GNN_dir is None:
        return None

    niche_cluster_conn_file = f'{options.GNN_dir}/consolidate_out_adj.csv.gz'
    if not os.path.isfile(niche_cluster_conn_file):
        niche_cluster_conn_file = f'{options.GNN_dir}/consolidate_out_adj.csv'
    if not os.path.isfile(niche_cluster_conn_file):  # skip if file not exist
        warning(f"Cannot find niche cluster connectivity file: {niche_cluster_conn_file}.")
        return None

    return np.loadtxt(f'{niche_cluster_conn_file}', delimiter=',')


def load_niche_cluster_score(options: Values) -> Optional[np.ndarray]:
    """
    Load the niche cluster score from the output of NT score.
    Args:
        options: Values, the options from optparse
    Returns:
        Optional[np.ndarray]: the niche cluster score
    """

    if options.NT_dir is None:
        return None

    niche_cluster_score_file = f'{options.NT_dir}/niche_cluster_score.csv.gz'
    if not os.path.isfile(niche_cluster_score_file):
        niche_cluster_score_file = f'{options.NT_dir}/niche_cluster_score.csv'
    if not os.path.isfile(niche_cluster_score_file):  # skip if file not exist
        warning(f"Cannot find niche cluster score file: {niche_cluster_score_file}.")
        return None

    return np.loadtxt(f'{niche_cluster_score_file}', delimiter=',')


def load_niche_level_niche_cluster_assign(options: Values) -> Optional[DataFrame]:
    """
    Load the niche cluster assignment for each niche level.
    Args:
        options: Values, the options from optparse
    Returns:
        Optional[DataFrame]: the niche cluster assignment for each niche level
    """

    if options.GNN_dir is None:
        return None

    niche_level_niche_cluster_assign_file = f'{options.GNN_dir}/niche_level_niche_cluster.csv.gz'
    if not os.path.isfile(niche_level_niche_cluster_assign_file):
        niche_level_niche_cluster_assign_file = f'{options.GNN_dir}/niche_level_niche_cluster.csv'
    if not os.path.isfile(niche_level_niche_cluster_assign_file):  # skip if file not exist
        warning(f"Cannot find niche level niche cluster assign file: {niche_level_niche_cluster_assign_file}.")
        return None

    return pd.read_csv(niche_level_niche_cluster_assign_file, index_col=0)


def load_cell_level_niche_cluster_assign(options: Values) -> Optional[DataFrame]:
    """
    Load the niche cluster assignment for each cell level.
    Args:
        options: Values, the options from optparse
    Returns:
        Optional[DataFrame]: the niche cluster assignment for each cell level
    """

    if options.GNN_dir is None:
        return None

    cell_level_niche_cluster_assign_file = f'{options.GNN_dir}/cell_level_niche_cluster.csv.gz'
    if not os.path.isfile(cell_level_niche_cluster_assign_file):
        cell_level_niche_cluster_assign_file = f'{options.GNN_dir}/cell_level_niche_cluster.csv'
    if not os.path.isfile(cell_level_niche_cluster_assign_file):  # skip if file not exist
        warning(f"Cannot find cell level niche cluster assign file: {cell_level_niche_cluster_assign_file}.")
        return None

    return pd.read_csv(cell_level_niche_cluster_assign_file, index_col=0)


def load_niche_level_max_niche_cluster(options: Values) -> Optional[DataFrame]:
    """
    Load the max niche cluster assignment for each niche level.
    Args:
        options: Values, the options from optparse
    Returns:
        Optional[DataFrame]: the max niche cluster assignment for each niche level
    """

    if options.GNN_dir is None:
        return None

    niche_level_max_niche_cluster_file = f'{options.GNN_dir}/niche_level_max_niche_cluster.csv.gz'
    if not os.path.isfile(niche_level_max_niche_cluster_file):
        niche_level_max_niche_cluster_file = f'{options.GNN_dir}/niche_level_max_niche_cluster.csv'
    if not os.path.isfile(niche_level_max_niche_cluster_file):  # skip if file not exist
        warning(f"Cannot find niche level max niche cluster file: {niche_level_max_niche_cluster_file}.")
        return None

    return pd.read_csv(niche_level_max_niche_cluster_file, index_col=0)


def load_cell_level_max_niche_cluster(options: Values) -> Optional[DataFrame]:
    """
    Load the max niche cluster assignment for each cell level.
    Args:
        options: Values, the options from optparse
    Returns:
        Optional[DataFrame]: the max niche cluster assignment for each cell level
    """

    if options.GNN_dir is None:
        return None

    cell_level_max_niche_cluster_file = f'{options.GNN_dir}/cell_level_max_niche_cluster.csv.gz'
    if not os.path.isfile(cell_level_max_niche_cluster_file):
        cell_level_max_niche_cluster_file = f'{options.GNN_dir}/cell_level_max_niche_cluster.csv'
    if not os.path.isfile(cell_level_max_niche_cluster_file):  # skip if file not exist
        warning(f"Cannot find cell level max niche cluster file: {cell_level_max_niche_cluster_file}.")
        return None

    return pd.read_csv(cell_level_max_niche_cluster_file, index_col=0)


def load_niche_hidden_features(options: Values) -> Optional[np.ndarray]:
    """
    Load the niche hidden features.
    Args:
        options: Values, the options from optparse
    Returns:
        Optional[np.ndarray]: the niche hidden features
    """

    if options.NT_dir is None:
        return None

    niche_hidden_features_file = f'{options.NT_dir}/niche_hidden_features.csv.gz'
    if not os.path.isfile(niche_hidden_features_file):
        niche_hidden_features_file = f'{options.NT_dir}/niche_hidden_features.csv'
    if not os.path.isfile(niche_hidden_features_file):  # skip if file not exist
        warning(f"Cannot find niche hidden features file: {niche_hidden_features_file}.")
        return None

    return np.loadtxt(f'{niche_hidden_features_file}', delimiter=',')


# ----------------------------
# Classes
# ----------------------------


class AnaData:
    """
    Class to store the data for analysis
    This class have the following attributes:
    - options: Values, the options from optparse
    - rel_params: Dict, the relative paths for params
    - meta_data_df: DataFrame, the original Cell ID and Cell Type
    - train_loss: Dict, the training loss
    - cell_type_codes: DataFrame, the cell type codes
    - cell_type_coding: np.ndarray, #N X #cell_type, the cell type coding matrix
    - cell_type_composition: DataFrame, the cell type composition
    - ct_embedding: DataFrame, #cell_type X #gene, the cell type embedding
    - niche_cluster_connectivity: np.ndarray, the niche cluster connectivity
    - niche_level_niche_cluster_assign: DataFrame, the niche cluster assignment for each niche level
    - cell_level_niche_cluster_assign: DataFrame, the niche cluster assignment for each cell level
    - niche_level_max_niche_cluster: DataFrame, the max niche cluster assignment for each niche level
    - cell_level_max_niche_cluster: DataFrame, the max niche cluster assignment for each cell level
    - train_loss: Dict, the training loss
    - NT_score: DataFrame, the NT score
    - niche_cluster_score: np.ndarray, the niche cluster score
    - niche_level_niche_cluster_assign: pd.DataFrame, the niche cluster assignment for each niche level
    - cell_level_niche_cluster_assign: pd.DataFrame, the niche cluster assignment for each cell level
    - niche_level_max_niche_cluster: pd.DataFrame, the max niche cluster assignment for each niche level
    - cell_level_max_niche_cluster: pd.DataFrame, the max niche cluster assignment for each cell level
    - niche_hidden_features: np.ndarray, the hidden features for each niche level
    """

    def __init__(self, options: Values) -> None:
        """
        Initialize the class with the options"""

        # save options
        self.options = options

        # meta_data_df
        if hasattr(self.options, 'NN_dir'):

            # get real path
            params = read_yaml_file(f'{options.NN_dir}/samples.yaml')
            self.rel_params = get_rel_params(options.NN_dir, params)
            # load meta data
            self.meta_data_df = pd.read_csv(get_meta_data_file(options.NN_dir))
        else:  # not NN_dir, only support for visualization of meta_input
            self.meta_data_df = pd.read_csv(self.options.meta_input)

        # ID name check
        if self.meta_data_df.columns[0] == 'Cell_ID':
            self.meta_data_df = self.meta_data_df.set_index('Cell_ID')
            self.options.spatial_res = 'cell'
            info(message='Cell level meta data loaded.')
            if 'Cell_Type' in self.meta_data_df.columns:
                # make the Cell_Type column categorical
                # the order of categories is the same as the order of appearance in the cell_type_codes
                self.meta_data_df['Cell_Type'] = self.meta_data_df['Cell_Type'].astype('category')
        elif self.meta_data_df.columns[0] == 'Spot_ID':
            self.meta_data_df = self.meta_data_df.set_index('Spot_ID')
            self.options.spatial_res = 'spot'
            info(message='Spot level meta data loaded.')
        else:
            raise ValueError('ID name in meta-data input should be either Cell_ID or Spot_ID.')

        # make the Sample column categorical
        self.meta_data_df['Sample'] = self.meta_data_df['Sample'].astype('str').astype('category')

    @property
    def train_loss(self):
        if not hasattr(self, '_train_loss') or self._train_loss is None:
            self._train_loss = load_loss_record_data(self.options)
        return self._train_loss

    @property
    def umap_embedding(self) -> Optional[np.ndarray]:
        if hasattr(self, '_umap_embedding'):
            pass
        if not os.path.isfile(f'{self.options.NN_dir}/UMAP_embedding.csv'):
            warning('UMAP_embedding.csv are required for clustering visualization. Skip the clustering visualization.')
            self._umap_embedding = None
        else:
            self._umap_embedding = np.loadtxt(f'{self.options.NN_dir}/UMAP_embedding.csv', delimiter=',')
        return self._umap_embedding

    @property
    def cell_type_codes(self) -> DataFrame:
        if not hasattr(self, '_cell_type_codes') or self._cell_type_codes is None:  # type: ignore
            self._cell_type_codes = pd.read_csv(f'{self.options.NN_dir}/cell_type_code.csv', index_col=0)
            # order the cell type in meta_data_df
            self.meta_data_df['Cell_Type'] = pd.Categorical(self.meta_data_df['Cell_Type'],
                                                            categories=self._cell_type_codes['Cell_Type'].tolist())
        return self._cell_type_codes

    @property
    def cell_type_coding(self) -> np.ndarray:
        if not hasattr(self, '_cell_type_coding'):
            self._cell_type_coding = pd.read_csv(f'{self.options.NN_dir}/ct_coding.csv', index_col=0).values
        return self._cell_type_coding

    def _load_cell_type_composition(self) -> None:
        data_df_list, data_2_df_list = [], []  # cell type composition, raw cell type composition
        for sample in self.rel_params['Data']:
            # cell type composition
            cell_type_composition_df = pd.read_csv(sample['Features'], header=None)
            cell_type_composition_df.columns = self.cell_type_codes.loc[np.arange(cell_type_composition_df.shape[1]),
                                                                        'Cell_Type'].tolist()
            sample_df = cell_type_composition_df
            # add index (cell ID)
            coordinates_df = pd.read_csv(sample['Coordinates'], index_col=0)
            sample_df.index = coordinates_df.index
            sample_df['Sample'] = [sample["Name"]] * sample_df.shape[0]
            data_df_list.append(sample_df)

            # raw cell type composition
            if not self.options.embedding_adjust:
                continue
            feature_file = f"{sample['Features'][:-27]}_Raw_CellTypeComposition.csv.gz"
            if not os.path.isfile(feature_file):
                continue
            cell_type_composition_df = pd.read_csv(feature_file, header=None)
            cell_type_composition_df.columns = self.cell_type_codes.loc[np.arange(cell_type_composition_df.shape[1]),
                                                                        'Cell_Type'].tolist()
            sample_df = cell_type_composition_df
            sample_df.index = coordinates_df.index
            sample_df['Sample'] = [sample["Name"]] * sample_df.shape[0]
            data_df_list.append(sample_df)

        data_df = pd.concat(data_df_list)
        data_2_df = pd.concat(data_2_df_list)
        if data_df.shape[0] == self.meta_data_df.shape[0]:  # number of niche consistency check
            if self.options.embedding_adjust:  # adjust cell type composition
                if data_2_df.shape[0] == self.meta_data_df.shape[0]:  # number of niche consistency check
                    self._adjust_cell_type_composition = data_df[self.cell_type_codes['Cell_Type'].tolist() +
                                                                 ['Sample']].loc[self.meta_data_df.index]
                    self._cell_type_composition = data_2_df[self.cell_type_codes['Cell_Type'].tolist() +
                                                            ['Sample']].loc[self.meta_data_df.index]
                else:
                    raise ValueError(
                        f"Number of niches in the adjust cell type composition file ({data_2_df.shape[0]}) does not match the number of cells in the meta data ({self.meta_data_df.shape[0]})."
                    )
            else:  # no adjust cell type composition
                self._cell_type_composition = data_df[self.cell_type_codes['Cell_Type'].tolist() +
                                                      ['Sample']].loc[self.meta_data_df.index]
        else:
            raise ValueError(
                f"Number of niches in the cell type composition file ({data_df.shape[0]}) does not match the number of cells in the meta data ({self.meta_data_df.shape[0]})."
            )

    @property
    def cell_type_composition(self) -> DataFrame:
        if not hasattr(self, '_cell_type_composition') or self._cell_type_composition is None:
            self._load_cell_type_composition()
        return self._cell_type_composition

    def _load_NT_score(self) -> Optional[DataFrame]:
        if self.options.NT_dir is None:
            return None
        if not os.path.isfile(f'{self.options.NT_dir}/NTScore.csv.gz'):
            warning(f"Cannot find NT score file: {self.options.NT_dir}/NTScore.csv.gz.")
            return None
        NTScore_df = pd.read_csv(f'{self.options.NT_dir}/NTScore.csv.gz', index_col=0)
        return NTScore_df.loc[self.meta_data_df.index]

    @property
    def adjust_cell_type_composition(self) -> DataFrame:
        if not self.options.embedding_adjust:
            warning('The embedding adjust is not enabled. Skip the adjust cell type composition loading.')
        if not hasattr(self, '_adjust_cell_type_composition'):
            self._load_cell_type_composition()
        return self._adjust_cell_type_composition

    @property
    def ct_embedding(self) -> Optional[DataFrame]:
        if hasattr(self, '_ct_embedding'):
            pass
        if os.path.isfile(f'{self.options.NN_dir}/ct_embedding.csv'):
            self._ct_embedding = pd.read_csv(f'{self.options.NN_dir}/ct_embedding.csv', index_col=0)
        else:
            warning('Cell type embedding is not available. Skip the cell type based visualization.')
            self._ct_embedding = None
        return self._ct_embedding

    @property
    def NT_score(self) -> Optional[DataFrame]:
        if not hasattr(self, '_NT_score') or self._NT_score is None:  # type: ignore
            self._NT_score = self._load_NT_score()
        return self._NT_score

    @property
    def niche_cluster_connectivity(self) -> Optional[np.ndarray]:
        if not hasattr(self, '_niche_cluster_connectivity') or self._niche_cluster_connectivity is None:  # type: ignore
            self._niche_cluster_connectivity = load_niche_cluster_connectivity(self.options)
        return self._niche_cluster_connectivity

    @property
    def niche_cluster_score(self) -> Optional[np.ndarray]:
        if not hasattr(self, '_niche_cluster_score') or self._niche_cluster_score is None:  # type: ignore
            self._niche_cluster_score = load_niche_cluster_score(self.options)
        return self._niche_cluster_score

    @property
    def niche_level_niche_cluster_assign(self) -> Optional[DataFrame]:
        if not hasattr(
                self,
                '_niche_level_niche_cluster_assign') or self._niche_level_niche_cluster_assign is None:  # type: ignore
            self._niche_level_niche_cluster_assign = load_niche_level_niche_cluster_assign(self.options)
            if self._niche_level_niche_cluster_assign is None:
                return None
            try:
                self._niche_level_niche_cluster_assign = self._niche_level_niche_cluster_assign.loc[
                    self.meta_data_df.index]
            except:
                pass
        return self._niche_level_niche_cluster_assign

    @property
    def cell_level_niche_cluster_assign(self) -> Optional[DataFrame]:
        if not hasattr(
                self,
                '_cell_level_niche_cluster_assign') or self._cell_level_niche_cluster_assign is None:  # type: ignore
            self._cell_level_niche_cluster_assign = load_cell_level_niche_cluster_assign(self.options)
            if self._cell_level_niche_cluster_assign is None:
                return None
            try:
                self._cell_level_niche_cluster_assign = self._cell_level_niche_cluster_assign.loc[
                    self.meta_data_df.index]
            except:
                pass
        return self._cell_level_niche_cluster_assign

    @property
    def niche_level_max_niche_cluster(self) -> Optional[DataFrame]:
        if not hasattr(self,
                       '_niche_level_max_niche_cluster') or self._niche_level_max_niche_cluster is None:  # type: ignore
            self._niche_level_max_niche_cluster = load_niche_level_max_niche_cluster(self.options)
            if self._niche_level_max_niche_cluster is None:
                return None
            try:
                self._niche_level_max_niche_cluster = self._niche_level_max_niche_cluster.loc[self.meta_data_df.index]
            except:
                pass
        return self._niche_level_max_niche_cluster

    @property
    def cell_level_max_niche_cluster(self) -> Optional[DataFrame]:
        if not hasattr(self,
                       '_cell_level_max_niche_cluster') or self._cell_level_max_niche_cluster is None:  # type: ignore
            self._cell_level_max_niche_cluster = load_cell_level_max_niche_cluster(self.options)
            if self._cell_level_max_niche_cluster is None:
                return None
            try:
                self._cell_level_max_niche_cluster = self._cell_level_max_niche_cluster.loc[self.meta_data_df.index]
            except:
                pass
        return self._cell_level_max_niche_cluster

    @property
    def niche_hidden_features(self) -> Optional[np.ndarray]:
        if not hasattr(self, '_niche_hidden_features') or self._niche_hidden_features is None:  # type: ignore
            self._niche_hidden_features = load_niche_hidden_features(self.options)
            if self._niche_hidden_features is None:
                return None
            try:
                self._niche_hidden_features = self._niche_hidden_features[self.meta_data_df.index]
            except:
                pass
        return self._niche_hidden_features
