from pathlib import Path
from typing import Union

import pandas as pd

from ..log import warning


def read_meta_data(meta_data_file: Union[str, Path]) -> pd.DataFrame:
    """
    Read meta data file.
    :param meta_data_file: str or Path, meta data file path.
    :return: pd.DataFrame, meta data.
    """

    return pd.read_csv(meta_data_file, header=0, index_col=False, sep=',')


def valid_meta_data(meta_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate original data.
    :param meta_data_df: pd.DataFrame, original data.
    :return: pd.DataFrame, original data.

    1) check if Cell_ID/Spot_ID, Sample, x, and y columns in the original data
    2) check if there any duplicated Cell_ID/Spot_ID in the original data
    2) make the Cell_Type column categorical (if exists)
    3) return original data with Cell_ID/Spot_ID, Sample, x, and y columns
    """

    # check if Cell_ID, Sample, Cell_Type, x, and y columns in the original data
    id_name = meta_data_df.columns[0]
    if id_name == 'Cell_ID':  # cell-level data
        pass
    elif id_name == 'Spot_ID':  # spot-level data
        pass
    else:
        raise ValueError('ID name in meta-data input should be either Cell_ID or Spot_ID.')
    if 'Sample' not in meta_data_df.columns:
        raise ValueError('Sample column is missing in the original data.')
    if 'x' not in meta_data_df.columns:
        raise ValueError('x column is missing in the original data.')
    if 'y' not in meta_data_df.columns:
        raise ValueError('y column is missing in the original data.')

    # check if there any duplicated Cell_ID/Spot_ID in the original data
    if meta_data_df[id_name].duplicated().any():
        warning(
            f'There are duplicated {id_name} in the original data. Sample name will added to {id_name} to distinguish them.'
        )
        meta_data_df[id_name] = meta_data_df['Sample'] + '_' + meta_data_df[id_name]
    if meta_data_df[id_name].isnull().any():
        raise ValueError(f'Duplicated {id_name} within same sample found!')

    if id_name == 'Cell_ID' and 'Cell_Type' in meta_data_df.columns:
        meta_data_df['Cell_Type'] = meta_data_df['Cell_Type'].astype(str).astype('category')
        # check if the cell type category number is less than 2
        if len(meta_data_df['Cell_Type'].cat.categories) < 2:
            raise ValueError('Cell_Type column found but less than 2 cell types.')
        meta_data_df = meta_data_df.dropna(subset=[id_name, 'Sample', 'Cell_Type', 'x', 'y'])
    else:
        meta_data_df = meta_data_df.dropna(subset=[id_name, 'Sample', 'x', 'y'])

    # make the Sample column string
    meta_data_df['Sample'] = meta_data_df['Sample'].astype(str).astype('category')

    return meta_data_df


def save_cell_type_code(save_dir: Union[str, Path], cell_types: pd.Series) -> None:
    """
    Save mappings of the categorical data.
    :param save_dir: str or Path, save directory.
    :param cell_types: pd.Series, cell types.
    :return: None.
    """

    # check if the cell type is categorical
    if not pd.api.types.is_categorical_dtype(cell_types):  # type: ignore
        cell_types = cell_types.astype('category')

    # save mappings of the categorical data
    cell_type_code = pd.DataFrame(enumerate(cell_types.cat.categories), columns=['Code', 'Cell_Type'])
    cell_type_code.to_csv(f'{save_dir}/cell_type_code.csv', index=False)


def save_meta_data(save_dir: Union[str, Path], meta_data_df: pd.DataFrame) -> None:
    """
    Save meta data.
    :param save_dir: str or Path, save directory.
    :param meta_data_df: pd.DataFrame, meta data.
    :return: None.
    """

    meta_data_df.to_csv(f'{save_dir}/meta_data.csv.gz', index=False)


def load_meta_data(save_dir: Union[str, Path], meta_data_file: Union[str, Path]) -> pd.DataFrame:
    """
    Load original data.
    :param save_dir: str or Path, save directory.
    :return: pd.DataFrame, original data.
    """

    # read original data file
    meta_data_df = read_meta_data(meta_data_file=meta_data_file)

    meta_data_df = valid_meta_data(meta_data_df=meta_data_df)

    save_meta_data(save_dir=save_dir, meta_data_df=meta_data_df)

    return meta_data_df
