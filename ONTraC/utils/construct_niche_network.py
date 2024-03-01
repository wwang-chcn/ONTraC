import pandas as pd


def load_original_data(data_file: str) -> pd.DataFrame:
    """
    Load original data
    :param data_file: str, original data file
    :return: pd.DataFrame, original data

    1) read original data file (csv format)
    2) check if Cell_ID, Sample, Cell_Type, x, and y columns in the original data
    3) return original data with Cell_ID, Sample, Cell_Type, x, and y columns
    """
    
    # read original data file
    ori_data_df = pd.read_csv(data_file)
    
    # check if Cell_ID, Sample, Cell_Type, x, and y columns in the original data
    if 'Cell_ID' not in ori_data_df.columns:
        raise ValueError('Cell_ID column is missing in the original data.')
    if 'Sample' not in ori_data_df.columns:
        raise ValueError('Sample column is missing in the original data.')
    if 'Cell_Type' not in ori_data_df.columns:
        raise ValueError('Cell_Type column is missing in the original data.')
    if 'x' not in ori_data_df.columns:
        raise ValueError('x column is missing in the original data.')
    if 'y' not in ori_data_df.columns:
        raise ValueError('y column is missing in the original data.')
    
    return ori_data_df.loc[:, ['Cell_ID', 'Sample', 'Cell_Type', 'x', 'y']]