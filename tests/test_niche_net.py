from optparse import Values
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from ONTraC.niche_net._niche_net import (build_knn_network,
                                         calc_cell_type_composition,
                                         calc_edge_index,
                                         calc_niche_weight_matrix)
from ONTraC.preprocessing.data import load_meta_data

from .utils import temp_dirs


@pytest.fixture
def options() -> Values:
    # Create an options object for testing
    _options = Values()
    _options.meta_input = 'tests/_data/test_metadata.csv'
    _options.decomposition_cell_type_composition_input = None
    _options.NN_dir = 'tests/_data/NN'
    _options.n_local = 2
    _options.n_neighbors = 5
    return _options


@pytest.fixture()
def sample_data_df(options: Values) -> pd.DataFrame:
    # Create sample data
    sample_data_df = pd.read_csv(options.meta_input)
    sample_data_df = sample_data_df[sample_data_df['Sample'] == 'S1']
    sample_data_df['Cell_Type'] = sample_data_df['Cell_Type'].astype('category')
    return sample_data_df


@pytest.fixture()
def sample_name() -> str:
    return 'S1'


@pytest.fixture()
def dis_matrix() -> np.ndarray:
    return np.array([[0., 1.41421356, 2.82842712, 4.24264069, 5.65685425, 7.07106781],
                     [0., 1.41421356, 1.41421356, 2.82842712, 4.24264069, 5.65685425],
                     [0., 1.41421356, 1.41421356, 2.82842712, 2.82842712, 4.24264069],
                     [0., 1.41421356, 1.41421356, 2.82842712, 2.82842712, 4.24264069],
                     [0., 1.41421356, 1.41421356, 2.82842712, 2.82842712, 4.24264069],
                     [0., 1.41421356, 1.41421356, 2.82842712, 2.82842712, 4.24264069],
                     [0., 1.41421356, 1.41421356, 2.82842712, 4.24264069, 5.65685425],
                     [0., 1.41421356, 2.82842712, 4.24264069, 5.65685425, 7.07106781]])


@pytest.fixture()
def indices_matrix() -> np.ndarray:
    return np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 3, 4, 5], [2, 1, 3, 0, 4, 5], [3, 4, 2, 5, 1, 0], [4, 5, 3, 2, 6, 1],
                     [5, 4, 6, 7, 3, 2], [6, 7, 5, 4, 3, 2], [7, 6, 5, 4, 3, 2]])


@pytest.fixture()
def edge_index() -> np.ndarray:
    return np.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [2, 0], [2, 1],
                     [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [3, 0], [3, 1], [3, 2], [3, 4], [3, 5], [3, 6], [3, 7],
                     [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [4, 6], [4, 7], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4],
                     [5, 6], [5, 7], [6, 2], [6, 3], [6, 4], [6, 5], [6, 7], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6]])


@pytest.fixture()
def niche_weight_matrix() -> csr_matrix:
    return csr_matrix([[
        1.00000000e+00, 7.78800783e-01, 3.67879441e-01, 1.05399223e-01, 1.83156386e-02, 1.93045410e-03, 0.00000000e+00,
        0.00000000e+00
    ],
                       [
                           3.67879441e-01, 1.00000000e+00, 3.67879441e-01, 1.83156389e-02, 1.23409799e-04,
                           1.12535168e-07, 0.00000000e+00, 0.00000000e+00
                       ],
                       [
                           1.83156389e-02, 3.67879441e-01, 1.00000000e+00, 3.67879441e-01, 1.83156389e-02,
                           1.23409799e-04, 0.00000000e+00, 0.00000000e+00
                       ],
                       [
                           1.23409799e-04, 1.83156389e-02, 3.67879441e-01, 1.00000000e+00, 3.67879441e-01,
                           1.83156389e-02, 0.00000000e+00, 0.00000000e+00
                       ],
                       [
                           0.00000000e+00, 1.23409799e-04, 1.83156389e-02, 3.67879441e-01, 1.00000000e+00,
                           3.67879441e-01, 1.83156389e-02, 0.00000000e+00
                       ],
                       [
                           0.00000000e+00, 0.00000000e+00, 1.23409799e-04, 1.83156389e-02, 3.67879441e-01,
                           1.00000000e+00, 3.67879441e-01, 1.83156389e-02
                       ],
                       [
                           0.00000000e+00, 0.00000000e+00, 1.12535168e-07, 1.23409799e-04, 1.83156389e-02,
                           3.67879441e-01, 1.00000000e+00, 3.67879441e-01
                       ],
                       [
                           0.00000000e+00, 0.00000000e+00, 1.93045410e-03, 1.83156386e-02, 1.05399223e-01,
                           3.67879441e-01, 7.78800783e-01, 1.00000000e+00
                       ]])


@pytest.fixture()
def ct_coding_matrix() -> np.ndarray:
    return np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.], [0., 1.]])


@pytest.fixture()
def cell_type_composition() -> np.ndarray:
    return np.array([[0.61003367, 0.38996633], [0.41949784, 0.58050216], [0.58483686, 0.41516314],
                     [0.41516314, 0.58483686], [0.58483686, 0.41516314], [0.41516314, 0.58483686],
                     [0.58050216, 0.41949784], [0.38996633, 0.61003367]])


def test_load_meta_data(options: Values) -> None:
    """
    Test the load_meta_data module.
    :param options: Values, options.
    :return: None.
    """

    with temp_dirs(options=options):
        # load_meta_data function should:
        # 1) read original data file (options.meta_input)
        # 2) retrun original data with Cell_ID, Sample, Cell_Type, x, and y columns
        # 3) Cell_ID should be unique
        # 4) Cell_Type should be categorical
        # 5) save `cell_type_code.csv` file in the preprocessing directory
        meta_data_df = load_meta_data(save_dir=options.NN_dir, meta_data_file=options.meta_input)

        # Check the expected output data shape
        assert meta_data_df.shape[0] == 16  # Check if the DataFrame shape is unchanged

        # Check if the expected columns are present
        assert 'Cell_ID' in meta_data_df.columns
        assert 'Sample' in meta_data_df.columns
        assert 'Cell_Type' in meta_data_df.columns
        assert 'x' in meta_data_df.columns
        assert 'y' in meta_data_df.columns

        # Check if Cell_ID is unique
        # TODO: need an duplicate Cell_ID in different samples case here
        # assert not ori_data_df['Cell_ID'].duplicated().any()

        # Check if Cell_Type is categorical
        assert meta_data_df['Cell_Type'].dtype.name == 'category'

        # Check if the `cell_type_code.csv` file is saved
        assert Path(f'{options.NN_dir}/cell_type_code.csv').exists()

        # Check the content of the `cell_type_code.csv` file
        gen_cell_type_code = pd.read_csv(f'{options.NN_dir}/cell_type_code.csv')
        assert gen_cell_type_code.equals(pd.DataFrame({'Code': [0, 1], 'Cell_Type': ['A', 'B']}))


def test_build_knn_network(options: Values, sample_data_df: pd.DataFrame, sample_name: str, dis_matrix: np.ndarray,
                           indices_matrix: np.ndarray) -> None:
    """
    Test the build_knn_network function.
    :param options: Values, options.
    :param sample_data_df: pd.DataFrame, sample data.
    :param sample_name: str, sample name.
    :param dis_matrix: np.ndarray, distance matrix.
    :param indices_matrix: np.ndarray, indices matrix.
    :return: None.
    """

    # Call the function
    gen_coordinates, gen_dis_matrix, gen_indices_matrix = build_knn_network(sample_name=sample_name,
                                                                            sample_meta_df=sample_data_df,
                                                                            n_neighbors=options.n_neighbors,
                                                                            n_local=options.n_local)

    # Check the output types
    assert isinstance(gen_coordinates, np.ndarray)
    assert isinstance(gen_dis_matrix, np.ndarray)
    assert isinstance(gen_indices_matrix, np.ndarray)

    # Check the shape of the output arrays
    assert gen_coordinates.shape == (8, 2)
    assert gen_dis_matrix.shape == (8, 6)
    assert gen_indices_matrix.shape == (8, 6)

    # Check if the coordinates are correct
    assert np.array_equal(gen_coordinates, np.array(sample_data_df[['x', 'y']].values))

    # Check if the distance matrix is correct
    assert np.allclose(a=gen_dis_matrix, b=dis_matrix, atol=1e-7)

    # Check if the indices matrix is correct
    assert np.array_equal(a1=gen_indices_matrix, a2=indices_matrix)


def test_calc_edge_index(options: Values, sample_data_df: pd.DataFrame, sample_name: str, indices_matrix: np.ndarray,
                         edge_index: np.ndarray) -> None:
    """
    Test the calc_edge_index function.
    :param options: Values, options.
    :param sample_data_df: pd.DataFrame, sample data.
    :param sample_name: str, sample name.
    :param indices_matrix: np.ndarray, indices matrix.
    :param edge_index: np.ndarray, edge index.
    :return: None.
    """

    # Call the function
    gen_edge_index = calc_edge_index(sample_name=sample_name,
                                     sample_meta_df=sample_data_df,
                                     indices_matrix=indices_matrix,
                                     n_neighbors=options.n_neighbors)

    # Check the output type
    assert isinstance(gen_edge_index, np.ndarray)

    # Check the shape of the output array
    assert gen_edge_index.shape == (48, 2)

    # Check if the edge index is correct
    assert np.array_equal(gen_edge_index, edge_index)


def test_calc_niche_weight_matrix(options: Values, sample_data_df: pd.DataFrame, dis_matrix: np.ndarray,
                                  indices_matrix: np.ndarray, niche_weight_matrix: csr_matrix) -> None:
    """
    Test the calc_niche_weight_matrix function.
    :param options: Values, options.
    :param sample_data_df: pd.DataFrame, sample data.
    :param dis_matrix: np.ndarray, distance matrix.
    :param indices_matrix: np.ndarray, indices matrix.
    :param niche_weight_matrix: csr_matrix, niche weight matrix.
    :return: None.
    """

    # Call the function
    gen_niche_weight_matrix = calc_niche_weight_matrix(sample_name='sample',
                                                       sample_meta_df=sample_data_df,
                                                       dis_matrix=dis_matrix,
                                                       indices_matrix=indices_matrix,
                                                       n_neighbors=options.n_neighbors,
                                                       n_local=options.n_local)

    # Check the output type
    assert isinstance(gen_niche_weight_matrix, csr_matrix)

    # Check the shape of the output matrix
    assert gen_niche_weight_matrix.shape == (8, 8)

    # Check if the niche weight matrix is correct
    assert np.allclose(a=gen_niche_weight_matrix.toarray(), b=niche_weight_matrix.toarray(), atol=1e-7)


def test_calc_cell_type_composition(niche_weight_matrix: csr_matrix, ct_coding_matrix: np.ndarray,
                                    cell_type_composition: np.ndarray) -> None:
    """
    Test the calc_cell_type_composition function.
    :param options: Values, options.
    :param sample_data_df: pd.DataFrame, sample data.
    :param niche_weight_matrix: csr_matrix, niche weight matrix.
    :param cell_type_composition: np.ndarray, cell type composition.
    :return: None.
    """

    # Call the function
    gen_cell_type_composition = calc_cell_type_composition(niche_weight_matrix=niche_weight_matrix,
                                                           ct_coding_matrix=ct_coding_matrix)

    # Check the output type
    assert isinstance(gen_cell_type_composition, np.ndarray)

    # Check the shape of the output array
    assert gen_cell_type_composition.shape == (8, 2)

    # Check if the cell type composition is correct
    assert np.allclose(a=gen_cell_type_composition, b=cell_type_composition, atol=1e-7)
