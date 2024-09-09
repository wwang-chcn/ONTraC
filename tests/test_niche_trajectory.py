from optparse import Values

import numpy as np
import pytest

from ONTraC.niche_trajectory._niche_trajectory import (
    get_niche_NTScore, get_niche_trajectory_path, trajectory_path_to_NC_score)


@pytest.fixture
def options() -> Values:
    # Create an options object for testing
    _options = Values()
    _options.equal_space = True  # TODO: test for equal_space=False
    return _options


def test_get_niche_trajectory_path() -> None:
    """
    Test the function get_niche_trajectory_path.
    
    :return: None.
    """

    # Test case: Adjacency matrix with 5 nodes
    adj_matrix = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
    expected_path = [0, 1, 2, 3, 4]
    assert get_niche_trajectory_path(adj_matrix) == expected_path


def test_trajectory_path_to_NC_score(options: Values):
    """
    Test the function trajectory_path_to_NC_score.
    :param options: Values, options.
    :return: None.
    """

    # Test case: Niche trajectory path with 6 clusters
    niche_trajectory_path = [1, 2, 3, 4, 5, 0]
    expected_NT_score = np.array([1., 0., 0.2, 0.4, 0.6, 0.8])
    gen_NT_score = trajectory_path_to_NC_score(options=options,
                                               niche_trajectory_path=niche_trajectory_path,
                                               niche_clustering_sum=np.array(
                                                   [132.15, 481.15, 458.1, 35.94, 785.2, 53.157]))
    assert np.allclose(gen_NT_score, expected_NT_score)


def test_get_niche_NTScore(options: Values):
    """
    Test the function get_niche_NTScore.
    :param options: Values, options.
    :return: None.
    """
    
    # Test case: Niche cluster loading with 3 clusters and adjacency matrix with 3 nodes
    niche_cluster_loading = np.array([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1], [0.6, 0.1, 0.3]])  # #niche x #niche_cluster
    niche_adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    expected_niche_cluster_score = np.array([0, 0.5, 1])
    expected_niche_level_NTScore = np.array([0.8, 0.35, 0.35])

    niche_cluster_score, niche_level_NTScore = get_niche_NTScore(options=options,
                                                                 niche_cluster_loading=niche_cluster_loading,
                                                                 niche_adj_matrix=niche_adj_matrix)

    assert np.allclose(niche_cluster_score, expected_niche_cluster_score)
    assert np.allclose(niche_level_NTScore, expected_niche_level_NTScore)
