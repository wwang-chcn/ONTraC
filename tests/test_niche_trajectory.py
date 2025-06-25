from unittest.mock import Mock

import numpy as np
import pandas as pd

from ONTraC.niche_trajectory._niche_trajectory import get_niche_NTScore, get_niche_trajectory_path, trajectory_path_to_NC_score


def test_get_niche_trajectory_path() -> None:
    """
    Test the function get_niche_trajectory_path.
    
    :return: None.
    """
    # Test case: Adjacency matrix with 4 nodes
    adj_matrix = np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]])
    expected_path = [0, 3, 2, 1]

    options = Mock()

    options.trajectory_construct = "BF"
    assert get_niche_trajectory_path(trajectory_construct_method=options.trajectory_construct,
                                     niche_adj_matrix=adj_matrix) == expected_path

    # Test case: Adjacency matrix with 5 nodes
    adj_matrix = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
    expected_path = [0, 1, 2, 3, 4]

    options = Mock()

    # BF
    assert get_niche_trajectory_path(trajectory_construct_method='BF', niche_adj_matrix=adj_matrix) == expected_path

    # Test case: More complex adjacency matrix with 6 nodes
    adj_matrix = np.array([[0., 0.02272882, 0.26394865, 0.25664073, 0.00442298, 0.29683277],
                           [0.02272882, 0., 0.00166663, 0.07405341, 0.57961249, 0.36278379],
                           [0.26394859, 0.00166663, 0., 0.56053263, 0.00179466, 0.00084385],
                           [0.25664064, 0.07405341, 0.56053275, 0., 0.09484559, 0.07408164],
                           [0.00442298, 0.57961255, 0.00179466, 0.0948456, 0., 0.35368276],
                           [0.29683274, 0.36278391, 0.00084385, 0.07408167, 0.35368285, 0.]])
    expected_path = [3, 2, 0, 5, 1, 4]

    options = Mock()

    options.trajectory_construct = "BF"
    assert get_niche_trajectory_path(trajectory_construct_method=options.trajectory_construct,
                                     niche_adj_matrix=adj_matrix) == expected_path


def test_trajectory_path_to_NC_score():
    """
    Test the function trajectory_path_to_NC_score.
    :param options: Values, options.
    :return: None.
    """

    options = Mock()

    # Test case: Niche trajectory path with 6 clusters
    niche_trajectory_path = [1, 2, 3, 4, 5, 0]
    expected_NT_score = np.array([1., 0., 0.2, 0.4, 0.6, 0.8])
    gen_NT_score = trajectory_path_to_NC_score(niche_trajectory_path=niche_trajectory_path,
                                               niche_clustering_sum=np.array(
                                                   [132.15, 481.15, 458.1, 35.94, 785.2, 53.157]))
    assert np.allclose(gen_NT_score, expected_NT_score)


def test_get_niche_NTScore():
    """
    Test the function get_niche_NTScore.
    :param options: Values, options.
    :return: None.
    """

    # Test case: Niche cluster loading with 3 clusters and adjacency matrix with 3 nodes
    niche_level_niche_cluster_assign_df = pd.DataFrame(np.array([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1],
                                                                 [0.6, 0.1, 0.3]]))  # #niche x #niche_cluster
    niche_adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    expected_niche_cluster_score = np.array([0, 0.5, 1])
    expected_niche_level_NTScore = np.array([0.8, 0.35, 0.35])

    options = Mock()
    options.trajectory_construct = "BF"

    niche_cluster_score, niche_level_NTScore_df = get_niche_NTScore(
        trajectory_construct_method=options.trajectory_construct,
        niche_level_niche_cluster_assign_df=niche_level_niche_cluster_assign_df,
        niche_adj_matrix=niche_adj_matrix)

    assert np.allclose(niche_cluster_score, expected_niche_cluster_score)
    assert np.allclose(niche_level_NTScore_df.values.reshape(-1), expected_niche_level_NTScore)
