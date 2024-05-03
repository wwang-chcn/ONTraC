import numpy as np
import pandas as pd


def gini(array: np.ndarray | pd.Series) -> float:
    """Calculate the Gini coefficient of a numpy array."""
    #
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    if isinstance(array, pd.Series):
        array = np.array(array)
    array = array.flatten()  # type: ignore
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)  # type: ignore
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)  # type: ignore
    # Number of array elements:
    n = array.shape[0]  # type: ignore
    # Index per array element:
    index = np.arange(1, n + 1)  # type: ignore
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # type: ignore
