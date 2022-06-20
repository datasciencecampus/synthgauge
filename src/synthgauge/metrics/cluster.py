""" Utility metrics derived from centroid-based clustering. """

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans

from ..utils import df_combine


def _get_cluster_labels(combined, method, k, random_state):
    """Apply the chosen clustering method to a dataset and return its
    final labels."""

    if method == "kmeans":
        numeric = combined.select_dtypes(include="number")
        model = KMeans(
            n_clusters=k, algorithm="elkan", random_state=random_state
        ).fit(numeric)

    elif method == "kprototypes":
        categorical_idxs = [
            i
            for i, dtype in enumerate(combined.dtypes)
            if not pd.api.types.is_numeric_dtype(dtype)
        ]
        model = KPrototypes(
            n_clusters=k, random_state=random_state, n_init=1
        ).fit(combined, categorical=categorical_idxs)

    else:
        raise ValueError(
            'Clustering method must be one of `"kmeans"` or'
            f'`"kprototypes"` not {method}.'
        )

    return model.labels_.astype(int)


def _get_cluster_proportions(labels, synthetic_indicator, k):
    """Calculate the proportion of each cluster that is synthetic."""

    proportions = []
    for ki in range(k):
        proportions.append(
            sum(synthetic_indicator[labels == ki]) / sum(labels == ki)
        )

    return np.array(proportions)


def clustered_MSD(combined, synthetic_indicator, method, k, random_state=None):
    """Clustered mean-squared difference

    This metric performs a cluster analysis on the data and then considers the
    proportion of synthetic data in each cluster. It measures the difference
    between this and the overall proportion, squares, and then takes the mean
    across clusters.

    Parameters
    ----------
    combined : 2d numpy array
        Array containing both the real and synthetic data. Rows are examples
        and columns are variables.
    synthetic_indicator : 1d numpy array
        Corresponding boolean array indicating which of the rows in `combined`
        are synthetic (1), and which are real (0).
    method : {'kmeans','kprototypes'}, default='kmeans'
        Clustering method to use. See sklearn.cluster.KMeans and
        kmodes.kprototypes.KPrototypes for details.
    k : int
        Integer indicating how many clusters to form during the analysis.
    random_state : int, RandomState instance or None, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    MSD_p : float
        Mean-squared difference between the within-cluster proportions of
        synthetic data and the overall proportion of synthetic data.

    Notes
    -----
    There is no obvious criterion for selecting the number of clusters `k`. One
    approach when comparing two different synthetic datasets against eachother
    could be to try several values of `k` with the original data to examine the
    sensitivity of the choice of `k`, and then go with the one that is most
    often the best choice.

    Additionally, it is important to note that this metric says nothing about
    how the data are distributed within clusters.
    """

    labels = _get_cluster_labels(combined, method, k, random_state)
    proportions = _get_cluster_proportions(labels, synthetic_indicator, k)

    # calculate error from ideal proportion
    ideal_prop = np.mean(synthetic_indicator)
    prop_square_error = np.square(proportions - ideal_prop)

    return np.mean(prop_square_error)


def multi_clustered_MSD(
    real,
    synth,
    feats=None,
    method="kmeans",
    k_min=10,
    k_max=40,
    random_state=None,
):
    """multiple clustered mean-squared difference

    This metric performs the `clustered_MSD` above multiple times.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feats: str or list of str, optional
        Numeric features to use. Non-numeric features will be filtered out. By
        default all numeric features are used.
    method : {'kmeans','kprototypes'}, default='kmeans'
        Clustering method to use. See sklearn.cluster.KMeans and
        kmodes.kprototypes.KPrototypes for details.
    k_min : int
        Minimum number of clusters to perform cluster analysis with.
    k_max : int
        Maximum number of clusters to perform cluster analysis with.
    random_state : int, RandomState instance or None, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    cluster_MSDs : 1d numpy array
        array containing MSDs from cluster analyses with different number of
        clusters

    Notes
    -----
    Since there is no obvious way to select the number of clusters `k`, one
    approach when comparing two different synthetic datasets against eachother
    could be to try several values of `k` with the original data to examine the
    sensitivity of the choice of `k`, and then go with the one that is most
    often the best choice.

    Additionally, it is important to note that this metric says nothing about
    how the data are distributed within clusters.

    """
    # combine data
    combined = df_combine(
        real, synth, feats=feats, source_val_real=0, source_val_synth=1
    )
    # remove source column
    synth_bool = combined.pop("source")

    cluster_MSDs = []

    for k in range(k_min, k_max + 1):
        # print(k)
        cluster_MSDs.append(
            clustered_MSD(
                combined, synth_bool, method, k, random_state=random_state
            )
        )

    return min(cluster_MSDs)


if __name__ == "__main__":
    pass
