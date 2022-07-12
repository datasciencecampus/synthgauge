"""Utility metrics derived from centroid-based clustering."""

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans

from ..utils import df_combine


def _get_cluster_labels(combined, method, k, random_state):
    """Apply the chosen clustering method to a dataset and return its
    final labels.

    Parameters
    ----------
    combined: pandas.DataFrame
        Dataframe containing the real and synthetic data.
    method: {"kmeans", "kprototypes"}
        Which clustering method to use.
    k: int
        Number of clusters to fit.
    random_state: int, optional
        Random seed for fitting clusters.

    Returns
    -------
    labels: np.ndarray
        Integer labels indicating cluster membership for each point in
        `combined`.
    """

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


def _get_cluster_proportions(labels, indicator):
    """Calculate the proportion of each cluster that is synthetic.

    Parameters
    ----------
    labels, indicator: array_like
        Arrays detailing cluster membership (`labels`) and which points
        are real or synthetic (`indicator`).

    Returns
    -------
    proportions: numpy.ndarray
        Array with synthetic data proportion of each cluster.
    """

    proportions = []
    for cluster in np.unique(labels):
        proportions.append(
            sum(indicator[labels == cluster]) / sum(labels == cluster)
        )

    return np.array(proportions)


def clustered_MSD(combined, synthetic_indicator, method, k, random_state=None):
    """Clustered mean-squared difference.

    This metric performs a cluster analysis on the data and then
    considers the proportion of synthetic data in each cluster. It
    measures the mean squared difference between this and the overall
    proportion across all clusters.

    Parameters
    ----------
    combined : pandas.DataFrame
        Dataframe containing the real and synthetic data.
    synthetic_indicator : array_like
        Integer-boolean array indicating which of the rows in `combined`
        are synthetic (1), and which are real (0).
    method : {"kmeans", "kprototypes"}
        Clustering method to use. Only k-means and k-prototypes
        are implemented. If using k-means, only numeric columns in
        `combined` are considered, while k-prototypes allows for
        mixed-type clustering.
    k : int
        Integer indicating how many clusters to fit to `combined`.
    random_state : int, optional
        The random seed used to fit the clustering algorithm, providing
        reproducible results.

    Returns
    -------
    float
        Mean-squared difference between the within-cluster proportions
        of synthetic data and the overall proportion of synthetic data.

    See Also
    --------
    sklearn.cluster.KMeans
    kmodes.kprototypes.KPrototypes

    Notes
    -----
    There is no obvious criterion for selecting the number of clusters
    `k`. One approach when comparing two different synthetic datasets
    against each other could be to try several values of `k` with the
    original data to examine the sensitivity of the choice of `k`, and
    then go with the one that is most often the best choice.

    Additionally, it is important to note that this metric says nothing
    about how the data are distributed within clusters.
    """

    labels = _get_cluster_labels(combined, method, k, random_state)
    proportions = _get_cluster_proportions(labels, synthetic_indicator)

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
    """Multiple clustered mean-squared difference (CMSD).

    This metric calculates `clustered_MSD` across a range of values for
    `k`, the number of clusters, and returns the minimum value.

    Parameters
    ----------
    real, synth : pandas.DataFrame
        Dataframes containing the real and synthetic data.
    feats : str or list of str or None, default None
        Feature(s) to use in the clustering. If `None` (default), all
        features in `real` and `synth` are used.
    method : {"kmeans", "kprototypes"}, default "kmeans"
        Clustering method to use. Only k-means and k-prototypes
        are implemented. If using k-means (default), only numeric
        columns are considered, while k-prototypes allows for mixed-type
        clustering.
    k_min, k_max : int
        Minimum and maximum number of clusters to use. Defaults are 10
        and 40, respectively.
    random_state : int, optional
        The random seed used to fit the clustering algorithm.

    Returns
    -------
    float
        The minimum observed clustered MSD.
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
