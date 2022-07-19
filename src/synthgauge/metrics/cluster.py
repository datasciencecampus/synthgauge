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
    combined : pandas.DataFrame
        Dataframe containing the real and synthetic data.
    method : {"kmeans", "kprototypes"}
        Which clustering method to use.
    k : int
        Number of clusters to fit.
    random_state : int, optional
        Random seed for fitting clusters.

    Returns
    -------
    labels : np.ndarray
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
    labels : array_like
        Cluster membership array.
    indicator : array_like
        Indicator of which data are real and which are synthetic.

    Returns
    -------
    proportions : numpy.ndarray
        Array with synthetic data proportion of each cluster.
    """

    proportions = []
    for cluster in np.unique(labels):
        proportions.append(
            sum(indicator[labels == cluster]) / sum(labels == cluster)
        )

    return np.array(proportions)


def clustered_msd(
    real,
    synth,
    feats=None,
    method="kmeans",
    k_min=2,
    k_max=10,
    random_state=None,
):
    """(Multiple) clustered mean-squared difference (MSD).

    This metric clusters the real and synthetic data together, measuring
    the synthetic utility according to its representation across the
    fitted clusters. Since there is often no obvious choice for the
    number of clusters, :math:`k`, we consider a range of values.

    For each value of :math:`k`, the chosen clustering method is fit
    and the proportion of synthetic data in each cluster is recorded.
    The clustered MSD is then calculated as the mean-squared difference
    between these proportions and the overall proportion of synthetic
    data.

    This collection of MSDs is summarised by taking its minimum to give
    the metric value.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feats : list of str or None, default None
        Features to use in the clustering. If `None` (default), all
        common features are used.
    method : {"kmeans", "kprototypes"}, default "kmeans"
        Clustering method to use. Only k-means and k-prototypes
        are implemented. If using k-means (default), only numeric
        columns are considered, while k-prototypes allows for mixed-type
        clustering.
    k_min : int, default 2
        Minimum number of clusters to use. Default of 2.
    k_max : int, default 10
        Maximum number of clusters to use. Default of 10.
    random_state : int, optional
        The random seed used to fit the clustering algorithm.

    Returns
    -------
    float
        The minimum observed clustered MSD.

    Notes
    -----
    This function can be used with a single value of `k` by setting
    `k_min` and `k_max` both to `k`. For instance, if a sensible number
    of clusters is known a priori.

    This metric says nothing about how appropriate the clustering method
    may be for the data at hand, nor how the data are distributed among
    the clusters. Both methods considered here have rather strong
    assumptions about the relative size and characteristics of the
    clusters in the data. As such, exploratory analysis is advised to
    determine whether such centroid-based clustering is well-suited.
    """

    combined = df_combine(real, synth, feats, "source", 0, 1)
    indicator = combined.pop("source")

    msds = []
    for k in range(k_min, k_max + 1):
        labels = _get_cluster_labels(combined, method, k, random_state)
        proportions = _get_cluster_proportions(labels, indicator)
        msd = np.mean(np.square(proportions - indicator.mean()))
        msds.append(msd)

    return min(msds)
