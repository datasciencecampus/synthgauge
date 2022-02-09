"""
clustering metric
"""
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
import numpy as np
from ..utils import df_combine


def clustered_MSD(combined, synthetic_indicator, method, k, random_state=None):
    """clustered mean-squared difference

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
    if method == 'kmeans':
        combined_numeric = combined.select_dtypes(include='number')
        clusters = KMeans(n_clusters=k, random_state=random_state).fit(
            combined_numeric).labels_
    if method == 'kprototype':
        cat_cols = combined.select_dtypes(
            include=['object', 'category']).columns
        cat_cols_i = [n for n, col in enumerate(
            combined.columns) if col in cat_cols]
        clusters = KPrototypes(n_clusters=k, random_state=random_state).fit(
            combined, categorical=cat_cols_i).labels_
    cluster_proportion = []
    for ki in range(k):
        cluster_proportion.append(sum(synthetic_indicator[clusters == ki])
                                  / sum(clusters == ki))
    cluster_proportion = np.array(cluster_proportion)

    # calculate error from ideal proportion
    ideal_prop = sum(synthetic_indicator)/len(combined)

    prop_square_error = np.square(cluster_proportion - ideal_prop)

    MSE_p = sum(prop_square_error)/len(prop_square_error)
    # print(MSE_p)
    return MSE_p


def multi_clustered_MSD(real, synth, feats=None, method='kmeans', k_min=10,
                        k_max=40, random_state=None):
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
    combined = df_combine(real, synth, feats=feats,
                          source_val_real=0, source_val_synth=1)
    # remove source column
    synth_bool = combined.pop('source')

    cluster_MSDs = []

    for k in range(k_min, k_max):
        # print(k)
        cluster_MSDs.append(clustered_MSD(
            combined, synth_bool, method, k, random_state=random_state))

    return min(cluster_MSDs)


if __name__ == '__main__':
    pass
