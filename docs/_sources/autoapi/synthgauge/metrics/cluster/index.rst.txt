:py:mod:`synthgauge.metrics.cluster`
====================================

.. py:module:: synthgauge.metrics.cluster

.. autoapi-nested-parse::

   clustering metric



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.metrics.cluster.clustered_MSD
   synthgauge.metrics.cluster.multi_clustered_MSD



.. py:function:: clustered_MSD(combined, synthetic_indicator, method, k, random_state=None)

   clustered mean-squared difference

   This metric performs a cluster analysis on the data and then considers the
   proportion of synthetic data in each cluster. It measures the difference
   between this and the overall proportion, squares, and then takes the mean
   across clusters.

   :param combined: Array containing both the real and synthetic data. Rows are examples
                    and columns are variables.
   :type combined: 2d numpy array
   :param synthetic_indicator: Corresponding boolean array indicating which of the rows in `combined`
                               are synthetic (1), and which are real (0).
   :type synthetic_indicator: 1d numpy array
   :param method: Clustering method to use. See sklearn.cluster.KMeans and
                  kmodes.kprototypes.KPrototypes for details.
   :type method: {'kmeans','kprototypes'}, default='kmeans'
   :param k: Integer indicating how many clusters to form during the analysis.
   :type k: int
   :param random_state: If int, random_state is the seed used by the random number generator;
                        If RandomState instance, random_state is the random number generator;
                        If None, the random number generator is the RandomState instance used
                        by `np.random`.
   :type random_state: int, RandomState instance or None, default: None

   :returns: **MSD_p** -- Mean-squared difference between the within-cluster proportions of
             synthetic data and the overall proportion of synthetic data.
   :rtype: float

   .. rubric:: Notes

   There is no obvious criterion for selecting the number of clusters `k`. One
   approach when comparing two different synthetic datasets against eachother
   could be to try several values of `k` with the original data to examine the
   sensitivity of the choice of `k`, and then go with the one that is most
   often the best choice.

   Additionally, it is important to note that this metric says nothing about
   how the data are distributed within clusters.


.. py:function:: multi_clustered_MSD(real, synth, feats=None, method='kmeans', k_min=10, k_max=40, random_state=None)

   multiple clustered mean-squared difference

   This metric performs the `clustered_MSD` above multiple times.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feats: Numeric features to use. Non-numeric features will be filtered out. By
                 default all numeric features are used.
   :type feats: str or list of str, optional
   :param method: Clustering method to use. See sklearn.cluster.KMeans and
                  kmodes.kprototypes.KPrototypes for details.
   :type method: {'kmeans','kprototypes'}, default='kmeans'
   :param k_min: Minimum number of clusters to perform cluster analysis with.
   :type k_min: int
   :param k_max: Maximum number of clusters to perform cluster analysis with.
   :type k_max: int
   :param random_state: If int, random_state is the seed used by the random number generator;
                        If RandomState instance, random_state is the random number generator;
                        If None, the random number generator is the RandomState instance used
                        by `np.random`.
   :type random_state: int, RandomState instance or None, default: None

   :returns: **cluster_MSDs** -- array containing MSDs from cluster analyses with different number of
             clusters
   :rtype: 1d numpy array

   .. rubric:: Notes

   Since there is no obvious way to select the number of clusters `k`, one
   approach when comparing two different synthetic datasets against eachother
   could be to try several values of `k` with the original data to examine the
   sensitivity of the choice of `k`, and then go with the one that is most
   often the best choice.

   Additionally, it is important to note that this metric says nothing about
   how the data are distributed within clusters.


