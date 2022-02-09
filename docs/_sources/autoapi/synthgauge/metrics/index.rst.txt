:py:mod:`synthgauge.metrics`
============================

.. py:module:: synthgauge.metrics


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   classification/index.rst
   cluster/index.rst
   correlation/index.rst
   privacy/index.rst
   propensity/index.rst
   univariate_distance/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.metrics.classification_comparison
   synthgauge.metrics.multi_clustered_MSD
   synthgauge.metrics.correlation_MSD
   synthgauge.metrics.correlation_ratio_MSE
   synthgauge.metrics.cramers_v_MSE
   synthgauge.metrics.TCAP
   synthgauge.metrics.min_NN_dist
   synthgauge.metrics.sample_overlap_score
   synthgauge.metrics.propensity_metrics
   synthgauge.metrics.feature_density_diff_mae
   synthgauge.metrics.jensen_shannon_distance
   synthgauge.metrics.jensen_shannon_divergence
   synthgauge.metrics.kolmogorov_smirnov
   synthgauge.metrics.kruskal_wallis
   synthgauge.metrics.kullback_leibler
   synthgauge.metrics.mann_whitney
   synthgauge.metrics.wasserstein
   synthgauge.metrics.wilcoxon



.. py:function:: classification_comparison(real, synth, key, target, sklearn_classifier, random_state=None, test_prop=0.2, **kwargs)

   Classification utility metric

   This metric fits two classification models to `real` and `synth`
   respectively, and then tests them both against withheld `real` data. We
   obtain utility scores by subtracting the precision, recall and f1 scores
   of the predictions obtained by the synth model from those obtained by the
   real model.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param key: list of column names to use as the input in the classification.
   :type key: list of strings
   :param target: column to use as target in the classification.
   :type target: str
   :param sklearn_classifier: classifier with fit and predict methods.
   :type sklearn_classifier: scikit-learn estimator
   :param random_state: Controls the shuffling steps during the train-test split and the
                        classification algorithm itself. Pass an int for reproducible output
                        across multiple function calls.
   :type random_state: int, RandomState instance or None, default=42
   :param test_prop: If float, should be between 0.0 and 1.0 and represent the proportion
                     of the dataset to include in the test split. If int, represents the
                     absolute number of test samples.
   :type test_prop: float or int, default=0.2

   :returns: **ClassificationResult** --

             precision_difference : float
                 precision of model trained on real data subtracted by precision of
                 model trained on synthetic data.
             recall_difference : float
                 recall of model trained on real data subtracted by recall of
                 model trained on synthetic data.
             f1_difference : float
                 f1 score of model trained on real data subtracted by f1 score of
                 model trained on synthetic data.
   :rtype: namedtuple

   .. rubric:: Notes

   Some preprocessing is carried out before the models are trained. Numeric
   features are scaled and categorical features are one-hot-encoded.

   A score of zero tells us the synthetic data is just as good as the real at
   training classifier models. Increases in these scores indicate poorer
   utility.


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


.. py:function:: correlation_MSD(real, synth, feats=None)

   correlation mean-squared-difference

   This metric calculates the mean squared difference between the correlation
   matrices for the real and synthetic datasets. This gives an indication of
   how well the synthetic data has retained inter-variable relationships.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feats: Numeric features to use. Non-numeric features will be filtered out. By
                 default all numeric features are used.
   :type feats: str or list of str, optional

   :returns: **MSD_c** -- Correlation mean squared difference.
   :rtype: float

   .. seealso:: :obj:`numpy.corrcoef`

   .. rubric:: Notes

   Correlations only make sense for numerical data, so this function first
   filters the columns by datatype.

   The smaller the number this function returns, the better the synthetic
   data captures the correlation between variables in the real data. This
   method is therefore good for comparing multiple synthetic datasets to see
   which is closest to the real. However, as this is not a test, there is no
   threshold value below which we can claim the datasets are statistically
   the same.


.. py:function:: correlation_ratio_MSE(real, synth, categorical_feats='auto', numerical_feats=None)

   Correlation Ratio Mean Squared Error

   This metric calculates the difference in association between categorical
   and continuous feature pairings in the real and synthetic datasets.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param categorical_feats: Categorical feature(s) in `real` and `synth` to include in comparison.
                             By default all object and categorical columns are selected.
   :type categorical_feats: str or list of str, optional
   :param numerical_feats: Numerical feature(s) in `real` and `synth` to include in comparison.
                           By default all columns not in `categorical_feats` are selected.
   :type numerical_feats: str or list of str, optional

   :returns: **corr_ratio_MSE** -- Mean squared error between `real` and `synth` in correlation ratio
             scores across valid feature pairs.
   :rtype: float

   .. rubric:: Notes

   If no categorical features are selected, columns of type `category` or
   `object` are used. If no numerical features are selected, all columns that
   are not listed as categorical features are used.


.. py:function:: cramers_v_MSE(real, synth, feats=None)

   Cramer's V Mean Squared Error

   This metric calculates the difference in association between categorical
   features in the real and synthetic datasets.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feats: Feature(s) in `real` and `synth` to include in comparison. By default
                 all object and categorical columns are selected.
   :type feats: str or list of str, optional

   :returns: **cramers_v_MSE** -- Mean squared error between `real` and `synth` in Cramer's V scores
             across feature pairs.
   :rtype: float

   .. rubric:: Notes

   This metric is only valid for categorical features so a warning is returned
   if any of the selected features appear to be numeric. If no features are
   selected, only the `category` and `object` types are used.


.. py:function:: TCAP(real, synth, key, target)

   Target Correct Attribution Probability TCAP

   This privacy metric calculates the average chance that the key-target
   pairings in the `synth` dataset reveal the true key-target pairings in the
   original, `real` dataset.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param key: List of features in `synth` to use as the key.
   :type key: list
   :param target: Feature to use as the target.
   :type target: str or list of str

   :returns: **TCAP** -- The average TCAP across the dataset.
   :rtype: float

   .. rubric:: Notes

   This metric provides an estimate of how well an intruder could infer
   attributes of groups in the real dataset by studying the synthetic. The
   choices for `key` and `target` will vary depending on the dataset in
   question but we would suggest the `key` features are those that could be
   readily available to an outsider and the `target` feature is one we
   wouldn't want them finding out, such as a protected characteristic.

   This method only works with categorical data, so binning of continuous data
   may be required.


.. py:function:: min_NN_dist(real, synth, feats=None, real_outliers_only=True, outlier_factor_threshold=2)

   Minimum Nearest Neighbour distance

   This privacy metric returns the smallest distance between any point in
   the `real` dataset and any point in the `synth` dataset. There is an
   option to only consider the outliers in the real dataset as these perhaps
   pose more of a privacy concern.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feats: Features to use. By default all features are used.
   :type feats: str or list of str, optional
   :param real_outliers_only: Boolean indicating whether to filter out inliers (default) or not.
   :type real_outliers_only: bool (default True)
   :param outlier_factor_threshold: Float influencing classification of ouliers. Increase to include
                                    fewer real points in nearest neighbour calculations.
   :type outlier_factor_threshold: float (default 2)

   :returns: **min_dist** -- Minimum manhattan distance between `real` and `synth` data.
   :rtype: float

   .. rubric:: Notes

   This privacy metric provides an insight into whether the synthetic dataset
   is too similar to the real dataset. It does this by calculating the
   minimum distance between the real records and the synthetic records.

   This metric assumes that categorical data is ordinal during distance
   calculations, or that it has already been suitably one-hot-encoded.


.. py:function:: sample_overlap_score(real, synth, feats=None, sample_size=0.2, runs=5, score_type='unique')

   Return percentage of overlap between real and synth data.

   Samples from both the real and synthetic datasets are compared for
   similarity. This similarity, or overlap score, is based on the
   exact matches of real data records within the synthetic data.

   :param real: DataFrame containing the real data.
   :type real: pandas.DataFrame
   :param synth: DataFrame containing the synthetic data.
   :type synth: pandas.DataFrame
   :param feats: The features that will be used to match records. By
                 default all features will be used.
   :type feats: str or list of str, optional.
   :param sample_size: The ratio (if `sample_size` between 0 and 1) or count
                       (`sample_size` > 1) of records to sample. Default is 0.2 or 20%.
   :type sample_size: float or int, optional
   :param runs: The number of times to compute the score. Total score is averaged
                across runs.
   :type runs: int, optional
   :param score_type: Method used for calculating the overlap score. If "unique", the
                      default, the score is the percentage of unique records in the real
                      sample that have a match within the synth data. If "sample" the
                      score is the percentage of all records within the real sample that
                      have a match within the synth sample.
   :type score_type: {"unique"|"sample"}

   :returns: Overlap score between `real` and `synth`
   :rtype: float


.. py:function:: propensity_metrics(real, synth, method='CART', feats=None, num_perms=20, **kwargs)

   Propensity metrics

   This function calculates three flavours of propensity mean-squared error,
   all of which quantify utility by measuring how well a classifier can be
   trained to distinguish `real` and `synth`. It returns the raw, observed
   value together with this value standardised by the expected value if the
   dataset was well synthesised, and also as a ratio of the expected value.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param method: Classification method to use.
   :type method: str, ['CART', 'LogisticRegression']
   :param feats: List of features in the dataset that will be used in propensity
                 calculations. By default all features will be used.
   :type feats: list of str, optional.
   :param num_perms: If CART method is used, this is the number of times to repeat process
                     when calculating the expected mean-squared error.
   :type num_perms: int
   :param \*\*kwargs: Keyword arguments passed to LogisticRegression function.
   :type \*\*kwargs: dict

   :returns: * **observed_p_MSE** (*float*) -- Observed propensity score mean squared error.
             * **standardised_p_MSE** (*float*) -- Standardised propensity score mean square-squared error.
             * **ratio_p_MSE** (*float*) -- Ratio of the propensity score mean square-squared error to the
               expected value.

   .. rubric:: Notes

   The standardisation and ratio operations transform the scales for the MSE
   into ones that make more sense for synthetic data. For the observed
   propensity the MSE score gives better utility as the value gets closer to
   zero, which is only attainable when the datasets are identical. However,
   when generating synthetic data we do not want to produce identical entries,
   but to acheive distributional similarity between the distribution of the
   observed data and the model used to generate the synthetic.

   The standardised score tends towards zero when the datasets are similar
   and increases otherwise.

   The ratio tends towards one when the datasets are similar and increases
   otherwise.


.. py:function:: feature_density_diff_mae(real, synth, feats=None, bins=10)

   Calculate Mean Absolute Error of feature densities.

   For each feature the difference between the density across the bins
   within `real` and `synth` is calculated. Finally the MAE across all
   features and bins is calculated. A value close to 0 indicates that
   a similar distribution for `feats` is observed between the real and
   synthetic datasets.

   :param real: DataFrame containing the real data.
   :type real: pandas.DataFrame
   :param synth: DataFrame containing the sythetic data.
   :type synth: pandas.DataFrame
   :param feats: The features that will be used to compute the densities. By
                 default all features in `real` will be used.
   :type feats: str or list of str, optional.
   :param bins: Bins to use for computing the density. This value is passed
                to `numpy.histogram_bin_edges` so can be any value accepted by
                that function. The default setting of 10 uses 10 bins.
   :type bins: str or int, optional

   :returns: Mean Absolute Error of feature densities.
   :rtype: float


.. py:function:: jensen_shannon_distance(real, synth, feature, bins='auto', **kwargs)

   Distance: Jensen-Shannon

   The Jensen-Shannon distance describes the difference between the `real` and
   `synth` distributions of the `feature` in terms of entropy. It is the
   square root of Jensen-Shannon divergence. It measures the distance between
   probabilities so the data are first discretised into bins.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feature: String indicating the feature of the datasets to compare. This must be
                   a continuous variable.
   :type feature: str
   :param bins: Number of bins to use when discretising data, if `None` data will be
                treated as categorical. If string chosen method will be used to
                calculate optimal bin width, 'auto' is the default. See
                numpy.histogram_bin_edges for further options.
   :type bins: int or str
   :param \*\*kwargs: Keyword arguments.
   :type \*\*kwargs: dict

   :returns: **distance** -- The computed distance between the distributions.
   :rtype: double

   .. seealso:: :obj:`divergence.jensen_shannon_divergence`, :obj:`scipy.spatial.distance.jensenshannon`

   .. rubric:: Notes

   This is a wrapper function for `scipy.spatial.distance.jensenshannon`.
   Since this function expects probability vectors the data is first
   discretised into evenly-spaced bins.

   We can think of the Jensen-Shannon distance as the amount of information,
   or entropy, encoded in the difference between the `real` and `synth`
   distributions of the `feature`.

   The distance is zero if the distributions are identical, and is bounded
   above by one if they are nothing alike. This method is therefore good
   for comparing multiple synthetic datasets, or features within a dataset,
   to see which is closest to the real. However, as this is not a test,
   there is no threshold distance below which we can claim the distributions
   are statistically the same.

   An optimal 'bins' value has not been suggested.

   .. rubric:: Examples

   >>> import pandas as pd
   >>> real_data = pd.DataFrame(get_real(500),
   ...                          columns = ['feat1', 'feat2', 'feat3'])
   >>> synth_data = pd.DataFrame(get_synth(500),
   ...                           columns = ['feat1', 'feat2', 'feat3'])

   The first feature appears to be more similar than the second across
   datasets.

   >>> jensen_shannon_distance(real_data, synth_data, 'feat1', bins = 20)
   0.11006632967333475 # random
   >>> jensen_shannon_distance(real_data, synth_data, 'feat2', bins = 20)
   0.43556476029981644 # random


.. py:function:: jensen_shannon_divergence(real, synth, feature, bins='auto', **kwargs)

   Divergence: Jensen-Shannon

   The Jensen-Shannon divergence describes the difference between the `real`
   and `synth` distributions of the `feature` in terms of entropy. It is the
   square of Jensen-Shannon distance.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feature: String indicating the feature of the datasets to compare. This must be
                   a continuous variable.
   :type feature: str
   :param bins: Number of bins to use when discretising data, if `None` data will be
                treated as categorical. If string chosen method will be used to
                calculate optimal bin width, 'auto' is the default. See
                numpy.histogram_bin_edges for further options.
   :type bins: int or str
   :param \*\*kwargs: Keyword arguments.
   :type \*\*kwargs: dict

   :returns: **divergence** -- The computed divergence between the distributions.
   :rtype: double

   .. seealso:: :obj:`distance.jensen_shannon_distance`, :obj:`scipy.spatial.distance.jensenshannon`

   .. rubric:: Notes

   This is a wrapper function that just squares the result of
   `distance.jensen_shannon_distance`, which in turn wraps
   `scipy.spatial.distance.jensenshannon`. Since this function expects
   probability vectors the data is first discretised into evenly-spaced bins.

   We can think of the Jensen-Shannon divergence as the amount of information,
   or entropy, encoded in the difference between the `real` and `synth`
   distributions of the `feature`.

   The distance is zero if the distributions are identical, and is bounded
   above by one if they are nothing alike. This method is therefore good
   for comparing multiple synthetic datasets, or features within a dataset,
   to see which is closest to the real. However, as this is not a test,
   there is no threshold distance below which we can claim the distributions
   are statistically the same.

   An optimal 'bins' value has not been suggested.

   .. rubric:: Examples

   >>> import pandas as pd
   >>> real_data = pd.DataFrame(get_real(500),
                                columns = ['feat1', 'feat2', 'feat3'])
   >>> synth_data = pd.DataFrame(get_synth(500),
                                 columns = ['feat1', 'feat2', 'feat3'])

   The first feature appears to be more similar than the second across
   datasets.

   >>> jensen_shannon_divergence(real_data, synth_data, 'feat1', bins = 20)
   0.11006632967333475 # random
   >>> jensen_shannon_divergence(real_data, synth_data, 'feat2', bins = 20)
   0.43556476029981644 # random


.. py:function:: kolmogorov_smirnov(real, synth, feature, **kwargs)

   Distance: Kolmogorov-Smirnov

   The Kolmogorov-Smirnov metric calculates the maximum difference between the
   cumulative distribution functions of the `feature` in the `real` and
   `synth` datasets.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feature: String indicating the feature of the datasets to compare. This must be
                   a continuous variable.
   :type feature: str
   :param \*\*kwargs: Keyword arguments.
   :type \*\*kwargs: dict

   :returns: * **statistic** (*float*) -- Kolmogorov-Smirnov statistic
             * **pvalue** (*float*) -- Two-tailed p-value

   .. seealso:: :obj:`scipy.stats.ks_2samp`

   .. rubric:: Notes

   This is a wrapper function for `scipy.stats.ks_2samp`, which tests whether
   two samples are drawn from the same distribution by calculating the maximum
   difference between their cumulative distribution functions.

   If the returned statistic is small or the p-value is high, then we cannot
   reject the hypothesis that the distributions are the same.

   This approach is only defined if the feature is continuous. The SciPy
   documentation further suggests this method works best when one of the
   samples has a size of only a few thousand.

   .. rubric:: Examples

   >>> import pandas as pd
   >>> real_data = pd.DataFrame(get_real(500),
   ...                          columns = ['feat1', 'feat2', 'feat3'])
   >>> synth_data = pd.DataFrame(get_synth(500),
   ...                           columns = ['feat1', 'feat2', 'feat3'])

   The first feature appears to come from the same distribution in both
   datasets.

   >>> kolmogorov_smirnov(real_data, synth_data, 'feat1')
   KstestResult(statistic=0.062, pvalue=0.2919248807417811) # random

   The second feature appears to come from different distributions in the
   datasets.

   >>> kolmogorov_smirnov(real_data, synth_data, 'feat2')
   KstestResult(statistic=0.274, pvalue=6.383314923658339e-17) # random


.. py:function:: kruskal_wallis(real, synth, feature, **kwargs)

   Hypothesis Test: Kruskal-Wallis

   The Kruskal-Wallis test compares the distributions of data by examining how
   well-mixed they are when pooled. This is acheived by ranking the pooled
   data. A low p-value suggests the data are not similar.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feature: String indicating the feature of the datasets to compare. This must be
                   a continuous variable.
   :type feature: str
   :param \*\*kwargs: Keyword arguments.
   :type \*\*kwargs: dict

   :returns: * **statistic** (*float*) -- The Kruskal-Wallis H statistic.
             * **pvalue** (*float*) -- The p-value for the test.

   .. seealso:: :obj:`scipy.stats.kruskal`

   .. rubric:: Notes

   This is a wrapper function for `scipy.stats.kruskal`.

   The null hypothesis for this test is that the medians of the distributions
   are equal. The alternative hypothesis is then that they are different. This
   would suggest that the synthetic and real data are not similarly
   distributed.

   We notice however that failure to reject the null hypothesis only suggests
   that the medians could be equal and says nothing else about how the data
   are distributed.

   .. rubric:: Examples

   >>> import pandas as pd
   >>> real_data = pd.DataFrame(get_real(500),
   ...                          columns = ['feat1', 'feat2', 'feat3'])
   >>> synth_data = pd.DataFrame(get_synth(500),
   ...                            columns = ['feat1', 'feat2', 'feat3'])

   If we were to choose our p-value threshold as 0.05, we would reach the
   conclusion that the distributions of the first feature are similar but the
   distributions of the second feature are not.

   >>> kruskal_wallis(real_data, synth_data, 'feat1', bins = 20)
   KruskalResult(statistic=1.4447530549450676, pvalue=0.22937173881858086)
   # random
   >>> kruskal_wallis(real_data, synth_data, 'feat2', bins = 20)
   KruskalResult(statistic=5.1566145854149, pvalue=0.023157995217201643)
   # random


.. py:function:: kullback_leibler(real, synth, feature, bins='auto', **kwargs)

   Divergence: Kullback-Leibler

   The Kullback-Leibler divergence describes how much the `real` distribution
   of the `feature` varies from the `synth` in terms of entropy. This is an
   assymmetric measure so does not describe the opposing variation. Since it
   measures the variation between probabilities, the data are first
   discretised into bins.


   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feature: String indicating the feature of the datasets to compare. This must be
                   a continuous variable.
   :type feature: str
   :param bins: Number of bins to use when discretising data, if `None` data will be
                treated as categorical. If string chosen method will be used to
                calculate optimal bin width, 'auto' is the default. See
                numpy.histogram_bin_edges for further options.
   :type bins: int or str
   :param \*\*kwargs: Keyword arguments.
   :type \*\*kwargs: dict

   :returns: **D** -- The calculated divergence.
   :rtype: float

   .. seealso:: :obj:`scipy.stats.entropy`

   .. rubric:: Notes

   This is a wrapper function for `scipy.stats.entropy`. Since this function
   expects a probability vector, the data is first discretised into evenly-
   spaced bins.

   We can think of the Kullback-Leibler divergence as a measure of surprise
   we might expect seeing an example from the real data, relative to the
   distribution of the synthetic.

   The divergence is zero if the distributions are identical, and is bounded
   above by one if they are nothing alike. This method is therefore good
   for comparing multiple synthetic datasets, or features within a dataset,
   to see which is closest to the real. However, as this is not a test,
   there is no threshold distance below which we can claim the distributions
   are statistically the same.

   An optimal 'bins' value has not been suggested.

   BUG: returns `inf` if no real data falls in any one of the bins - dividing
   by zero error.

   .. rubric:: Examples

   >>> import pandas as pd
   >>> real_data = pd.DataFrame(get_real(500),
                                columns = ['feat1', 'feat2', 'feat3'])
   >>> synth_data = pd.DataFrame(get_synth(500),
                                 columns = ['feat1', 'feat2', 'feat3'])

   The first feature appears to be more similar than the second across
   datasets.

   >>> kullback_leibler(real_data, synth_data, 'feat1', bins = 20)
   0.03389133708660097 # random
   >>> kullback_leibler(real_data, synth_data, 'feat2', bins = 20)
   0.58739109417064730 # random


.. py:function:: mann_whitney(real, synth, feature, **kwargs)

   Hypothesis Test: Mann-Whitney

   The Mann-Whitney test compares the distributions of data by examining how
   well-mixed they are when pooled. This is acheived by ranking the pooled
   data. A low p-value suggests the data are not similar.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feature: String indicating the feature of the datasets to compare. This must be
                   a continuous variable.
   :type feature: str
   :param \*\*kwargs: Keyword arguments.
   :type \*\*kwargs: dict

   :returns: * **statistic** (*float*) -- The Mann-Whitney U statistic, in particular U for `synth`.
             * **pvalue** (*float*) -- Two-sided p-value assuming an asymptotic normal distribution.

   .. seealso:: :obj:`scipy.stats.mannwhitneyu`

   .. rubric:: Notes

   This is a wrapper function for `scipy.stats.mannwhitneyu`.

   The null hypothesis for this test is that for randomly selected real and
   synthetic values, the probability that the real value is greater than the
   synthetic is the same as the probability that the synthetic value is
   greater than the real.

   We reject this hypothesis if the p-value is suitably small. This would in
   turn suggest that the synthetic and real data are not similarly
   distributed.

   .. rubric:: Examples

   >>> import pandas as pd
   >>> real_data = pd.DataFrame(get_real(500),
   ...                          columns = ['feat1', 'feat2', 'feat3'])
   >>> synth_data = pd.DataFrame(get_synth(500),
   ...                            columns = ['feat1', 'feat2', 'feat3'])

   If we were to choose our p-value threshold as 0.05, we would reach the
   conclusion that the distributions of the first feature are similar but
   the distributions of the second feature are not.

   >>> mann_whitney(real_data, synth_data, 'feat1', bins = 20)
   MannwhitneyuResult(statistic=126910.0, pvalue=0.6758436855431454) # random
   >>> mann_whitney(real_data, synth_data, 'feat2', bins = 20)
   MannwhitneyuResult(statistic=134107.0, pvalue=0.04613704446362845) # random


.. py:function:: wasserstein(real, synth, feature, **kwargs)

   Distance: Wasserstein

   The Wasserstein distance, or Earth Mover's distance, can be thought of as
   calculating the amount of "work" required to move from the distribution of
   the synthetic data to the distribution of the real data.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feature: String indicating the feature of the datasets to compare. This must be
                   a continuous variable.
   :type feature: str
   :param \*\*kwargs: Keyword arguments.
   :type \*\*kwargs: dict

   :returns: **distance** -- The computed distance between the distributions.
   :rtype: float

   .. seealso:: :obj:`scipy.stats.wasserstein_distance`

   .. rubric:: Notes

   This is a wrapper function for `scipy.stats.wasserstein_distance`.

   Computationally, we can find the Wasserstein distance by calculating the
   area between the cumulative distribution functions for the two
   distributions.

   If :math:`s` is the `synth` distribution of the `feature`, :math:`r` is the
   `real` distribution of the `feature` and :math:`R` and :math:`S` are their
   respective cumulative distribution functions, then

   .. math::

       wasserstein(s, r) = \int_{-\infty}^{+\infty} |S-R|

   The distance is zero if the distributions are identical, and increases as
   they become less alike. This method is therefore good for comparing
   multiple synthetic datasets, or features within a dataset, to see which is
   closest to the real. However, as this is not a test, there is no threshold
   distance below which we can claim the distributions are statistically the
   same.

   .. rubric:: Examples

   >>> import pandas as pd
   >>> real_data = pd.DataFrame(get_real(500),
   ...                          columns = ['feat1', 'feat2', 'feat3'])
   >>> synth_data = pd.DataFrame(get_synth(500),
   ...                           columns = ['feat1', 'feat2', 'feat3'])

   The first feature appears to be more similar than the second across
   datasets.

   >>> wasserstein(real_data, synth_data, 'feat1')
   0.0688192355094602 # random
   >>> wasserstein(real_data, synth_data, 'feat2')
   0.8172329918412307 # random


.. py:function:: wilcoxon(real, synth, feature, **kwargs)

   Hypothesis Test: Wilcoxon

   The Wilcoxon test compares the distributions of paired data. It does this
   by ranking the differences.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param feature: String indicating the feature of the datasets to compare. This must be
                   a continuous variable.
   :type feature: str
   :param \*\*kwargs: Keyword arguments.
   :type \*\*kwargs: dict

   :returns: * **statistic** (*float*) -- The sum of the ranks of the differences above or below zero, whichever
               is greater.
             * **pvalue** (*float*) -- Two-sided p-value.

   .. seealso:: :obj:`scipy.stats.wilcoxon`

   .. rubric:: Notes

   This is a wrapper function for `scipy.stats.wilcoxon`.

   The null hypothesis for this test is that the median of the paired
   differences is zero. The alternative hypothesis is that it is different
   from zero. This would suggest that the synthetic and real data are not
   similarly distributed.

   This test only makes sense when the synthetic and real data are paired.
   That is, each synthetic datapoint is matched to a real one. In which case,
   it is required that data are ordered to reflect this.

   .. rubric:: Examples

   >>> import pandas as pd
   >>> real_data = pd.DataFrame(get_real(500),
   ...                          columns = ['feat1', 'feat2', 'feat3'])
   >>> synth_data = pd.DataFrame(get_synth(500),
   ...                            columns = ['feat1', 'feat2', 'feat3'])

   If we were to choose our p-value threshold as 0.05, we would reach the
   conclusion that the distributions of the first feature are similar but of
   the second feature are not.

   >>> wilcoxon(real_data, synth_data, 'feat1', bins = 20)
   WilcoxonResult(statistic=58917.0, pvalue=0.25131501183065175) # random
   >>> wilcoxon(real_data, synth_data, 'feat2', bins = 20)
   WilcoxonResult(statistic=54474.0, pvalue=0.011678503879013464) # random


