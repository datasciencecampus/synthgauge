:py:mod:`synthgauge.metrics.privacy`
====================================

.. py:module:: synthgauge.metrics.privacy


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.metrics.privacy.get_WEAP
   synthgauge.metrics.privacy.TCAP
   synthgauge.metrics.privacy.find_outliers
   synthgauge.metrics.privacy.min_NN_dist
   synthgauge.metrics.privacy.sample_overlap_score



.. py:function:: get_WEAP(synth, key, target)

   Get the Within Equivalence class Attribution Probabilities WEAP

   For each record in the synthetic dataset, this function returns the
   proportion across the whole dataset that these `key` values are matched
   with this `target` value.

   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param key: List of features in `synth` to use as the key.
   :type key: list
   :param target: Feature to use as the target.
   :type target: str or list of str

   :returns: A series object containing the WEAP scores for each record in `synth`.
   :rtype: pandas.Series

   .. rubric:: Notes

   This function is intended to only be used within `TCAP()` to determine
   which synthetic records are most likely to pose an attribution risk.


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


.. py:function:: find_outliers(data, outlier_factor_threshold)

   Find Outliers

   This function returns whether each row in `data` can be considered an
   outlier.

   :param data:
   :type data: pandas dataframe
   :param outlier_factor_threshold: Float influencing classification of ouliers. Increasing this threshold
                                    means that fewer points are considered outliers.
   :type outlier_factor_threshold: float

   :returns: **outlier_bool** -- List indicating which rows of `data` are outliers.
   :rtype: list of bool

   .. rubric:: Notes

   Most inliers will have an outlier factor of less than one, however there
   are no clear rules that determine when a data point is an outlier. This
   is likely to vary from dataset to dataset and, as such, we recommend
   tuning `outlier_factor_threshold` to suit.


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


