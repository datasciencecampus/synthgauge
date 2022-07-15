:py:mod:`synthgauge.metrics.correlation`
========================================

.. py:module:: synthgauge.metrics.correlation

.. autoapi-nested-parse::

   Correlation-based utility metrics.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.metrics.correlation._mean_squared_difference
   synthgauge.metrics.correlation._cramers_v
   synthgauge.metrics.correlation._pairwise_cramers_v
   synthgauge.metrics.correlation.correlation_msd
   synthgauge.metrics.correlation._correlation_ratio
   synthgauge.metrics.correlation.correlation_ratio_msd



.. py:function:: _mean_squared_difference(x, y)

   Calculate the mean-squared difference (error) between two numeric
   objects or two arrays.


.. py:function:: _cramers_v(var1, var2)

   Cramer's V.

   Measures the association between two nominal categorical variables.

   :param var1: Measurements for the first variable.
   :type var1: pandas.Series
   :param var2: Measurements for the second variable.
   :type var2: pandas.Series

   :returns: The association between the two variables.
   :rtype: float

   .. rubric:: Notes

   Wikipedia suggests that this formulation of Cramer's V tends to
   overestimate the strength of an association and poses a corrected
   version. However, since we are only concerned with how associations
   compare and not what the actual values are, we continue to use this
   simpler version.


.. py:function:: _pairwise_cramers_v(data)

   Compute pairwise Cramer's V for the columns of `data`.


.. py:function:: correlation_msd(real, synth, method='pearson', feats=None)

   Mean-squared difference in correlation coefficients.

   This metric calculates the mean squared difference between the
   correlation matrices for the real and synthetic datasets. This gives
   an indication of how well the synthetic data has retained bivariate
   relationships.

   :param real: Dataframe containing the real data.
   :type real: pandas.DataFrame
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas.DataFrame
   :param method:
   :type method: {"pearson", "spearman", "cramers_v"}, default "pearson"
   :param feats: Features to measure correlation across. If `method="cramers_v"`,
                 all numeric columns will be filtered out. Likewise, for the
                 other correlation methods, all non-numeric columns are removed.
                 If `None` (default), all common features that satisfy the needs
                 of `method` are used.
   :type feats: list of str or None, default None

   :returns: Mean-squared difference of correlation coefficients.
   :rtype: float

   .. seealso:: :obj:`numpy.corrcoef`

   .. rubric:: Notes

   The smaller the number this function returns, the better the
   synthetic data captures the correlation between variables in the
   real data. This method is therefore good for comparing multiple
   synthetic datasets to see which is closest to the real. However, as
   this is not a test, there is no threshold value below which we can
   claim the datasets are statistically the same.

   We only consider the coefficients above the main diagonal when
   calculating the MSD. If we included the entire matrix, we would
   double-count each pair of features as well as including the trivial
   ones along the main diagonal.


.. py:function:: _correlation_ratio(categorical, continuous)

   Categorical-continuous correlation ratio.

   Calculates the correlation ratio for categorical-continuous
   association. Describes the possibility of deducing the corresponding
   category for a given continuous value.

   Missing values are not permitted in either series. Any rows with a
   missing value are dropped from both series before calculating the
   ratio.

   Returns a value in the range [0, 1] where 0 means a category can not
   be determined given a continuous measurement and 1 means it can with
   absolute certainty.

   :param categorical: Categorical feature measurements.
   :type categorical: pandas.Series
   :param continuous: Continuous feature measurements.
   :type continuous: pandas.Series

   :returns: The categorical-continuous association ratio.
   :rtype: float

   .. rubric:: Notes

   See https://en.wikipedia.org/wiki/Correlation_ratio for details.


.. py:function:: correlation_ratio_msd(real, synth, categorical=None, numeric=None)

   Correlation ratio mean-squared difference.

   This metric calculates the mean-squared difference in association
   between categorical and continuous feature pairings in the real and
   synthetic datasets.

   :param real: Dataframe containing the real data.
   :type real: pandas.DataFrame
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas.DataFrame
   :param categorical: Categorical features in `real` and `synth` to include in
                       comparison. If `None` (default), uses all common object-type
                       columns.
   :type categorical: list of str or None, default None
   :param numeric: Numerical features in `real` and `synth` to include in
                   comparison. If `None` (default), uses all common columns not
                   selected by `categorical`.
   :type numeric: list of str or None, default None

   :returns: Mean squared difference between `real` and `synth` in
             correlation ratio scores across all categorical-continuous
             feature pairs.
   :rtype: float


