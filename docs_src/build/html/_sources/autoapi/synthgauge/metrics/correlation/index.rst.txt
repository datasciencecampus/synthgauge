:py:mod:`synthgauge.metrics.correlation`
========================================

.. py:module:: synthgauge.metrics.correlation


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.metrics.correlation.correlation_MSD
   synthgauge.metrics.correlation.cramers_v
   synthgauge.metrics.correlation.cramers_v_MSE
   synthgauge.metrics.correlation.correlation_ratio
   synthgauge.metrics.correlation.correlation_ratio_MSE



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


.. py:function:: cramers_v(var1, var2)

   Cramer's V

   Measures the association between two nominal categorical variables.

   :param var1: Series object containing the values for one of the variables to be used
                in the comparison.
   :type var1: pandas.Series
   :param var2: Series object containing the values for the other variable to be used
                in the comparison.
   :type var2: pandas.Series

   :returns: **Cramers_V** -- The amount of association between the two variables.
   :rtype: float

   .. rubric:: Notes

   Wikipedia suggests that this formulation of Cramer's V tends to
   overestimate the strength of an association and poses a corrected version.
   However, since we are only concerned with how associations compare and not
   what the actual values are, we continue to use this simpler version.


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


.. py:function:: correlation_ratio(categorical, continuous)

   Correlation ratio

   Calculates the correlation ratio for categorical-continuous association.
   Describes the possibility of deducing the corresponding category for a
   given continuous value.

   Returns a value in the range [0,1] where 0 means a category can not be
   determined given a continuous measurement and 1 means it can with absolute
   certainty.

   :param categorical: A sequence of categorical measurements
   :type categorical: Pandas Series
   :param continuous: A sequence of continuous measurements
   :type continuous: Pandas Series

   :returns:
   :rtype: float in the range [0,1]

   .. rubric:: Notes

   See https://en.wikipedia.org/wiki/Correlation_ratio for more details.


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


