:py:mod:`synthgauge.metrics.propensity`
=======================================

.. py:module:: synthgauge.metrics.propensity


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.metrics.propensity.propensity_MSE
   synthgauge.metrics.propensity.expected_p_MSE
   synthgauge.metrics.propensity.stdev_p_MSE
   synthgauge.metrics.propensity.perm_expected_sd_p_MSE
   synthgauge.metrics.propensity.p_MSE_ratio
   synthgauge.metrics.propensity.standardised_p_MSE
   synthgauge.metrics.propensity.propensity_metrics



.. py:function:: propensity_MSE(real, synth, method, **kwargs)

   Propensity mean-squared error

   We think of the propensity of an example as the estimated probability it is
   classified as being synthetic. For a good synthetic dataset we would expect
   this to be the same as the proportion of synthetic examples used in
   training. The mean squared error from this proportion is hence a suitable
   utility metric.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param method: Classification method to use.
   :type method: str, ['CART', 'LogisticRegression']
   :param \*\*kwargs: Keyword arguments passed to classification function.
   :type \*\*kwargs: dict

   :returns: **MSE_p** -- Propensity score mean squared error.
   :rtype: float

   .. seealso:: :obj:`sklearn.linear_model.LogisticRegression`, :obj:`sklearn.tree.DecisionTreeClassifier`

   .. rubric:: Notes

   Propensity scores represent probabilities of group membership. By modelling
   whether an example is synthetic or not, we can use propensity scores as a
   measure of utility.

   This returns zero if the distributions are identical, and is bounded
   above by 1-c if they are nothing alike, where c is the proportion of
   synthetic data. This method is therefore good for comparing multiple
   synthetic datasets. However, as this is not a test, there is no threshold
   distance below which we can claim the distributions are statistically the
   same.

   This function assumes that some preprocessing has been carried out so that
   the data is ready to be passed to the classification function. Encoding
   of categorical data is performed, but, for example, scaling is not. Without
   this erroneous results may be returned. The logistic regression can fail to
   converge if many variables are considered. Anecdotally, this doesn't seem
   to drastically impact the propensity scores, although this should be
   investigated formally. `**kwargs` are passed to the classification model so
   it can be tuned.

   Using a CART model as a classifier is recommended in the literature however
   we also support the use of logistic regression.

   https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssa.12358


.. py:function:: expected_p_MSE(real, synth)

   Expected propensity mean-squared error

   This is the expected propensity mean-squared error under the null case
   that the `real` and `synth` datasets are distributionally similar.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe

   :returns: **MSE_p** -- Expected propensity score mean square-squared error.
   :rtype: float

   .. rubric:: Notes

   This expectation is used to standardise `propensity_MSE` to an
   interpretable scale.

   It has been shown that the null case is distributed as a multiple of a
   :math:`\chi^2` distribution with :math:`k-1` degrees of freedom, which has
   expectation:

   .. math::

       (k-1)(1-c)^2c/N

   where :math:`k` is the number of predictors in the model, :math:`c` is the
   proportion of synthetic data, and :math:`N` is the total number of examples
   Further explanation and derivation of this formulation can be found here:
   https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssa.12358 Appendix A1.


.. py:function:: stdev_p_MSE(real, synth)

   Standard deviation propensity mean-squared error

   This is the standard deviation of the propensity mean-squared error under
   the null case that the `real` and `synth` datasets are distributionally
   similar.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe

   :returns: **st_dev** -- Expected propensity score mean square-squared error standard deviation.
   :rtype: float

   .. rubric:: Notes

   This standard deviation is used to standardise `propensity_MSE` to an
   interpretable scale.

   It has been shown that the null case is distributed as a multiple of a
   :math:`\chi^2` distribution with :math:`k-1` degrees of freedom, which has
   standard deviation:

   .. math::

       \sqrt{2(k-1)}(1-c)^2c/N

   where :math:`k` is the number of predictors in the model, :math:`c` is the
   proportion of synthetic data, and :math:`N` is the total number of examples
   Further explanation and derivation of this formulation can be found here:
   https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssa.12358 Appendix A1.


.. py:function:: perm_expected_sd_p_MSE(real, synth, num_perms=20, **kwargs)

   Permutation Expected and Standard Deviation Propensity Mean Squared Error

   Repeat pMSE calculation several times but while randomly permutating the
   boolean indicator column. This should approximate propensity mean squared
   error results for 'properly' synthesised datasets when it is not possible
   to calculate these directly, in particular when CART is used.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param num_perms: The number of times to repeat the process.
   :type num_perms: int
   :param \*\*kwargs: Keyword arguments passed to DecisionTreeClassifier function.
   :type \*\*kwargs: dict

   :returns: * **mean** (*float*) -- Mean of the propensity_MSE scores over all the repititions.
             * **sd** (*float*) -- Standard deviation of the propensity_MSE scores over all the
               repititions.

   .. rubric:: Notes

   This function is only intended to be used within `propensity_metrics()`.

   This function is stochastic so will return a differnet result every time.


.. py:function:: p_MSE_ratio(real, synth, method='CART', feats=None, **kwargs)

   Propensity mean-squared error ratio

   This is the ratio of observed propensity mean-squared error, to that
   expected under the null case that the synthetic and real datasets are
   similarly distributed.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param method: Classification method to use.
   :type method: str, ['CART', 'LogisticRegression']
   :param feats: List of features in the dataset that will be used in propensity
                 calculations. By default all features will be used.
   :type feats: list of str, optional.
   :param \*\*kwargs: Keyword arguments passed to classifier function.
   :type \*\*kwargs: dict

   :returns: **ratio** -- Ratio of the propensity score mean square-squared error.
   :rtype: float

   .. rubric:: Notes

   This standardisation transforms the scale for the MSE into one that makes
   more sense for synthetic data. Before, the MSE score gave better utility
   as the value got closer to zero, which is only attainable when the datasets
   are identical. However, when generating synthetic data we do not want to
   produce identical entries, but to acheive distributional similarity between
   the distribution of the observed data and the model used to generate the
   synthetic.

   This ratio tends towards one when the datasets are similar and increases
   otherwise.


.. py:function:: standardised_p_MSE(real, synth, method='CART', feats=None, **kwargs)

   Standardised propensity mean-squared error

   This is the difference between the observed propensity mean-squared error
   and that expected under the null case that the synthetic and real datasets
   are similarly distributed, scaled by the standard deviation.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param method: Classification method to use.
   :type method: str, ['CART', 'LogisticRegression']
   :param feats: List of features in the dataset that will be used in propensity
                 calculations. By default all features will be used.
   :type feats: list of str, optional.
   :param \*\*kwargs: Keyword arguments passed to LogisticRegression function.
   :type \*\*kwargs: dict

   :returns: **ratio** -- Standardised propensity score mean square-squared error.
   :rtype: float

   .. rubric:: Notes

   This standardisation transforms the scale for the MSE into one that makes
   more sense for synthetic data. Before, the MSE score gave better utility
   as the value got closer to zero, which is only attainable when the datasets
   are identical. However, when generating synthetic data we do not want to
   produce identical entries, but to acheive distributional similarity between
   the distribution of the observed data and the model used to generate the
   synthetic.

   This metric tends towards zero when the datasets are similar and increases
   otherwise.


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


