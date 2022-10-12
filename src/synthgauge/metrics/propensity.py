"""Propensity-based utility metrics."""

import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.special
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

from ..utils import df_combine


def _combine_encode_and_pop(real, synth):
    """Get the combined, encoded real and synthetic data, and their
    origins.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.

    Returns
    -------
    combined : pandas.DataFrame
        The combined data with categorical columns one-hot encoded.
    indicator : numpy.ndarray
        An indicator for whether the data is real (0) or synthetic (1).
    """

    combined = df_combine(real, synth, source_val_real=0, source_val_synth=1)
    combined = pd.get_dummies(combined, drop_first=True)
    indicator = combined.pop("source").values

    return combined, indicator


def _get_propensity_scores(data, labels, method, **kwargs):
    """Fit a propensity model to the data and extract its scores.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe to fit propensity model to.
    labels : numpy.ndarray
        Indicator for which data are real (0) or synthetic (1).
    method : {"cart", "logr"}
        Which propensity model to use.
    **kwargs : dict, optional
        Keyword arguments passed to propensity model.

    Returns
    -------
    scores : numpy.ndarray
        Propensity score for each point in `data`.
    """

    if method == "logr":
        model = LogisticRegression
        data = PolynomialFeatures(
            2, interaction_only=True, include_bias=False
        ).fit_transform(data)

    if method == "cart":
        model = DecisionTreeClassifier

    scores = model(**kwargs).fit(data, labels).predict_proba(data)[:, 1]

    return scores


def pmse(combined, indicator, method, **kwargs):
    """Calculate the propensity score mean-squared error (pMSE).

    Parameters
    ----------
    combined : pandas.DataFrame
        The combined set of real and synthetic data.
    indicator : numpy.ndarray
        An indicator for which data are real (0) or synthetic (1).
    method : {"cart", "logr"}
        Which propensity model to use. Must be either CART (`"cart"`) or
        logistic regression with first-order interactions (`"logr"`).
    **kwargs : dict, optional
        Keyword arguments passed to propensity model.

    Returns
    -------
    float
        Propensity score mean-squared error.

    See Also
    --------
    sklearn.linear_model.LogisticRegression
    sklearn.tree.DecisionTreeClassifier

    Notes
    -----
    Propensity scores represent probabilities of group membership. By
    modelling whether an example is synthetic or not, we can use
    propensity scores as a measure of utility.

    This returns zero if the distributions are identical, and is bounded
    above by :math:`1 - c` if they are nothing alike, where :math:`c` is
    the proportion of the data that is synthetic. This method is
    therefore good for comparing multiple synthetic datasets. However,
    as this is not a test, there is no threshold distance below which we
    can claim the distributions are statistically the same.

    This function assumes that some preprocessing has been carried out
    so that the data is ready to be passed to the classification
    function. Encoding of categorical data is performed, but, for
    example, scaling is not. Without this, erroneous results may be
    returned. The logistic regression can fail to converge if many
    variables are considered. Anecdotally, this doesn't seem to
    drastically impact the propensity scores, although this should be
    investigated formally.

    Using a CART model as a classifier is recommended in the literature
    however we also support the use of logistic regression. For further
    details, see: https://doi.org/10.1111/rssa.12358
    """

    scores = _get_propensity_scores(combined, indicator, method, **kwargs)
    ideal = indicator.mean()

    return np.mean(np.square(scores - ideal))


def _pmse_logr_statistics(combined, indicator):
    """Calculate the location and scale of pMSE in the null case where
    the real and synthetic datasets are formed from identical processes.

    Parameters
    ----------

    combined : pandas.DataFrame
        Dataframe containing the combined real and synthetic data.
    indicator : numpy.ndarray
        Indicator for whether data are real (0) or synthetic (1).

    Returns
    -------
    loc : float
        Expectation of pMSE in the null case.
    scale : float
        Standard deviation of pMSE in the null case.

    Notes
    -----
    It has been shown that the null case is distributed as a multiple of
    a :math:`\\chi^2` distribution with :math:`k-1` degrees of freedom.

    Therefore, its expectation is:

    .. math::

        E(pMSE) = \\frac{(k - 1)(1 - c)^2}{N}

    and its standard deviation is:

    .. math::

        sd(pMSE) = \\frac{c \\sqrt{2(k - 1)} (1 - c)^2}{N}

    where :math:`k` is the number of predictors used in the model,
    :math:`c` is the proportion of synthetic data, and :math:`N` is the
    total number of data points.

    Here, all features and first-order interactions are used in the
    model. Let :math:`m` be the number of features, then:

    .. math::

        k = m + \\binom{m}{2}

    Further explanation and derivation of these results can be found at:
    https://doi.org/10.1111/rssa.12358
    """

    num_rows, num_cols = combined.shape
    num_predictors = num_cols + scipy.special.binom(num_cols, 2)
    prop_synth = indicator.mean()

    loc = (num_predictors - 1) * (1 - prop_synth) ** 2 / num_rows
    scale = (
        prop_synth
        * np.sqrt(2 * (num_predictors - 1))
        * (1 - prop_synth) ** 2
        / num_rows
    )

    return loc, scale


def _pmse_cart_stats_boot(combined, indicator, trials, **kwargs):
    """Null statistic estimation for CART propensity via bootstrapping.

    Estimate the location and scale of pMSE in the null case by taking
    several bootstrapped samples of the real data, each being twice the
    length of the original data. Half of the data are listed as real
    while the other half are synthetic. These labelled data are then run
    through the usual CART pMSE calculation procedure to produce a set.

    The set of pMSE values are then summarised to give the estimated
    mean and standard deviation.

    Parameters
    ----------
    combined : pandas.DataFrame
        Dataframe containing the stacked real and synthetic data.
    trials : int
        Number of samples to take in the estimate.
    **kwargs : dict, optional
        Keyword arguments passed to
        `sklearn.tree.DecisionTreeClassifier`.

    Returns
    -------
    loc : float
        Estimated expectation of pMSE in the null case.
    scale : float
        Estimated standard deviation of pMSE in the null case.

    Notes
    -----
    This estimation method removes any dependency on the synthetic data
    being assessed, improving on the permutation-based method set out in
    https://doi.org/10.1111/rssa.12358.

    A description of this method and its justification are presented in
    https://doi.org/10.29012/jpc.748. This implementation is adapted
    from the source code submitted with the aforementioned paper:
    https://github.com/ClaireMcKayBowen/Code-for-NIST-PSCR-Differential-Privacy-Synthetic-Data-Challenge

    Note that the `random_state` keyword argument is used to
    (independently) bootstrap samples and to fit the CART model. Without
    specifying this, the results will not be reproducible.
    """

    rng = np.random.default_rng(kwargs.get("random_state", None))
    original = combined[indicator == 0]
    rows = len(original)

    pmses = []
    for perm in range(trials):
        sampled = original.iloc[rng.integers(rows, size=2 * rows), :]
        indicator = np.repeat([0, 1], rows)
        pmses.append(pmse(sampled, indicator, method="cart", **kwargs))

    return np.mean(pmses), np.std(pmses)


def _pmse_cart_stats_perm(combined, indicator, num_perms, **kwargs):
    """Null statistic estimation for CART propensity via permutations.

    Estimate the location and scale of pMSE in the null case by
    repeating pMSE calculations on permuations of the indicator column
    using a CART model. The set of calculations are then summarised
    using the mean or standard deviation, respectively.

    Parameters
    ----------
    combined : pandas.DataFrame
        Dataframe containing the combined real and synthetic data.
    indicator : numpy.ndarray
        Indicator for whether data are real (0) or synthetic (1).
    num_perms : int
        The number of permutations to consider.
    **kwargs : dict, optional
        Keyword arguments passed to
        `sklearn.tree.DecisionTreeClassifer`.

    Returns
    -------
    loc : float
        Estimated expectation of pMSE in the null case.
    scale : float
        Estimated standard deviation of pMSE in the null case.

    Notes
    -----
    When using a CART propensity model, the number of predictors is
    unknown and the results used in `_pmse_logr_statistics` do not
    apply. To circumvent this, taking several permutations should
    approximate the results for "properly" synthesised data without
    knowing :math:`k` a priori. Further details of this approach are
    available in https://doi.org/10.1111/rssa.12358.

    Note that the `random_state` keyword argument is used to
    (independently) create the permutations and to fit the CART model.
    Without specifying this, the results will not be reproducible.
    """

    rng = np.random.default_rng(kwargs.get("random_state", None))

    pmses = []
    for _ in range(num_perms):
        rng.shuffle(indicator)
        pmses.append(pmse(combined, indicator, method="cart", **kwargs))

    return np.mean(pmses), np.std(pmses)


def _pmse_cart_statistics(combined, indicator, num_perms, estimator, **kwargs):
    """Null statistic estimation for CART-based propensity scores.

    Estimation can either be carried out using permutations of the
    combined indicator or via bootstrapping. The former is the original
    method presented in https://doi.org/10.1111/rssa.12358, while the
    latter (from https://doi.org/10.29012/jpc.748) improves on this by
    removing dependence on the synthetic data.

    Parameters
    ----------
    combined : pandas.DataFrame
        Dataframe containing the real and synthetic data.
    indicator : numpy.ndarray
        Indicator for whether data are real (0) or synthetic (1).
    num_perms : int
        Number of trials to consider in the estimate.
    estimator : {"perm", "boot"}
        Which estimation process to use. By default, permutations are
        used to ensure back-compatibility.
    **kwargs : dict, optional
        Keyword arguments to be passed to
        `sklearn.tree.DecisionTreeClassifier`.

    Returns
    -------
    loc : float
        Estimated expectation of pMSE in the null case.
    scale : float
        Estimated standard deviation of pMSE in the null case.

    Warns
    -----
    FutureWarning
        The permutation method is flawed and will be removed in a
        future minor version. See https://doi.org/10.29012/jpc.748 for
        details.

    Raises
    ------
    ValueError
        If the estimation method, `estimator`, is not valid.
    """

    if estimator == "perm":
        message = (
            "The permutation method is flawed and will be removed in a future "
            "release. Consider using `estimator='boot'` instead."
        )
        warnings.warn(message, FutureWarning)

        loc, scale = _pmse_cart_stats_perm(
            combined, indicator, num_perms, **kwargs
        )

    elif estimator == "boot":
        loc, scale = _pmse_cart_stats_boot(
            combined, indicator, num_perms, **kwargs
        )

    else:
        raise ValueError("Estimation method must be `perm` or `boot`.")

    return loc, scale


def pmse_ratio(
    combined, indicator, method, num_perms=None, estimator="perm", **kwargs
):
    """The propensity score mean-squared error ratio.

    This is the ratio of observed pMSE to that expected under the null
    case, i.e.

    .. math::

        ratio(pMSE) = \\frac{pMSE}{E(pMSE)}

    Parameters
    ----------
    combined : pandas.DataFrame
        Dataframe containing the combined real and synthetic data.
    indicator : numpy.ndarray
        Indicator for whether data are real (0) or synthetic (1).
    method : {"cart", "logr"}
        Which propensity model to use. Must be either CART (`"cart"`) or
        logistic regression with first-order interactions (`"logr"`).
    num_perms : int, optional
        Number of permutations to consider when estimating the null case
        statistics with a CART model.
    estimator : {"perm", "boot"}
        Which estimation process to use with a CART model. By default,
        permutations are used to ensure back-compatibility.
    **kwargs : dict, optional
        Keyword arguments passed to the propensity model classifier.

    Returns
    -------
    float
        The observed-to-null pMSE ratio.

    Notes
    -----
    The interpretation of this metric makes more sense for synthetic
    data. The pMSE alone gives better utility as the value gets closer
    to zero, which is only attainable when the datasets are identical.
    However, when generating synthetic data, we do not want to produce
    identical entries. Rather, we want to achieve similarity between the
    distributions of the real and synthetic datasets.

    This ratio tends towards one when this is achieved, and increases
    otherwise.

    Note that the `random_state` keyword argument is used to
    (independently) create the permutations and to fit the model when
    using a CART model. Without specifying this, the results will not be
    reproducible.
    """

    observed = pmse(combined, indicator, method, **kwargs)

    if method == "logr":
        loc, _ = _pmse_logr_statistics(combined, indicator)
    if method == "cart":
        loc, _ = _pmse_cart_statistics(
            combined, indicator, num_perms, estimator, **kwargs
        )

    return observed / loc


def pmse_standardised(
    combined, indicator, method, num_perms=None, estimator="perm", **kwargs
):
    """The standardised propensity score mean-squared error.

    This takes the observed pMSE and standardises it against the null
    case, i.e.

    .. math::

        stand(pMSE) = (pMSE - E(pMSE)) / sd(pMSE)

    Parameters
    ----------
    combined : pandas.DataFrame
        Dataframe containing the combined real and synthetic data.
    indicator : numpy.ndarray
        Indicator for whether data are real (0) or synthetic (1).
    method : {"cart", "logr"}
        Which propensity model to use. Must be either CART (`"cart"`) or
        logistic regression with first-order interactions (`"logr"`).
    num_perms : int, optional
        Number of permutations to consider when estimating the null case
        statistics with a CART model.
    estimator : {"perm", "boot"}
        Which estimation process to use with a CART model. By default,
        permutations are used to ensure back-compatibility.
    **kwargs : dict, optional
        Keyword arguments passed to the propensity model.

    Returns
    -------
    float
        The null-standardised pMSE.

    Notes
    -----
    The interpretation of this metric makes more sense for synthetic
    data. The pMSE alone indicates better utility as it gets closer to
    zero, which is only attainable when the datasets are identical.
    However, when generating synthetic data, we do not want to produce
    identical entries. Rather, we want to achieve similarity between the
    distributions of the real and synthetic datasets.

    This standardised value tends towards zero when this is achieved,
    and increases in magnitude otherwise.

    Note that the `random_state` keyword argument is used to
    (independently) create the permutations and to fit the model when
    using a CART model. Without specifying this, the results will not be
    reproducible.
    """

    observed = pmse(combined, indicator, method, **kwargs)

    if method == "logr":
        loc, scale = _pmse_logr_statistics(combined, indicator)
    if method == "cart":
        loc, scale = _pmse_cart_statistics(
            combined, indicator, num_perms, estimator, **kwargs
        )

    return (observed - loc) / scale


def propensity_metrics(
    real,
    synth,
    method="cart",
    feats=None,
    num_perms=20,
    estimator="perm",
    **kwargs,
):
    """Propensity score-based metrics.

    This function calculates three metrics based on the propensity score
    mean-squared error (pMSE), all of which quantify utility by
    measuring the distinguishability of the synthetic data. That is, how
    readily real and synthetic data can be identified.

    To do this, the datasets are combined and their origins tracked by a
    boolean indicator. This combined dataset is then used to fit a
    binary classification model (CART or logistic regression with
    first-order interactions) with the indicator as the target. The
    propensity score for each row is then extracted and summarised to
    give a metric.

    The returned metrics are the observed pMSE along with the pMSE ratio
    and standardised pMSE. These second two metrics are given relative
    to the null case where the real and synthetic data are produced from
    identical processes.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    method : {"cart", "logr"}, default "cart"
        Which propensity model to use. Must be either CART (`"cart"`) or
        logistic regression with first-order interactions (`"logr"`).
    feats : list of str or None, default None
        List of features in the dataset to be used in the propensity
        model. If `None` (default), all common features are used.
    num_perms : int, default 20
        Number of permutations to consider when estimating the null case
        statistics with a CART model.
    estimator : {"perm", "boot"}
        Which estimation process to use with a CART model. By default,
        permutations are used to ensure back-compatibility.
    **kwargs : dict, optional
        Keyword arguments passed to the propensity model.

    Returns
    -------
    observed : float
        The observed pMSE.
    standard : float
        The null-standardised pMSE.
    ratio : float
        The observed-null pMSE ratio.

    Raises
    ------
    ValueError
        If `method` is not one of `'cart'` or `'logr'`.

    See Also
    --------
    sklearn.linear_model.LogisticRegression
    sklearn.tree.DecisionTreeClassifier
    synthgauge.metrics.propensity.pmse
    synthgauge.metrics.propensity.pmse_ratio
    synthgauge.metrics.propensity.pmse_standardised

    Notes
    -----
    For the CART model, `sklearn.tree.DecisionTreeClassifier` is used.
    Meanwhile, the logistic regression model uses
    `sklearn.linear_model.LogisticRegression`.

    Note that the `random_state` keyword argument is used to
    (independently) create the permutations and to fit the model when
    using a CART model. Without specifying this, the results will not be
    reproducible.

    Details on these metrics can be found at:
    https://doi.org/10.1111/rssa.12358
    """

    if method not in ("cart", "logr"):
        raise ValueError(
            f"Propensity method must be 'cart' or 'logr' not {method}."
        )

    feats = feats or real.columns.intersection(synth.columns)
    combined, indicator = _combine_encode_and_pop(real[feats], synth[feats])

    if method == "logr":
        loc, scale = _pmse_logr_statistics(combined, indicator)
    if method == "cart":
        loc, scale = _pmse_cart_statistics(
            combined, indicator, num_perms, estimator, **kwargs
        )

    observed = pmse(combined, indicator, method, **kwargs)
    standard = (observed - loc) / scale
    ratio = observed / loc

    PropensityResult = namedtuple(
        "PropensityResult",
        ("pmse", "pmse_standardised", "pmse_ratio"),
    )

    return PropensityResult(observed, standard, ratio)


def specks(real, synth, classifier, **kwargs):
    """Propensity score comparison via the Kolmogorov-Smirnov distance.

    The SPECKS metric was originally presented in
    https://arxiv.org/pdf/1803.06763.pdf and works as follows:

        1. Stack the real and synthetic data, and create a variable
           indicating whether each record is real (0) or synthetic (1).
        2. Calculate the propensity score for each record using a binary
           classifier on the indicator variable.
        3. Compute the Kolmogorov-Smirnov distance between the empirical
           CDFs for the real and synthetic propensity scores.

    The Kolmogorov-Smirnov distance is defined as the maximum difference
    between two empirical distributions. Therefore, it is bounded
    between zero and one. If the synthetic data properly resembles the
    original data then they will be indistinguishable, leading to close
    empirical CDFs.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    classifier : scikit-learn estimator
        Any `scikit-learn`-style classifier class with a `predict_proba`
        method.
    **kwargs : dict, optional
        Keyword arguments to be passed to `classifer`.

    Returns
    -------
    float
        The Kolmogorov-Smirnov distance between the real and synthetic
        propensity score CDFs.

    Notes
    -----
    The combined dataset is one-hot-encoded before being passed to the
    classifier so categorical features can be handled.

    The paper introducing SPECKS has also been published in METRON:
    https://doi.org/10.1007/s40300-021-00201-0.
    """

    combined, indicator = _combine_encode_and_pop(real, synth)
    scores = (
        classifier(**kwargs)
        .fit(combined, indicator)
        .predict_proba(combined)[:, 1]
    )

    return ks_2samp(scores[indicator == 0], scores[indicator == 1]).statistic
