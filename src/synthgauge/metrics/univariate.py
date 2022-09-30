"""Univariate utility metrics."""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


def _get_bin_counts(real, synth, feature, bins):
    """Discretise real and synthetic features, and return the bin counts
    for each. Used by the Kullback-Leibler and Jensen-Shannon functions.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data. respectively.
    feature : str
        Feature to be processed into bin counts. Must be continuous.
    bins : int or str or None
        The binning method to use. If `int`, is the number of bins. If
        `str`, must be a method accepted by `numpy.histogram_bin_edges`.
        If `None`, the feature is assumed to be categorical and counts
        are taken for every value that appears in either dataset.

    Returns
    -------
    real_counts, synth_counts : np.ndarray
        The bin counts for the real and synthetic data, respectively.
    """

    real, synth = real[feature].values, synth[feature].values
    combined = np.concatenate((real, synth))

    if bins is None:
        values = np.unique(combined)
        real_counts = np.array([sum(real == val) for val in values])
        synth_counts = np.array([sum(synth == val) for val in values])
    else:
        edges = np.histogram_bin_edges(combined, bins=bins)
        real_counts, _ = np.histogram(real, edges)
        synth_counts, _ = np.histogram(synth, edges)

    return real_counts, synth_counts


def kullback_leibler(real, synth, feature, bins="auto", **kwargs):
    """Kullback-Leibler divergence.

    Describes how much the synthetic feature distribution varies from
    the real distribution in terms of relative entropy. The divergence
    is assymmetric and does not satisfy the triangle inequality. Thus,
    it does not describe "distance" in the mathematical sense.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature of the datasets to compare. This must be continuous.
    bins : int or str or None, default "auto"
        The binning method to use. If `int`, is the number of bins. If
        `str`, must be a method accepted by `numpy.histogram_bin_edges`.
        If `None`, the feature is assumed to be categorical and counts
        are taken for each value in either dataset.
    **kwargs : dict, optional
        Keyword arguments for `scipy.stats.entropy`.

    Returns
    -------
    float
        The computed divergence between the distributions.

    See Also
    --------
    scipy.stats.entropy

    Notes
    -----
    This is a wrapper function for `scipy.stats.entropy`. Since this
    function expects a probability vector, the data is first discretised
    into evenly-spaced bins.

    We can think of the Kullback-Leibler divergence as a measure of
    surprise we might expect seeing an example from the real data,
    relative to the distribution of the synthetic.

    The divergence is zero if the distributions are identical, and
    larger values indicate that the two discretised distributions are
    further from one another. Note that the KL divergence may be
    infinite - the most common case where this happens is when the
    synthetic data does not contain any data in at least one bin.

    An optimal 'bins' value has not been suggested.

    Examples
    --------
    >>> import pandas as pd
    >>> real = pd.DataFrame(
    ...     {"a": [0, 1, 2, 3], "b": ["cat", "cow", "dog", "emu"]}
    ... )
    >>> synth = pd.DataFrame(
    ...     {"a": [3, 1, 2, 0], "b": ["cat", "cat", "cow", "dog"]}
    ... )

    The first feature is replicated up to a re-ordering in the synthetic
    data, so its KL divergence is zero:

    >>> kullback_leibler(real, synth, "a")
    0.0

    However, the second feature does not include the ``"emu"`` category
    in the synthetic data, so the divergence is infinite:

    >>> kullback_leibler(real, synth, "b", bins=None)
    inf

    """

    real_counts, synth_counts = _get_bin_counts(real, synth, feature, bins)

    return stats.entropy(real_counts, synth_counts, **kwargs)


def jensen_shannon_divergence(real, synth, feature, bins="auto", **kwargs):
    """Jensen-Shannon divergence.

    Also known as the information radius, the Jensen-Shannon divergence
    describes the similarity between two probability distributions in
    terms of entropy. This divergence modifies the Kullback-Leibler
    divergence to be symmetric and finite (between 0 and 1).

    The divergence does not satisfy the triangle inequality. Thus, it
    does not describe "distance" in the mathematical sense. Taking its
    square root provides a metric known as the Jensen-Shannon distance.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature of the datasets to compare. This must be continuous.
    bins : int or str or None, default "auto"
        The binning method to use. If `int`, is the number of bins. If
        `str`, must be a method accepted by `numpy.histogram_bin_edges`.
        If `None`, the feature is assumed to be categorical and counts
        are taken for each value in either dataset.
    **kwargs : dict, optional
        Keyword arguments for `scipy.spatial.distance.jensenshannon`.

    Returns
    -------
    float
        The computed divergence between the distributions.

    See Also
    --------
    synthgauge.metrics.univariate_distance.jensen_shannon_distance
    scipy.spatial.distance.jensenshannon

    Notes
    -----
    This is a wrapper of
    `synthgauge.metrics.univariate_distance.jensen_shannon_distance`,
    which in turn wraps `scipy.spatial.distance.jensenshannon`. Since
    this function expects probability vectors, the data is first
    discretised into evenly-spaced bins.

    We can think of the Jensen-Shannon divergence as the amount of
    information, or entropy, encoded in the difference between the
    real and synthetic distributions of the feature.

    The divergence is zero if the distributions are identical, and is
    bounded above by one if they are nothing alike. This method is
    therefore good for comparing multiple synthetic datasets, or
    features within a dataset, to see which is closest to the real.
    However, as this is not a test, there is no threshold distance below
    which we can claim the distributions are statistically the same.

    An optimal 'bins' value has not been suggested.

    Examples
    --------
    >>> import pandas as pd
    >>> real = pd.DataFrame(
    ...     {"a": [0, 1, 3, 4], "b": ["cat", "cow", "dog", "emu"]}
    ... )
    >>> synth = pd.DataFrame(
    ...     {"a": [3, 3, 2, 0], "b": ["cat", "cat", "cow", "dog"]}
    ... )

    The second feature appears to be more similar than the first across
    datasets:

    >>> jensen_shannon_divergence(real, synth, "a")
    0.1732867951399863
    >>> jensen_shannon_divergence(real, synth, "b", bins=None)
    0.10788077716941784

    """

    return jensen_shannon_distance(real, synth, feature, bins, **kwargs) ** 2


def jensen_shannon_distance(real, synth, feature, bins="auto", **kwargs):
    """Jensen-Shannon distance.

    Describes the difference between two distributions in terms of
    entropy. Calculated as the square root of the Jensen-Shannon
    divergence, the Jensen-Shannon distance satisfies the mathematical
    definition of a metric.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature of the datasets to compare. This must be continuous.
    bins : int or str or None, default "auto"
        The binning method to use. If `int`, is the number of bins. If
        `str`, must be a method accepted by `numpy.histogram_bin_edges`.
        If `None`, the feature is assumed to be categorical and counts
        are taken for each value in either dataset.
    **kwargs : dict, optional
        Keyword arguments for `scipy.spatial.distance.jensenshannon`.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    See Also
    --------
    synthgauge.metrics.univariate_distance.jensen_shannon_divergence
    scipy.spatial.distance.jensenshannon

    Notes
    -----
    This is a wrapper for `scipy.spatial.distance.jensenshannon`. Since
    this function expects probability vectors, the data is first
    discretised into evenly-spaced bins.

    We can think of the Jensen-Shannon distance as the amount of
    information, or entropy, encoded in the difference between the
    `real` and `synth` distributions of the `feature`.

    The distance is zero if the distributions are identical, and is
    bounded above by one if they are nothing alike. This method is
    therefore good for comparing multiple synthetic datasets, or
    features within a dataset, to see which is closest to the real.
    However, as this is not a test, there is no threshold distance below
    which we can claim the distributions are statistically the same.

    An optimal 'bins' value has not been suggested.

    Examples
    --------
    >>> import pandas as pd
    >>> real = pd.DataFrame(
    ...     {"a": [0, 1, 3, 4], "b": ["cat", "cow", "dog", "emu"]}
    ... )
    >>> synth = pd.DataFrame(
    ...     {"a": [3, 3, 2, 0], "b": ["cat", "cat", "cow", "dog"]}
    ... )

    The second feature appears to be more similar than the first across
    datasets:

    >>> jensen_shannon_distance(real, synth, "a")
    0.41627730557884884
    >>> jensen_shannon_distance(real, synth, "b", bins=None)
    0.328452092654953

    """

    real_counts, synth_counts = _get_bin_counts(real, synth, feature, bins)

    return jensenshannon(real_counts, synth_counts, **kwargs)


def wasserstein(real, synth, feature, **kwargs):
    """The (first) Wasserstein distance.

    Also known as the "Earth Mover's" distance, this metric can be
    thought of as calculating the amount of "work" required to move from
    the distribution of the synthetic data to the distribution of the
    real data.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature of the datasets to compare. This must be continuous.
    **kwargs : dict, optional
        Keyword arguments for `scipy.stats.wasserstein_distance`.

    Returns
    -------
    float
        The computed distance between the distributions.

    See Also
    --------
    scipy.stats.wasserstein_distance

    Notes
    -----
    This is a wrapper for `scipy.stats.wasserstein_distance`.
    Computationally, we can find the Wasserstein distance by calculating
    the area between the cumulative distribution functions for the two
    distributions.

    If :math:`s` is the synthetic feature distribution, :math:`r` is the
    real feature distribution, and :math:`R` and :math:`S` are their
    respective cumulative distribution functions, then

    .. math::

        W(s, r) = \\int_{-\\infty}^{+\\infty} |S - R|

    The distance is zero if the distributions are identical and
    increases as they become less alike. This method is therefore good
    for comparing multiple synthetic datasets, or features within a
    dataset, to see which is closest to the real. However, as this is
    not a test, there is no threshold distance below which we can claim
    the distributions are statistically the same.

    Examples
    --------
    >>> import pandas as pd
    >>> real = pd.DataFrame(
    ...     {"a": [0, 1, 3, 4, 3, 4], "b": [3, 7, 2, 1, 7, 4]}
    ... )
    >>> synth = pd.DataFrame(
    ...     {"a": [3, 3, 2, 0, 2, 3], "b": [1, 5, 1, 1, 6, 3]}
    ... )

    The first feature appears to be more similar than the second across
    datasets:

    >>> wasserstein(real, synth, "a")
    0.6666666666666667
    >>> wasserstein(real, synth, "b")
    1.166666666666667

    """

    return stats.wasserstein_distance(real[feature], synth[feature], **kwargs)


def kolmogorov_smirnov(real, synth, feature, **kwargs):
    """Kolmogorov-Smirnov test.

    The Kolmogorov-Smirnov test statistic is the maximum difference
    between the cumulative distribution functions of the real and
    synthetic features.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Name of the feature to compare. This must be continuous.
    **kwargs : dict, optional
        Keyword arguments for `scipy.stats.ks_2samp`.

    Returns
    -------
    statistic, pvalue : float
        Kolmogorov-Smirnov test statistic.
    pvalue : float
        Two-tailed p-value.

    See Also
    --------
    scipy.stats.ks_2samp

    Notes
    -----
    This is a wrapper for `scipy.stats.ks_2samp`, which tests whether
    two samples are drawn from the same distribution by calculating the
    maximum difference between their cumulative distribution functions.

    If the returned statistic is small or the p-value is high, then we
    cannot reject the hypothesis that the distributions are the same.

    This approach is only defined if the feature is continuous. The
    documentation further suggests this method works best when one of
    the samples has a size of only a few thousand.

    Examples
    --------
    >>> import pandas as pd
    >>> real = pd.DataFrame(
    ...     {"a": [0, 1, 3, 3, 4, 4], "b": [1, 4, 5, 6, 7, 7]}
    ... )
    >>> synth = pd.DataFrame(
    ...     {"a": [0, 2, 2, 3, 3, 3], "b": [1, 1, 1, 2, 2, 3]}
    ... )

    The first feature appears to come from the same distribution in both
    datasets.

    >>> kolmogorov_smirnov(real, synth, "a")
    KstestResult(statistic=0.3333333333333333, pvalue=0.9307359307359307)

    The second feature appears to come from different distributions in
    the datasets.

    >>> kolmogorov_smirnov(real, synth, "b")
    KstestResult(statistic=0.8333333333333334, pvalue=0.025974025974025972)

    """

    return stats.ks_2samp(real[feature], synth[feature], **kwargs)


def kruskal_wallis(real, synth, feature, **kwargs):
    """Kruskal-Wallis H test.

    The Kruskal-Wallis test seeks to determine whether two sets of data
    originated from the same distribution. This is acheived by pooling
    and ranking the datasets. A low p-value suggests the two sets
    originate from different distributions and are not similar.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature of the datasets to compare. This must be continuous.
    **kwargs : dict, optional
        Keyword arguments for `scipy.stats.kruskal`.

    Returns
    -------
    statistic : float
        The Kruskal-Wallis H statistic.
    pvalue : float
        The p-value for the test.

    See Also
    --------
    scipy.stats.kruskal

    Notes
    -----
    This is a wrapper function for `scipy.stats.kruskal`.

    The null hypothesis for this test is that the medians of the
    distributions are equal. The alternative hypothesis is then that
    they are different. This would suggest that the synthetic and real
    data are not similarly distributed.

    We notice, however, that failure to reject the null hypothesis only
    suggests that the medians could be equal and says nothing else about
    how the data are distributed.

    Examples
    --------
    >>> import pandas as pd
    >>> real = pd.DataFrame(
    ...     {"a": [0, 1, 3, 3, 4, 4], "b": [1, 4, 5, 6, 7, 7]}
    ... )
    >>> synth = pd.DataFrame(
    ...     {"a": [0, 2, 2, 3, 3, 3], "b": [1, 1, 1, 2, 2, 3]}
    ... )

    The test for the first feature suggests that the data are similarly
    distributed according to their medians.

    >>> kruskal_wallis(real, synth, "a")
    KruskalResult(statistic=0.5646387832699667, pvalue=0.45239722100817814)

    The second feature test is much clearer that the data are drawn from
    distributions with different medians.

    >>> kruskal_wallis(real, synth, "b")
    KruskalResult(statistic=4.877737226277376, pvalue=0.02720526089960062)

    """

    return stats.kruskal(real[feature], synth[feature], **kwargs)


def mann_whitney(real, synth, feature, **kwargs):
    """Mann-Whitney U test.

    The Mann-Whitney test compares two sets of data by examining how
    well-mixed they are when pooled. This is acheived by ranking the
    pooled data. A low p-value suggests the data are not similar.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature of the datasets to compare. This must be continuous.
    **kwargs : dict, optional
        Keyword arguments for `scipy.stats.mannwhitneyu`.

    Returns
    -------
    statistic : float
        The Mann-Whitney U statistic, in particular U for `synth`.
    pvalue : float
        Two-sided p-value assuming an asymptotic normal distribution.

    See Also
    --------
    scipy.stats.mannwhitneyu

    Notes
    -----
    This is a wrapper function for `scipy.stats.mannwhitneyu`.

    The null hypothesis for this test is that for randomly selected real
    and synthetic values, the probability that the real value is greater
    than the synthetic is the same as the probability that the synthetic
    value is greater than the real.

    We reject this hypothesis if the p-value is suitably small. This
    would in turn suggest that the synthetic and real data are not
    similarly distributed.

    Examples
    --------
    >>> import pandas as pd
    >>> real = pd.DataFrame(
    ...     {"a": [0, 1, 3, 3, 4, 4], "b": [1, 4, 5, 6, 7, 7]}
    ... )
    >>> synth = pd.DataFrame(
    ...     {"a": [0, 2, 2, 3, 3, 3], "b": [1, 1, 1, 2, 2, 3]}
    ... )

    If we were to choose our p-value threshold as 0.05, we would reach
    the conclusion that the distributions of the first feature are
    similar but the distributions of the second feature are not.

    >>> mann_whitney(real, synth, "a")
    MannwhitneyuResult(statistic=22.5, pvalue=0.5041764308016705)
    >>> mann_whitney(real, synth, "b")
    MannwhitneyuResult(statistic=31.5, pvalue=0.033439907088311766)

    """

    return stats.mannwhitneyu(
        real[feature], synth[feature], alternative="two-sided", **kwargs
    )


def wilcoxon(real, synth, feature, **kwargs):
    """Wilcoxon signed-rank test.

    In this use, the Wilcoxon test compares the distributions of paired
    data. It does this by ranking the pairwise differences between the
    real and synthetic data.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature of the datasets to compare. This must be continuous.
    **kwargs : dict, optional
        Keyword arguments for `scipy.stats.wilcoxon`.

    Returns
    -------
    statistic : float
        The sum of the ranks of the differences above or below zero,
        whichever is greater.
    pvalue : float
        Two-sided p-value.

    See Also
    --------
    scipy.stats.wilcoxon

    Notes
    -----
    This is a wrapper function for `scipy.stats.wilcoxon`.

    The null hypothesis for this test is that the median of the paired
    differences is zero. The alternative hypothesis is that it is
    different from zero. This would suggest that the synthetic and real
    data are not similarly distributed.

    This test only makes sense when the synthetic and real data are
    paired. That is, each synthetic datum is matched to a real one. In
    which case, it is required that data are ordered to reflect this.

    Examples
    --------
    >>> import pandas as pd
    >>> real = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5, 8, 9]})
    >>> synth = pd.DataFrame({"a": [2, 0, 3, 7, 7, 4, 2, 1]})

    By eye, you might think these distributions are quite different from
    one another. However, the Wilcoxon test suggests that these two
    datasets were drawn from similar distributions.

    >>> wilcoxon(real, synth, "a")
    WilcoxonResult(statistic=17.0, pvalue=0.9453125)

    """

    return stats.wilcoxon(real[feature], synth[feature], **kwargs)
