from scipy.stats import (
    entropy, kruskal, ks_2samp, mannwhitneyu, wasserstein_distance)
from scipy.stats import wilcoxon as scipy_wc
from scipy.spatial.distance import jensenshannon
import pandas as pd
import numpy as np

from ..utils import feature_density_diff


def kolmogorov_smirnov(real, synth, feature, **kwargs):
    """ Distance: Kolmogorov-Smirnov

    The Kolmogorov-Smirnov metric calculates the maximum difference between the
    cumulative distribution functions of the `feature` in the `real` and
    `synth` datasets.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feature : str
        String indicating the feature of the datasets to compare. This must be
        a continuous variable.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    statistic : float
        Kolmogorov-Smirnov statistic
    pvalue : float
        Two-tailed p-value

    See Also
    --------
    scipy.stats.ks_2samp

    Notes
    -----
    This is a wrapper function for `scipy.stats.ks_2samp`, which tests whether
    two samples are drawn from the same distribution by calculating the maximum
    difference between their cumulative distribution functions.

    If the returned statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions are the same.

    This approach is only defined if the feature is continuous. The SciPy
    documentation further suggests this method works best when one of the
    samples has a size of only a few thousand.

    Examples
    --------
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
    """
    return ks_2samp(real[feature], synth[feature], **kwargs)


def wasserstein(real, synth, feature, **kwargs):
    """ Distance: Wasserstein

    The Wasserstein distance, or Earth Mover's distance, can be thought of as
    calculating the amount of "work" required to move from the distribution of
    the synthetic data to the distribution of the real data.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feature : str
        String indicating the feature of the datasets to compare. This must be
        a continuous variable.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    See Also
    --------
    scipy.stats.wasserstein_distance

    Notes
    -----
    This is a wrapper function for `scipy.stats.wasserstein_distance`.

    Computationally, we can find the Wasserstein distance by calculating the
    area between the cumulative distribution functions for the two
    distributions.

    If :math:`s` is the `synth` distribution of the `feature`, :math:`r` is the
    `real` distribution of the `feature` and :math:`R` and :math:`S` are their
    respective cumulative distribution functions, then

    .. math::

        wasserstein(s, r) = \\int_{-\\infty}^{+\\infty} |S-R|

    The distance is zero if the distributions are identical, and increases as
    they become less alike. This method is therefore good for comparing
    multiple synthetic datasets, or features within a dataset, to see which is
    closest to the real. However, as this is not a test, there is no threshold
    distance below which we can claim the distributions are statistically the
    same.

    Examples
    --------
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

    """
    return wasserstein_distance(real[feature], synth[feature], **kwargs)


def jensen_shannon_distance(real, synth, feature, bins='auto', **kwargs):
    """ Distance: Jensen-Shannon

    The Jensen-Shannon distance describes the difference between the `real` and
    `synth` distributions of the `feature` in terms of entropy. It is the
    square root of Jensen-Shannon divergence. It measures the distance between
    probabilities so the data are first discretised into bins.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feature : str
        String indicating the feature of the datasets to compare. This must be
        a continuous variable.
    bins : int or str
        Number of bins to use when discretising data, if `None` data will be
        treated as categorical. If string chosen method will be used to
        calculate optimal bin width, 'auto' is the default. See
        numpy.histogram_bin_edges for further options.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    distance : double
        The computed distance between the distributions.

    See Also
    --------
    divergence.jensen_shannon_divergence
    scipy.spatial.distance.jensenshannon

    Notes
    -----
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

    Examples
    --------
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

    """
    if bins is None:
        values = set(np.concatenate(
            [np.unique(real[feature]), np.unique(synth[feature])]))
        real_bincounts = [(real[feature].values == val).sum()
                          for val in values]
        synth_bincounts = [(synth[feature].values == val).sum()
                           for val in values]
    else:
        # assign bin edges based on joined data
        bin_edges = np.histogram_bin_edges(np.concatenate((real[feature],
                                                           synth[feature])),
                                           bins=bins)
        # count the number of points in each bin seperately
        real_bincounts = np.histogram(real[feature], bin_edges)[0]
        synth_bincounts = np.histogram(synth[feature], bin_edges)[0]
    return jensenshannon(real_bincounts, synth_bincounts, **kwargs)


def feature_density_diff_mae(real, synth, feats=None, bins=10):
    """ Calculate Mean Absolute Error of feature densities.

    For each feature the difference between the density across the bins
    within `real` and `synth` is calculated. Finally the MAE across all
    features and bins is calculated. A value close to 0 indicates that
    a similar distribution for `feats` is observed between the real and
    synthetic datasets.

    Parameters
    ----------
    real: pandas.DataFrame
        DataFrame containing the real data.
    synth: pandas.DataFrame
        DataFrame containing the sythetic data.
    feats: str or list of str, optional.
        The features that will be used to compute the densities. By
        default all features in `real` will be used.
    bins: str or int, optional
        Bins to use for computing the density. This value is passed
        to `numpy.histogram_bin_edges` so can be any value accepted by
        that function. The default setting of 10 uses 10 bins.

    Returns
    -------
    float:
        Mean Absolute Error of feature densities.
    """
    if isinstance(feats, pd.Index):
        feats = feats
    elif isinstance(feats, str):
        feats = [feats]
    else:
        feats = feats or real.columns
    diffs = [feature_density_diff(real, synth, f, bins)[0] for f in feats]
    return np.mean(np.abs(np.concatenate(diffs)))


def kullback_leibler(real, synth, feature, bins='auto', **kwargs):
    """ Divergence: Kullback-Leibler

    The Kullback-Leibler divergence describes how much the `real` distribution
    of the `feature` varies from the `synth` in terms of entropy. This is an
    assymmetric measure so does not describe the opposing variation. Since it
    measures the variation between probabilities, the data are first
    discretised into bins.


    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feature : str
        String indicating the feature of the datasets to compare. This must be
        a continuous variable.
    bins : int or str
        Number of bins to use when discretising data, if `None` data will be
        treated as categorical. If string chosen method will be used to
        calculate optimal bin width, 'auto' is the default. See
        numpy.histogram_bin_edges for further options.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    D : float
        The calculated divergence.

    See Also
    --------
    scipy.stats.entropy

    Notes
    -----
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

    Examples
    --------
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

    """
    if bins is None:
        values = set(np.concatenate(
            [np.unique(real[feature]), np.unique(synth[feature])]))
        real_bincounts = [(real[feature].values == val).sum()
                          for val in values]
        synth_bincounts = [(synth[feature].values == val).sum()
                           for val in values]
    else:
        # assign bin edges based on joined data
        bin_edges = np.histogram_bin_edges(np.concatenate((real[feature],
                                                           synth[feature])),
                                           bins=bins)
        # count the number of points in each bin seperately
        real_bincounts = np.histogram(real[feature], bin_edges)[0]
        synth_bincounts = np.histogram(synth[feature], bin_edges)[0]
    return entropy(real_bincounts, synth_bincounts, **kwargs)


def jensen_shannon_divergence(real, synth, feature, bins='auto', **kwargs):
    """ Divergence: Jensen-Shannon

    The Jensen-Shannon divergence describes the difference between the `real`
    and `synth` distributions of the `feature` in terms of entropy. It is the
    square of Jensen-Shannon distance.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feature : str
        String indicating the feature of the datasets to compare. This must be
        a continuous variable.
    bins : int or str
        Number of bins to use when discretising data, if `None` data will be
        treated as categorical. If string chosen method will be used to
        calculate optimal bin width, 'auto' is the default. See
        numpy.histogram_bin_edges for further options.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    divergence : double
        The computed divergence between the distributions.

    See Also
    --------
    distance.jensen_shannon_distance
    scipy.spatial.distance.jensenshannon

    Notes
    -----
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

    Examples
    --------
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

    """
    return jensen_shannon_distance(real, synth, feature, bins, **kwargs) ** 2


def mann_whitney(real, synth, feature, **kwargs):
    """ Hypothesis Test: Mann-Whitney

    The Mann-Whitney test compares the distributions of data by examining how
    well-mixed they are when pooled. This is acheived by ranking the pooled
    data. A low p-value suggests the data are not similar.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feature : str
        String indicating the feature of the datasets to compare. This must be
        a continuous variable.
    **kwargs : dict
        Keyword arguments.

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

    The null hypothesis for this test is that for randomly selected real and
    synthetic values, the probability that the real value is greater than the
    synthetic is the same as the probability that the synthetic value is
    greater than the real.

    We reject this hypothesis if the p-value is suitably small. This would in
    turn suggest that the synthetic and real data are not similarly
    distributed.

    Examples
    --------
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
    """
    return mannwhitneyu(real[feature], synth[feature],
                        alternative='two-sided', **kwargs)


def wilcoxon(real, synth, feature, **kwargs):
    """ Hypothesis Test: Wilcoxon

    The Wilcoxon test compares the distributions of paired data. It does this
    by ranking the differences.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feature : str
        String indicating the feature of the datasets to compare. This must be
        a continuous variable.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    statistic : float
        The sum of the ranks of the differences above or below zero, whichever
        is greater.
    pvalue : float
        Two-sided p-value.

    See Also
    --------
    scipy.stats.wilcoxon

    Notes
    -----
    This is a wrapper function for `scipy.stats.wilcoxon`.

    The null hypothesis for this test is that the median of the paired
    differences is zero. The alternative hypothesis is that it is different
    from zero. This would suggest that the synthetic and real data are not
    similarly distributed.

    This test only makes sense when the synthetic and real data are paired.
    That is, each synthetic datapoint is matched to a real one. In which case,
    it is required that data are ordered to reflect this.

    Examples
    --------
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
    """
    return scipy_wc(real[feature], y=synth[feature], **kwargs)


def kruskal_wallis(real, synth, feature, **kwargs):
    """ Hypothesis Test: Kruskal-Wallis

    The Kruskal-Wallis test compares the distributions of data by examining how
    well-mixed they are when pooled. This is acheived by ranking the pooled
    data. A low p-value suggests the data are not similar.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feature : str
        String indicating the feature of the datasets to compare. This must be
        a continuous variable.
    **kwargs : dict
        Keyword arguments.

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

    The null hypothesis for this test is that the medians of the distributions
    are equal. The alternative hypothesis is then that they are different. This
    would suggest that the synthetic and real data are not similarly
    distributed.

    We notice however that failure to reject the null hypothesis only suggests
    that the medians could be equal and says nothing else about how the data
    are distributed.

    Examples
    --------
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
    """
    return kruskal(real[feature], synth[feature], **kwargs)


if __name__ == '__main__':
    pass
