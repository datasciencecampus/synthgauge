"""Correlation-based utility metrics."""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def _mean_squared_difference(x, y):
    """Calculate the mean-squared difference (error) between two numeric
    objects or two arrays."""

    return np.nanmean((np.array(x) - np.array(y)) ** 2)


def _cramers_v(var1, var2):
    """Cramer's V.

    Measures the association between two nominal categorical variables.

    Parameters
    ----------
    var1 : pandas.Series
        Measurements for the first variable.
    var2 : pandas.Series
        Measurements for the second variable.

    Returns
    -------
    float
        The association between the two variables.

    Notes
    -----
    Wikipedia suggests that this formulation of Cramer's V tends to
    overestimate the strength of an association and poses a corrected
    version. However, since we are only concerned with how associations
    compare and not what the actual values are, we continue to use this
    simpler version.
    """

    confusion = np.array(pd.crosstab(var1, var2))
    chi2, *_ = chi2_contingency(confusion, correction=False)

    return np.sqrt((chi2 / np.sum(confusion)) / (min(confusion.shape) - 1))


def _pairwise_cramers_v(data):
    """Compute pairwise Cramer's V for the columns of `data`."""

    results = []
    for x, y in itertools.product(data.columns, repeat=2):
        results.append(_cramers_v(data[x], data[y]))

    size = data.shape[1]
    results = np.array(results).reshape((size, size))

    return pd.DataFrame(results, index=data.columns, columns=data.columns)


def correlation_msd(real, synth, method="pearson", feats=None):
    """Mean-squared difference in correlation coefficients.

    This metric calculates the mean squared difference between the
    correlation matrices for the real and synthetic datasets. This gives
    an indication of how well the synthetic data has retained bivariate
    relationships.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    method : {"pearson", "spearman", "cramers_v"}, default "pearson"
    feats : list of str or None, default None
        Features to measure correlation across. If `method="cramers_v"`,
        all numeric columns will be filtered out. Likewise, for the
        other correlation methods, all non-numeric columns are removed.
        If `None` (default), all common features that satisfy the needs
        of `method` are used.

    Returns
    -------
    float
        Mean-squared difference of correlation coefficients.

    See Also
    --------
    numpy.corrcoef

    Notes
    -----
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
    """

    feats = feats or real.columns.intersection(synth.columns)

    if method == "cramers_v":
        real = real[feats].select_dtypes(exclude="number")
        synth = synth[feats].select_dtypes(exclude="number")
        rcorr, scorr = _pairwise_cramers_v(real), _pairwise_cramers_v(synth)
    else:
        real = real[feats].select_dtypes(include="number")
        synth = synth[feats].select_dtypes(include="number")
        rcorr, scorr = real.corr(method=method), synth.corr(method=method)

    idxs = np.triu(np.ones(len(rcorr)), k=1).astype(bool)
    rcorr, scorr = rcorr.where(idxs), scorr.where(idxs)

    return _mean_squared_difference(rcorr, scorr)


def _correlation_ratio(categorical, continuous):
    """Categorical-continuous correlation ratio.

    Calculates the correlation ratio for categorical-continuous
    association. Describes the possibility of deducing the corresponding
    category for a given continuous value.

    Missing values are not permitted in either series. Any rows with a
    missing value are dropped from both series before calculating the
    ratio.

    Returns a value in the range [0, 1] where 0 means a category can not
    be determined given a continuous measurement and 1 means it can with
    absolute certainty.

    Parameters
    ----------
    categorical : pandas.Series
        Categorical feature measurements.
    continuous : pandas.Series
        Continuous feature measurements.

    Returns
    -------
    float
        The categorical-continuous association ratio.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Correlation_ratio for details.
    """

    combined = pd.concat((categorical, continuous), axis=1).dropna()
    categorical, continuous = combined.values.T

    categories = np.unique(categorical)
    category_means = np.zeros(len(categories))
    category_counts = np.zeros(len(categories))

    for i, cat in enumerate(categories):
        cts_in_cat = continuous[categorical == cat]
        category_means[i] = np.mean(cts_in_cat)
        category_counts[i] = len(cts_in_cat)

    total_mean = np.mean(continuous)
    numerator = np.sum(category_counts * ((category_means - total_mean) ** 2))
    denominator = np.sum((continuous - total_mean) ** 2)

    return np.sqrt(numerator / denominator)


def correlation_ratio_msd(real, synth, categorical=None, numeric=None):
    """Correlation ratio mean-squared difference.

    This metric calculates the mean-squared difference in association
    between categorical and continuous feature pairings in the real and
    synthetic datasets.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical : list of str or None, default None
        Categorical features in `real` and `synth` to include in
        comparison. If `None` (default), uses all common object-type
        columns.
    numeric : list of str or None, default None
        Numerical features in `real` and `synth` to include in
        comparison. If `None` (default), uses all common columns not
        selected by `categorical`.

    Returns
    -------
    float
        Mean squared difference between `real` and `synth` in
        correlation ratio scores across all categorical-continuous
        feature pairs.
    """

    common = real.columns.intersection(synth.columns)
    categorical = (
        categorical
        or real[common].select_dtypes(include=("object", "category")).columns
    )
    numeric = numeric or common.difference(categorical)

    real_corr_ratio, synth_corr_ratio = [], []
    for cat_feat, num_feat in itertools.product(categorical, numeric):
        real_corr_ratio.append(
            _correlation_ratio(real[cat_feat], real[num_feat])
        )
        synth_corr_ratio.append(
            _correlation_ratio(synth[cat_feat], synth[num_feat])
        )

    return _mean_squared_difference(real_corr_ratio, synth_corr_ratio)
