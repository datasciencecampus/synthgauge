from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from numpy import corrcoef
from scipy.stats import chi2_contingency
from itertools import combinations, product
import warnings


def correlation_MSD(real, synth, feats=None):
    """correlation mean-squared-difference

    This metric calculates the mean squared difference between the correlation
    matrices for the real and synthetic datasets. This gives an indication of
    how well the synthetic data has retained inter-variable relationships.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feats: str or list of str, optional
        Numeric features to use. Non-numeric features will be filtered out. By
        default all numeric features are used.

    Returns
    -------
    MSD_c : float
        Correlation mean squared difference.

    See Also
    --------
    numpy.corrcoef

    Notes
    -----
    Correlations only make sense for numerical data, so this function first
    filters the columns by datatype.

    The smaller the number this function returns, the better the synthetic
    data captures the correlation between variables in the real data. This
    method is therefore good for comparing multiple synthetic datasets to see
    which is closest to the real. However, as this is not a test, there is no
    threshold value below which we can claim the datasets are statistically
    the same.
    """
    if isinstance(feats, pd.Index):
        feats = feats
    elif isinstance(feats, str):
        feats = [feats]
    else:
        feats = feats or real.columns.to_list()
    # select only numeric variables
    real_numeric = real[feats].select_dtypes(np.number)
    synth_numeric = synth[feats].select_dtypes(np.number)
    # calculate correlation matrices
    real_corr = corrcoef(real_numeric, rowvar=False)
    synth_corr = corrcoef(synth_numeric, rowvar=False)
    # return mean squared difference
    return mean_squared_error(real_corr, synth_corr)


def cramers_v(var1, var2):
    """Cramer's V

    Measures the association between two nominal categorical variables.

    Parameters
    ----------
    var1 : pandas.Series
        Series object containing the values for one of the variables to be used
        in the comparison.
    var2 : pandas.Series
        Series object containing the values for the other variable to be used
        in the comparison.

    Returns
    -------
    Cramers_V : float
        The amount of association between the two variables.

    Notes
    -----
    Wikipedia suggests that this formulation of Cramer's V tends to
    overestimate the strength of an association and poses a corrected version.
    However, since we are only concerned with how associations compare and not
    what the actual values are, we continue to use this simpler version.
    """
    confusion_matrix = np.array(pd.crosstab(var1, var2))
    # Chi-squared test statistic, sample size, and minimum of rows and columns
    X2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = np.sum(confusion_matrix)
    minDim = min(confusion_matrix.shape)-1
    # Calculate Cramer's V
    return(np.sqrt((X2/n) / minDim))


def cramers_v_MSE(real, synth, feats=None):
    """Cramer's V Mean Squared Error

    This metric calculates the difference in association between categorical
    features in the real and synthetic datasets.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feats: str or list of str, optional
        Feature(s) in `real` and `synth` to include in comparison. By default
        all object and categorical columns are selected.

    Returns
    -------
    cramers_v_MSE : float
        Mean squared error between `real` and `synth` in Cramer's V scores
        across feature pairs.

    Notes
    -----
    This metric is only valid for categorical features so a warning is returned
    if any of the selected features appear to be numeric. If no features are
    selected, only the `category` and `object` types are used.
    """
    # check features are categorical if supplied
    if isinstance(feats, str):
        feats = [feats]
    if feats is None:
        feats = real.select_dtypes(include=['object', 'category']).columns
    else:
        non_cat = real[feats].select_dtypes(
            exclude=['object', 'category']).columns
        if len(non_cat) > 0:
            warnings.warn('Selected features include numeric types:'
                          f'{non_cat} If these should not be included, rerun '
                          'specifying different features. Otherwise, they will'
                          ' be assumed to be encoded categories.')
    # find all possible feature combinations
    feat_combinations = list(combinations(feats, 2))
    real_cramers_v = []
    synth_cramers_v = []
    # loop through feature combinations
    for feat1, feat2 in feat_combinations:
        real_cramers_v.append(cramers_v(real[feat1], real[feat2]))
        synth_cramers_v.append(cramers_v(synth[feat1], synth[feat2]))
    return mean_squared_error(real_cramers_v, synth_cramers_v)


def correlation_ratio(categorical, continuous):
    """Correlation ratio

    Calculates the correlation ratio for categorical-continuous association.
    Describes the possibility of deducing the corresponding category for a
    given continuous value.

    Returns a value in the range [0,1] where 0 means a category can not be
    determined given a continuous measurement and 1 means it can with absolute
    certainty.

    Parameters
    ----------
    categorical : Pandas Series
        A sequence of categorical measurements
    continuous : Pandas Series
        A sequence of continuous measurements

    Returns
    -------
    float in the range [0,1]

    Notes
    -----
    See https://en.wikipedia.org/wiki/Correlation_ratio for more details.
    """
    categories = pd.unique(categorical)
    category_means = np.zeros(len(categories))
    category_counts = np.zeros(len(categories))
    for i, cat in enumerate(categories):
        cts_in_cat = continuous[categorical == cat]
        category_means[i] = np.mean(cts_in_cat)
        category_counts[i] = len(cts_in_cat)
    total_mean = np.mean(continuous)
    numerator = np.sum(category_counts*((category_means-total_mean)**2))
    denominator = np.sum((continuous-total_mean)**2)
    return np.sqrt(numerator/denominator)


def correlation_ratio_MSE(real, synth, categorical_feats='auto',
                          numerical_feats=None):
    """Correlation Ratio Mean Squared Error

    This metric calculates the difference in association between categorical
    and continuous feature pairings in the real and synthetic datasets.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    categorical_feats: str or list of str, optional
        Categorical feature(s) in `real` and `synth` to include in comparison.
        By default all object and categorical columns are selected.
    numerical_feats: str or list of str, optional
        Numerical feature(s) in `real` and `synth` to include in comparison.
        By default all columns not in `categorical_feats` are selected.

    Returns
    -------
    corr_ratio_MSE : float
        Mean squared error between `real` and `synth` in correlation ratio
        scores across valid feature pairs.

    Notes
    -----
    If no categorical features are selected, columns of type `category` or
    `object` are used. If no numerical features are selected, all columns that
    are not listed as categorical features are used.
    """
    if categorical_feats == 'auto':
        categorical_feats = real.select_dtypes(
            include=['object', 'category']).columns
    if numerical_feats is None:
        numerical_feats = list(set(real.columns)-set(categorical_feats))
    real_corr_ratio = []
    synth_corr_ratio = []
    for cat_feat, num_feat in list(product(categorical_feats,
                                           numerical_feats)):
        real_corr_ratio.append(correlation_ratio(
            real[cat_feat], real[num_feat]))
        synth_corr_ratio.append(correlation_ratio(
            synth[cat_feat], synth[num_feat]))
    return mean_squared_error(real_corr_ratio, synth_corr_ratio)


if __name__ == '__main__':
    pass
