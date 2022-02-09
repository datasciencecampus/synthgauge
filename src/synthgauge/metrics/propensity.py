import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
import random
import warnings
from collections import namedtuple
from ..utils import df_combine


def propensity_MSE(real, synth, method, **kwargs):
    """Propensity mean-squared error

    We think of the propensity of an example as the estimated probability it is
    classified as being synthetic. For a good synthetic dataset we would expect
    this to be the same as the proportion of synthetic examples used in
    training. The mean squared error from this proportion is hence a suitable
    utility metric.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    method : str, ['CART', 'LogisticRegression']
        Classification method to use.
    **kwargs : dict
        Keyword arguments passed to classification function.

    Returns
    -------
    MSE_p : float
        Propensity score mean squared error.

    See Also
    --------
    sklearn.linear_model.LogisticRegression
    sklearn.tree.DecisionTreeClassifier

    Notes
    -----
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
    """
    if method not in ['CART', 'LogisticRegression']:
        raise ValueError(
            "method must be either 'CART' or 'LogisticRegression'")
    # combine data
    combined = df_combine(real, synth, source_val_real=0, source_val_synth=1)
    # remove source column
    synth_bool = combined.pop('source')
    # encode categorical variables
    combined_encoded = pd.get_dummies(combined, drop_first=True)
    if method == 'LogisticRegression':
        # add interactions
        combined_encoded = PolynomialFeatures(2, interaction_only=True,
                                              include_bias=False) \
            .fit_transform(combined_encoded)

        model = LogisticRegression(**kwargs)
    if method == 'CART':
        model = DecisionTreeClassifier(**kwargs)
    model.fit(combined_encoded, synth_bool)

    # calculate propensity for each example
    props = model.predict_proba(combined_encoded)[:, 1]

    ideal_prop = len(synth)/len(combined_encoded)

    props_square_error = np.square(props - ideal_prop)

    MSE_p = sum(props_square_error)/len(props_square_error)

    return MSE_p


def expected_p_MSE(real, synth):
    """ Expected propensity mean-squared error

    This is the expected propensity mean-squared error under the null case
    that the `real` and `synth` datasets are distributionally similar.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.

    Returns
    -------
    MSE_p : float
        Expected propensity score mean square-squared error.

    Notes
    -----
    This expectation is used to standardise `propensity_MSE` to an
    interpretable scale.

    It has been shown that the null case is distributed as a multiple of a
    :math:`\\chi^2` distribution with :math:`k-1` degrees of freedom, which has
    expectation:

    .. math::

        (k-1)(1-c)^2c/N

    where :math:`k` is the number of predictors in the model, :math:`c` is the
    proportion of synthetic data, and :math:`N` is the total number of examples
    Further explanation and derivation of this formulation can be found here:
    https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssa.12358 Appendix A1.

    """
    # =(k−1)(1−c)^2*c/N
    num_vars = real.shape[1]
    num_vars_and_interactions = num_vars*(num_vars+1)/2
    total_num_examples = real.shape[0]+synth.shape[0]
    prop_synth = synth.shape[0]/total_num_examples
    return ((num_vars_and_interactions)*(1.0-prop_synth)
            ** 2*prop_synth/total_num_examples)


def stdev_p_MSE(real, synth):
    """ Standard deviation propensity mean-squared error

    This is the standard deviation of the propensity mean-squared error under
    the null case that the `real` and `synth` datasets are distributionally
    similar.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.

    Returns
    -------
    st_dev : float
        Expected propensity score mean square-squared error standard deviation.

    Notes
    -----
    This standard deviation is used to standardise `propensity_MSE` to an
    interpretable scale.

    It has been shown that the null case is distributed as a multiple of a
    :math:`\\chi^2` distribution with :math:`k-1` degrees of freedom, which has
    standard deviation:

    .. math::

        \\sqrt{2(k-1)}(1-c)^2c/N

    where :math:`k` is the number of predictors in the model, :math:`c` is the
    proportion of synthetic data, and :math:`N` is the total number of examples
    Further explanation and derivation of this formulation can be found here:
    https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/rssa.12358 Appendix A1.
    """
    num_vars = real.shape[1]
    num_vars_and_interactions = num_vars*(num_vars+1)/2
    total_num_examples = real.shape[0]+synth.shape[0]
    prop_synth = synth.shape[0]/total_num_examples
    return ((2*(num_vars_and_interactions))
            ** 0.5*(1.0-prop_synth)**2*prop_synth/total_num_examples)


def perm_expected_sd_p_MSE(real, synth, num_perms=20, **kwargs):
    """Permutation Expected and Standard Deviation Propensity Mean Squared Error

    Repeat pMSE calculation several times but while randomly permutating the
    boolean indicator column. This should approximate propensity mean squared
    error results for 'properly' synthesised datasets when it is not possible
    to calculate these directly, in particular when CART is used.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    num_perms : int
        The number of times to repeat the process.
    **kwargs : dict
        Keyword arguments passed to DecisionTreeClassifier function.

    Returns
    -------
    mean : float
        Mean of the propensity_MSE scores over all the repititions.
    sd : float
        Standard deviation of the propensity_MSE scores over all the
        repititions.

    Notes
    -----
    This function is only intended to be used within `propensity_metrics()`.

    This function is stochastic so will return a differnet result every time.
    """
    perm_MSEs = np.zeros(num_perms)
    for i in range(num_perms):
        combined = df_combine(
            real, synth, source_val_real=0, source_val_synth=1)
        # remove source column
        synth_bool = combined.pop('source').tolist()
        random.shuffle(synth_bool)
        # encode categorical variables
        combined_encoded = pd.get_dummies(combined, drop_first=True)
        model = DecisionTreeClassifier(**kwargs)
        model.fit(combined_encoded, synth_bool)
        # calculate propensity for each example
        props = model.predict_proba(combined_encoded)[:, 1]

        ideal_prop = len(synth)/len(combined_encoded)

        props_square_error = np.square(props - ideal_prop)

        MSE_p = sum(props_square_error)/len(props_square_error)

        perm_MSEs[i] = MSE_p
    return np.mean(perm_MSEs), np.std(perm_MSEs)


def p_MSE_ratio(real, synth, method='CART', feats=None, **kwargs):
    """ Propensity mean-squared error ratio

    This is the ratio of observed propensity mean-squared error, to that
    expected under the null case that the synthetic and real datasets are
    similarly distributed.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    method : str, ['CART', 'LogisticRegression']
        Classification method to use.
    feats : list of str, optional.
        List of features in the dataset that will be used in propensity
        calculations. By default all features will be used.
    **kwargs : dict
        Keyword arguments passed to classifier function.

    Returns
    -------
    ratio : float
        Ratio of the propensity score mean square-squared error.

    Notes
    -----
    This standardisation transforms the scale for the MSE into one that makes
    more sense for synthetic data. Before, the MSE score gave better utility
    as the value got closer to zero, which is only attainable when the datasets
    are identical. However, when generating synthetic data we do not want to
    produce identical entries, but to acheive distributional similarity between
    the distribution of the observed data and the model used to generate the
    synthetic.

    This ratio tends towards one when the datasets are similar and increases
    otherwise.
    """
    warnings.simplefilter("always", category=DeprecationWarning)
    warnings.warn(
        "p_MSE_ratio is now contained within `propensity_metrics`.",
        DeprecationWarning)
    if isinstance(feats, pd.Index):
        feats = feats
    elif isinstance(feats, str):
        feats = [feats]
    else:
        feats = feats or real.columns.to_list()
    if method == 'Logistic Regression':
        exp_p_MSE = expected_p_MSE(real[feats], synth[feats])
    if method == 'CART':
        exp_p_MSE, _ = perm_expected_sd_p_MSE(
            real[feats], synth[feats], num_perms=2, **kwargs)
    obs_p_MSE = propensity_MSE(
        real[feats], synth[feats], method=method, **kwargs)
    return obs_p_MSE/exp_p_MSE


def standardised_p_MSE(real, synth, method='CART', feats=None, **kwargs):
    """ Standardised propensity mean-squared error

    This is the difference between the observed propensity mean-squared error
    and that expected under the null case that the synthetic and real datasets
    are similarly distributed, scaled by the standard deviation.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    method : str, ['CART', 'LogisticRegression']
        Classification method to use.
    feats : list of str, optional.
        List of features in the dataset that will be used in propensity
        calculations. By default all features will be used.
    **kwargs : dict
        Keyword arguments passed to LogisticRegression function.

    Returns
    -------
    ratio : float
        Standardised propensity score mean square-squared error.

    Notes
    -----
    This standardisation transforms the scale for the MSE into one that makes
    more sense for synthetic data. Before, the MSE score gave better utility
    as the value got closer to zero, which is only attainable when the datasets
    are identical. However, when generating synthetic data we do not want to
    produce identical entries, but to acheive distributional similarity between
    the distribution of the observed data and the model used to generate the
    synthetic.

    This metric tends towards zero when the datasets are similar and increases
    otherwise.
    """
    warnings.simplefilter("always", category=DeprecationWarning)
    warnings.warn(
        "standardised_p_MSE is now contained within `propensity_metrics`.",
        DeprecationWarning)
    if isinstance(feats, pd.Index):
        feats = feats
    elif isinstance(feats, str):
        feats = [feats]
    else:
        feats = feats or real.columns.to_list()

    if method == 'LogisticRegression':
        exp_p_MSE = expected_p_MSE(real[feats], synth[feats])
        std_p_MSE = stdev_p_MSE(real[feats], synth[feats])

    if method == 'CART':
        exp_p_MSE, std_p_MSE = perm_expected_sd_p_MSE(
            real[feats], synth[feats], **kwargs)

    obs_p_MSE = propensity_MSE(
        real[feats], synth[feats], method=method, **kwargs)
    return (obs_p_MSE - exp_p_MSE)/std_p_MSE


def propensity_metrics(real, synth, method='CART', feats=None, num_perms=20,
                       **kwargs):
    """ Propensity metrics

    This function calculates three flavours of propensity mean-squared error,
    all of which quantify utility by measuring how well a classifier can be
    trained to distinguish `real` and `synth`. It returns the raw, observed
    value together with this value standardised by the expected value if the
    dataset was well synthesised, and also as a ratio of the expected value.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    method : str, ['CART', 'LogisticRegression']
        Classification method to use.
    feats : list of str, optional.
        List of features in the dataset that will be used in propensity
        calculations. By default all features will be used.
    num_perms : int
        If CART method is used, this is the number of times to repeat process
        when calculating the expected mean-squared error.
    **kwargs : dict
        Keyword arguments passed to LogisticRegression function.

    Returns
    -------
    observed_p_MSE : float
        Observed propensity score mean squared error.
    standardised_p_MSE : float
        Standardised propensity score mean square-squared error.
    ratio_p_MSE : float
        Ratio of the propensity score mean square-squared error to the
        expected value.


    Notes
    -----
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
    """
    if isinstance(feats, pd.Index):
        feats = feats
    elif isinstance(feats, str):
        feats = [feats]
    else:
        feats = feats or real.columns.to_list()

    if method == 'LogisticRegression':
        exp_p_MSE = expected_p_MSE(real[feats], synth[feats])
        std_p_MSE = stdev_p_MSE(real[feats], synth[feats])

    if method == 'CART':
        exp_p_MSE, std_p_MSE = perm_expected_sd_p_MSE(
            real[feats], synth[feats], num_perms=num_perms, **kwargs)

    obs_p_MSE = propensity_MSE(
        real[feats], synth[feats], method=method, **kwargs)
    standardised_p_MSE = (obs_p_MSE - exp_p_MSE)/std_p_MSE
    ratio_p_MSE = obs_p_MSE/exp_p_MSE

    PropensityResult = namedtuple('PropensityResult', ('observed_p_MSE',
                                                       'standardised_p_MSE',
                                                       'ratio_p_MSE'))

    return PropensityResult(obs_p_MSE, standardised_p_MSE, ratio_p_MSE)


if __name__ == '__main__':
    pass
