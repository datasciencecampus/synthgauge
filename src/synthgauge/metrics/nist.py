"""Functions for the generic measures from the 2018 NIST competition."""

import itertools

import numpy as np
import pandas as pd


def _numeric_edges(real, synth, bins):
    """Find the bin edges for the numeric features."""

    numeric = real.select_dtypes(include="number").columns.intersection(
        synth.columns
    )
    edges = {col: np.histogram_bin_edges(real[col], bins) for col in numeric}

    return edges


def _discretise_datasets(real, synth, bins):
    """Discretise the numeric features of each dataset."""

    rout, sout = real.copy(), synth.copy()
    edges = _numeric_edges(rout, sout, bins)
    for col, edge in edges.items():
        rout.loc[:, col] = pd.cut(rout[col], edge)
        sout.loc[:, col] = pd.cut(sout[col], edge)

    return rout, sout


def _kway_marginal_score(real, synth, features):
    """Get the transformed score for a single set of features.

    Note that the datasets should have their numeric features
    discretised already.
    """

    rmarg = real.groupby(features).size() / len(real)
    smarg = synth.groupby(features).size() / len(synth)

    return 1 - sum(abs(rmarg - smarg)) / 2


def kway_marginals(real, synth, k=3, trials=100, bins=100, seed=None):
    r"""A measure based on the similarity of a set of k-way marginals.

    This measure works as follows:

        1. Discretise all numeric features (based on the orignal data).
        2. Randomly select `trials` sets of `k` features and calculate
           the corresponding marginal counts for each dataset.
        3. Calculate summed absolute deviation in the counts across all
           bins and marginal sets.
        4. Transform the summed absolute deviations, :math:`s`, to form
           a set of scores :math:`S = \left[1-s/2 | for each s\right]`.
           This transformation ensures the scores are in :math:`[0, 1]`.
           These extremes represent the worst- and best-case scenarios,
           respectively.
        5. Return the mean score.

    The NIST competition utilised a set of 100 three-way marginals.
    Details can be found at https://doi.org/10.6028/NIST.TN.2151.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    k : int, default 3
        Number of features to include in each k-way marginal. Default
        uses 3 (as done in the NIST competition).
    trials : int, default 100
        Maximum number of marginals to consider to estimate the overall
        score. If there are fewer `k`-way combinations than `trials`,
        tries all. Default uses 100 (as done in the NIST competition).
    bins : int or str, default 100
        Binning method for sampled numeric features. Can be anything
        accepted by `numpy.histogram`. Default uses 100 bins (as done in
        the NIST competition).
    seed : int or None, default None
        Random number seed. If `None`, results will not be reproducible.

    Returns
    -------
    score : float
        The mean transformed sum absolute deviation in k-way densities.
    """

    disreal, dissynth = _discretise_datasets(real, synth, bins)
    prng = np.random.default_rng(seed)

    choices = list(
        itertools.combinations(real.columns.intersection(synth.columns), r=k)
    )
    marginals = prng.choice(
        choices, size=min(trials, len(choices)), replace=False
    ).tolist()

    scores = [
        _kway_marginal_score(disreal, dissynth, marginal)
        for marginal in marginals
    ]

    return np.mean(scores)


def _make_rule(data, row, column, prng):
    """Given a column, make a rule for it."""

    values = data[column].unique()
    observed = row[column]

    if pd.api.types.is_numeric_dtype(values):
        rule = (observed, prng.uniform(0, values.max() - values.min()))
    else:
        rule = {observed}
        while True:
            new = prng.choice(values)
            if new in rule:
                break
            rule.add(new)

    return rule


def _create_test_cases(data, trials, prob, seed):
    """Create a collection of HOC test cases.

    For each test case, sample a row. Iterate over the columns,
    including them with some probability and generating them a rule for
    the test case. This rule is determined by the data type of the
    column:

      - Numeric columns use a random subrange from the whole dataset
      - Categoric columns use a random subset of the elements in the
        entire dataset

    Both of these types of rules always include the observed value in
    the row of the associated column; this means that the test will
    always be satisfied by at least one row when it comes to evaluation.
    """

    prng = np.random.default_rng(seed)

    cases = []
    for _ in range(trials):
        row = data.iloc[prng.integers(0, len(data)), :]
        case = {
            column: _make_rule(data, row, column, prng)
            for column in data.columns
            if prng.random() <= prob
        }
        cases.append(case)

    return cases


def _evaluate_test_cases(data, cases):
    """Evaluate the test cases on a dataset.

    Each test case's score is set as the proportion of the dataset for
    which all rules in the test case are satisfied. Each type of rule is
    satisfied differently:

      - Numeric rules are satisfied if the observed value lies within
        the rule's subrange
      - Categoric rules are satisfied if the observed value lies in the
        rule's subset
    """

    results = []
    for case in cases:
        result = pd.DataFrame()
        for col, rule in case.items():
            if isinstance(rule, tuple):
                result[col] = abs(data[col] - rule[0]) <= rule[1]
            else:
                result[col] = data[col].isin(rule)

        results.append(result.min(axis=1).mean())

    return results


def hoc(real, synth, trials=300, prob=0.1, seed=None):
    r"""A measure based on Higher Order Conjunctions (HOC).

    This measure compares the relative sizes of randomly selected pools
    of "similar" rows in the real and synthetic data. This measure of
    similarity is defined across a set of randomly genereated test
    cases applied to each dataset. Each test case consists of a set of
    rules.

    The :math:`i`-th test calculates the fraction of records satisfying
    its rules in the real data, :math:`f_{ri}`, and the synthetic,
    denoted :math:`f_{si}`. Their dissimilarity in test :math:`i` is
    quantified as:

    .. math::

        d_i = \ln\left(\max(f_{si}, 10^{-6})\right) - \ln(f_{ri})

    These dissimilarities are summarised as:

    .. math::

        \Delta = \sqrt{\frac{1}{N} \sum_{i=1}^{N} d_i^2}

    where :math:`N` is the number of test cases. Finally, this is
    transformed to a HOC score:

    .. math::

        HOC = \max \left(0, 1 + \frac{\Delta}{\ln(10^{-3})}\right)

    This measure is bounded between 0 and 1, indicating whether the
    datasets are nothing alike or identical based on the test cases,
    respectively. In the original text this score is multiplied by 1000
    to make it human-readable. Full details are available in
    https://doi.org/10.6028/NIST.TN.2151.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    trials : int, default 300
        Number of test cases to create. Default of 300 as in the
        competition.
    prob : float, default 0.1
        Probability of any column being included in a test case. Default
        of 0.1 as in the competition.
    seed : int or None, default None
        Random number seed. If `None`, results will not be reproducible.

    Returns
    -------
    score : float
        The overall HOC score.

    Notes
    -----
    It is possible that some test cases will be "empty", i.e. when no
    columns are selected. In this scenario, the score for that case will
    be `np.nan` rather than it being resampled.
    """

    cases = _create_test_cases(real, trials, prob, seed)
    real_scores = _evaluate_test_cases(real, cases)
    synth_scores = _evaluate_test_cases(synth, cases)

    dissims = (
        np.log(max(si, 1e-6)) - np.log(ri)
        for ri, si in zip(real_scores, synth_scores)
    )

    delta = np.sqrt(sum(d**2 for d in dissims) / trials)
    score = max(0, 1 + delta / np.log(1e-3))

    return score
