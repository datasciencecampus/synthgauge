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
    """The first generic measure based on similarity of k-way marginals.

    In essence, calculate the summed absolute difference in density
    across an array of randomly sampled k-way marginals. Transform and
    summarise these deviations to give a single score between 0 and 1.
    These extremes represent the worst and best case scenarios,
    respectively.

    The NIST competition utilised 3-way marginals only. Details can be
    found at https://doi.org/10.6028/NIST.TN.2151.

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
    observed = row[column].values[0]
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


def _create_test_cases(data, trials=300, prob=0.1, seed=None):
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

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the "original" data.
    trials : int, default 300
        Number of test cases to create. Default of 300 as in the
        competition.
    prob : float, default 0.1
        Probability of any column being included in a test case.
    seed : int or None, default None
        Random number seed. If `None`, results will not be reproducible.

    Returns
    -------
    cases : list
        List of column rules for each test case.
    """

    prng = np.random.default_rng(seed)

    cases = []
    for _ in range(trials):

        row = data.sample(1, random_state=prng)
        case = {
            column: _make_rule(data, row, column, prng)
            for column in data.columns
            if prng.random() < prob
        }
        cases.append(case)

    return cases


def _evaluate_test_cases(data, cases):
    """Evaluate the test cases on a dataset."""

    results = []
    for case in cases:
        result = pd.DataFrame()
        for col, rule in case.items():
            if isinstance(rule, tuple):
                result[col] = abs(data[col] - rule[0]) < rule[1]
            else:
                result[col] = data[col].isin(rule)

        results.append(result.min(axis=1).mean())

    return results
