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
        accepted by `numpy.histogram_bin_edges`. Default uses 100 bins
        (as done in the NIST competition).
    seed : int or None, default None
        Random number seed. If `None`, results will not be reproducible.

    Returns
    -------
    score : float
        The mean transformed summed absolute deviation in k-way densities.
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
