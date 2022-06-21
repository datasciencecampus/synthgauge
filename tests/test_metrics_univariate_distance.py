"""Property-based tests for univariate distance metrics."""

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy import stats

from synthgauge.metrics import univariate_distance as univariate

from .utils import datasets


@given(datasets(column_spec={"a": "float"}))
def test_kolmogorov_smirnov(datasets):
    """Check that the Kolmogorov-Smirnov test can be run correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty or real.equals(synth)))

    result = univariate.kolmogorov_smirnov(real, synth, "a")
    stat, pval = result

    assert isinstance(result, stats._stats_py.KstestResult)
    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert pval >= 0 and pval <= 1


@given(datasets(column_spec={"a": "float"}))
def test_wasserstein(datasets):
    """Check that the Wasserstein distance can be calculated."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    distance = univariate.wasserstein(real, synth, "a")

    assert isinstance(distance, float)
    assert np.isnan(distance) or distance >= 0


@given(
    datasets(
        column_spec={"a": "float"},
        allow_nan=False,
        min_value=0,
        max_value=1000,
    ),
    st.one_of(st.none(), st.integers(1, 10)),
)
def test_jensen_shannon_distance(datasets, bins):
    """Check that the Jensen-Shannon distance can be calculated."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    distance = univariate.jensen_shannon_distance(real, synth, "a", bins)

    assert isinstance(distance, float)
    assert np.isnan(distance) or (distance >= 0 and distance <= 1)


@given(
    datasets(
        column_spec={"a": "float"},
        allow_nan=False,
        min_value=0,
        max_value=1000,
    ),
    st.one_of(st.none(), st.integers(1, 10)),
)
def test_jensen_shannon_divergence(datasets, bins):
    """Check that the Jensen-Shannon divergence can be calculated."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    divergence = univariate.jensen_shannon_divergence(real, synth, "a", bins)

    assert isinstance(divergence, float)
    assert np.isnan(divergence) or (divergence >= 0 and divergence <= 1)
    assert (
        divergence
        == univariate.jensen_shannon_distance(real, synth, "a", bins) ** 2
    )


@given(
    datasets(available_dtypes=("object",), min_columns=2),
    st.one_of(st.none(), st.sampled_from(("a", ["a", "b"]))),
    st.integers(1, 10),
)
def test_feature_density_diff_mae(datasets, feats, bins):
    """Check that the feature density difference mean absolute error
    can be calculated correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    mae = univariate.feature_density_diff_mae(real, synth, feats, bins)

    assert isinstance(mae, float)
    assert mae >= 0 and mae <= bins


@given(
    datasets(
        column_spec={"a": "float"},
        min_value=0,
        max_value=1000,
        allow_nan=False,
    ),
    st.one_of(st.none(), st.integers(1, 10)),
)
def test_kullback_leibler(datasets, bins):
    """Check that the Kullback-Leibler divergence can be calculated.

    We ensure that the real dataset has at least as many unique values
    as the synthetic dataset. Failure to do this means the divergence
    is not bounded above by 1."""

    real, synth = datasets
    if synth["a"].nunique() > real["a"].nunique():
        real, synth = synth, real

    assume(not (real.empty or synth.empty))

    divergence = univariate.kullback_leibler(real, synth, "a", bins)

    assert isinstance(divergence, float)
    assert np.isinf(divergence) or (divergence >= 0 and divergence <= 1)


@given(datasets(column_spec={"a": "float"}))
def test_mann_whitney(datasets):
    """Check that the Mann-Whitney test can be run correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    result = univariate.mann_whitney(real, synth, "a")
    stat, pval = result

    assert isinstance(result, stats._mannwhitneyu.MannwhitneyuResult)
    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert (np.isnan(stat) and np.isnan(pval)) or (pval >= 0 and pval <= 1)


@given(datasets(column_spec={"a": "float"}))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_wilcoxon(datasets):
    """Check that the Wilcoxon test can be run correctly."""

    real, synth = datasets
    assume(
        len(real) == len(synth)
        and not (real.empty or synth.empty or real.equals(synth))
    )

    result = univariate.wilcoxon(real, synth, "a")
    stat, pval = result

    assert isinstance(result, stats._morestats.WilcoxonResult)
    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert (np.isnan(stat) and np.isnan(pval)) or (pval >= 0 and pval <= 1)


@given(datasets(column_spec={"a": "float"}))
def test_kruskal_wallis(datasets):
    """Check that the Kruskal-Wallis test can be run correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty or real.equals(synth)))

    result = univariate.kruskal_wallis(real, synth, "a")
    stat, pval = result

    assert isinstance(result, stats._stats_py.KruskalResult)
    assert isinstance(stat, float)
    assert isinstance(pval, float)
    assert (np.isnan(stat) and np.isnan(pval)) or (pval >= 0 and pval <= 1)
