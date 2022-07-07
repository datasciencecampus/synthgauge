"""Property-based tests for the propensity score-based metrics."""

import string

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from synthgauge.metrics import propensity

from .utils import datasets


def _reduce_synthetic(synth, seed):
    """Cut off the synthetic data at a random point."""

    rng = np.random.default_rng(seed)
    nrows = int(len(synth) * rng.uniform(0, 0.75))
    synth = synth.iloc[:nrows, :]

    return synth


@given(datasets())
def test_combined_and_pop(datasets):
    """Check that two datasets can be concatenated and their origins
    preserved."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    combined, indicator = propensity._combine_and_pop(real, synth)

    assert isinstance(combined, pd.DataFrame)
    assert isinstance(indicator, np.ndarray)

    assert set(indicator) == {0, 1}
    assert len(combined) == len(real) + len(synth)
    assert len(indicator) == len(combined)
    assert sum(indicator) == len(synth)


@given(st.sampled_from(("logr", "cart")), st.integers(0, 100))
@settings(
    deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_get_propensity_scores(real, synth, method, seed):
    """Check that a set of propensity scores can be obtained."""

    combined, indicator = propensity._combine_and_pop(real, synth)

    scores = propensity._get_propensity_scores(
        combined, indicator, method, random_state=seed
    )

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(combined),)
    assert np.logical_and(scores >= 0, scores <= 1).all()


@given(st.sampled_from(("logr", "cart")), st.integers(0, 100))
@settings(
    deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_pmse(real, synth, method, seed):
    """Check that either model can be used to obtain a propensity score
    mean-squared error. We first drop a random set of synthetic rows so
    that the bounds on pMSE can be verified."""

    synth = _reduce_synthetic(synth, seed)
    combined, indicator = propensity._combine_and_pop(real, synth)
    synth_prop = np.mean(indicator)

    pmse = propensity.pmse(combined, indicator, method, random_state=seed)

    assert isinstance(pmse, float)
    assert pmse >= 0 and pmse <= 1 - synth_prop

    if synth_prop > 0.5:
        assert pmse <= synth_prop**2
    else:
        assert pmse <= synth_prop**2 - 2 * synth_prop + 1


@given(st.integers(0, 100))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_pmse_logr_statistics(real, synth, seed):
    """Check that the theoretic null-pMSE statistics are calculated
    correctly."""

    synth = _reduce_synthetic(synth, seed)
    combined, indicator = propensity._combine_and_pop(real, synth)

    nrows, ncols = combined.shape
    npreds = ncols * (ncols + 1) / 2

    synth_prop = indicator.mean()

    loc, scale = propensity._pmse_logr_statistics(combined, indicator)

    assert isinstance(loc, float)
    assert isinstance(scale, float)

    assert scale > 0

    if synth_prop > 0.5:
        assert loc <= synth_prop**2
    else:
        assert loc <= synth_prop**2 - 2 * synth_prop + 1

    assert np.isclose(loc, (npreds - 1) * (1 - synth_prop) ** 2 / nrows)

    assert np.isclose(
        scale,
        synth_prop * np.sqrt(2 * (npreds - 1)) * (1 - synth_prop) ** 2 / nrows,
    )


@given(st.integers(0, 100))
@settings(
    deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_pmse_cart_statistics(real, synth, seed):
    """Check that the permutation-based null statistics can be estimated
    correctly."""

    synth = _reduce_synthetic(synth, seed)
    combined, indicator = propensity._combine_and_pop(real, synth)
    synth_prop = indicator.mean()

    loc, scale = propensity._pmse_cart_statistics(
        combined, indicator, num_perms=10, random_state=seed
    )

    assert isinstance(loc, float)
    assert isinstance(scale, float)

    assert scale > 0

    if synth_prop > 0.5:
        assert loc <= synth_prop**2
    else:
        assert loc <= synth_prop**2 - 2 * synth_prop + 1


@given(st.sampled_from(("logr", "cart")), st.integers(0, 100))
@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_pmse_ratio(real, synth, method, seed):
    """Check that the pMSE-ratio can be calculated correctly."""

    combined, indicator = propensity._combine_and_pop(real, synth)

    ratio = propensity.pmse_ratio(
        combined, indicator, method, num_perms=10, random_state=seed
    )

    assert isinstance(ratio, float)
    assert ratio > 0


@given(st.sampled_from(("logr", "cart")), st.integers(0, 100))
@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_pmse_standardised(real, synth, method, seed):
    """Check that the standardised pMSE can be calculated."""

    synth = _reduce_synthetic(synth, seed)
    combined, indicator = propensity._combine_and_pop(real, synth)

    standard = propensity.pmse_standardised(
        combined, indicator, method, num_perms=10, random_state=seed
    )

    assert isinstance(standard, float)


@given(st.sampled_from(("logr", "cart")), st.integers(0, 100))
@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_propensity_metrics(real, synth, method, seed):
    """Check that the propensity metric wrapper function returns its
    named tuple. Further tests for the individual metrics are above."""

    synth = _reduce_synthetic(synth, seed)
    result = propensity.propensity_metrics(
        real, synth, method=method, random_state=seed
    )

    assert repr(result).startswith("PropensityResult")
    assert result._fields == (
        "observed_p_MSE",
        "standardised_p_MSE",
        "ratio_p_MSE",
    )
    assert all(isinstance(val, float) for val in result)


@given(st.text(string.ascii_letters, min_size=1))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_propensity_metrics_error(real, synth, method):
    """Check that a ValueError is raised if the propensity model is not
    one of 'cart' or 'logr'."""

    match = f"Propensity method must be 'cart' or 'logr' not {method}."
    with pytest.raises(ValueError, match=match):
        _ = propensity.propensity_metrics(real, synth, method)
