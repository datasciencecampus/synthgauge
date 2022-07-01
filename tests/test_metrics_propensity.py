"""Property-based tests for the propensity score-based metrics."""

import string

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from synthgauge.metrics import propensity

from .utils import blood_type_feats


def _reduce_synthetic(synth, seed):
    """Cut off the synthetic data at a random point."""

    rng = np.random.default_rng(seed)
    nrows = int(len(synth) * rng.uniform(0, 0.75))
    synth = synth.iloc[:nrows, :]

    return synth


@given(st.sampled_from(("LogisticRegression", "CART")), st.integers(0, 100))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_propensity_MSE(real, synth, method, seed):
    """Check that either model can be used to obtain a propensity score
    mean-squared error. We first drop a random set of synthetic rows so
    that the bounds on pMSE can be verified."""

    synth = _reduce_synthetic(synth, seed)
    synth_prop = len(synth) / (len(real) + len(synth))

    pmse = propensity.propensity_MSE(real, synth, method)

    assert isinstance(pmse, float)
    assert pmse >= 0

    if synth_prop > 0.5:
        assert pmse <= synth_prop**2
    else:
        assert pmse <= synth_prop**2 - 2 * synth_prop + 1


@given(st.text(string.ascii_letters, min_size=1))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_propensity_MSE_method_error(real, synth, method):
    """Check that a ValueError is raised if the propensity model is not
    one of CART or LogisticRegression."""

    match = "^method must be either 'CART' or 'LogisticRegression'$"
    with pytest.raises(ValueError, match=match):
        _ = propensity.propensity_MSE(real, synth, method)


@given(st.integers(0, 100))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_expected_p_MSE(real, synth, seed):
    """Check that the expected pMSE is calculated correctly."""

    synth = _reduce_synthetic(synth, seed)
    ncols = len(real.columns)
    total_rows = len(real) + len(synth)
    synth_prop = len(synth) / total_rows

    expected = propensity.expected_p_MSE(real, synth)

    assert isinstance(expected, float)

    if synth_prop > 0.5:
        assert expected <= synth_prop**2
    else:
        assert expected <= synth_prop**2 - 2 * synth_prop + 1

    assert np.isclose(
        expected * total_rows / ((ncols * (ncols + 1) / 2) - 1),
        synth_prop * (1 - synth_prop) ** 2,
    )


@given(st.integers(0, 100))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_stdev_p_MSE(real, synth, seed):
    """Check that the std. dev. of pMSE is calculated correctly."""

    synth = _reduce_synthetic(synth, seed)
    ncols = len(real.columns)
    total_rows = len(real) + len(synth)
    synth_prop = len(synth) / total_rows

    stddev = propensity.stdev_p_MSE(real, synth)

    assert isinstance(stddev, float)
    assert np.isclose(
        stddev * total_rows / (2 * ((ncols * (ncols + 1) / 2) - 1)) ** 0.5,
        synth_prop * (1 - synth_prop) ** 2,
    )


@given(st.integers(0, 100))
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)
def test_perm_expected_sd_p_MSE(real, synth, seed):
    """Check that the permutation-based expectation and std. dev. can
    be calculated for a pair of datasets."""

    rng = np.random.default_rng(seed)
    nrows = int(len(synth) * rng.uniform(0, 0.75))
    synth = synth.iloc[:nrows, :]

    synth_prop = len(synth) / (len(real) + len(synth))

    expected, stddev = propensity.perm_expected_sd_p_MSE(
        real, synth, num_perms=10
    )
    assert isinstance(expected, float)
    assert isinstance(stddev, float)

    assert stddev >= 0

    if synth_prop > 0.5:
        assert expected <= synth_prop**2
    else:
        assert expected <= synth_prop**2 - 2 * synth_prop + 1


@given(
    st.sampled_from(("CART", "LogisticRegression")),
    blood_type_feats,
    st.integers(0, 100),
)
@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_p_MSE_ratio(real, synth, method, feats, seed):
    """Check that the pMSE-ratio can be calculated correctly."""

    rng = np.random.default_rng(seed)
    nrows = int(len(synth) * rng.uniform(0, 0.5))
    synth = synth.iloc[:nrows, :]

    ratio = propensity.p_MSE_ratio(real, synth, method, feats)

    assert isinstance(ratio, float)
    assert ratio > 0

    if method == "LogisticRegression":
        assert ratio >= 1


@given(
    st.sampled_from(("CART", "LogisticRegression")),
    blood_type_feats,
    st.integers(0, 100),
)
@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_standardised_p_MSE(real, synth, method, feats, seed):
    """Check that the standardised pMSE can be calculated correctly."""

    rng = np.random.default_rng(seed)
    nrows = int(len(synth) * rng.uniform(0, 0.5))
    synth = synth.iloc[:nrows, :]

    standardised = propensity.standardised_p_MSE(real, synth, method, feats)

    assert isinstance(standardised, float)


@given(st.sampled_from(("CART", "LogisticRegression")), blood_type_feats)
@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_propensity_metrics(real, synth, method, feats):
    """Check that the propensity metric wrapper function returns its
    named tuple. Further tests for the individual metrics are above."""

    result = propensity.propensity_metrics(real, synth, method, feats)

    assert repr(result).startswith("PropensityResult")
    assert result._fields == (
        "observed_p_MSE",
        "standardised_p_MSE",
        "ratio_p_MSE",
    )
    assert all(isinstance(val, float) for val in result)
