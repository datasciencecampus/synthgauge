"""Property-based tests for the correlation-based metrics."""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from synthgauge.metrics import correlation

from .utils import datasets, resolve_features


@given(
    datasets(min_columns=2, available_dtypes=("float",), allow_nan=False),
    st.sampled_from((None, ["a", "b"])),
)
def test_correlation_MSD(datasets, feats):
    """Check that the mean-squared difference in Pearson's correlation
    can be found correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    columns = resolve_features(feats, real)
    msd = correlation.correlation_MSD(real, synth, feats)

    assert isinstance(msd, float)
    assert (
        np.isnan(msd)
        or (msd >= 0 and msd <= len(columns))
        or np.isclose(msd, len(columns))
    )


@given(datasets(2, 2, available_dtypes=("object", "bool")))
def test_cramers_v(datasets):
    """Check that our implementation of Cramer's V correlation is
    sensible."""

    data, _ = datasets
    assume(not data.empty)

    cramers = correlation.cramers_v(data["a"], data["b"])

    assert isinstance(cramers, float)
    assert (
        np.isnan(cramers)
        or (cramers >= 0 and cramers <= 1)
        or np.isclose(cramers, 1)
    )


@given(
    datasets(2, 2, available_dtypes=("object", "bool")),
    st.sampled_from((None, ["a", "b"])),
)
def test_cramers_v_MSD(datasets, feats):
    """Check that the Cramer's V MSD can be calculated correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    columns = resolve_features(feats, real)
    msd = correlation.cramers_v_MSE(real, synth, feats)

    assert isinstance(msd, float)
    assert (
        np.isnan(msd)
        or (msd >= 0 and msd <= len(columns))
        or np.isclose(msd, len(columns))
    )


@given(
    datasets(
        min_value=0, max_value=100, column_spec={"a": "int", "b": "object"}
    )
)
def test_cramers_v_MSD_warning(datasets):
    """Check that a warning is sent if numeric columns are passed."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    with pytest.warns(
        UserWarning, match="Selected features include numeric types"
    ):
        _ = correlation.cramers_v_MSE(real, synth, feats=["a", "b"])


@given(
    datasets(
        column_spec={"a": "object", "b": "float"},
        min_value=0,
        max_value=1000,
        allow_nan=False,
    )
)
def test_correlation_ratio(datasets):
    """Check that the categorical-continuous association can be
    calculated correctly."""

    data, _ = datasets
    assume(not data.empty)

    ratio = correlation.correlation_ratio(data["a"], data["b"])

    assert isinstance(ratio, float)
    assert (
        np.isnan(ratio) or (ratio >= 0 and ratio <= 1) or np.isclose(ratio, 1)
    )


@given(
    datasets(
        column_spec={"a": "object", "b": "float"},
        min_value=0,
        max_value=1000,
        allow_nan=False,
    ),
    st.sampled_from((None, ["a"])),
    st.sampled_from((None, ["b"])),
)
def test_correlation_ratio_MSE(datasets, categorical, numeric):
    """Check that the categorical-continuous association mean-squared
    error can be calculated correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    msd = correlation.correlation_ratio_MSE(real, synth, categorical, numeric)

    assert isinstance(msd, float)
    assert np.isnan(msd) or (msd >= 0 and msd <= 1) or np.isclose(msd, 1)
