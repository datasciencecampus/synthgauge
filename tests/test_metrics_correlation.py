"""Property-based tests for the correlation-based metrics."""

import numpy as np
import pandas as pd
from hypothesis import assume, given
from hypothesis import strategies as st

from synthgauge.metrics import correlation

from .utils import datasets


@given(datasets(2, 2, available_dtypes=("object", "bool")))
def test_cramers_v(datasets):
    """Check that our implementation of Cramer's V correlation is
    sensible."""

    data, _ = datasets
    assume(not data.empty)

    cramers = correlation._cramers_v(data["a"], data["b"])

    assert isinstance(cramers, float)
    assert (
        np.isnan(cramers)
        or (cramers >= 0 and cramers <= 1)
        or np.isclose(cramers, 1)
    )


@given(datasets(2, 5, available_dtypes=["object"]))
def test_pairwise_cramers_v(datasets):
    """Check that the pairwise Cramer's V metric can be computed for an
    all-object dataframe."""

    data, _ = datasets
    assume(not data.empty and (data.nunique() > 1).all())

    cramers = correlation._pairwise_cramers_v(data)
    values = cramers.values

    assert isinstance(cramers, pd.DataFrame)
    assert cramers.index.equals(data.columns)
    assert cramers.columns.equals(data.columns)
    assert np.isclose(values.diagonal(), 1).all()
    assert values.sum() < values.size or np.isclose(
        values.sum(), values.size
    )  # test [0, 1] while avoiding floating-point errors


@given(
    datasets(2, available_dtypes=("float",), allow_nan=False),
    st.sampled_from(("pearson", "spearman")),
    st.sampled_from((None, ["a", "b"])),
)
def test_correlation_msd_numeric(datasets, method, feats):
    """Check that the mean-squared difference in correlation can be
    found correctly for either Pearson's or Spearman's method."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    msd = correlation.correlation_msd(real, synth, method, feats)

    assert isinstance(msd, float)
    assert np.isnan(msd) or (msd >= 0 and msd < 4) or np.isclose(msd, 4)


@given(
    datasets(2, available_dtypes=("object", "bool")),
    st.sampled_from((None, ["a", "b"])),
)
def test_correlation_msd_cramers_v(datasets, feats):
    """Check that the Cramer's V MSD can be calculated correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    msd = correlation.correlation_msd(real, synth, "cramers_v", feats)

    assert isinstance(msd, float)
    assert np.isnan(msd) or (msd >= 0 and msd < 1) or np.isclose(msd, 1)


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

    ratio = correlation._correlation_ratio(data["a"], data["b"])

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

    msd = correlation.correlation_ratio_msd(real, synth, categorical, numeric)

    assert isinstance(msd, float)
    assert np.isnan(msd) or (msd >= 0 and msd <= 1) or np.isclose(msd, 1)
