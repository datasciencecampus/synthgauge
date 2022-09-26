"""Tests for the NIST 2018 metrics."""

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from synthgauge.metrics import nist

from .utils import datasets


@given(
    datasets(
        column_spec={"a": "int", "b": "int", "c": "object"},
        min_value=0,
        max_value=100,
        allow_nan=False,
    ),
    st.sampled_from(["auto", 10, 100]),
)
def test_numeric_edges(datasets, bins):
    """Test the numeric feature edges can be found correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    edges = nist._numeric_edges(real, synth, bins)

    assert isinstance(edges, dict)
    assert list(edges.keys()) == ["a", "b"]

    for col, edge in edges.items():
        assert isinstance(edge, np.ndarray)
        assert edge.ndim == 1
        assert min(edge) <= real[col].min()
        assert max(edge) >= real[col].max()
        if isinstance(bins, int):
            assert edge.size == bins + 1


@given(
    datasets(
        column_spec={"a": "int", "b": "int", "c": "object"},
        min_value=0,
        max_value=100,
        allow_nan=False,
    ),
    st.sampled_from(["auto", 10, 100]),
)
def test_discretise_datasets(datasets, bins):
    """Test that a real-synthetic pair can be discretised."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    disreal, dissynth = nist._discretise_datasets(real, synth, bins)

    for disc, orig in zip((disreal, dissynth), (real, synth)):
        assert isinstance(disc, pd.DataFrame)
        assert disc.columns.equals(orig.columns)
        assert disc["c"].equals(orig["c"])


@given(
    datasets(available_dtypes=("object",), min_columns=3),
    st.lists(st.sampled_from(["a", "b", "c"]), unique=True, min_size=1).map(
        sorted
    ),
)
@settings(deadline=400)
def test_kway_marginal_score(datasets, features):
    """Test that the transformed marginal score can be calculated."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    score = nist._kway_marginal_score(real, synth, features)

    assert isinstance(score, float)
    assert np.isnan(score) or (score >= 0 and score <= 1)


@given(
    datasets(
        column_spec={"a": "int", "b": "int", "c": "object"},
        min_value=0,
        max_value=100,
        allow_nan=False,
    ),
    st.integers(1, 100),
)
@settings(deadline=800)
def test_kway_marginals(datasets, seed):
    """Test that the k-way marginal score can be calculated."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    score = nist.kway_marginals(real, synth, k=2, trials=3, seed=seed)

    assert isinstance(score, float)
    assert np.isnan(score) or (score >= 0 and score <= 1)


@given(datasets(min_value=0, allow_nan=False), st.integers(1, 100))
def test_make_rule(datasets, seed):
    """Test that a valid rule can be made."""

    data, _ = datasets
    assume(not data.empty)

    prng = np.random.default_rng(seed)
    row = data.iloc[prng.integers(0, len(data)), :]
    column = prng.choice(data.columns)
    observed = row[column]
    values = data[column].unique()

    parts = nist._make_rule(data, row, column, prng)

    if pd.api.types.is_numeric_dtype(data[column]):
        assert isinstance(parts, tuple)
        assert len(parts) == 2
        assert parts[0] == observed
        assert parts[1] >= 0 and parts[1] <= values.max() - values.min()
    else:
        assert isinstance(parts, set)
        assert observed in parts
        assert parts.issubset(values)


@given(
    datasets(min_value=0, allow_nan=False),
    st.sampled_from([1, 5, 10]),
    st.floats(0.1, 1.0),
    st.integers(1, 100),
)
def test_create_test_cases(datasets, trials, prob, seed):
    """Test that a collection of HOC tests can be created."""

    data, _ = datasets
    assume(not data.empty)

    columns = data.columns
    cases = nist._create_test_cases(data, trials, prob, seed)

    assert isinstance(cases, list)
    assert len(cases) == trials

    for case in cases:
        assert isinstance(case, dict)
        assert len(case) <= len(columns)
        for col, rule in case.items():
            assert col in columns
            if pd.api.types.is_numeric_dtype(data[col]):
                assert isinstance(rule, tuple)
            else:
                assert isinstance(rule, set)


@given(datasets(min_value=0, allow_nan=False), st.integers(1, 100))
def test_evaluate_test_cases(datasets, seed):
    """Test that a collection of HOC tests can be evaluated."""

    data, _ = datasets
    assume(not data.empty)

    cases = nist._create_test_cases(data, 10, 1, seed)
    results = nist._evaluate_test_cases(data, cases)

    assert isinstance(results, list)
    assert len(results) == len(cases)

    for result, case in zip(results, cases):
        if case == {}:
            assert np.isnan(result)
        else:
            assert result >= 1 / len(data) and result <= 1


@given(datasets(min_value=0, allow_nan=False), st.integers(1, 100))
def test_hoc(datasets, seed):
    """Test that the HOC score can be evaluated."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    score = nist.hoc(real, synth, 10, 1, seed)

    assert isinstance(score, (float, int))
    assert score >= 0 and score <= 1
