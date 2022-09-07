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
