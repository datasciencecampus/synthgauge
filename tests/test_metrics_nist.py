"""Tests for the NIST 2018 metrics."""

from hypothesis import assume, given
from hypothesis import strategies as st

from synthgauge.metrics import nist

from .utils import datasets


@given(datasets(), st.integers(0, 100))
def test_three_way_marginals(datasets, seed):
    """Test that the 3-way marginal score can be calculated."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    score = nist.three_way_marginals(real, synth, seed)

    assert isinstance(score, float)
    assert score >= 0 and score <= 1
