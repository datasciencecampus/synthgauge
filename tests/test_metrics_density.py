"""Property-based tests for the feature density MAD functions."""

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from synthgauge.metrics import density

from .utils import datasets


@given(datasets(available_dtypes=["object"]), st.integers(1, 10))
@settings(deadline=None)
def test_feature_density_diff(datasets, bins):
    """Check that histogram-based density differences can be computed
    correctly. For the ease of testing, we only look at object (str)
    features and datasets with unique rows that are not identical."""

    real, synth = datasets
    assume(
        not (
            real.empty
            or synth.empty
            or real["a"].drop_duplicates().equals(synth["a"].drop_duplicates())
        )
    )

    num_categories = pd.concat((real, synth))["a"].nunique()

    diffs, edges = density._feature_density_diff(
        real, synth, feature="a", bins=bins
    )

    assert isinstance(diffs, np.ndarray)
    assert len(diffs) == bins
    limit = bins + 1e-10
    assert all(diff >= -limit and diff <= limit for diff in diffs)

    assert isinstance(edges, np.ndarray)
    assert np.array_equal(edges, np.linspace(0, num_categories - 1, bins + 1))


@given(
    datasets(available_dtypes=("object",), min_columns=2),
    st.one_of(st.none(), st.sampled_from(("a", ["a", "b"]))),
    st.integers(1, 10),
)
def test_feature_density_diff_mae(datasets, feats, bins):
    """Check that the feature density difference mean absolute
    difference can be calculated correctly."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    mad = density.feature_density_mad(real, synth, feats, bins)

    assert isinstance(mad, float)
    assert mad >= 0 and mad <= bins
