"""Property-based tests for the dataset creation functions."""

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from synthgauge import datasets


@st.composite
def data_and_labels(draw, rlims=(10, 1000), clims=(2, 6)):
    """Create a toy array and labels for testing."""

    shape = draw(st.tuples(st.integers(*rlims), st.integers(*clims)))

    data = draw(
        arrays(float, shape, elements=st.floats(-100, 100, allow_nan=False))
    )

    labels = draw(
        arrays(int, (len(data),), elements=st.integers(0, len(data)))
    )

    return data, labels


@given(
    data_and_labels(),
    st.floats(0, 100, allow_nan=False),
    st.floats(0, 0.5, allow_nan=False),
    st.integers(0, 100),
)
def test_adjust_data_elements(data_labels, noise, nan_prop, seed):
    """Check that a set of data and labels can be turned into a
    dataframe correctly."""

    data, labels = data_labels

    nrows, ncols = data.shape
    df = datasets._adjust_data_elements(data, labels, noise, nan_prop, seed)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (nrows, ncols + 1)
    assert df.isna().sum().sum() == int(nan_prop * df.size)

    if not noise and not nan_prop:
        assert np.array_equal(df.iloc[:, :-1], data)
        assert np.array_equal(df.iloc[:, -1], labels)


@given(
    st.floats(0, 100, allow_nan=False),
    st.floats(0, 0.5, allow_nan=False),
    st.integers(0, 100),
)
def test_make_blood_types_df(noise, nan_prop, seed):
    """Check that a toy blood types dataset can be made."""

    data = datasets.make_blood_types_df(noise, nan_prop, seed)

    assert isinstance(data, pd.DataFrame)
    assert data.shape == (1000, 6)
    assert data.columns.tolist() == [
        "age",
        "height",
        "weight",
        "hair_colour",
        "eye_colour",
        "blood_type",
    ]
    assert set(data["hair_colour"]).issubset(
        {np.nan, "Black", "Blonde", "Brown", "Red"}
    )
    assert set(data["eye_colour"]).issubset({np.nan, "Blue", "Brown", "Green"})
    assert set(data["blood_type"]).issubset({np.nan, "A", "AB", "B", "O"})
