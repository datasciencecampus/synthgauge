"""Property-based tests for the utility functions."""

import string

import numpy as np
import pandas as pd
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

from synthgauge import utils


@st.composite
def datasets(
    draw,
    max_columns=5,
    available_dtypes=("float", "object"),
    min_value=None,
    max_value=None,
    allow_nan=True,
):
    """Create a pair of datasets to act as 'real' and 'synthetic'."""

    ncols = draw(st.integers(1, max_columns))
    names = string.ascii_lowercase[: ncols - 1]
    dtypes = (draw(st.sampled_from(available_dtypes)) for _ in names)

    columns = []
    for name, dtype in zip(names, dtypes):

        if dtype == "object":
            elements = st.text(alphabet=string.ascii_letters, min_size=1)
        elif dtype == "float":
            elements = st.floats(
                min_value, max_value, allow_infinity=False, allow_nan=allow_nan
            )
        elif dtype == "int":
            elements = st.integers(min_value, max_value)
        elif dtype == "bool":
            elements = st.booleans()
        else:
            raise ValueError(
                "Available data types are int, float, object and bool."
            )

        columns.append(column(name, dtype=dtype, elements=elements))

    real = draw(data_frames(columns=columns))
    synth = draw(data_frames(columns=columns))

    return real, synth


@given(datasets(), st.one_of(st.none(), st.sampled_from(("a", ["a"]))))
def test_df_combine(datasets, feats):
    """Check that two datasets can be combined when features are named
    individually, given as a list or not specified."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    if isinstance(feats, str):
        columns = list([feats])
    elif isinstance(feats, list):
        columns = list(feats)
    else:
        columns = list(real.columns)

    combined = utils.df_combine(real, synth, feats)

    assert isinstance(combined, pd.DataFrame)
    assert len(combined) == len(real) + len(synth)
    assert list(combined.columns) == columns + ["source"]


@given(
    datasets(),
    st.one_of(st.none(), st.sampled_from(("a", ["a"]))),
    st.booleans(),
)
def test_df_separate(datasets, feats, drop):
    """Check that a dataset can be separated into two parts."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    combined = pd.concat((real, synth))
    combined["source"] = ["real"] * len(real) + ["synth"] * len(synth)

    separates = utils.df_separate(
        combined, "source", feats, drop_source_col=drop
    )

    assert isinstance(separates, tuple)

    for original, separate in zip((real, synth), separates):
        assert isinstance(separate, pd.DataFrame)
        assert len(original) == len(separate)

        if drop is False:
            assert "source" in separate.columns
            separate = separate.drop(columns="source")
        else:
            assert "source" not in separate.columns

        if isinstance(feats, (str, list)):
            original = pd.DataFrame(original[feats])

        assert original.equals(separate)


@given(
    datasets(allow_nan=False),
    st.one_of(st.none(), st.sampled_from(("a", ["a"]))),
)
def test_launder(datasets, feats):
    """Check that two datasets can have their features 'laundered'."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    laundered = utils.launder(real, synth, feats)

    for label, original, clean in zip(
        ("real", "synth"), (real, synth), laundered
    ):

        if isinstance(feats, str):
            assert np.array_equal(original[[feats]].values, clean.values)
            assert list(clean.columns) == [f"{feats}_{label}"]
        elif isinstance(feats, list):
            assert np.array_equal(original[feats].values, clean.values)
            assert list(clean.columns) == [f"{feat}_{label}" for feat in feats]
        else:
            assert np.array_equal(original.values, clean.values)
            assert list(clean.columns) == [
                f"{feat}_{label}" for feat in original.columns
            ]
