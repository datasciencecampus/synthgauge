"""Utility functions for property-based tests."""

import string

from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

import synthgauge as sg


@st.composite
def datasets(
    draw,
    max_columns=3,
    available_dtypes=("float", "object"),
    min_value=None,
    max_value=None,
    allow_nan=True,
):
    """Create a pair of datasets to act as 'real' and 'synth' in tests."""

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


@st.composite
def evaluators(draw, **kwargs):
    """Create an Evaluator instance."""

    real, synth = draw(datasets(**kwargs))
    evaluator = sg.Evaluator(real, synth)

    return evaluator
