"""Utility functions for property-based tests."""

import string

from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

import synthgauge as sg


def resolve_features(feats, data):
    """Resolve the specified features so they are always a list."""

    if isinstance(feats, str):
        columns = list([feats])
    elif isinstance(feats, list):
        columns = list(feats)
    else:
        columns = list(data.columns)

    return columns


@st.composite
def datasets(
    draw,
    min_columns=1,
    max_columns=3,
    available_dtypes=("float", "object"),
    min_value=None,
    max_value=None,
    allow_nan=True,
    column_spec=None,
):
    """Create a pair of datasets to act as 'real' and 'synth' in tests."""

    if column_spec is None:
        ncols = draw(st.integers(min_columns, max_columns))
        names = string.ascii_lowercase[: ncols - 1]
        dtypes = (draw(st.sampled_from(available_dtypes)) for _ in names)

    else:
        names = column_spec.keys()
        dtypes = column_spec.values()

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
