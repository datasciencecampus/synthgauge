"""Utility functions for property-based tests."""

import string

from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

import synthgauge as sg

available_columns = (
    "age",
    "height",
    "weight",
    "hair_colour",
    "eye_colour",
    "blood_type",
)

blood_type_feats = st.one_of(
    st.none(),
    st.lists(
        st.sampled_from(available_columns), min_size=1, max_size=4, unique=True
    ),
)


def resolve_features(feats, data):
    """Resolve the specified features so they are always a list."""

    return feats or data.columns.to_list()


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
        names = string.ascii_lowercase[:ncols]
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


@st.composite
def joint_params(draw):
    """Get a valid set of joint-plot parameters. Recycled by the 3D
    histogram and crosstab plot tests in `test_plot.py` and
    `test_evaluator.py`."""

    x, y = draw(
        st.lists(
            st.sampled_from(available_columns),
            min_size=2,
            max_size=2,
            unique=True,
        )
    )

    remaining_columns = list(set(available_columns).difference({x, y}))
    groupby = draw(st.one_of(st.none(), st.sampled_from(remaining_columns)))

    return x, y, groupby
