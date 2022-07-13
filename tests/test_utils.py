"""Property-based tests for the utility functions."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from synthgauge import utils

from .utils import datasets, resolve_features


@given(datasets(), st.sampled_from((None, ["a"])))
def test_df_combine(datasets, feats):
    """Check that two datasets can be combined when features are named
    individually, given as a list or not specified."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    columns = resolve_features(feats, real)

    combined = utils.df_combine(real, synth, feats)

    assert isinstance(combined, pd.DataFrame)
    assert len(combined) == len(real) + len(synth)
    assert list(combined.columns) == columns + ["source"]


@given(datasets(), st.sampled_from((None, ["a"])), st.booleans())
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


@given(datasets(allow_nan=False), st.sampled_from((None, ["a"])))
def test_launder(datasets, feats):
    """Check that two datasets can have their features 'laundered'."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    laundered = utils.launder(real, synth, feats)

    for label, original, clean in zip(
        ("real", "synth"), (real, synth), laundered
    ):

        if isinstance(feats, str):
            assert np.array_equal(original[[feats]], clean)
            assert list(clean.columns) == [f"{feats}_{label}"]
        elif isinstance(feats, list):
            assert np.array_equal(original[feats], clean)
            assert list(clean.columns) == [f"{feat}_{label}" for feat in feats]
        else:
            assert np.array_equal(original, clean)
            assert list(clean.columns) == [
                f"{feat}_{label}" for feat in original.columns
            ]


@given(datasets(available_dtypes=["object"]), st.sampled_from((None, ["a"])))
def test_cat_encode_categorical(datasets, feats):
    """Check that a dataset with object-only columns can be categorised
    and integer-encoded."""

    data, _ = datasets
    assume(not data.empty)

    columns = resolve_features(feats, data)

    out, cats = utils.cat_encode(data, feats)

    assert isinstance(out, pd.DataFrame)
    assert set(data[columns].columns) == set(out.columns)

    assert isinstance(cats, dict)
    assert set(cats.keys()) == set(columns)

    for col, categories in cats.items():

        assert pd.api.types.is_integer_dtype(out[col])
        assert isinstance(categories, pd.Index)
        assert data[col].to_list() == [categories[code] for code in out[col]]


@given(datasets(available_dtypes=("float",)), st.booleans())
def test_cat_encode_numeric(datasets, return_all):
    """Check that a warning is sent if the dataset to be encoded has
    numeric columns.

    Also, since we are only using numeric columns, check the output is
    empty unless we request everything be returned."""

    data, _ = datasets
    assume(not data.empty)

    columns = list(data.columns)

    with pytest.warns(
        UserWarning, match="^Selected features include non-object types"
    ):
        out, cats = utils.cat_encode(data, columns, return_all=return_all)

    assert cats == {}

    if return_all:
        assert out.equals(data)
    else:
        assert out.empty
        assert out.index.equals(data.index)
        assert out.columns.equals(pd.Index([]))


@given(datasets(allow_nan=False), st.booleans())
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_cat_encode_mixed(datasets, force):
    """Check that a mixed-type dataset can be cat-encoded. We also check
    the `force` parameter here."""

    data, _ = datasets
    assume(not data.empty)

    dtypes = data.dtypes.to_dict()
    assume(len(set(dtypes.values())) > 1)

    columns = list(data.columns)

    with pytest.warns(
        UserWarning, match="^Selected features include non-object types"
    ):
        out, cats = utils.cat_encode(data, columns, force=force)

    if force:
        assert set(out.columns) == set(columns)
        assert cats.keys() == dtypes.keys()

    else:
        numeric_columns = set(data.select_dtypes(include="number").columns)
        assert set(data.columns).difference(out.columns) == numeric_columns

    for col, categories in cats.items():
        assert pd.api.types.is_numeric_dtype(out[col])
        assert isinstance(categories, pd.Index)
        assert data[col].to_list() == [categories[code] for code in out[col]]


@given(datasets(available_dtypes=["object"]))
def test_cat_encode_convert_only(datasets):
    """Check that a dataset can be categorised without being encoded."""

    data, _ = datasets
    assume(not data.empty)

    out, cats = utils.cat_encode(data, convert_only=True)

    columns = data.columns

    assert cats is None
    assert isinstance(out, pd.DataFrame)
    assert set(columns) == set(out.columns)
    assert all(np.array_equal(data[col], (out[col].values)) for col in columns)

    for col in columns:
        assert np.array_equal(data[col], out[col])
        assert isinstance(out[col].dtype, pd.CategoricalDtype)
        assert set(out[col].cat.categories) == set(data[col].unique())


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

    diffs, edges = utils.feature_density_diff(
        real, synth, feature="a", bins=bins
    )

    assert isinstance(diffs, np.ndarray)
    assert len(diffs) == bins
    limit = bins + 1e-10
    assert all(diff >= -limit and diff <= limit for diff in diffs)

    assert isinstance(edges, np.ndarray)
    assert np.array_equal(edges, np.linspace(0, num_categories - 1, bins + 1))
