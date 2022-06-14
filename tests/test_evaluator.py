""" Tests for the Evaluator class. """

import itertools as it
import pickle
import string

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

import synthgauge as sg
from synthgauge.metrics import __all__ as implemented_metrics


@st.composite
def datasets(draw, max_columns=5, available_dtypes=("float", "object")):
    """Create a pair of datasets to act as 'real' and 'synthetic'."""

    ncols = draw(st.integers(1, max_columns))
    names = string.ascii_lowercase[: ncols - 1]
    dtypes = (draw(st.sampled_from(available_dtypes)) for _ in names)

    columns = []
    for name, dtype in zip(names, dtypes):
        elements = st.text(min_size=1) if dtype == "object" else None
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


@given(datasets())
def test_init(datasets):
    """Test the instantiation of an Evaluator object."""

    real, synth = datasets
    columns = real.columns

    evaluator = sg.Evaluator(real, synth)

    assert evaluator.real_data.equals(real)
    assert evaluator.synth_data.equals(synth)
    assert evaluator.feature_names.equals(columns)
    assert evaluator.metrics == {}
    assert evaluator.metric_results == {}
    assert list(evaluator.combined_data.columns) == list(columns) + ["source"]


@given(datasets())
def test_init_uncommon_features(datasets):
    """Check a warning is sent if the datasets do not have exactly the
    same columns."""

    real, synth = datasets
    real["xyz"] = 0

    with pytest.warns(UserWarning, match="^Features xyz are not"):
        evaluator = sg.Evaluator(real, synth)

    assert evaluator.feature_names.equals(synth.columns)


@given(datasets(), st.one_of(st.just("drop"), st.text()))
@settings(deadline=None)
def test_init_missing_values(datasets, handle_nans):
    """Check missing values are only dropped when `handle_nans = "drop"`."""

    real, synth = datasets
    real = pd.concat((real, pd.DataFrame({"a": [np.nan]})))
    nrows = len(real)

    evaluator = sg.Evaluator(real, synth, handle_nans)

    if handle_nans == "drop":
        assert len(evaluator.real_data) < nrows
    else:
        assert len(evaluator.real_data) == nrows


@given(datasets())
def test_describe_methods(datasets):
    """Check that a numeric summary table is returned."""

    real, synth = datasets
    assume(not (real.empty | synth.empty))

    common = real.columns.intersection(synth.columns)
    num_columns = real[common].select_dtypes(include="number").columns
    cat_columns = real[common].select_dtypes(exclude="number").columns

    evaluator = sg.Evaluator(real, synth)

    if num_columns.any():

        num_summary_idx = [
            "_".join(item)
            for item in it.product(num_columns, ("real", "synth"))
        ]
        num_summary_cols = [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]

        num_summary = evaluator.describe_numeric()

        assert isinstance(num_summary, pd.DataFrame)
        assert list(num_summary.index) == num_summary_idx
        assert list(num_summary.columns) == num_summary_cols

    if cat_columns.any():

        cat_summary_idx = [
            "_".join(item)
            for item in it.product(cat_columns, ("real", "synth"))
        ]
        cat_summary_cols = ["count", "unique", "most_frequent", "freq"]

        cat_summary = evaluator.describe_categorical()

        assert isinstance(cat_summary, pd.DataFrame)
        assert list(cat_summary.index) == cat_summary_idx
        assert list(cat_summary.columns) == cat_summary_cols


@given(
    evaluators(),
    st.sampled_from(implemented_metrics),
    st.one_of(st.none(), st.text(min_size=1)),
    st.dictionaries(st.text(min_size=1), st.text(min_size=1)),
)
def test_add_metric_implemented(evaluator, metric, alias, kwargs):
    """Check that any implemented metric can be added."""

    evaluator.add_metric(metric, alias, **kwargs)

    assert isinstance(evaluator.metrics, dict)

    if alias is None:
        assert evaluator.metrics == {metric: {**kwargs}}
    else:
        assert evaluator.metrics == {alias: {"metric_name": metric, **kwargs}}


@given(evaluators(), st.text(alphabet=string.ascii_letters, min_size=1))
def test_add_metric_not_implemented(evaluator, invalid_metric):
    """Check a NotImplementedError gets thrown by invalid metric names."""

    match = f"Metric '{invalid_metric}' is not implemented"
    with pytest.raises(NotImplementedError, match=match):
        evaluator.add_metric(invalid_metric)


@given(evaluators(), st.text(min_size=1), st.functions())
def test_add_custom_metric(evaluator, name, func):
    """Check that a (nonsense) custom metric can be added."""

    evaluator.add_custom_metric(name, func)

    assert evaluator.metrics == {name: {"metric_func": func}}


@given(evaluators(), st.sampled_from(implemented_metrics))
def test_copy_metrics(evaluator, metric):
    """Check that an Evaluator can copy another instance's metrics."""

    other = sg.Evaluator(evaluator.real_data, evaluator.synth_data)
    other.add_metric(metric)
    evaluator.copy_metrics(other)

    assert evaluator.metrics == other.metrics


@given(evaluators(), st.none())
def test_copy_metrics_not_evaluator(evaluator, other):
    """Check that a TypeError gets thrown when trying to copy metrics from
    something other than another Evaluator instance."""

    with pytest.raises(TypeError):
        evaluator.copy_metrics(other)


@given(
    evaluator=evaluators(),
    metric=st.sampled_from(implemented_metrics),
    overwrite=st.booleans(),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_save_and_load_metrics_valid(tmp_path, evaluator, metric, overwrite):
    """Check that valid metrics can be stored and loaded."""

    path = tmp_path / "metrics.pkl"
    evaluator.add_metric(metric, metric_alias="test")
    evaluator.save_metrics(path)

    old_metrics = dict(evaluator.metrics)
    new_metrics = {"xyz": {"metric_func": lambda x: 0}}
    new_metric_name = list(new_metrics.keys())[0]

    evaluator = sg.Evaluator(evaluator.real_data, evaluator.synth_data)
    evaluator.add_custom_metric(
        new_metric_name, new_metrics[new_metric_name]["metric_func"]
    )

    evaluator.load_metrics(path, overwrite)

    assert path.exists()

    if overwrite:
        assert evaluator.metrics == old_metrics
    else:
        assert evaluator.metrics == {**old_metrics, **new_metrics}


@given(evaluator=evaluators(), invalid_metric=st.text(min_size=1))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_load_metrics_invalid(tmp_path, evaluator, invalid_metric):
    """Check that loading any non-implemented metrics raises a ValueError."""

    path = tmp_path / "metrics.pkl"
    metrics = {invalid_metric: {"metric_name": invalid_metric}}
    with open(path, "wb") as f:
        pickle.dump(metrics, f)

    with pytest.raises(ValueError):
        evaluator.load_metrics(path)