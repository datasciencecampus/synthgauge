""" Tests for the Evaluator class. """

import itertools as it
import pickle
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import synthgauge as sg
from synthgauge.metrics import __all__ as implemented_metrics

from .utils import datasets, evaluators, joint_params


@given(datasets())
def test_init(datasets):
    """Test the instantiation of an Evaluator object."""

    real, synth = datasets
    columns = real.columns

    evaluator = sg.Evaluator(real, synth)

    assert evaluator.real_data.equals(real)
    assert evaluator.synth_data.equals(synth)
    assert evaluator.feature_names == list(columns)
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

    assert evaluator.feature_names == list(synth.columns)


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
    assume(not (real.empty or synth.empty))

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
    key = metric if alias is None else alias

    assert isinstance(evaluator.metrics, dict)
    assert evaluator.metrics == {key: {"name": metric, **kwargs}}


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

    assert evaluator.metrics == {name: {"name": name, "func": func}}


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
    new_metric_name = "xyz"
    new_metrics = {
        new_metric_name: {
            "func": lambda x: 0,
            "name": new_metric_name,
        }
    }

    evaluator = sg.Evaluator(evaluator.real_data, evaluator.synth_data)
    evaluator.add_custom_metric(
        new_metric_name, new_metrics[new_metric_name]["func"]
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
    metrics = {invalid_metric: {"name": invalid_metric}}
    with open(path, "wb") as f:
        pickle.dump(metrics, f)

    with pytest.raises(ValueError):
        evaluator.load_metrics(path)


@given(evaluators(), st.sampled_from(implemented_metrics))
def test_drop_metric_present(evaluator, metric):
    """Check that a metric can be dropped from an Evaluator."""

    evaluator.add_metric(metric)
    dropped = evaluator.drop_metric(metric)

    assert evaluator.metrics == {}
    assert dropped is None


@given(evaluators(), st.sampled_from(implemented_metrics))
def test_drop_metric_absent(evaluator, metric):
    """Check that None is returned when attempting to remove a metric that is
    not present in an Evaluator."""

    dropped = evaluator.drop_metric(metric)

    assert evaluator.metrics == {}
    assert dropped is None


@given(
    evaluators(available_dtypes=("float",), allow_nan=False),
    st.sampled_from(("wasserstein", "kolmogorov_smirnov", "kruskal_wallis")),
    st.booleans(),
    st.booleans(),
)
def test_evaluate_implemented_metric(evaluator, metric, use_alias, as_df):
    """Check that an implemented metric can be evaluated and presented.
    To ease writing this test, we only consider numeric data and a small
    selection of univariate metrics with the same signature."""

    assume(
        not (
            evaluator.real_data.empty
            or evaluator.synth_data.empty
            or evaluator.real_data["a"].equals(evaluator.synth_data["a"])
        )
    )

    alias = metric if use_alias else None
    evaluator.add_metric(metric, alias, feature="a")

    results = evaluator.evaluate(as_df)

    if as_df:
        assert isinstance(results, pd.DataFrame)
        assert list(results.columns) == ["value"]

        if len(results) > 1:
            evaluator_results = list(*evaluator.metric_results.values())
        else:
            evaluator_results = list(evaluator.metric_results.values())

        assert evaluator_results == results["value"].to_list()
        idxs = list(results.index)

    else:
        assert isinstance(results, dict)
        assert evaluator.metric_results == results
        idxs = list(results.keys())

    assert idxs == [metric] or [
        "-".join((metric, item)) for item in ("statistic", "pvalue")
    ]


@given(evaluators(), st.functions(like=lambda r, s: None, returns=st.floats()))
def test_evaluate_custom_metric(evaluator, func):
    """Check that a custom (nonsense) metric can be evaluated and presented."""

    evaluator.add_custom_metric("xyz", func)

    results = evaluator.evaluate()

    assert isinstance(results, dict)
    assert evaluator.metric_results == results
    assert list(results.keys()) == ["xyz"]
    assert isinstance(results.get("xyz"), float)


@given(
    st.tuples(
        st.floats(1, 5, allow_nan=False), st.floats(1, 5, allow_nan=False)
    )
)
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ],
)
def test_plot_histograms(real, synth, size):
    """Check that the histograms method produces a figure. Full tests
    of the underlying function are in `tests/test_plot.py`."""

    evaluator = sg.Evaluator(real, synth)

    fig = evaluator.plot_histograms(figsize=size)

    assert isinstance(fig, plt.Figure)
    assert np.allclose(fig.get_size_inches(), size, atol=0.01)


@given(
    joint_params(),
    st.sampled_from(("real", "synth", "combined")),
)
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_plot_histogram3d(real, synth, params, which_data):
    """Check that the 3D histogram method produces a figure. Full tests
    of the underlying function are in `tests/test_plot.py`."""

    x, y, _ = params
    evaluator = sg.Evaluator(real, synth)

    fig = evaluator.plot_histogram3d(which_data, x, y)
    ax = fig.axes[0]

    assert isinstance(fig, plt.Figure)
    assert ax.get_xlabel() == x
    assert ax.get_ylabel() == y


@given(st.sampled_from(("pearson", "cramers_v", "spearman")))
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_plot_correlation(real, synth, method):
    """Check that the correlation method produces a figure. Full tests
    of the underlying function are in `tests/test_plot.py`."""

    evaluator = sg.Evaluator(real, synth)

    fig = evaluator.plot_correlation(method=method)
    rax, sax, dax, empty, rbar, sbar, dbar = fig.axes

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == len(
        ("real", "synth", "diff", "empty", "rbar", "sbar", "dbar")
    )


@given(joint_params())
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_plot_crosstab(real, synth, params):
    """Check that the crosstab method produces a figure. Full tests of
    the underlying function are in `tests/test_plot.py`."""

    x, y, _ = params
    evaluator = sg.Evaluator(real, synth)

    fig = evaluator.plot_crosstab(x, y)
    rax, sax, _ = fig.axes

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == len(("REAL", "SYNTH", "cbar"))

    for ax in (rax, sax):
        assert ax.get_xlabel() == x
        assert ax.get_ylabel() == y


@given(st.sampled_from(("age", "height", "weight")))
@settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_plot_qq(real, synth, feature):
    """Check that the QQ-plot method produces a figure. Full tests of
    the underlying function are in `tests/test_plot.py`."""

    evaluator = sg.Evaluator(real, synth)

    fig = evaluator.plot_qq(feature)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
