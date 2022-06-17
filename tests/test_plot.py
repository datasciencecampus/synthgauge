"""Property-based tests for plotting functions."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from seaborn.axisgrid import JointGrid

from synthgauge import plot
from synthgauge.datasets import make_blood_types_df

from .utils import resolve_features

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
    st.sampled_from(available_columns),
    st.lists(
        st.sampled_from(available_columns), min_size=1, max_size=4, unique=True
    ),
)

plot_settings = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=15,
    deadline=None,
)


@st.composite
def joint_params(draw):
    """Get a valid set of joint-plot parameters. Recycled by the 3D
    histogram and crosstab plot tests."""

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


@pytest.fixture
def real():
    """Make some real (noiseless) data."""

    return make_blood_types_df(0, 0)


@pytest.fixture
def synth():
    """Make some synthetic (noisy) data."""

    return make_blood_types_df(1, 0)


@given(
    st.lists(st.text(min_size=1), min_size=1),
    st.sampled_from(("x", "y")),
    st.integers(1, 10),
)
@plot_settings
def test_suggest_label_rotation(labels, axis, limit):
    """Check that axis tick label rotation can be flagged correctly."""

    _, ax = plt.subplots()
    getattr(ax, f"set_{axis}ticks")(range(len(labels)))
    getattr(ax, f"set_{axis}ticklabels")(labels)

    rotate = plot.suggest_label_rotation(ax, axis, limit)
    assert rotate is any(len(label) > limit for label in labels)


@given(feats=blood_type_feats)
@plot_settings
def test_plot_histograms(real, feats):
    """Check that a histogram (or set thereof) can be created from a dataset
    regardless of its data types."""

    columns = resolve_features(feats, real)
    num_columns = len(columns)

    fig = plot.plot_histograms(real, feats)
    axes = fig.axes
    num_axes = len(axes)

    assert isinstance(fig, plt.Figure)

    if num_columns % 2 == 0:
        assert num_axes == num_columns
    else:
        assert num_axes == num_columns + 1
        assert not axes[-1].has_data()

    for col, ax in zip(columns, axes):

        assert isinstance(ax, plt.Subplot)
        assert ax.get_xlabel() == col
        assert ax.get_ylabel().lower() == "count"

        column = real[col]
        xmin, xmax = ax.get_xlim()

        if pd.api.types.is_numeric_dtype(column):
            assert xmin < min(column) and xmax > max(column)
        else:
            assert (xmin, xmax) == (-0.5, column.nunique() - 0.5)


@given(joint_params())
@plot_settings
def test_plot_joint(real, params):
    """Check that a joint plot can be created from two columns of a
    dataset."""

    x, y, groupby = params

    fig = plot.plot_joint(real, x, y, groupby=groupby)
    ax, axx, axy = fig.ax_joint, fig.ax_marg_x, fig.ax_marg_y

    assert isinstance(fig, JointGrid)

    assert isinstance(ax, plt.Subplot)
    assert ax.get_xlabel() == x
    assert ax.get_ylabel() == y
    if groupby is None:
        assert ax.get_legend() is None
    else:
        assert ax.get_legend().get_title().get_text() == groupby

    for name, col in zip(("x", "y"), (x, y)):
        column = real[col]
        vmin, vmax = sorted(getattr(ax, f"get_{name}lim")())

        if pd.api.types.is_numeric_dtype(column):
            assert vmin < min(column) and vmax > max(column)
        else:
            assert vmin, vmax == (-0.5, column.nunique() - 0.5)

    assert isinstance(axx, plt.Subplot)
    assert axx.get_xlabel() == x
    assert axx.get_ylabel().lower() == "count"
    assert axx.get_legend() is None

    assert isinstance(axy, plt.Subplot)
    assert axy.get_xlabel().lower() == "count"
    assert axy.get_ylabel() == y
    assert axy.get_legend() is None


@given(joint_params())
@plot_settings
def test_plot_histogram3d(real, params):
    """Check that a 3D histogram can be created from two columns of a
    dataset."""

    x, y, _ = params

    fig = plot.plot_histogram3d(real, x, y)
    ax = fig.axes[0]

    assert isinstance(fig, plt.Figure)

    assert (
        str(ax.__class__)
        == "<class 'matplotlib.axes._subplots.Axes3DSubplot'>"
    )
    assert ax.get_title() == "3D Histogram"
    assert ax.get_xlabel() == x
    assert ax.get_ylabel() == y
    assert ax.get_zlabel() == "$count$"

    for name, col in zip(("x", "y"), (x, y)):
        column = real[col]
        vmin, vmax = sorted(getattr(ax, f"get_{name}lim")())

        if pd.api.types.is_numeric_dtype(column):
            assert vmin < min(column) and vmax > max(column)
        else:
            assert vmin, vmax == (-0.5, column.nunique() - 0.5)


@given(method=st.sampled_from(("pearson", "spearman", "cramers_v")))
@plot_settings
def test_plot_correlation_methods(real, method):
    """Check that a correlation heatmap can be created for a single
    dataset with any of the implemented methods."""

    fig = plot.plot_correlation(real, method=method)
    ax, cbar = fig.axes

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Subplot)
    assert isinstance(cbar, plt.Subplot)

    assert ax.get_title() == "DataFrame 1 Correlation"
    assert cbar.get_label() == "<colorbar>"

    if method == "cramers_v":
        columns = list(real.select_dtypes(exclude="number").columns)
    else:
        columns = list(real.select_dtypes(include="number").columns)

    for col, xlab, ylab in zip(
        columns, ax.get_xticklabels(), ax.get_yticklabels()
    ):
        assert {col, xlab.get_text(), ylab.get_text()} == {col}


@given(method=st.sampled_from(("pearson", "spearman", "cramers_v")))
@plot_settings
def test_plot_correlation_method_errors(real, method):
    """Check that a ValueError is raised if only categorical columns are
    passed with a numeric correlation method or only numeric columns
    with `cramers_v`. Also catches the `if` statement for when `feats`
    is a single column."""

    if method == "cramers_v":
        column, match = "age", "^No categorical columns"
    else:
        column, match = "blood_type", "^No numeric columns"

    with pytest.raises(ValueError, match=match):
        _ = plot.plot_correlation(real, feats=column, method=method)


@given(method=st.sampled_from(("pearson", "spearman", "cramers_v")))
@plot_settings
def test_plot_correlation_two_datasets(real, synth, method):
    """Check that an array of correlation heatmaps can be produced when
    two dataframes are passed."""

    fig = plot.plot_correlation(real, synth, method=method)
    rax, sax, rbar, sbar = fig.axes

    assert isinstance(fig, plt.Figure)
    assert all(isinstance(ax, plt.Subplot) for ax in fig.axes)
    assert all(bar.get_label() == "<colorbar>" for bar in (rbar, sbar))

    if method == "cramers_v":
        columns = list(real.select_dtypes(exclude="number").columns)
    else:
        columns = list(real.select_dtypes(include="number").columns)

    for i, ax in enumerate((rax, sax)):
        assert ax.get_title() == f"DataFrame {i + 1} Correlation"
        for col, xlab, ylab in zip(
            columns, ax.get_xticklabels(), ax.get_yticklabels()
        ):
            assert {col, xlab.get_text(), ylab.get_text()} == {col}


@given(method=st.sampled_from(("pearson", "spearman", "cramers_v")))
@plot_settings
def test_plot_correlation_difference(real, synth, method):
    """Check that an additional heatmap can be produced for the
    absolute difference between two correlation matrices."""

    fig = plot.plot_correlation(real, synth, plot_diff=True, method=method)
    rax, sax, dax, empty, rbar, sbar, dbar = fig.axes

    assert isinstance(fig, plt.Figure)
    assert all(isinstance(ax, plt.Subplot) for ax in fig.axes)
    assert not empty.has_data()
    assert all(bar.get_label() == "<colorbar>" for bar in (rbar, sbar, dbar))

    dmin, dmax = dbar.get_ylim()
    assert dmin == 0 and dmax <= abs(
        max(rbar.get_ylim()) - min(sbar.get_ylim())
    )

    if method == "cramers_v":
        columns = list(real.select_dtypes(exclude="number").columns)
    else:
        columns = list(real.select_dtypes(include="number").columns)

    for i, ax in enumerate((rax, sax, dax)):
        if ax is dax:
            assert ax.get_title() == "Correlation Difference"
        else:
            assert ax.get_title() == f"DataFrame {i + 1} Correlation"

        for col, xlab, ylab in zip(
            columns, ax.get_xticklabels(), ax.get_yticklabels()
        ):
            assert {col, xlab.get_text(), ylab.get_text()} == {col}


@given(params=joint_params())
@plot_settings
def test_plot_crosstab(real, synth, params):
    """Check that an array of heatmaps can be produced correctly."""

    x, y, _ = params
    fig = plot.plot_crosstab(real, synth, x, y)
    rax, sax, cbar = fig.axes

    assert isinstance(fig, plt.Figure)
    assert isinstance(cbar, plt.Axes)
    assert rax.get_title() == "REAL" and sax.get_title() == "SYNTH"
    assert cbar.get_label() == "<colorbar>"

    for ax, data in zip((rax, sax), (real, synth)):
        assert isinstance(ax, plt.Subplot)
        assert ax.get_xlabel() == x
        assert ax.get_ylabel() == y

        total_count = sum(ax.collections[0]._A)
        binning_error = sum(
            pd.api.types.is_numeric_dtype(data[col]) for col in (x, y)
        )
        nrows = len(data)
        assert total_count in range(nrows - binning_error, nrows + 1)


@given(
    columns=st.lists(
        st.sampled_from(("age", "height", "weight")),
        unique=True,
        min_size=2,
        max_size=2,
    )
)
@plot_settings
def test_plot_crosstab_error(real, synth, columns):
    """Check that a TypeError is thrown when neither binning method is
    specified for a crosstab plot of two numeric columns."""

    with pytest.raises(
        TypeError, match="^`x_bins` and `y_bins` must not be None$"
    ):
        x, y = columns
        _ = plot.plot_crosstab(real, synth, x, y, x_bins=None, y_bins=None)


@given(
    params=joint_params(),
    cmap=st.sampled_from(("viridis", "ch:s=.25,rot=-.25", "light:coral")),
)
@plot_settings
def test_plot_crosstab_colour_palette(real, synth, params, cmap):
    """Check that a colour palette (map) can be passed to a crosstab
    plot correctly, following through to each heatmap and the colour
    bar.

    This test serves to cover this issue (#1) in the repository:
        https://github.com/datasciencecampus/synthgauge/issues/1
    """

    x, y, _ = params
    fig = plot.plot_crosstab(real, synth, x, y, cmap=cmap)
    rax, sax, cbar = fig.axes

    assert rax.collections[0].cmap is sax.collections[0].cmap
    assert rax.collections[0].cmap is cbar.collections[-1].cmap
