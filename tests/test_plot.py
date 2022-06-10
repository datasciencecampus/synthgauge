import pytest
from matplotlib.figure import Figure
from seaborn.axisgrid import JointGrid

from synthgauge.datasets import make_blood_types_df
from synthgauge.plot import (
    plot_correlation,
    plot_crosstab,
    plot_histogram3d,
    plot_histograms,
    plot_joint,
)


@pytest.fixture
def make_real():
    """Real data."""
    return make_blood_types_df(0, 0)


@pytest.fixture
def make_synth():
    """Synthetic data that differs from the real data."""
    return make_blood_types_df(1, 0)


def test_plot_hist(make_real):
    """Check correct figure type returned with number of
    axes matching number of columns in the DataFrame.
    """
    fig = plot_histograms(make_real)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == len(make_real.columns)


@pytest.mark.parametrize(
    "x, y",
    [("age", "height"), ("age", "eye_colour"), ("eye_colour", "hair_colour")],
)
def test_plot_joint(make_real, x, y):
    """Check correct figure type returned with expected axes"""
    fig = plot_joint(make_real, x, y)

    # Check figure type
    assert isinstance(fig, JointGrid)

    # # Check axes labels are as expected
    assert fig.ax_joint.get_xlabel() == x
    assert fig.ax_joint.get_ylabel() == y


@pytest.mark.parametrize(
    "x, y",
    [("age", "height"), ("age", "eye_colour"), ("eye_colour", "hair_colour")],
)
def test_plot_hist3d_numeric(make_real, x, y):
    """Check correct figure type returned with expected axes"""
    fig = plot_histogram3d(make_real, x, y)

    # Check figure type
    assert isinstance(fig, Figure)

    # Check axes labels are as expected
    assert fig.axes[0].get_xlabel() == x
    assert fig.axes[0].get_ylabel() == y


@pytest.mark.parametrize(
    "feats, method, plot_diff",
    [
        (None, "pearson", False),
        (None, "pearson", True),
        (["age", "height"], "spearman", False),
        (["eye_colour", "hair_colour"], "cramers_v", False),
    ],
)
def test_plot_correlation(make_real, make_synth, feats, method, plot_diff):
    """Check correct figure type returned with correct number of axes."""
    fig = plot_correlation(
        make_real, make_synth, feats=feats, method=method, plot_diff=plot_diff
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4 if not plot_diff else 6

    # Get features from first axis
    ax0 = fig.axes[0]
    x_labels = [label.get_text() for label in ax0.get_xticklabels()]
    y_labels = [label.get_text() for label in ax0.get_xticklabels()]

    # Check feature lables match
    assert x_labels == y_labels

    # Check correct features used
    if feats is None:
        feats = make_real.columns
    if method in ["pearson", "spearman"]:
        expected_feats = (
            make_real[feats].select_dtypes(include="number").columns
        )
    elif method == "cramers_v":
        expected_feats = (
            make_real[feats].select_dtypes(exclude="number").columns
        )
    assert expected_feats.to_list() == x_labels


@pytest.mark.parametrize(
    "feats, method,",
    [
        (["age", "height"], "cramers_v"),
        (["eye_colour", "hair_colour"], "pearson"),
    ],
)
def test_plot_correlation_method(make_real, make_synth, feats, method):
    with pytest.raises(ValueError):
        plot_correlation(make_real, make_synth, feats=feats, method=method)


@pytest.mark.parametrize(
    "x, y, x_bins, y_bins",
    [
        ("age", "height", "auto", "auto"),
        ("eye_colour", "hair_colour", 5, 5),
        ("height", "eye_colour", 10, "auto"),
    ],
)
def test_plot_crosstab(make_real, make_synth, x, y, x_bins, y_bins):
    """ """
    fig = plot_crosstab(make_real, make_synth, x, y, x_bins, y_bins)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize(
    "x, y, x_bins, y_bins",
    [("age", "height", None, None), ("height", "eye_colour", None, "auto")],
)
def test_plot_crosstab_bins(make_real, make_synth, x, y, x_bins, y_bins):
    with pytest.raises(TypeError):
        plot_crosstab(make_real, make_synth, x, y, x_bins, y_bins)
