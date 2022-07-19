"""Functions for visually evaluating synthetic data."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.dtypes.common import is_numeric_dtype

from .metrics.correlation import _pairwise_cramers_v
from .metrics.density import _feature_density_diff
from .utils import cat_encode

sns.set_theme()


def plot_histograms(df, feats=None, groupby=None, figcols=2, figsize=None):
    """Plot feature distributions.

    Plot a histogram (or countplot for categorical data) for each
    feature. Where multiple features are provided a grid will be
    generated to store all the plots.

    Optionally, a groupby feature can be specified to apply a grouping
    prior to calculating the distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the feature(s) to be plotted.
    feats : list of str or None, default None
        Features in to plot. If `None` (default), all features are used.
    groupby : str, optional
        Feature on which to group data.
    figcols : int, default 2
        Number of columns to use in the figure.
    figsize : tuple of float, optional
        Size of figure in inches `(width, height)`. Defaults to
        `matplotlib.pyplot.rcParams["figure.figsize"]`.

    Returns
    -------
    matplotlib.figure.Figure
    """

    feats = feats or df.columns

    n_rows = int(np.ceil(len(feats) / figcols))
    fig, axes = plt.subplots(n_rows, figcols, figsize=figsize)

    for feat, ax in zip(feats, axes.ravel()):
        plotter = sns.histplot if is_numeric_dtype(df[feat]) else sns.countplot
        plotter(data=df, x=feat, ax=ax, hue=groupby)

    # Turn off axes with no data
    for ax in axes.ravel():
        if not ax.has_data():
            ax.set_visible(False)

    fig.tight_layout()

    return fig


def _order_categorical(feat):
    """Order a feature only if it is categorical.

    Parameters
    ----------
    feat : pd.Series
        The feature to be ordered.

    Returns
    -------
    pd.Series
        The ordered feature.
    """

    return feat.cat.as_ordered() if hasattr(feat, "cat") else feat


def plot_joint(
    df, x, y, groupby=None, x_bins="auto", y_bins="auto", figsize=6
):
    """Plot bivariate and univariate graphs.

    Convenience function that leverages `seaborn`. For more granular
    control, refer to `seaborn.JointGrid` and `seaborn.jointplot`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature(s) to plot.
    x : str
        Feature to plot on the x-axis and -margin.
    y : str
        Feature to plot on the y-axis and -margin.
    groupby : str, optional
        Feature on which to group data.
    x_bins, y_bins : array_like or int or str, default "auto"
        Binning method for axis. If `array_like`, must be sequence of
        bin edges. If `int`, specifies the number of bins to use. If
        `str`, can be anything accepted by `numpy.histogram_bin_edges`.
        Defaults to `"auto"`.
    figsize: int, default 6
        Size of each side of the figure in inches (it will be square).
        Defaults to six inches.

    Returns
    -------
    seaborn.axisgrid.JointGrid
    """

    grid = sns.JointGrid(height=figsize)

    sns.histplot(
        data=df,
        x=_order_categorical(df[x]),
        y=_order_categorical(df[y]),
        hue=groupby,
        alpha=0.5,
        ax=grid.ax_joint,
    )

    # For margins can use countplot or hist depending on data type.
    # No legends are shown for these marginal plots.
    for side, feat, bins in zip(("x", "y"), (x, y), (x_bins, y_bins)):

        plot_kwargs = {
            side: feat,
            "ax": getattr(grid, f"ax_marg_{side}"),
            "hue": groupby,
        }

        if is_numeric_dtype(df[feat]):
            plotter = sns.histplot
            plot_kwargs["bins"] = bins
        else:
            plotter = sns.countplot

        ax = plotter(data=df, **plot_kwargs)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    return grid


def plot_histogram3d(df, x, y, x_bins="auto", y_bins="auto", figsize=None):
    """Plot 3D histogram of two features.

    This is similar to a 2D histogram plot with an extra axis added
    to display the count for each feature-wise pair as 3D bars.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature(s) to plot.
    x : str
        Feature to plot on the x-axis and -margin.
    y : str
        Feature to plot on the y-axis and -margin.
    x_bins, y_bins : array_like or int or str, default "auto"
        Binning method for axis. If the corresponding feature is
        categorical, the bins will be set to the cardinality of that
        feature. If `array_like`, must be sequence of bin edges. If
        `int`, specifies the number of bins to use. If `str`, can be
        anything accepted by `numpy.histogram_bin_edges`. Defaults to
        `"auto"`.
    figsize: tuple of float, optional
        Size of figure in inches `(width, height)`. Defaults to
        `matplotlib.pyplot.rcParams["figure.figsize"]`.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # Encode categorical data
    cat_feats = df.select_dtypes(include=("object", "category")).columns
    cat_labels = dict()

    if cat_feats.any():
        df, cat_labels = cat_encode(df, cat_feats, return_all=True)

    # Determine bins
    bins_xy = []
    for feat, bins in zip([x, y], [x_bins, y_bins]):
        bins = df[feat].nunique() if feat in cat_feats else bins
        bins_xy.append(np.histogram_bin_edges(df[feat], bins))

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")

    # Compute 2D histogram
    hist, xedges, yedges = np.histogram2d(df[x], df[y], bins=bins_xy)

    # Determine bar anchors and thickness
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()

    dx = (xedges[1] - xedges[0]) * 0.8
    dy = (yedges[1] - yedges[0]) * 0.8
    dz = hist.ravel()
    zpos = np.zeros_like(dz) + 0.1

    # Plot the 3D bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort="average")

    # Set title and labels
    ax.set_xlabel(x, fontsize=16)
    ax.set_ylabel(y, fontsize=16)
    ax.set_zlabel("$count$", fontsize=16, rotation=1)
    ax.set_title("3D Histogram")

    # Set tick lables for categorical features
    if x in cat_labels.keys():
        ax.set_xticks(xedges[1:])
        ax.set_xticklabels(cat_labels[x])
    if y in cat_labels.keys():
        ax.set_yticks(yedges[1:])
        ax.set_yticklabels(cat_labels[y])

    return fig


def plot_correlation(
    *dfs,
    feats=None,
    method="pearson",
    plot_diff=False,
    figcols=2,
    figsize=None,
    **kwargs,
):
    """Plot correlation between features in a dataframe.

    For each dataframe provided a subplot is generated showing a
    correlation heatmap of the features. For numeric features, the
    method can be anything supported by `pandas.DataFrame.corr`; for
    categorical or object feature types `"cramers_v"` must be specified.
    If the method does not match the data type, an error is raised.

    The `plot_diff` parameter will also include a difference plot, i.e.
    correlation difference between two dataframes. This is only used
    when two dataframes are provided.

    Parameters
    ----------
    *dfs : pandas.DataFrame
        Any number of dataframes to plot.
    feats : list of str or None, default None
        Features to plot. Must be present in all elements of `dfs`.
        If `None` (default), uses features common to all dataframes.
    method : {"pearson", "spearman", "cramers_v"}, default "pearson"
        Correlation method. See `pandas.DataFrame.corr` for more details
        on `"pearson"` and `"spearman"`. When `"cramers_v"` is
        specified, correlation is calculated using
        `synthgauge.metrics.correlation.cramers_v`.
    plot_diff : bool, default False
        If `True` and exactly two dataframes are provided, will also
        plot a heatmap of the absolute differences between the
        respective datasets' correlations.
    figcols : int, default 2
        Number of columns to use in the figure. Only used when `feats`
        contains more than one feature.
    figsize : tuple of float, optional
        Size of figure in inches `(width, height)`. Defaults to
        `matplotlib.pyplot.rcParams["figure.figsize"]`.
    **kwargs : dict, optional
        Any other keyword arguments to be passed to `seaborn.heatmap`.
        For example `annot=True` will turn on cell annotations. See
        documentation for more examples.

    Raises
    ------
    ValueError
        If `method` does not match the data type(s) of `feats`.

    Returns
    -------
    matplotlib.figure.Figure
    """

    feats = feats or list(set.intersection(*(set(df.columns) for df in dfs)))

    corr_results = []
    for df in dfs:

        if method.lower() in ("pearson", "spearman"):
            data = df[feats].select_dtypes(include="number")
            if len(data.columns) == 0:
                raise ValueError(
                    f"No numeric columns available for method: {method}"
                )

            corr_results.append(
                data.corr(method=method)
                .dropna(axis=0, how="all")
                .dropna(axis=1, how="all")
            )

        if method.lower() == "cramers_v":
            data = df[feats].select_dtypes(include=["object", "category"])
            if len(data.columns) == 0:
                raise ValueError(
                    f"No categorical columns available for method: {method}"
                )

            corr_results.append(
                _pairwise_cramers_v(data)
                .dropna(axis=0, how="all")
                .dropna(axis=1, how="all")
            )

    # Get min and max to set consistant colourbar
    corr_values = np.array(corr_results)
    vmin = corr_values.min()
    vmax = corr_values.max()

    # For now only perform diff if 2 DataFrames are given; no more.
    # TODO: Allow all diff permutations?
    if len(corr_results) == 2 and plot_diff:
        corr_diff = np.abs(corr_results[0] - corr_results[1])
        corr_results.append(corr_diff)

    n_subplots = len(corr_results)
    ncols = 1 if n_subplots == 1 else figcols
    nrows = int(np.ceil(n_subplots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for ax_num, (ax, corr) in enumerate(
        zip(np.array(axes).ravel(), corr_results)
    ):
        if ax_num == n_subplots - 1 and plot_diff:
            sp_title = "Correlation Difference"
            # Ignore vmin and vmax for this plot as scale will be
            # different to the others
            vmin = None
            vmax = None
        else:
            sp_title = f"DataFrame {ax_num + 1} Correlation"

        sns.heatmap(corr, ax=ax, vmin=vmin, vmax=vmax, **kwargs)

        ax.set_title(sp_title)

    # Turn off axes with no data
    for ax in np.array(axes).ravel():
        if not ax.has_data():
            ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_crosstab(
    real,
    synth,
    x,
    y,
    x_bins="auto",
    y_bins="auto",
    figsize=None,
    cmap="rocket",
    **kwargs,
):
    """Plot cross-tabulation heatmap for two features.

    The two-feature crosstab calculation is performed and plotted as a
    heatmap. One heatmap is shown for the real data and one for the
    synthetic. Numeric features are discretised using the `*_bins`
    arguments.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    x : str
        Feature to plot on the x-axis and -margin.
    y : str
        Feature to plot on the y-axis and -margin.
    x_bins, y_bins : array_like or int or str, default "auto"
        Binning method for axis. If `array_like`, must be sequence of
        bin edges. If `int`, specifies the number of bins to use. If
        `str`, can be anything accepted by `numpy.histogram_bin_edges`.
        Defaults to `"auto"`. Only used for numeric features.
    figsize : tuple of float, optional
        Size of figure in inches `(width, height)`. Defaults to
        `matplotlib.pyplot.rcParams["figure.figsize"]`.
    cmap : str, default "rocket"
        Palette name for heatmap and colour bar. See the documentation
        for `seaborn.color_palette` on available palette formats.
        Defaults to `"rocket"`.
    **kwargs : dict, optional
        Any other keyword arguments to be passed to `seaborn.heatmap`.
        For example, `annot=True` will turn on cell annotations. See
        documentation for more examples.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # Collect x and y values
    real_x, real_y = real[x], real[y]
    synth_x, synth_y = synth[x], synth[y]
    all_x = pd.concat((real_x, synth_x))
    all_y = pd.concat((real_y, synth_y))

    # Discretise numeric features
    if is_numeric_dtype(all_x):
        x_bins = np.histogram_bin_edges(all_x.dropna(), x_bins)
        real_x = pd.cut(real_x, x_bins)
        synth_x = pd.cut(synth_x, x_bins)

    if is_numeric_dtype(all_y):
        y_bins = np.histogram_bin_edges(all_y.dropna(), y_bins)
        real_y = pd.cut(real_y, y_bins)
        synth_y = pd.cut(synth_y, y_bins)

    freq_real = pd.crosstab(real_x, real_y, dropna=False)
    freq_synth = pd.crosstab(synth_x, synth_y, dropna=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Use same scale for real and synth
    vmin = min(freq_real.values.min(), freq_synth.values.min())
    vmax = max(freq_real.values.max(), freq_synth.values.max())

    cmap = sns.color_palette(cmap, as_cmap=True)
    mpbl = mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), cmap=cmap)

    for freq, ax, title in zip(
        (freq_real, freq_synth), axes, ("REAL", "SYNTH")
    ):
        sns.heatmap(
            freq.T,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar=False,
            ax=ax,
            **kwargs,
        )
        ax.set_title(title)

    fig.colorbar(mpbl, ax=axes, shrink=0.8)

    return fig


def plot_qq(real, synth, feature, n_quantiles=None, figsize=None):
    """Generate a Q-Q plot for a feature of real and synthetic data.

    Quantile-quantile (Q-Q) plots are used to visualise two sets of
    numeric data to see if they are generated from the same
    distribution.

    In this case, it is used to provide some insight into the
    feature distributions for the synthetic and real data. If the
    scatter plot shows a straight line, then it can be inferred that the
    two distributions are similar and therefore the synthetically
    generated data follows the same distribution as the real data.

    See `Q-Q Plot <https://en.wikipedia.org/wiki/Q-Q_plot>`_ for more
    information.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        Feature to plot. Must be in `real` and `synth`.
    n_quantiles : int or None, default None
        Number of quantiles to calculate. If `None` (default), uses the
        length of `real`.
    figsize: tuple of float, optional
        Size of figure in inches `(width, height)`. Defaults to
        `matplotlib.pyplot.rcParams["figure.figsize"]`.

    Raises
    ------
    TypeError
        If `feature` is not a numeric data type feature.

    Returns
    -------
    matplotlib.figure.Figure
    """

    dtype = real[feature].dtype
    if not is_numeric_dtype(dtype):
        raise TypeError(f"The feature must be numeric not of type: {dtype}")

    n_quantiles = n_quantiles or len(real)

    qs = np.linspace(0, 1, n_quantiles)
    x = np.quantile(real[feature], qs)
    y = np.quantile(synth[feature], qs)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(x, y)
    ax.set_xlabel("real data quantiles")
    ax.set_ylabel("synth data quantiles")
    ax.set_title(f"Q-Q Plot for {feature}")

    # Plot X = Y
    min_xy = min(x.min(), y.min())
    max_xy = max(x.max(), y.max())
    ax.plot([min_xy, max_xy], [min_xy, max_xy])

    return fig


def plot_feat_density_diff(
    real, synth, feats=None, feat_bins=10, diff_bins=10, figsize=None
):
    """Plot real and synth feature density differences.

    For each feature, the density difference between `real` and `synth`
    is calculated using `metrics.density._feature_density_diff`.

    If a single feature is provided in `feats`, the plot shows the raw
    density differences for each bin in that feature.

    Where multiple features are provided, the density differences are
    pd.concatenated into a flattened array and a histogram plotted. The
    histogram represents the distribution of differences in densities
    across all features and bins.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feats : list of str or None, default None
        Features used to compute the densities. If `None` (default), all
        common features are used.
    feat_bins : str or int, default 10
        Bins to use for computing the feature densities. This value is
        passed to `numpy.histogram_bin_edges` so can be any value
        accepted by that function. By default, uses 10 bins.
    diff_bins : str or int, default 10
        Bins to use when computing the multiple-feature difference
        histogram. This value is passed to `numpy.histogram_bin_edges`
        so can be any value accepted by that function. By default, uses
        10 bins.

    Returns
    -------
    matplotlib.figure.Figure
    """

    feats = feats or real.columns.intersection(synth.columns)

    if len(feats) == 1:
        feature = feats[0]
        diff_hist, diff_edges = _feature_density_diff(
            real, synth, feature, feat_bins
        )
        xlabel = f"{feature} Binned"
        ylabel = "Density Difference"
        title = f"Feature Density Difference for {feature}"

    else:
        # TODO: option to have different bins for each feature
        diffs = [
            _feature_density_diff(real, synth, feat, feat_bins)[0]
            for feat in feats
        ]

        diff_hist, diff_edges = np.histogram(
            np.concatenate(diffs), bins=diff_bins
        )

        xlabel = "Difference Bins"
        ylabel = "Count"
        title = "Histogram of Density Differences"

    fig, ax = plt.subplots(figsize=figsize)

    # default bar width is too large so use scaled bin size
    bar_width = (diff_edges[1] - diff_edges[0]) * 0.8
    ax.bar(diff_edges[:-1], diff_hist, align="edge", width=bar_width)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig
