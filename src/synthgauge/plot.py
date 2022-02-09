from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pandas import crosstab, cut, DataFrame, Index
from pandas.core.dtypes.common import (is_categorical_dtype, is_numeric_dtype,
                                       is_object_dtype)
import seaborn as sns
from .utils import cat_encode, feature_density_diff
from .metrics.correlation import cramers_v

sns.set_theme()


def suggest_label_rotation(ax, axis='x', char_lim=5):
    """Advise if labels should be rotated.

    For the specifed axis the maximum characters for the
    labels is calculated. If this is above `char_lim` the function
    will return `True` to suggest that labels should be rotated

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axes to analyse.
    axis: str, optional
        Default is 'x'.
    char_lim: int, optional
        Default is `5`.

    Returns
    -------
    bool
        True if labels should be rotated, otherwise False.

    """
    tick_func = f'get_{axis.lower()}ticklabels'
    labels = getattr(ax, tick_func)()
    max_chars = max([len(label.get_text()) for label in labels])
    return max_chars > char_lim


def plot_histograms(df, feats=None, groupby=None, figcols=2, figsize=None):
    """Plot feature distributions.

    Plot a histogram (or countplot for categorical data) for each feature.
    Where multiple features are provided a grid will be generated to store all
    plots.

    Optionally, a groupby feature can be specified to apply a grouping prior
    to calculating the distribution

    Parameters
    ----------
    df: pandas.DataFrame
    feats: str or list of str
        Feature(s) to plot. Must be column(s) in `df`.
    groupby: str, default=None
        Feature on which to group data. Default value of ``None`` means no
        grouping is used.
    figcols: int, default=2
        Number of columns to use in figure. Only used when `feats` contains
        more than one feature.
    figsize: tuple of float, optional
        Size of figure in inches (W,H). Defaults to
        ``rcParams["figure.figsize"](default: [6.4, 4.8])``.

    Returns
    -------
    matplotlib.figure.Figure

    """

    if isinstance(feats, Index):
        feats = feats
    elif isinstance(feats, str):
        feats = [feats]
    else:
        feats = feats or df.columns

    n_rows = int(np.ceil(len(feats) / figcols))
    fig, axes = plt.subplots(n_rows, figcols, figsize=figsize)

    for ft, ax in zip(feats, axes.ravel()):
        if is_categorical_dtype(df[ft]) or is_object_dtype(df[ft]):
            dist_func = sns.countplot
        elif is_numeric_dtype(df[ft]):
            dist_func = sns.histplot

        dist_func(data=df, x=ft, ax=ax, hue=groupby)

    # Turn off axes with no data
    for ax in axes.ravel():
        if not ax.has_data():
            ax.set_visible(False)

    fig.tight_layout()

    return fig


def plot_joint(df, x, y, groupby=None, x_bins='auto', y_bins='auto',
               figsize=None):
    """Plot bivariate and univariate graphs.

    Convenience function that leverages seaborn. For more granular control
    refer to ``seaborn.JointGrid`` and ``seaborn.jointplot``.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the feature(s) to plot.
    x: str
        Feature to plot on x axis and and margin. Must be column in `df`.
    y: str
        Feature to plot on y axis and and margin. Must be column in `df`.
    groupby: str, default=None
        Feature on which to group data. Default value of ``None`` means no
        grouping is used.
    x_bins, y_bins: array_like or int or str, default='auto'
        If ``array_like`` must be sequence of bin edges. If ``int``, specifies
        the number of bins to use. If ``str``, specifies the method used to
        calculate the optimal bin width. Default=``auto``. For all possible
        values see ``numpy.histogram_bin_edges``.
    figsize: int, default=None
        Size of each side of the figure in inches (it will be square). Default
        value of ``None`` will use default seaborn setting.

    Returns
    -------
    seaborn.axisgrid.JointGrid

    """
    figsize = figsize or 6
    g = sns.JointGrid(height=figsize)

    # If a feature is categorical it must be used 'as_ordered' for plotting a
    # histogram
    def order(ft): return ft.cat.as_ordered() if hasattr(ft, 'cat') else ft
    sns.histplot(data=df, x=order(df[x]), y=order(df[y]), hue=groupby,
                 alpha=0.5, ax=g.ax_joint)

    # For margins can use countplot or hist depending on data type.
    # No legends are shown for these marginal plots.
    if is_categorical_dtype(df[x]) or is_object_dtype(df[x]):
        ax = sns.countplot(data=df, x=x, hue=groupby, ax=g.ax_marg_x)
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()
    else:
        sns.histplot(data=df, x=x, hue=groupby, bins=x_bins, ax=g.ax_marg_x,
                     legend=False)

    if is_categorical_dtype(df[y]) or is_object_dtype(df[y]):
        ax = sns.countplot(data=df, y=y, hue=groupby, ax=g.ax_marg_y)
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()
    else:
        sns.histplot(data=df, y=y, hue=groupby, bins=y_bins, ax=g.ax_marg_y,
                     legend=False)

    return g


def plot_histogram3d(df, x, y, x_bins='auto', y_bins='auto', figsize=None):
    """Plot 3D histogram of two features.

    This is similar to a 2D histogram plot with an extra axis added
    to display the count for each featurewise pair as 3D bars.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the feature(s) to plot.
    x: str
        Feature to plot on x axis and and margin. Must be column in `df`.
    y: str
        Feature to plot on y axis and and margin. Must be column in `df`.
    x_bins, y_bins: array_like or int or str, default='auto'
        If ``array_like`` must be sequence of bin edges. If ``int``, specifies
        the number of bins to use. If ``str``, specifies the method used to
        calculate the optimal bin width. Default= ``auto``. For all possible
        values see ``numpy.histogram_bin_edges``. If `x` or `y` is categorical
        the bins will be set to the cardinality of the feature.
    figsize: int, default=None
        Size of each side of the figure in inches (it will be square). Default
        value of ``None`` will use default plot settings.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Encode categorical data
    cat_feats = [ft_name for ft_name, ft_series in df.iteritems()
                 if is_categorical_dtype(ft_series)
                 or is_object_dtype(ft_series)]
    cat_labels = dict()

    if len(cat_feats) > 0:
        df, cat_labels = cat_encode(df, cat_feats, return_all=True)

    # Determine bins
    bins_xy = []
    for ft, ft_bins in zip([x, y], [x_bins, y_bins]):
        if ft in cat_feats:
            # If categorical set bins to cardinality
            bins_xy.append(np.histogram_bin_edges(df[ft], df[ft].nunique()))
        else:
            # otherwise compute bins as specified
            bins_xy.append(np.histogram_bin_edges(df[ft], ft_bins))

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')

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
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    # Set title and labels
    ax.set_xlabel(x, fontsize=16)
    ax.set_ylabel(y, fontsize=16)
    ax.set_zlabel('$count$', fontsize=16, rotation=1)
    ax.set_title('3D Histogram')

    # Set tick lables for categorical features
    if x in cat_labels.keys():
        ax.set_xticks(xedges[1:])
        ax.set_xticklabels(cat_labels[x])
    if y in cat_labels.keys():
        ax.set_yticks(yedges[1:])
        ax.set_yticklabels(cat_labels[y])

    return fig


def plot_correlation(*df, feats=None, method='pearson', plot_diff=False,
                     figcols=2, figsize=None, **kwargs):
    """Plot correlation between features in a DataFrame.

    For each dataframe provided a subplot is generated showing a
    correlation heatmap of the features. For numeric features the method can
    be any method supported by ``pandas.DataFrame.corr``; for categorical or
    object feature types ``'cramers_v'`` must be specified. If the method does
    not match the data type an error is raised.

    The `plot_diff` parameter will also include a difference plot i.e.
    correlation difference between two dataframes. This is only used when
    two dataframes are provided.

    Parameters
    ----------
    *df: pandas.DataFrame
        Any number of DataFrame objects to plot.
    feats: str or list of str
        Feature(s) to plot. Must be column(s) in `df`.
    method: {'pearson'|'spearman'|'cramers_v'}, default='pearson'
        Correlation method. See ``pandas.DataFrame.corr`` for more details on
        ``'pearson'`` and ``'spearman'``. When ``'cramers_v'`` is specified,
        correlation is calculated using
        ``synthgauge.metrics.correlation.cramers_v``.
    plot_diff: bool, default=False
        If True and exactly two data frames are provided will also plot
        a heatmap of the absolute differences between the respective datasets'
        correlations.
    figcols: int, default=2
        Number of columns to use in figure. Only used when `feats` contains
        more than one feature.
    figsize: tuple of float, optional
        Size of figure in inches (W,H). Defaults to
        ``rcParams["figure.figsize"](default: [6.4, 4.8])``.
    **kwargs: dict, optional
        Any other keyword arguments to be passed to ``seaborn.heatmap``. For
        example ``annot=True`` will turn on cell annotations. See documentation
        for more examples.

    Returns
    -------
    matplotlib.figure.Figure
    """
    def method_cramers_v(df):
        """Compute cramers_v

        Compute pairwise cramers_v correlation for the entire dataframe.
        """
        results = np.array([])
        for x, y in product(df.columns, repeat=2):
            results = np.append(results, cramers_v(df[x], df[y]))

        size = np.sqrt(results.size).astype(int)
        results = results.reshape((size, size))

        return DataFrame(results, index=df.columns, columns=df.columns)

    if isinstance(feats, str):
        feats = [feats]

    corr_results = list()

    for d in df:
        # If no features specified use all columns
        if feats is None:
            feats = d.columns

        if method.lower() in ['pearson', 'spearman']:
            data = d[feats].select_dtypes(include='number')
            if len(data.columns) == 0:
                raise ValueError("No numeric columns available for method: "
                                 f"{method}")

            corr_results.append(data.corr(method=method).dropna(
                0, 'all').dropna(1, 'all'))

        elif method.lower() == 'cramers_v':
            data = d[feats].select_dtypes(include=['object', 'category'])
            if len(data.columns) == 0:
                raise ValueError("No categorical columns available for "
                                 f"method: {method}")

            corr_results.append(method_cramers_v(data).dropna(
                0, 'all').dropna(1, 'all'))

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
    nrows = int(np.ceil(n_subplots/ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for ax_num, (ax, corr) in enumerate(zip(np.array(axes).ravel(),
                                            corr_results)):
        if ax_num == n_subplots - 1 and plot_diff:
            sp_title = 'Correlation Difference'
            # Ignore vmin and vmax for this plot as scale will be different
            # to the others
            vmin = None
            vmax = None
        else:
            sp_title = f"DataFrame {ax_num+1} Correlation"

        sns.heatmap(corr, ax=ax, vmin=vmin, vmax=vmax, **kwargs)

        ax.set_title(sp_title)

    # Turn off axes with no data
    for ax in np.array(axes).ravel():
        if not ax.has_data():
            ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_crosstab(real, synth, x, y, x_bins='auto', y_bins='auto',
                  figsize=None, **kwargs):
    """Plot crosstab heatmap for two features.

    The two-feature crosstab calculation is performed and plotted as a heatmap.
    One heatmap is shown for the `real` data and one for the `synthetic`.
    Numeric features are discretised using the corresponding bins argument.

    Parameters
    ----------
    real: pandas.DataFrame
        DataFrame containing the real data.
    synth: pandas.DataFrame
        DataFrame containing the sythetic data.
    x: str
        Feature to plot on x axis. Must be column in `df`.
    y: str
        Feature to plot on y axis. Must be column in `df`.
    x_bins, y_bins: array_like or int or str, default='auto'
        If ``array_like`` must be sequence of bin edges. If ``int``, specifies
        the number of bins to use. If ``str``, specifies the method used to
        calculate the optimal bin width. Default= ``auto``. For all possible
        values see ``numpy.histogram_bin_edges``.
        Only used for numeric features.
    figsize: tuple of float, optional
        Size of figure in inches (W,H). Defaults to
        ``rcParams["figure.figsize"](default: [6.4, 4.8])``.
    **kwargs: dict, optional
        Any other keyword arguments to be passed to ``seaborn.heatmap``. For
        example ``annot=True`` will turn on cell annotations. See documentation
        for more examples.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Collect x and y values
    real_x, real_y = real[x], real[y]
    synth_x, synth_y = synth[x], synth[y]
    all_x = real_x.append(synth_x)
    all_y = real_y.append(synth_y)

    # Discretise numeric features
    try:
        if is_numeric_dtype(all_x):
            x_bins = np.histogram_bin_edges(all_x.dropna(), x_bins)
            real_x = cut(real_x, x_bins)
            synth_x = cut(synth_x, x_bins)

        if is_numeric_dtype(all_y):
            y_bins = np.histogram_bin_edges(all_y.dropna(), y_bins)
            real_y = cut(real_y, y_bins)
            synth_y = cut(synth_y, y_bins)
    except TypeError:
        raise TypeError('`x_bins` and `y_bins` must not be None')

    freq_real = crosstab(real_x, real_y, dropna=False)
    freq_synth = crosstab(synth_x, synth_y, dropna=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Use same scale for real and synth
    vmin = min(freq_real.values.min(), freq_synth.values.min())
    vmax = max(freq_real.values.max(), freq_synth.values.max())
    mpbl = mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax),
                                 cmap=sns.color_palette("rocket",
                                 as_cmap=True))

    sns.heatmap(freq_real, ax=axes[0], vmin=vmin, vmax=vmax, cbar=False,
                **kwargs)
    axes[0].set_title('REAL')

    sns.heatmap(freq_synth, ax=axes[1], vmin=vmin, vmax=vmax, cbar=False,
                **kwargs)
    axes[1].set_title('SYNTH')

    fig.colorbar(mpbl, ax=axes, shrink=0.8)

    return fig


def plot_qq(real, synth, feature, n_quantiles=None, figsize=None):
    """Plot a Q-Q plot.

    Generate a Quantile-Quantile plot of `feature` for the real and
    synthetic data. Q-Q plots are used to visualise two datasets
    to see if they are generated from the same distribution.

    In this case, it is used to provide some insight into the distribution
    of feature `feature` in the synthetic data versus the real data. If the
    plot is a straight line then it can be inferred that the two
    distributions are similar and therefore the synthetically generated
    data follows the same distribution as the real data.

    See `Q-Q Plot <https://en.wikipedia.org/wiki/Q-Q_plot>`_ for more
    information.

    Parameters
    ----------
    real: pandas.DataFrame
        DataFrame containing the real data.
    synth: pandas.DataFrame
        DataFrame containing the sythetic data.
    feature: str
        Feature to plot.
    n_quantiles: int, optional
        Number of quantiles to calculate. If ``None`` (the default) and
        ``real[feature]`` has the same length as ``synth[feature]`` then
        `n_quantiles` is set to the number of elements. If the lengths are
        different then n_quantiles is set to the length of ``real[feature]``.
    figsize: tuple of float, optional
        Size of figure in inches (W,H). Defaults to
        ``rcParams["figure.figsize"](default: [6.4, 4.8])``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if n_quantiles is None:
        if len(real[feature]) == len(synth[feature]):
            # Matching lengths do not require computing quantiles
            x = np.sort(real[feature])
            y = np.sort(synth[feature])
        else:
            # Determine n_quantiles from real data
            n_quantiles = len(real[feature])
            qs = np.linspace(0, 1, n_quantiles)
            x = np.quantile(real[feature], qs)
            y = np.quantile(synth[feature], qs)

    else:
        qs = np.linspace(0, 1, n_quantiles)
        x = np.quantile(real[feature], qs)
        y = np.quantile(synth[feature], qs)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(x, y)
    ax.set_xlabel('real data quantiles')
    ax.set_ylabel('synth data quantiles')
    ax.set_title(f'Q-Q Plot for "{feature}"')

    fig.canvas.draw()
    if suggest_label_rotation(ax):
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Plot X = Y
    min_xy = min(x.min(), y.min())
    max_xy = max(x.max(), y.max())
    ax.plot([min_xy, max_xy], [min_xy, max_xy])

    return fig


def plot_feat_density_diff(real, synth, feats=None, feat_bins=10,
                           diff_bins=10, figsize=None):
    """Plot real and synth feature density differences.

    For each feature the feature density difference between `real` and
    `synth` is calculated using ``synthgauge.utils.feature_density_diff``.

    If a single feature is provided the plot shows the raw density
    differences for each bin.

    Where multiple feature are provided, the density differences are
    concatenated into a flattened array and a histogram plotted. The
    histogram represents the distribution of differences in densities
    across all features and bins.

    Parameters
    ----------
    real: pandas.DataFrame
        DataFrame containing the real data.
    synth: pandas.DataFrame
        DataFrame containing the sythetic data.
    feats: str or list of str, optional.
        The features that will be used to compute the densities. By
        default all features in `real` will be used.
    feat_bins: str or int, optional
        Bins to use for computing the feature densities. This value is passed
        to ``numpy.histogram_bin_edges`` so can be any value accepted by
        that function. The default setting of ``10`` uses 10 bins.
    diff_bins: str or int, optional
        Bins to use for the difference histogram. Only used when
        multiple features are provided. This value is passed to
        ``numpy.histogram_bin_edges`` so can be any value accepted by
        that function. The default setting of ``10`` uses 10 bins.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes containing the plot.
    """
    if isinstance(feats, Index):
        feats = feats
    elif isinstance(feats, str):
        feats = [feats]
    else:
        feats = feats or real.columns

    if len(feats) == 1:
        diff_hist, diff_edges = feature_density_diff(real, synth, feats,
                                                     feat_bins)
        xlabel = f'{feats[0]} Binned'
        ylabel = 'Density Difference'
        title = f'Feature Density Difference for {feats[0]}'

    else:
        # TODO: option to have different bins for each feature
        diffs = [feature_density_diff(real, synth, f, feat_bins)[0]
                 for f in feats]

        diff_hist, diff_edges = np.histogram(np.concatenate(diffs),
                                             bins=diff_bins)

        xlabel = 'Difference Bins'
        ylabel = 'Count'
        title = 'Histogram of Density Differences'

    fig, ax = plt.subplots(figsize=figsize)

    # default bar width is too large so use scaled bin size
    bar_width = (diff_edges[1] - diff_edges[0]) * 0.8
    ax.bar(diff_edges[:-1], diff_hist, align='edge', width=bar_width)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax


if __name__ == '__main__':
    pass
