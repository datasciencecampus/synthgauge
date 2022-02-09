:py:mod:`synthgauge.plot`
=========================

.. py:module:: synthgauge.plot


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.plot.suggest_label_rotation
   synthgauge.plot.plot_histograms
   synthgauge.plot.plot_joint
   synthgauge.plot.plot_histogram3d
   synthgauge.plot.plot_correlation
   synthgauge.plot.plot_crosstab
   synthgauge.plot.plot_qq
   synthgauge.plot.plot_feat_density_diff



.. py:function:: suggest_label_rotation(ax, axis='x', char_lim=5)

   Advise if labels should be rotated.

   For the specifed axis the maximum characters for the
   labels is calculated. If this is above `char_lim` the function
   will return `True` to suggest that labels should be rotated

   :param ax: Axes to analyse.
   :type ax: matplotlib.axes.Axes
   :param axis: Default is 'x'.
   :type axis: str, optional
   :param char_lim: Default is `5`.
   :type char_lim: int, optional

   :returns: True if labels should be rotated, otherwise False.
   :rtype: bool


.. py:function:: plot_histograms(df, feats=None, groupby=None, figcols=2, figsize=None)

   Plot feature distributions.

   Plot a histogram (or countplot for categorical data) for each feature.
   Where multiple features are provided a grid will be generated to store all
   plots.

   Optionally, a groupby feature can be specified to apply a grouping prior
   to calculating the distribution

   :param df:
   :type df: pandas.DataFrame
   :param feats: Feature(s) to plot. Must be column(s) in `df`.
   :type feats: str or list of str
   :param groupby: Feature on which to group data. Default value of ``None`` means no
                   grouping is used.
   :type groupby: str, default=None
   :param figcols: Number of columns to use in figure. Only used when `feats` contains
                   more than one feature.
   :type figcols: int, default=2
   :param figsize: Size of figure in inches (W,H). Defaults to
                   ``rcParams["figure.figsize"](default: [6.4, 4.8])``.
   :type figsize: tuple of float, optional

   :returns:
   :rtype: matplotlib.figure.Figure


.. py:function:: plot_joint(df, x, y, groupby=None, x_bins='auto', y_bins='auto', figsize=None)

   Plot bivariate and univariate graphs.

   Convenience function that leverages seaborn. For more granular control
   refer to ``seaborn.JointGrid`` and ``seaborn.jointplot``.

   :param df: DataFrame containing the feature(s) to plot.
   :type df: pandas.DataFrame
   :param x: Feature to plot on x axis and and margin. Must be column in `df`.
   :type x: str
   :param y: Feature to plot on y axis and and margin. Must be column in `df`.
   :type y: str
   :param groupby: Feature on which to group data. Default value of ``None`` means no
                   grouping is used.
   :type groupby: str, default=None
   :param x_bins: If ``array_like`` must be sequence of bin edges. If ``int``, specifies
                  the number of bins to use. If ``str``, specifies the method used to
                  calculate the optimal bin width. Default=``auto``. For all possible
                  values see ``numpy.histogram_bin_edges``.
   :type x_bins: array_like or int or str, default='auto'
   :param y_bins: If ``array_like`` must be sequence of bin edges. If ``int``, specifies
                  the number of bins to use. If ``str``, specifies the method used to
                  calculate the optimal bin width. Default=``auto``. For all possible
                  values see ``numpy.histogram_bin_edges``.
   :type y_bins: array_like or int or str, default='auto'
   :param figsize: Size of each side of the figure in inches (it will be square). Default
                   value of ``None`` will use default seaborn setting.
   :type figsize: int, default=None

   :returns:
   :rtype: seaborn.axisgrid.JointGrid


.. py:function:: plot_histogram3d(df, x, y, x_bins='auto', y_bins='auto', figsize=None)

   Plot 3D histogram of two features.

   This is similar to a 2D histogram plot with an extra axis added
   to display the count for each featurewise pair as 3D bars.

   :param df: DataFrame containing the feature(s) to plot.
   :type df: pandas.DataFrame
   :param x: Feature to plot on x axis and and margin. Must be column in `df`.
   :type x: str
   :param y: Feature to plot on y axis and and margin. Must be column in `df`.
   :type y: str
   :param x_bins: If ``array_like`` must be sequence of bin edges. If ``int``, specifies
                  the number of bins to use. If ``str``, specifies the method used to
                  calculate the optimal bin width. Default= ``auto``. For all possible
                  values see ``numpy.histogram_bin_edges``. If `x` or `y` is categorical
                  the bins will be set to the cardinality of the feature.
   :type x_bins: array_like or int or str, default='auto'
   :param y_bins: If ``array_like`` must be sequence of bin edges. If ``int``, specifies
                  the number of bins to use. If ``str``, specifies the method used to
                  calculate the optimal bin width. Default= ``auto``. For all possible
                  values see ``numpy.histogram_bin_edges``. If `x` or `y` is categorical
                  the bins will be set to the cardinality of the feature.
   :type y_bins: array_like or int or str, default='auto'
   :param figsize: Size of each side of the figure in inches (it will be square). Default
                   value of ``None`` will use default plot settings.
   :type figsize: int, default=None

   :returns:
   :rtype: matplotlib.figure.Figure


.. py:function:: plot_correlation(*df, feats=None, method='pearson', plot_diff=False, figcols=2, figsize=None, **kwargs)

   Plot correlation between features in a DataFrame.

   For each dataframe provided a subplot is generated showing a
   correlation heatmap of the features. For numeric features the method can
   be any method supported by ``pandas.DataFrame.corr``; for categorical or
   object feature types ``'cramers_v'`` must be specified. If the method does
   not match the data type an error is raised.

   The `plot_diff` parameter will also include a difference plot i.e.
   correlation difference between two dataframes. This is only used when
   two dataframes are provided.

   :param \*df: Any number of DataFrame objects to plot.
   :type \*df: pandas.DataFrame
   :param feats: Feature(s) to plot. Must be column(s) in `df`.
   :type feats: str or list of str
   :param method: Correlation method. See ``pandas.DataFrame.corr`` for more details on
                  ``'pearson'`` and ``'spearman'``. When ``'cramers_v'`` is specified,
                  correlation is calculated using
                  ``synthgauge.metrics.correlation.cramers_v``.
   :type method: {'pearson'|'spearman'|'cramers_v'}, default='pearson'
   :param plot_diff: If True and exactly two data frames are provided will also plot
                     a heatmap of the absolute differences between the respective datasets'
                     correlations.
   :type plot_diff: bool, default=False
   :param figcols: Number of columns to use in figure. Only used when `feats` contains
                   more than one feature.
   :type figcols: int, default=2
   :param figsize: Size of figure in inches (W,H). Defaults to
                   ``rcParams["figure.figsize"](default: [6.4, 4.8])``.
   :type figsize: tuple of float, optional
   :param \*\*kwargs: Any other keyword arguments to be passed to ``seaborn.heatmap``. For
                      example ``annot=True`` will turn on cell annotations. See documentation
                      for more examples.
   :type \*\*kwargs: dict, optional

   :returns:
   :rtype: matplotlib.figure.Figure


.. py:function:: plot_crosstab(real, synth, x, y, x_bins='auto', y_bins='auto', figsize=None, **kwargs)

   Plot crosstab heatmap for two features.

   The two-feature crosstab calculation is performed and plotted as a heatmap.
   One heatmap is shown for the `real` data and one for the `synthetic`.
   Numeric features are discretised using the corresponding bins argument.

   :param real: DataFrame containing the real data.
   :type real: pandas.DataFrame
   :param synth: DataFrame containing the sythetic data.
   :type synth: pandas.DataFrame
   :param x: Feature to plot on x axis. Must be column in `df`.
   :type x: str
   :param y: Feature to plot on y axis. Must be column in `df`.
   :type y: str
   :param x_bins: If ``array_like`` must be sequence of bin edges. If ``int``, specifies
                  the number of bins to use. If ``str``, specifies the method used to
                  calculate the optimal bin width. Default= ``auto``. For all possible
                  values see ``numpy.histogram_bin_edges``.
                  Only used for numeric features.
   :type x_bins: array_like or int or str, default='auto'
   :param y_bins: If ``array_like`` must be sequence of bin edges. If ``int``, specifies
                  the number of bins to use. If ``str``, specifies the method used to
                  calculate the optimal bin width. Default= ``auto``. For all possible
                  values see ``numpy.histogram_bin_edges``.
                  Only used for numeric features.
   :type y_bins: array_like or int or str, default='auto'
   :param figsize: Size of figure in inches (W,H). Defaults to
                   ``rcParams["figure.figsize"](default: [6.4, 4.8])``.
   :type figsize: tuple of float, optional
   :param \*\*kwargs: Any other keyword arguments to be passed to ``seaborn.heatmap``. For
                      example ``annot=True`` will turn on cell annotations. See documentation
                      for more examples.
   :type \*\*kwargs: dict, optional

   :returns:
   :rtype: matplotlib.figure.Figure


.. py:function:: plot_qq(real, synth, feature, n_quantiles=None, figsize=None)

   Plot a Q-Q plot.

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

   :param real: DataFrame containing the real data.
   :type real: pandas.DataFrame
   :param synth: DataFrame containing the sythetic data.
   :type synth: pandas.DataFrame
   :param feature: Feature to plot.
   :type feature: str
   :param n_quantiles: Number of quantiles to calculate. If ``None`` (the default) and
                       ``real[feature]`` has the same length as ``synth[feature]`` then
                       `n_quantiles` is set to the number of elements. If the lengths are
                       different then n_quantiles is set to the length of ``real[feature]``.
   :type n_quantiles: int, optional
   :param figsize: Size of figure in inches (W,H). Defaults to
                   ``rcParams["figure.figsize"](default: [6.4, 4.8])``.
   :type figsize: tuple of float, optional

   :returns:
   :rtype: matplotlib.figure.Figure


.. py:function:: plot_feat_density_diff(real, synth, feats=None, feat_bins=10, diff_bins=10, figsize=None)

   Plot real and synth feature density differences.

   For each feature the feature density difference between `real` and
   `synth` is calculated using ``synthgauge.utils.feature_density_diff``.

   If a single feature is provided the plot shows the raw density
   differences for each bin.

   Where multiple feature are provided, the density differences are
   concatenated into a flattened array and a histogram plotted. The
   histogram represents the distribution of differences in densities
   across all features and bins.

   :param real: DataFrame containing the real data.
   :type real: pandas.DataFrame
   :param synth: DataFrame containing the sythetic data.
   :type synth: pandas.DataFrame
   :param feats: The features that will be used to compute the densities. By
                 default all features in `real` will be used.
   :type feats: str or list of str, optional.
   :param feat_bins: Bins to use for computing the feature densities. This value is passed
                     to ``numpy.histogram_bin_edges`` so can be any value accepted by
                     that function. The default setting of ``10`` uses 10 bins.
   :type feat_bins: str or int, optional
   :param diff_bins: Bins to use for the difference histogram. Only used when
                     multiple features are provided. This value is passed to
                     ``numpy.histogram_bin_edges`` so can be any value accepted by
                     that function. The default setting of ``10`` uses 10 bins.
   :type diff_bins: str or int, optional

   :returns: The matplotlib axes containing the plot.
   :rtype: matplotlib.axes._subplots.AxesSubplot


