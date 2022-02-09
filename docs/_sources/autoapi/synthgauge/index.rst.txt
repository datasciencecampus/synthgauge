:py:mod:`synthgauge`
====================

.. py:module:: synthgauge


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   metrics/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   datasets/index.rst
   evaluate/index.rst
   plot/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   synthgauge.Evaluator




Attributes
~~~~~~~~~~

.. autoapisummary::

   synthgauge.__version__


.. py:class:: Evaluator(real, synth, handle_nans='drop')

   Bases: :py:obj:`object`

   

   .. py:method:: _feat_valid(self, feat)

      Checks if feature is valid common feature



   .. py:method:: describe_numeric(self)

      Summarise numerical features

      This function uses `pandas.DataFrame.describe` to calculate
      summary statistics for each numeric real and synthetic feature.

      :returns: Descriptive statistics for each numeric feature.
      :rtype: pandas.DataFrame


   .. py:method:: describe_categorical(self)

      Summarise categorical features

      This function uses `pandas.DataFrame.describe` to calculate
      summary statistics for each object-type real and synthetic
      feature.

      :returns: Descriptive statistics for each object-type feature.
      :rtype: pandas.DataFrame


   .. py:method:: add_metric(self, metric_name, metric_alias=None, **metric_kwargs)

      Add a metric to the evaluator

      Metrics and their arguments are recorded to be run at
      a later time. This allows metric customisation but ensures
      that the same metric configuration is applied consistently i.e.
      once added the parameters do not require resupplying for each
      execution of the metric. Supplying a metric alias allows the same
      metric to be used multiple times with different parameters.

      Note that self.real_data and self.synth_data will be passed
      automatically to metrics that expect these arguments. They should
      not be declared in `metric_kwargs`.

      :param metric_name: Name of the metric. Must match function name in metrics.
      :type metric_name: str
      :param metric_alias: Alias to be given to this use of the metric. Allows the same metric
                           to be used multiple times with different parameters within the same
                           evaluator instance.
      :type metric_alias: str, optional
      :param \*\*metric_kwargs: Extra arguments to `metric_name`: refer to each metric
                                documentation for a list of all possible arguments.
      :type \*\*metric_kwargs: dict, optional


   .. py:method:: add_custom_metric(self, metric_name, metric_func, **metric_kwargs)

      Add a custom metric to the Evaluator object.

      To enhance customisability, this function allows users to add metrics
      from outwith SynthGauge to the Evaluator.

      A custom metric is a function that accepts the real and synthetic
      dataframes as the first and second positional arguments respectively.
      Any other parameters must be defined as keyword arguments. The metric
      function can return a value of any desired type although scalar numeric
      values are recommended, or namedtuples when there are multiple
      outputs.

      :param metric_name: Name of the metric. This is what will appear in the results table.
      :type metric_name: str
      :param metric_func: Function to be called during the evaluation step. The first two
                          arguments will be ``self.real`` and ``self.synth``.
      :type metric_func: function
      :param \*\*metric_kwargs: Extra arguments to be passed to `metric_func` during evaluation.
      :type \*\*metric_kwargs: dict, optional


   .. py:method:: copy_metrics(self, Other)

      Copy metrics from another Evaluator object

      To facilitate consistent comparisons of different synthetic datasets,
      this function copies the metrics dictionary from another evaluator
      object.

      :param Other: The other evaluator object from which the metrics dictionary will
                    be copied.
      :type Other: Evaluator


   .. py:method:: save_metrics(self, filename)

      Save metrics to disk

      Save the Evaluator's current metrics to a pickle file.

      :param filename: Path to pickle file to save the metrics.


   .. py:method:: load_metrics(self, filename, overwrite=False)

      Load metrics from disk

      Update or overwrite the Evaluator's current metrics from a pickle
      file.

      :param filename: Path to metrics pickle file.
      :type filename: str
      :param overwrite: If True, all current metrics will be replaced with the loaded
                        metrics. Default is False which will update the current metric
                        dictionary with the loaded metrics.
      :type overwrite: bool, optional


   .. py:method:: metrics(self)
      :property:

      Return __metrics


   .. py:method:: combined_data(self)
      :property:

      Return combined real and synthetic data


   .. py:method:: drop_metric(self, metric)

      Drops the metric from the evaluator

      The metric `metric` will be removed from the metric
      catalogue for the evaluator.

      Note: To update the metric parameters see add_metric.

      :param metric: Name or alias of the metric to remove from the metrics stored for
                     this evaluator instance.
      :type metric: str


   .. py:method:: evaluate(self, as_df=False)

      Compute metrics for real and synth data

      Runs through the given metrics in self.__metrics and executes
      each with corresponding arguments. The results are returned as
      either a dictionary or DataFrame.

      Results are also silenty stored as a dictionary in
      `self.metric_results`.

      :param as_df: If True the results will be returned as a pandas DataFrame,
                    otherwise a dictionary is returned. Default is False.
      :type as_df: bool, optional

      :returns: * *pandas.DataFrame* -- If `as_df` is True a DataFrame is returned. The rows
                  represent metric names and the columns their values.
                * *dict* -- If `as_df` is False (the default) a dictionary is returned.
                  The keys represent metric names and the values metric values.


   .. py:method:: plot_histograms(self, figcols=2, figsize=None)

      Plot grid of feature distributions

      Convenience wrapper for plot.plot_histograms. This function uses the
      combined real and synthetic data sets and groups by 'source'.


   .. py:method:: plot_histogram3d(self, data, x, y, x_bins='auto', y_bins='auto', figsize=None)

      Plot 3D histogram

      Convenience wrapper for plot.plot_histogram3d.

      :param data: Dataframe to pass to plotting function. Either "real" to pass
                   `E.real_data`, "synth" to pass E.synth_data or "combined" to
                   pass E.combined_data.
      :type data: {"real"|"synth"|"combined"}:


   .. py:method:: plot_correlation(self, feats=None, method='pearson', figcols=2, figsize=None, **kwargs)

      Plot correlation heatmaps

      Plot correlation heatmaps for `self.real_data`, `self.synth_data`.
      See plot.plot_correlation for details.



   .. py:method:: plot_crosstab(self, x, y, figsize=None, **kwargs)

      Plot pairwise crosstabulation

      Convenience wrapper for plot.plot_crosstab. Automatically sets `real`
      and `synth` parameters to the correspondin data in the Evaluator.


   .. py:method:: plot_qq(self, feature, n_quantiles=None, figsize=None)

      Plot Q-Q plot

      Plot Quantile-Quantile plot. See plot.plot_qq for details.

      :param feature: Feature to plot.
      :type feature: str
      :param \*\*kwargs: Keyword arguments to pass to plot.plot_qq.
      :type \*\*kwargs: dict, optional



.. py:data:: __version__
   

   

