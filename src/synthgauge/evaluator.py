"""The core class for evaluating datasets."""

import pickle
import warnings
from copy import deepcopy

import pandas as pd

from . import metrics, plot, utils


class Evaluator:
    """The central class in `synthgauge`, used to hold and evaluate data
    via metrics and visualisation.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    handle_nans : str, default "drop"
        Whether to drop missing values. If yes, use "drop" (default).

    Returns
    -------
    synthgauge.Evaluator
        An `Evaluator` object ready for metric and visual evaluation.
    """

    def __init__(self, real, synth, handle_nans="drop"):
        common_feats = real.columns.intersection(synth.columns)
        ignore_feats = real.columns.union(synth.columns).difference(
            common_feats
        )

        if len(ignore_feats) > 0:
            msg = (
                f"Features {', '.join(ignore_feats)} are not common to "
                "`real` and `synth` and will be ignored in further analysis."
            )

            warnings.warn(msg)

        self.feature_names = list(common_feats)

        # Metrics is private to apply some validation
        self.__metrics = dict()  # assign metrics and kwargs
        self.metric_results = dict()  # store results

        # Handle NaNs
        if handle_nans == "drop":
            real.dropna(inplace=True)
            synth.dropna(inplace=True)

        self.real_data = real
        self.synth_data = synth
        self.combined_data = utils.df_combine(self.real_data, self.synth_data)

    def describe_numeric(self):
        """Summarise numeric features.

        This function uses `pandas.DataFrame.describe` to calculate
        summary statistics for each numeric feature in `self.real_data`
        and `self.synth_data`.

        Returns
        -------
        pandas.DataFrame
            Descriptive statistics for each numeric feature.
        """

        real, synth = utils.launder(self.real_data, self.synth_data)

        return pd.concat(
            [
                real.describe(include="number"),
                synth.describe(include="number"),
            ],
            axis=1,
        ).T.sort_index()

    def describe_categorical(self):
        """Summarise categorical features.

        This function uses `pandas.DataFrame.describe` to calculate
        summary statistics for each object-type feature in
        `self.real_data` and `self.synth_data`.

        Returns
        -------
        pandas.DataFrame
            Descriptive statistics for each object-type feature.
        """

        real, synth = utils.launder(self.real_data, self.synth_data)

        return (
            pd.concat(
                [
                    real.describe(include=["category", "object"]),
                    synth.describe(include=["category", "object"]),
                ],
                axis=1,
            )
            .T.sort_index()
            .rename(columns={"top": "most_frequent"})
        )

    def add_metric(self, name, alias=None, **kwargs):
        """Add a metric to the evaluator.

        Metrics and their arguments are recorded to be run at a later
        time. This allows metric customisation but ensures that the same
        metric configuration is applied consistently, i.e. once added,
        the parameters do not require resupplying for each execution of
        the metric. Supplying a metric alias allows the same metric to
        be used multiple times with different parameters.

        Note that `self.real_data` and `self.synth_data` will be passed
        automatically to metrics that expect these arguments. They
        should not be declared in `metric_kwargs`.

        Parameters
        ----------
        name : str
            Name of the metric. Must match one of the functions in
            `synthgauge.metrics`.
        alias : str, optional
            Alias to be given to this use of the metric in the results
            table. Allows the same metric to be used multiple times with
            different parameters. If not specified, `name` is used.
        **kwargs : dict, optional
            Keyword arguments for the metric. Refer to the associated
            metric documentation for details.
        """

        try:
            getattr(metrics, name)
            kwargs["name"] = name
            alias = name if alias is None else alias
            self.__metrics.update({alias: kwargs})

        except AttributeError:
            raise NotImplementedError(f"Metric '{name}' is not implemented")

    def add_custom_metric(self, alias, func, **kwargs):
        """Add a custom metric to the evaluator.

        A custom metric uses any user-defined function that accepts the
        real and synthetic dataframes as the first and second positional
        arguments, respectively. Any other parameters must be defined as
        keyword arguments. The metric function can return a value of any
        desired type although scalar numeric values are recommended, or
        `collections.namedtuples` when there are multiple outputs.

        Parameters
        ----------
        alias : str
            Alias for the metric to appear in the results table.
        func : function
            Top-level metric function to be called during the evaluation
            step. The first two arguments of `func` must be `self.real`
            and `self.synth`.
        **kwargs : dict, optional
            Keyword arguments to be passed to `func`.
        """

        kwargs.update({"func": func, "name": alias})
        self.__metrics.update({alias: kwargs})

    def copy_metrics(self, other):
        """Copy metrics from another evaluator.

        To facilitate consistent comparisons of different synthetic
        datasets, this function copies the metrics dictionary from
        another `Evaluator` instance.

        Parameters
        ----------
        other : Evaluator
            The other evaluator from which the metrics dictionary will
            be copied.
        """

        if not isinstance(other, Evaluator):
            raise TypeError("`other` must be of class Evaluator")

        self.__metrics = deepcopy(other.metrics)

    def save_metrics(self, filename):
        """Save the current metrics dictionary to disk via `pickle`.

        Parameters
        ----------
        filename : str
            Path to pickle file to save the metrics.
        """

        with open(filename, "wb") as f:
            pickle.dump(self.metrics, f)

    def load_metrics(self, filename, overwrite=False):
        """Load metrics from disk.

        Update or overwrite the current metric dictionary from a pickle.

        Parameters
        ----------
        filename : str
            Path to metrics pickle file.
        overwrite : bool, default False
            If `True`, all current metrics will be replaced with the
            loaded metrics. Default is `False`, which will update the
            current metric dictionary with the loaded metrics.
        """

        with open(filename, "rb") as f:
            new_metrics = pickle.load(f)

        invalid_metrics = []
        for k, v in new_metrics.items():
            if getattr(metrics, v["name"], None) is None:
                invalid_metrics.append(k)

        if len(invalid_metrics) > 0:
            invalid_str = ", ".join(invalid_metrics)
            raise ValueError(f"Invalid metrics encountered in: {invalid_str}.")

        if overwrite:
            self.__metrics = new_metrics
        else:
            self.__metrics.update(new_metrics)

    @property
    def metrics(self):
        """Return __metrics."""

        return self.__metrics

    def drop_metric(self, metric):
        """Drops the named metric from the metrics dictionary.

        Parameters
        ----------
        metric : str
            Key (name or alias, if specified) of the metric to remove.
        """

        try:
            del self.__metrics[metric]
        except KeyError:
            pass

    def evaluate(self, as_df=False):
        """Compute metrics for real and synth data.

        Run through the metrics dictionary and execute each with its
        corresponding arguments. The results are returned as either a
        dictionary or dataframe.

        Results are also silently stored as a dictionary in
        `self.metric_results`.

        Parameters
        ----------
        as_df : bool, default False
            If `True`, the results will be returned as a
            `pandas.DataFrame`, otherwise a dictionary is returned.
            Default is `False`.

        Returns
        -------
        pandas.DataFrame
            If `as_df` is `True`. Each row corresponds to a metric-value
            pair. Metrics with multiple values have multiple rows.
        dict
            If `as_df` is `False`. The keys are the metric names and
            the values are the metric values (grouped). Metrics with
            multiple values are assigned to a single key.
        """

        results = dict.fromkeys(self.__metrics.keys())

        metrics_copy = deepcopy(self.__metrics)
        for metric, kwargs in metrics_copy.items():
            metric_name = kwargs.pop("name")
            if metric_name in metrics.__dict__.keys():
                metric_func = getattr(metrics, metric_name)
            else:
                metric_func = kwargs.pop("func")
            results[metric] = metric_func(
                self.real_data, self.synth_data, **kwargs
            )

        self.metric_results = dict(results)

        if as_df:
            tidy_results = {}
            for k, v in self.metric_results.items():
                try:
                    for vk, vv in v._asdict().items():
                        tidy_results[k + "-" + vk] = vv
                except AttributeError:
                    tidy_results[k] = v

            return pd.DataFrame(tidy_results, index=["value"]).T

        else:
            return results

    def plot_histograms(self, figcols=2, figsize=None):
        """Plot grid of feature distributions.

        Convenience wrapper for `synthgauge.plot.plot_histograms`. This
        function uses the combined real and synthetic data sets and
        groups by `'source'`.
        """

        return plot.plot_histograms(
            self.combined_data,
            feats=self.feature_names,
            groupby="source",
            figcols=figcols,
            figsize=figsize,
        )

    def plot_histogram3d(
        self, data, x, y, x_bins="auto", y_bins="auto", figsize=None
    ):
        """Plot 3D histogram.

        Convenience wrapper for `synthgauge.plot.plot_histogram3d`.

        Parameters
        ----------
        data: {"real", "synth", "combined"}
            Dataframe to pass to plotting function. Either `"real"` to
            pass `self.real_data`, `"synth"` to pass `self.synth_data`
            or `"combined"` to pass `self.combined_data`.
        x : str
            Column to plot along the x-axis.
        y : str
            Column to plot alont the y-axis.
        """

        return plot.plot_histogram3d(
            getattr(self, f"{data}_data"),
            x=x,
            y=y,
            x_bins=x_bins,
            y_bins=y_bins,
            figsize=figsize,
        )

    def plot_correlation(
        self, feats=None, method="pearson", figcols=2, figsize=None, **kwargs
    ):
        """Plot a grid of correlation heatmaps.

        Convenience wrapper for `synthgauge.plot.plot_correlation`. Each
        dataset (real and synthetic) has a plot, as well as one for the
        differences in their correlations.
        """

        return plot.plot_correlation(
            self.real_data,
            self.synth_data,
            feats=feats,
            method=method,
            plot_diff=True,
            figcols=figcols,
            figsize=figsize,
            **kwargs,
        )

    def plot_crosstab(self, x, y, figsize=None, **kwargs):
        """Plot pairwise cross-tabulation.

        Convenience wrapper for `synthgauge.plot.plot_crosstab`.
        Automatically sets `real` and `synth` parameters to the
        corresponding data in `self`.
        """

        return plot.plot_crosstab(
            self.real_data,
            self.synth_data,
            x=x,
            y=y,
            figsize=figsize,
            **kwargs,
        )

    def plot_qq(self, feature, n_quantiles=None, figsize=None):
        """Plot quantile-quantile plot.

        Convenience wrapper for `synthgauge.plot.plot_qq`.

        Parameters
        ----------
        feature : str
            Feature to plot.
        **kwargs : dict, optional
            Keyword arguments to pass to `synthgauge.plot.plot_qq`.
        """

        return plot.plot_qq(
            self.real_data, self.synth_data, feature, n_quantiles, figsize
        )
