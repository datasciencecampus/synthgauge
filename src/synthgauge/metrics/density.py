"""Mean absolute difference in feature densities."""

import numpy as np

from .. import utils


def _feature_density_diff(real, synth, feature, bins=10):
    """Computes the difference between real and synth feature densities.

    For the specified feature the density is computed across `bins` in
    both the real and synthetic data. The per-bin difference is computed
    and returned along with the bin edges that were used.

    Prior to calculating the densities. all values are converted to
    numeric via `synthgauge.utils.cat_encode`.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feature : str
        The feature that will be used to compute the density.
    bins : str or int, default 10
        Bins to use for computing the density. This value is passed
        to `numpy.histogram_bin_edges` so can be any value accepted by
        that function. Default uses 10 bins.

    Returns
    -------
    hist_diff : numpy.ndarray
        The difference in feature density for each of the bins.
    bin_edges : numpy.ndarray
        The edges of the bins.
    """

    combined = utils.df_combine(real, synth, feats=[feature])
    encoded, _ = utils.cat_encode(combined, feats=[feature], return_all=True)
    enc_real, enc_synth = utils.df_separate(encoded, "source")

    bin_edges = np.histogram_bin_edges(encoded[feature], bins=bins)

    real_hist, _ = np.histogram(
        enc_real[feature], bins=bin_edges, density=True
    )
    synth_hist, _ = np.histogram(
        enc_synth[feature], bins=bin_edges, density=True
    )

    hist_diff = synth_hist - real_hist

    return hist_diff, bin_edges


def feature_density_mad(real, synth, feats=None, bins=10):
    """Mean absolute difference of feature densities.

    For each feature the difference between the density across the bins
    within `real` and `synth` is calculated. Finally the MAE across all
    features and bins is calculated. A value close to 0 indicates that
    the real and synthetic datasets have a similar set of feature
    distributions.

    Parameters
    ----------
    real : pandas.DataFrame
        DataFrame containing the real data.
    synth : pandas.DataFrame
        DataFrame containing the sythetic data.
    feats : list of str or None, default None
        The features that will be used to compute the densities. If
        `None` (default), all common features are used.
    bins : str or int, default 10
        Binning method for discretising the data. Can be anything
        accepted by `numpy.histogram_bin_edges`. Default uses 10 bins.

    Returns
    -------
    float
        Mean absolute error of feature densities.
    """

    feats = feats or real.columns.intersection(synth.columns)
    diffs = [
        _feature_density_diff(real, synth, feat, bins)[0] for feat in feats
    ]

    return np.mean(np.abs(np.concatenate(diffs)))
