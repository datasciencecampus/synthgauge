"""Privacy metrics."""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .. import utils


def _get_weap_scores(synth, key, target):
    """Within Equivalence class Attribution Probabilities (WEAP).

    For each record in the synthetic dataset, this function returns the
    proportion across the whole dataset that a set of `key` values are
    matched with this `target` value.

    Parameters
    ----------
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    key : list of str
        List of features in `synth` to use as the key.
    target : str
        Feature to use as the target.

    Returns
    -------
    pandas.Series
        A series object containing the WEAP scores for each record in
        `synth`.

    Notes
    -----
    This function is intended to only be used within `TCAP()` to
    determine which synthetic records are most likely to pose an
    attribution risk.
    """

    synth = synth.copy()
    key_and_target = [*key, target]

    key_target_vc = synth.value_counts(subset=key_and_target).reset_index()
    key_target_vc.columns = key_and_target + ["key_target_frequency"]
    key_vc = synth.value_counts(key).reset_index()
    key_vc.columns = key + ["key_frequency"]

    synth = synth.merge(key_target_vc)
    synth = synth.merge(key_vc)

    return synth["key_target_frequency"] / synth["key_frequency"]


def tcap_score(real, synth, key, target):
    """Target Correct Attribution Probability (TCAP) score.

    This privacy metric calculates the average chance that the
    key-target pairings in a synthetic dataset reveal the true
    key-target pairings in associated real dataset.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    key : list of str
        List of features in `synth` to use as the key.
    target : str
        Feature to use as the target.

    Returns
    -------
    float
        The average TCAP score across the dataset.

    Notes
    -----
    This metric provides an estimate of how well an intruder could infer
    attributes of groups in the real dataset by studying the synthetic.
    The choices for `key` and `target` will vary depending on the
    dataset in question but we would suggest the `key` features are
    those that could be readily available to an outsider and the
    `target` feature is one we wouldn't want them finding out, such as a
    protected characteristic.

    This method only works with categorical data, so binning of
    continuous data may be required.

    Full details may be found in:

    Taub and Elliott (2019). The Synthetic Data Challenge. The Hague,
    The Netherlands: Joint UNECE/Eurostat Work Session on Statistical
    Data Confidentiality, Session 3.
    """

    scores = _get_weap_scores(synth, key, target)

    if sum(scores == 1) == 0:
        return 0

    synth_reduced = synth[scores == 1][[*key, target]]
    synth_reduced.drop_duplicates(inplace=True)
    synth_reduced.rename(
        columns={target: target + "_synthguess"}, inplace=True
    )

    combined = real.merge(synth_reduced, how="left", on=key)
    target_matches = combined[target] == combined[target + "_synthguess"]

    return np.mean(target_matches)


def _find_outliers(data, threshold, n_neighbours):
    """Identify local outliers using the nearest-neighbour principle.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe to be assessed for outliers.
    threshold : float
        Float influencing classification of outliers. Increasing this
        threshold means that fewer points are considered outliers.
    n_neighbours : int
        Number of neighbours to consider in outlier detection.

    Returns
    -------
    outlier_bool : list of bool
        List indicating which rows of `data` are outliers.

    Notes
    -----
    Most inliers will have an outlier factor of less than one, however
    there are no clear rules that determine when a data point is an
    outlier. This is likely to vary from dataset to dataset and, as
    such, we recommend tuning `outlier_factor_threshold` to suit.
    """

    lof = LocalOutlierFactor(n_neighbors=n_neighbours)
    lof.fit_predict(data)

    outlier_factor = -lof.negative_outlier_factor_
    outlier_bool = outlier_factor < threshold

    return outlier_bool


def min_nearest_neighbour(
    real,
    synth,
    feats=None,
    outliers_only=True,
    threshold=2,
    n_neighbours=5,
):
    """Minimum nearest-neighbour distance.

    This privacy metric returns the smallest distance between any point
    in the real dataset and any point in the synthetic dataset. There is
    an option to only consider the outliers in the real dataset as these
    perhaps pose more of a privacy concern.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feats : list of str or None, default None
        Features in `real` and `synth` to use when calculating
        distance. If `None` (default), all common features are used.
    outliers_only : bool, default True
        Boolean indicating whether to filter out the real data inliers
        (default) or not.
    threshold : number, default 2
        Outlier decision threshold. Increase to include fewer points
        from `real` in nearest-neighbour calculations.
    n_neighbours : int, default 5
        Number of neighbours to consider when identifying local
        outliers.

    Returns
    -------
    float
        Minimum Manhattan distance between `real` and `synth` data.

    Notes
    -----
    This privacy metric provides an insight into whether the synthetic
    dataset is too similar to the real dataset. It does this by
    calculating the minimum distance between the real records and the
    synthetic records.

    This metric assumes that categorical data is ordinal during distance
    calculations, or that it has already been suitably one-hot-encoded.
    """

    combined = utils.df_combine(real, synth, feats=feats)
    combined_recode, _ = utils.cat_encode(combined, return_all=True)
    real, synth = utils.df_separate(
        combined_recode,
        source_col_name="source",
        source_val_real=0,
        source_val_synth=1,
    )

    if outliers_only:
        outliers = _find_outliers(real, threshold, n_neighbours)
        real = real[outliers]

    neigh = NearestNeighbors(n_neighbors=1, radius=100, p=1).fit(real)
    distances, _ = neigh.kneighbors(synth, return_distance=True)

    return np.min(distances)


def _get_sample(data, feats, n_samples, seed, label):
    """Take a sample from the data and count the feature frequencies."""

    return (
        data[feats]
        .sample(n_samples, random_state=seed)
        .assign(**{f"{label}_count": 1})
        .groupby(feats)
        .count()
        .reset_index()
    )


def sample_overlap_score(
    real,
    synth,
    feats=None,
    sample_size=0.2,
    runs=5,
    seed=None,
    score_type="unique",
):
    """Return percentage of overlap between real and synth data based on
    random sampling.

    Samples from both the real and synthetic datasets are compared for
    similarity. This similarity, or overlap score, is based on the
    exact matches of real data records within the synthetic data.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feats : list of str or None, default None
        The features used to match records. If `None` (default), all
        common features are used.
    sample_size : float or int, default 0.2
        The ratio (if `sample_size` between 0 and 1) or count
        (`sample_size` > 1) of records to sample. Default is 0.2 (20%).
    runs : int, default 5
        The number of sampling runs to use when computing the score.
    seed : int, optional
        Random number seed used for sampling.
    score_type : {"unique", "sample"}, default "unique"
        Method used for calculating the overlap score. If "unique"
        (default), the score is the percentage of unique records in the
        real sample that have a match within the synthetic data. If
        "sample", the score is the percentage of all records within the
        real sample that have a match within the synth sample.

    Returns
    -------
    overlap_score : float
        Estimated overlap score between `real` and `synth`.
    """

    feats = feats or real.columns.intersection(synth.columns).to_list()

    min_num_rows = min(real.shape[0], synth.shape[0])
    if 0 <= sample_size <= 1:
        n_samples = int(min_num_rows * sample_size)
    else:
        n_samples = min(min_num_rows, sample_size)

    scores = []
    for _ in range(runs):

        sample_real = _get_sample(real, feats, n_samples, seed, "real")
        sample_synth = _get_sample(synth, feats, n_samples, seed, "synth")

        duplicates = sample_real.merge(
            sample_synth,
            how="left",
            on=feats,
            suffixes=("_real", "_synth"),
            indicator="_match",
        )

        if score_type == "unique":
            score = duplicates._match.value_counts(normalize=True).both
        if score_type == "sample":
            score = (
                duplicates[duplicates._match == "both"].real_count.sum()
                / n_samples
            )

        scores.append(score)

    return np.mean(scores)
