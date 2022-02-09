from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import pandas as pd
import numpy as np
from ..utils import cat_encode, df_combine, df_separate


def get_WEAP(synth, key, target):
    """Get the Within Equivalence class Attribution Probabilities WEAP

    For each record in the synthetic dataset, this function returns the
    proportion across the whole dataset that these `key` values are matched
    with this `target` value.

    Parameters
    ----------
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    key : list
        List of features in `synth` to use as the key.
    target : str or list of str
        Feature to use as the target.

    Returns
    -------
    pandas.Series
        A series object containing the WEAP scores for each record in `synth`.

    Notes
    -----
    This function is intended to only be used within `TCAP()` to determine
    which synthetic records are most likely to pose an attribution risk.
    """
    synth = synth.copy()
    key_target_vc = synth.value_counts(subset=key + target).reset_index()
    key_target_vc.columns = key + target + ['key_target_frequency']
    key_vc = synth.value_counts(key).reset_index()
    key_vc.columns = key + ['key_frequency']

    synth = synth.merge(key_target_vc)
    synth = synth.merge(key_vc)

    return synth['key_target_frequency']/synth['key_frequency']


def TCAP(real, synth, key, target):
    """ Target Correct Attribution Probability TCAP

    This privacy metric calculates the average chance that the key-target
    pairings in the `synth` dataset reveal the true key-target pairings in the
    original, `real` dataset.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    key : list
        List of features in `synth` to use as the key.
    target : str or list of str
        Feature to use as the target.

    Returns
    -------
    TCAP : float
        The average TCAP across the dataset.

    Notes
    -----
    This metric provides an estimate of how well an intruder could infer
    attributes of groups in the real dataset by studying the synthetic. The
    choices for `key` and `target` will vary depending on the dataset in
    question but we would suggest the `key` features are those that could be
    readily available to an outsider and the `target` feature is one we
    wouldn't want them finding out, such as a protected characteristic.

    This method only works with categorical data, so binning of continuous data
    may be required.
    """
    if not isinstance(target, list):
        target = [target]
    WEAP_scores = get_WEAP(synth, key, target)
    if sum(WEAP_scores == 1) == 0:
        return 0
    synth_reduced = synth[WEAP_scores == 1][key+target]
    synth_reduced.drop_duplicates(inplace=True)
    synth_reduced.rename(
        columns={target[0]: target[0]+'_synthguess'}, inplace=True)

    combined = real.merge(synth_reduced, how='left', on=key)
    target_matches = combined[target[0]] == combined[target[0]+'_synthguess']

    return sum(target_matches)/len(target_matches)


def find_outliers(data, outlier_factor_threshold):
    """Find Outliers

    This function returns whether each row in `data` can be considered an
    outlier.

    Parameters
    ----------
    data : pandas dataframe
    outlier_factor_threshold : float
        Float influencing classification of ouliers. Increasing this threshold
        means that fewer points are considered outliers.

    Returns
    -------
    outlier_bool : list of bool
        List indicating which rows of `data` are outliers.

    Notes
    -----
    Most inliers will have an outlier factor of less than one, however there
    are no clear rules that determine when a data point is an outlier. This
    is likely to vary from dataset to dataset and, as such, we recommend
    tuning `outlier_factor_threshold` to suit.
    """
    lof = LocalOutlierFactor(n_neighbors=5)
    lof.fit_predict(data)
    outlier_factor = -lof.negative_outlier_factor_
    outlier_bool = outlier_factor < outlier_factor_threshold
    return outlier_bool


def min_NN_dist(real, synth, feats=None, real_outliers_only=True,
                outlier_factor_threshold=2):
    """ Minimum Nearest Neighbour distance

    This privacy metric returns the smallest distance between any point in
    the `real` dataset and any point in the `synth` dataset. There is an
    option to only consider the outliers in the real dataset as these perhaps
    pose more of a privacy concern.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    feats: str or list of str, optional
        Features to use. By default all features are used.
    real_outliers_only : bool (default True)
        Boolean indicating whether to filter out inliers (default) or not.
    outlier_factor_threshold : float (default 2)
        Float influencing classification of ouliers. Increase to include
        fewer real points in nearest neighbour calculations.

    Returns
    -------
    min_dist : float
        Minimum manhattan distance between `real` and `synth` data.

    Notes
    -----
    This privacy metric provides an insight into whether the synthetic dataset
    is too similar to the real dataset. It does this by calculating the
    minimum distance between the real records and the synthetic records.

    This metric assumes that categorical data is ordinal during distance
    calculations, or that it has already been suitably one-hot-encoded.
    """
    combined = df_combine(real, synth, feats=feats)
    combined_recode, _ = cat_encode(combined, return_all=True)
    real, synth = df_separate(combined_recode, source_col_name='source',
                              source_val_real=0, source_val_synth=1)
    if real_outliers_only:
        outlier_bool = find_outliers(real, outlier_factor_threshold)
        real = real[outlier_bool]
    near_neigh = NearestNeighbors(n_neighbors=1, radius=100, p=1)
    near_neigh.fit(real)
    dist, ind = near_neigh.kneighbors(synth, return_distance=True)
    return int(min(dist))


def sample_overlap_score(real, synth, feats=None, sample_size=0.2, runs=5,
                         score_type='unique'):
    """ Return percentage of overlap between real and synth data.

    Samples from both the real and synthetic datasets are compared for
    similarity. This similarity, or overlap score, is based on the
    exact matches of real data records within the synthetic data.

    Parameters
    ----------
    real: pandas.DataFrame
        DataFrame containing the real data.
    synth: pandas.DataFrame
        DataFrame containing the synthetic data.
    feats: str or list of str, optional.
        The features that will be used to match records. By
        default all features will be used.
    sample_size: float or int, optional
        The ratio (if `sample_size` between 0 and 1) or count
        (`sample_size` > 1) of records to sample. Default is 0.2 or 20%.
    runs: int, optional
        The number of times to compute the score. Total score is averaged
        across runs.
    score_type: {"unique"|"sample"}
        Method used for calculating the overlap score. If "unique", the
        default, the score is the percentage of unique records in the real
        sample that have a match within the synth data. If "sample" the
        score is the percentage of all records within the real sample that
        have a match within the synth sample.

    Returns
    -------
    float:
        Overlap score between `real` and `synth`
    """
    if isinstance(feats, pd.Index):
        feats = feats
    elif isinstance(feats, str):
        feats = [feats]
    else:
        feats = feats or real.columns.to_list()
    if 0 <= sample_size <= 1:
        nsamples = int(real.shape[0] * sample_size)
    else:
        if sample_size > real.shape[0]:
            nsamples = real.shape[0]
        else:
            nsamples = sample_size
    scores = []
    for r in range(runs):
        sample_real = real[feats].sample(nsamples).assign(real_count=1) \
            .groupby(feats).count().reset_index()
        sample_synth = synth[feats].sample(nsamples).assign(synth_count=1) \
            .groupby(feats).count().reset_index()
        duplicates = sample_real.merge(sample_synth, how='left', on=feats,
                                       suffixes=('_real', '_synth'),
                                       indicator='_match')
        if score_type == 'unique':
            score = duplicates._match.value_counts(normalize=True).both
        elif score_type == 'sample':
            score = duplicates[duplicates._match ==
                               'both'].real_count.sum() / nsamples
        scores.append(score)
    return np.mean(scores)


if __name__ == "__main__":
    pass
