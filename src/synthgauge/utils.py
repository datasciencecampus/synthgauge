import numpy as np
import pandas as pd
import pickle
import warnings


def load_pickle(src):
    """Load pickle file into object.

    """

    with open(src, 'rb') as pf:
        data = pickle.load(pf)
    return data


def df_combine(real, synth, feats=None, source_col_name='source',
               source_val_real='real', source_val_synth='synth'):
    """ Combine separate dataframes of real and synthetic data.

    The dataframes are concatenated along axis=0 and a source column is
    added to distinguish the real data from the sythetic data.

    Optionally, specific features can be selected.

    Parameters
    ----------
    real: pandas.DataFrame
        DataFrame containing the real, unsythesised data.
    synth: pandas.DataFrame
        DataFrame containing the synthesised data.
    feats: str or list of str, optional
        Features to combine. By default all common features are used.
    source_col_name: str, optional
        Name of the source column. This is added to the combined DataFrame
        and filled with the `source_val_real` and `source_val_synth`
        values to signify the real and synthetic data respectively.
        Defaults to "source".
    source_val_real: any, optional
        Value to use in `source_col_name` column to signify the real data.
        Any value accepted by pandas.DataFrame. Defaults to "real".
    source_val_synth: any, optional
        Value to use in `source_col_name` column to signify the sythetic
        data. Any value accepted by pandas.DataFrame. Defaults to "synth".

    Returns
    -------
    pandas.DataFrame
        The result of the concatenation of the input DataFrame objects.
    """

    feats = feats or real.columns.intersection(synth.columns)
    if isinstance(feats, str):
        feats = [feats]

    real = real[feats].copy()
    real[source_col_name] = source_val_real

    synth = synth[feats].copy()
    synth[source_col_name] = source_val_synth

    data = pd.concat([real, synth])

    return data


def df_separate(data, source_col_name, feats=None, source_val_real='real',
                source_val_synth='synth', drop_source_col=True):
    """ Separate a dataframe into real and synthetic data.

    The dataframe is split using a source column and real and synthetic
    flags. Optionally, specific features can be selected.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to split into real and synthetic components.
    source_col_name: str
        Name of the column used to signify real versus synthetic data.
    feats: str or list of str, optional
        Features to separate. By default all features are used.
    source_val_real: any, optional
        Value in `source_col_name` column signifying the real data. Any
        value accepted by pandas.DataFrame. Defaults to "real".
    source_val_synth: any, optional
        Value in `source_col_name` column signifying the synthetic data.
        Any value accepted by pandas.DataFrame. Defaults to "synth".
    drop_source_col: bool, optional
        If True, the default, the `source_col_name` column will be dropped
        from the output DataFrames.

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        Two DataFrame objects containing the real data and synthetic
        data respectively.
    """
    feats = feats or data.columns
    if isinstance(feats, str):
        feats = [feats]

    real = data[data[source_col_name] == source_val_real][feats].copy()
    synth = data[data[source_col_name] == source_val_synth][feats].copy()

    if drop_source_col:
        real.drop(columns=source_col_name, inplace=True)
        synth.drop(columns=source_col_name, inplace=True)

    return real, synth


def launder(real, synth, feats=None, suffix_real='real', suffix_synth='synth'):
    """ Launder feature names and optionally filter.

    To provide clear distinction between the real and synthetic
    features, each dataframe is updated to append suffixes to the feature
    names.

    Optionally, specific features can be selected.

    Parameters
    ----------
    real: pandas.DataFrame
        DataFrame containing the real, unsythesised data
    synth: pandas.DataFrame
        DataFrame containing the synthesised data
    feats: str or list of str, optional
        Features to launder. By default all common features are used.
    suffix_real: str, optional
        Suffix to append to columns in `real`. Default is "real".
    suffix_synth: str, optional
        Suffix to append to columns in `synth`. Default is "synth".

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        Two DataFrame objects containing the real data and synthetic
        data respectively.
    """

    feats = feats or real.columns.intersection(synth.columns)
    if isinstance(feats, str):
        feats = [feats]

    real = real[feats].copy()
    synth = synth[feats].copy()

    real.columns = [f'{c}_{suffix_real}' for c in real.columns]
    synth.columns = [f'{c}_{suffix_synth}' for c in synth.columns]

    return real, synth


def cat_encode(df, feats=None, return_all=False, convert_only=False,
               force=False):
    """ Convert object features to categories.

    Generates a new version of the input datframe with the
    specified features categorically encoded with integer labels.
    Optionally, the features can be returned as the 'category' type
    with no encoding.

    Before performing the conversion a check is made to identify any
    speficied features that are not 'object' type and therefore less
    suited to categorical encoding. A warning is raised for these features
    and they will be ignored from subsequent encoding steps unless
    `force` is `True`.

    Parameters
    ----------
    df: pandas.DataFrame
        Input dataframe to be converted.
    feats: str or list of str, optional
        Feature(s) in `df` to convert to category. By default all object
        columns are selected.
    return_all: bool, default=False
        If True all features in `df` will be returned with only those
        specified in `feats` converted. Otherwise only the converted
        features are returned. This is the default.
    convert_only: bool, default=False
        If True the features will only be converted to the 'category'
        type and NOT encoded.
    force: bool, optional
        If true, all specified features will be encoded regardless of
        data type.


    Returns
    -------
    pandas.DataFrame
        DataFrame with converted features
    dict or NoneType
        A dictionary mapping each encoded feature to its categories. If
        convert_only=True NoneType is returned.

    Warns
    -----
    UserWarning
        A warning is given if any of the specified features for
        conversion are not an 'object' type.

    """
    all_obj = df.select_dtypes(include=['object', 'category']).columns
    feats = feats or all_obj
    if isinstance(feats, str):
        feats = [feats]

    # Check for non-object type features
    non_obj = list(set(feats).difference(all_obj))
    if len(non_obj) > 0:
        warnings.warn('Selected features include non-object types:'
                      f'{non_obj} Is this intended? If so, rerun and '
                      'specify `force_encode=True`. Otherwise they will '
                      'be dropped or, if `return_all==True`, passed '
                      'through unchanged.')

    if return_all:
        out_df = df.copy()
    else:
        out_df = df[feats].copy()

    cat_dict = {} if not convert_only else None

    if force:
        enc_fts = feats
    else:
        enc_fts = list(set(feats).difference(non_obj))

    for ft in enc_fts:
        out_df[ft] = out_df[ft].astype('category')

        if not convert_only:
            cat_dict.update({ft: out_df[ft].cat.categories})
            out_df[ft] = out_df[ft].cat.codes

    return out_df, cat_dict


def feature_density_diff(real, synth, feature, bins=10):
    """Computes the difference between real and synth feature densities.

    For the specified feature the density is computed across `bins` in both
    the `real` and `synth` data sets. The per-bin difference is computed and
    returned along with the bin edges that were used.

    Prior to calculating the densities all values are converted to numeric
    via `utils.cat_encode`.

    Parameters
    ----------
    real: pandas.DataFrame
        DataFrame containing the real data.
    synth: pandas.DataFrame
        DataFrame containing the sythetic data.
    feature: str
        The feature that will be used to compute the density.
    bins: str or int, optional
        Bins to use for computing the density. This value is passed
        to `numpy.histogram_bin_edges` so can be any value accepted by
        that function. The default setting of 10 uses 10 bins.

    Returns
    -------
    hist_diff: np.ndarry
        The difference in feature density for each of the bins.
    bin_edges: np.ndarray
        The edges of the bins.

    """
    enc_real, _ = cat_encode(real, feats=feature)
    enc_synth, _ = cat_encode(synth, feats=feature)

    full_range = np.concatenate((enc_real[feature], enc_synth[feature]))
    bin_edges = np.histogram_bin_edges(full_range, bins=bins)

    real_hist, _ = np.histogram(enc_real[feature], bins=bin_edges,
                                density=True)
    synth_hist, _ = np.histogram(enc_synth[feature], bins=bin_edges,
                                 density=True)

    hist_diff = synth_hist - real_hist

    return hist_diff, bin_edges


if __name__ == '__main__':
    pass
