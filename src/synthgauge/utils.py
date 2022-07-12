"""Utility functions for handling real and synthetic data."""

import warnings

import numpy as np
import pandas as pd


def df_combine(
    real,
    synth,
    feats=None,
    source_col_name="source",
    source_val_real="real",
    source_val_synth="synth",
):
    """Combine separate dataframes of real and synthetic data.

    The dataframes are concatenated along the first axis (rows) and a
    source column is added to distinguish the real data from the
    synthetic data. Optionally, specific features can be selected.

    Parameters
    ----------
    real, synth : pandas.DataFrame
        Dataframes containing the real and synthetic data.
    feats : str or list of str or None, default None
        Features to combine. If `None` (default), all common features
        are used.
    source_col_name : str, default "source"
        Name of the source column. This is added to the combined dataset
        and filled with the `source_val_real` and `source_val_synth`
        values to signify the real and synthetic data respectively.
        Defaults to `"source"`.
    source_val_real : any, default "real"
        Value to use in `source_col_name` column to signify the real
        data. Defaults to `"real"`.
    source_val_synth : any, default "synth"
        Value to use in `source_col_name` column to signify the
        synthetic data. Defaults to `"synth"`.

    Returns
    -------
    combined : pandas.DataFrame
        The combined dataframe.
    """

    feats = feats or real.columns.intersection(synth.columns)
    if isinstance(feats, str):
        feats = [feats]

    real = real[feats].copy()
    real[source_col_name] = source_val_real

    synth = synth[feats].copy()
    synth[source_col_name] = source_val_synth

    combined = pd.concat([real, synth])

    return combined


def df_separate(
    data,
    source_col_name,
    feats=None,
    source_val_real="real",
    source_val_synth="synth",
    drop_source_col=True,
):
    """Separate a dataframe into real and synthetic data.

    The dataframe is split using a source column and real and synthetic
    flags. Optionally, specific features can be selected.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe to split into real and synthetic components.
    source_col_name : str
        Name of the column used to signify real versus synthetic data.
    feats : str or list of str or None, default None
        Features to separate. If `None` (default), uses all features.
    source_val_real : any, default "real"
        Value in `source_col_name` column signifying the real data.
        Defaults to `"real"`.
    source_val_synth : any, default "synth"
        Value in `source_col_name` column signifying the synthetic data.
        Defaults to `"synth"`.
    drop_source_col : bool, default True
        Indicates whether the `source_col_name` column should be
        dropped from the outputs (default) or not.

    Returns
    -------
    real, synth : pandas.DataFrame
        Dataframes containing the real data and synthetic data.
    """

    if isinstance(feats, str):
        columns = [feats]
    elif isinstance(feats, list):
        columns = list(feats)
    else:
        columns = list(data.columns)

    columns = columns + [source_col_name]

    real = data[data[source_col_name] == source_val_real][columns].copy()
    synth = data[data[source_col_name] == source_val_synth][columns].copy()

    if drop_source_col:
        real.drop(columns=source_col_name, inplace=True, errors="ignore")
        synth.drop(columns=source_col_name, inplace=True, errors="ignore")

    return real, synth


def launder(real, synth, feats=None, suffix_real="real", suffix_synth="synth"):
    """Launder feature names and optionally filter.

    To provide clear distinction between the real and synthetic
    features, each dataframe is updated to append suffixes to the
    feature names. Optionally, specific features can be selected.

    Parameters
    ----------
    real, synth : pandas.DataFrame
        Dataframes containing the real and synthetic data.
    feats : str or list of str or None, default None
        Features to launder. If `None` (default), all common features
        are used.
    suffix_real : str, default "real"
        Suffix to append to columns in `real`. Default is `"real"`.
    suffix_synth : str, default "synth"
        Suffix to append to columns in `synth`. Default is `"synth"`.

    Returns
    -------
    real, synth : pandas.DataFrame
        Laundered versions of the real and synthetic data.
    """

    feats = feats or real.columns.intersection(synth.columns)
    if isinstance(feats, str):
        feats = [feats]

    real = real[feats].copy()
    synth = synth[feats].copy()

    real.columns = [f"{c}_{suffix_real}" for c in real.columns]
    synth.columns = [f"{c}_{suffix_synth}" for c in synth.columns]

    return real, synth


def cat_encode(
    df, feats=None, return_all=False, convert_only=False, force=False
):
    """Convert object features to categories.

    Generates a new version of the input dataframe with the specified
    features categorically encoded with integer labels. Optionally, the
    features can be returned as `category` data type with no encoding.

    Before performing the conversion, a check is made to identify any
    speficied features that are not `object`-type and thus less suited
    to categorical encoding. A warning is raised for these features and
    they will be ignored from subsequent encoding steps unless `force`
    is set to `True`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to be converted.
    feats : str or list of str or None, default None
        Feature(s) in `df` to convert to categorical. If `None`
        (default), all object-type columns are selected.
    return_all : bool, default False
        If `True`, all features in `df` will be returned regardless of
        whether they were converted. If `False` (default), only the
        converted features are returned.
    convert_only : bool, default False
        If `True`, the features will only be converted to the `category`
        data-type without being integer-encoded.
    force : bool, default False
        If `True`, all features in `feats` will be encoded regardless of
        their data-type.

    Warns
    -----
    UserWarning
        A warning is given if any of the features in `feats` are not of
        an `object` data type.

    Returns
    -------
    out_df : pandas.DataFrame
        DataFrame with (at least) the converted features.
    cat_dict : dict or NoneType
        A dictionary mapping each encoded feature to its categories. If
        `convert_only=True`, returns `None`.
    """

    all_obj = df.select_dtypes(include=["object", "category"]).columns
    feats = feats or all_obj
    if isinstance(feats, str):
        feats = [feats]

    # Check for non-object type features
    non_obj = list(set(feats).difference(all_obj))
    if len(non_obj) > 0:
        warnings.warn(
            f"Selected features include non-object types: {non_obj}\n"
            "Is this intended? If so, rerun with `force=True`. "
            "Otherwise, they will be dropped, unless "
            "`return_all=True`, where they will pass through unchanged."
        )

    cat_dict = {} if not convert_only else None

    if force:
        enc_fts = feats
    else:
        enc_fts = list(set(feats).difference(non_obj))

    if return_all:
        out_df = df.copy()
    else:
        out_df = df[enc_fts].copy()

    for ft in enc_fts:
        out_df[ft] = out_df[ft].astype("category")

        if not convert_only:
            cat_dict.update({ft: out_df[ft].cat.categories})
            out_df[ft] = out_df[ft].cat.codes

    return out_df, cat_dict


def feature_density_diff(real, synth, feature, bins=10):
    """Computes the difference between real and synth feature densities.

    For the specified feature the density is computed across `bins` in
    both the real and synthetic data. The per-bin difference is computed
    and returned along with the bin edges that were used.

    Prior to calculating the densities. all values are converted to
    numeric via `synthgauge.utils.cat_encode`.

    Parameters
    ----------
    real, synth : pandas.DataFrame
        Dataframes containing the real and synthetic data.
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

    combined = df_combine(real, synth, feats=feature)
    encoded, _ = cat_encode(combined, feats=feature, return_all=True)
    enc_real, enc_synth = df_separate(encoded, "source")

    bin_edges = np.histogram_bin_edges(encoded[feature], bins=bins)

    real_hist, _ = np.histogram(
        enc_real[feature], bins=bin_edges, density=True
    )
    synth_hist, _ = np.histogram(
        enc_synth[feature], bins=bin_edges, density=True
    )

    hist_diff = synth_hist - real_hist

    return hist_diff, bin_edges


if __name__ == "__main__":
    pass
