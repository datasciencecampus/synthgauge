"""Utility functions for handling real and synthetic data."""

import warnings

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
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feats : list of str or None, default None
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

    real = real[feats].copy()
    real[source_col_name] = source_val_real

    synth = synth[feats].copy()
    synth[source_col_name] = source_val_synth

    combined = pd.concat([real, synth], ignore_index=True)

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
    feats : list of str or None, default None
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
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    """

    feats = list(feats) if feats is not None else list(data.columns)
    feats.append(source_col_name)

    real = data[data[source_col_name] == source_val_real][feats].copy()
    synth = data[data[source_col_name] == source_val_synth][feats].copy()

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
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feats : list of str or None, default None
        Features to launder. If `None` (default), all common features
        are used.
    suffix_real : str, default "real"
        Suffix to append to columns in `real`. Default is `"real"`.
    suffix_synth : str, default "synth"
        Suffix to append to columns in `synth`. Default is `"synth"`.

    Returns
    -------
    real : pandas.DataFrame
        Dataframe containing the laundered real data.
    synth : pandas.DataFrame
        Dataframe containing the laundered synthetic data.
    """

    feats = feats or real.columns.intersection(synth.columns)

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
    feats : list of str or None, default None
        Features in `df` to convert to categorical. If `None` (default),
        all object-type columns are selected.
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
        Dataframe with (at least) the converted features.
    cat_dict : dict or NoneType
        A dictionary mapping each encoded feature to its categories. If
        `convert_only=True`, returns as `None`.
    """

    all_cat_cols = df.select_dtypes(include=("object", "category")).columns
    feats = pd.Index(feats) if feats is not None else all_cat_cols

    # Check for non-object type features
    non_cat_cols = feats.difference(all_cat_cols)
    if non_cat_cols.any():
        warnings.warn(
            "Selected features include non-object types: "
            f"{non_cat_cols.to_list()}."
            "\nIs this intended? If so, rerun with `force=True`. "
            "If not, they will be dropped, unless `return_all=True`, "
            "where they will pass through unchanged."
        )

    cat_dict = {} if not convert_only else None

    feats_to_encode = feats if force else feats.difference(non_cat_cols)
    out_df = df.copy() if return_all else df[feats_to_encode].copy()

    for feature in feats_to_encode:
        out_df[feature] = out_df[feature].astype("category")

        if not convert_only:
            feature_cat = out_df[feature].cat
            cat_dict[feature] = feature_cat.categories
            out_df[feature] = feature_cat.codes

    return out_df, cat_dict
