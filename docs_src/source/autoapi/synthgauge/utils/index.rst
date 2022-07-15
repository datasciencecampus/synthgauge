:py:mod:`synthgauge.utils`
==========================

.. py:module:: synthgauge.utils

.. autoapi-nested-parse::

   Utility functions for handling real and synthetic data.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.utils.df_combine
   synthgauge.utils.df_separate
   synthgauge.utils.launder
   synthgauge.utils.cat_encode



.. py:function:: df_combine(real, synth, feats=None, source_col_name='source', source_val_real='real', source_val_synth='synth')

   Combine separate dataframes of real and synthetic data.

   The dataframes are concatenated along the first axis (rows) and a
   source column is added to distinguish the real data from the
   synthetic data. Optionally, specific features can be selected.

   :param real: Dataframe containing the real data.
   :type real: pandas.DataFrame
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas.DataFrame
   :param feats: Features to combine. If `None` (default), all common features
                 are used.
   :type feats: list of str or None, default None
   :param source_col_name: Name of the source column. This is added to the combined dataset
                           and filled with the `source_val_real` and `source_val_synth`
                           values to signify the real and synthetic data respectively.
                           Defaults to `"source"`.
   :type source_col_name: str, default "source"
   :param source_val_real: Value to use in `source_col_name` column to signify the real
                           data. Defaults to `"real"`.
   :type source_val_real: any, default "real"
   :param source_val_synth: Value to use in `source_col_name` column to signify the
                            synthetic data. Defaults to `"synth"`.
   :type source_val_synth: any, default "synth"

   :returns: **combined** -- The combined dataframe.
   :rtype: pandas.DataFrame


.. py:function:: df_separate(data, source_col_name, feats=None, source_val_real='real', source_val_synth='synth', drop_source_col=True)

   Separate a dataframe into real and synthetic data.

   The dataframe is split using a source column and real and synthetic
   flags. Optionally, specific features can be selected.

   :param data: Dataframe to split into real and synthetic components.
   :type data: pandas.DataFrame
   :param source_col_name: Name of the column used to signify real versus synthetic data.
   :type source_col_name: str
   :param feats: Features to separate. If `None` (default), uses all features.
   :type feats: list of str or None, default None
   :param source_val_real: Value in `source_col_name` column signifying the real data.
                           Defaults to `"real"`.
   :type source_val_real: any, default "real"
   :param source_val_synth: Value in `source_col_name` column signifying the synthetic data.
                            Defaults to `"synth"`.
   :type source_val_synth: any, default "synth"
   :param drop_source_col: Indicates whether the `source_col_name` column should be
                           dropped from the outputs (default) or not.
   :type drop_source_col: bool, default True

   :returns: * **real** (*pandas.DataFrame*) -- Dataframe containing the real data.
             * **synth** (*pandas.DataFrame*) -- Dataframe containing the synthetic data.


.. py:function:: launder(real, synth, feats=None, suffix_real='real', suffix_synth='synth')

   Launder feature names and optionally filter.

   To provide clear distinction between the real and synthetic
   features, each dataframe is updated to append suffixes to the
   feature names. Optionally, specific features can be selected.

   :param real: Dataframe containing the real data.
   :type real: pandas.DataFrame
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas.DataFrame
   :param feats: Features to launder. If `None` (default), all common features
                 are used.
   :type feats: list of str or None, default None
   :param suffix_real: Suffix to append to columns in `real`. Default is `"real"`.
   :type suffix_real: str, default "real"
   :param suffix_synth: Suffix to append to columns in `synth`. Default is `"synth"`.
   :type suffix_synth: str, default "synth"

   :returns: * **real** (*pandas.DataFrame*) -- Dataframe containing the laundered real data.
             * **synth** (*pandas.DataFrame*) -- Dataframe containing the laundered synthetic data.


.. py:function:: cat_encode(df, feats=None, return_all=False, convert_only=False, force=False)

   Convert object features to categories.

   Generates a new version of the input dataframe with the specified
   features categorically encoded with integer labels. Optionally, the
   features can be returned as `category` data type with no encoding.

   Before performing the conversion, a check is made to identify any
   speficied features that are not `object`-type and thus less suited
   to categorical encoding. A warning is raised for these features and
   they will be ignored from subsequent encoding steps unless `force`
   is set to `True`.

   :param df: Input dataframe to be converted.
   :type df: pandas.DataFrame
   :param feats: Features in `df` to convert to categorical. If `None` (default),
                 all object-type columns are selected.
   :type feats: list of str or None, default None
   :param return_all: If `True`, all features in `df` will be returned regardless of
                      whether they were converted. If `False` (default), only the
                      converted features are returned.
   :type return_all: bool, default False
   :param convert_only: If `True`, the features will only be converted to the `category`
                        data-type without being integer-encoded.
   :type convert_only: bool, default False
   :param force: If `True`, all features in `feats` will be encoded regardless of
                 their data-type.
   :type force: bool, default False

   :Warns: **UserWarning** -- A warning is given if any of the features in `feats` are not of
           an `object` data type.

   :returns: * **out_df** (*pandas.DataFrame*) -- Dataframe with (at least) the converted features.
             * **cat_dict** (*dict or NoneType*) -- A dictionary mapping each encoded feature to its categories. If
               `convert_only=True`, returns as `None`.


