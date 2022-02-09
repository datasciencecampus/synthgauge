:py:mod:`synthgauge.utils`
==========================

.. py:module:: synthgauge.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.utils.load_pickle
   synthgauge.utils.df_combine
   synthgauge.utils.df_separate
   synthgauge.utils.launder
   synthgauge.utils.cat_encode
   synthgauge.utils.feature_density_diff



.. py:function:: load_pickle(src)

   Load pickle file into object.




.. py:function:: df_combine(real, synth, feats=None, source_col_name='source', source_val_real='real', source_val_synth='synth')

   Combine separate dataframes of real and synthetic data.

   The dataframes are concatenated along axis=0 and a source column is
   added to distinguish the real data from the sythetic data.

   Optionally, specific features can be selected.

   :param real: DataFrame containing the real, unsythesised data.
   :type real: pandas.DataFrame
   :param synth: DataFrame containing the synthesised data.
   :type synth: pandas.DataFrame
   :param feats: Features to combine. By default all common features are used.
   :type feats: str or list of str, optional
   :param source_col_name: Name of the source column. This is added to the combined DataFrame
                           and filled with the `source_val_real` and `source_val_synth`
                           values to signify the real and synthetic data respectively.
                           Defaults to "source".
   :type source_col_name: str, optional
   :param source_val_real: Value to use in `source_col_name` column to signify the real data.
                           Any value accepted by pandas.DataFrame. Defaults to "real".
   :type source_val_real: any, optional
   :param source_val_synth: Value to use in `source_col_name` column to signify the sythetic
                            data. Any value accepted by pandas.DataFrame. Defaults to "synth".
   :type source_val_synth: any, optional

   :returns: The result of the concatenation of the input DataFrame objects.
   :rtype: pandas.DataFrame


.. py:function:: df_separate(data, source_col_name, feats=None, source_val_real='real', source_val_synth='synth', drop_source_col=True)

   Separate a dataframe into real and synthetic data.

   The dataframe is split using a source column and real and synthetic
   flags. Optionally, specific features can be selected.

   :param data: DataFrame to split into real and synthetic components.
   :type data: pandas.DataFrame
   :param source_col_name: Name of the column used to signify real versus synthetic data.
   :type source_col_name: str
   :param feats: Features to separate. By default all features are used.
   :type feats: str or list of str, optional
   :param source_val_real: Value in `source_col_name` column signifying the real data. Any
                           value accepted by pandas.DataFrame. Defaults to "real".
   :type source_val_real: any, optional
   :param source_val_synth: Value in `source_col_name` column signifying the synthetic data.
                            Any value accepted by pandas.DataFrame. Defaults to "synth".
   :type source_val_synth: any, optional
   :param drop_source_col: If True, the default, the `source_col_name` column will be dropped
                           from the output DataFrames.
   :type drop_source_col: bool, optional

   :returns: Two DataFrame objects containing the real data and synthetic
             data respectively.
   :rtype: pandas.DataFrame, pandas.DataFrame


.. py:function:: launder(real, synth, feats=None, suffix_real='real', suffix_synth='synth')

   Launder feature names and optionally filter.

   To provide clear distinction between the real and synthetic
   features, each dataframe is updated to append suffixes to the feature
   names.

   Optionally, specific features can be selected.

   :param real: DataFrame containing the real, unsythesised data
   :type real: pandas.DataFrame
   :param synth: DataFrame containing the synthesised data
   :type synth: pandas.DataFrame
   :param feats: Features to launder. By default all common features are used.
   :type feats: str or list of str, optional
   :param suffix_real: Suffix to append to columns in `real`. Default is "real".
   :type suffix_real: str, optional
   :param suffix_synth: Suffix to append to columns in `synth`. Default is "synth".
   :type suffix_synth: str, optional

   :returns: Two DataFrame objects containing the real data and synthetic
             data respectively.
   :rtype: pandas.DataFrame, pandas.DataFrame


.. py:function:: cat_encode(df, feats=None, return_all=False, convert_only=False, force=False)

   Convert object features to categories.

   Generates a new version of the input datframe with the
   specified features categorically encoded with integer labels.
   Optionally, the features can be returned as the 'category' type
   with no encoding.

   Before performing the conversion a check is made to identify any
   speficied features that are not 'object' type and therefore less
   suited to categorical encoding. A warning is raised for these features
   and they will be ignored from subsequent encoding steps unless
   `force` is `True`.

   :param df: Input dataframe to be converted.
   :type df: pandas.DataFrame
   :param feats: Feature(s) in `df` to convert to category. By default all object
                 columns are selected.
   :type feats: str or list of str, optional
   :param return_all: If True all features in `df` will be returned with only those
                      specified in `feats` converted. Otherwise only the converted
                      features are returned. This is the default.
   :type return_all: bool, default=False
   :param convert_only: If True the features will only be converted to the 'category'
                        type and NOT encoded.
   :type convert_only: bool, default=False
   :param force: If true, all specified features will be encoded regardless of
                 data type.
   :type force: bool, optional

   :returns: * *pandas.DataFrame* -- DataFrame with converted features
             * *dict or NoneType* -- A dictionary mapping each encoded feature to its categories. If
               convert_only=True NoneType is returned.

   :Warns: **UserWarning** -- A warning is given if any of the specified features for
           conversion are not an 'object' type.


.. py:function:: feature_density_diff(real, synth, feature, bins=10)

   Computes the difference between real and synth feature densities.

   For the specified feature the density is computed across `bins` in both
   the `real` and `synth` data sets. The per-bin difference is computed and
   returned along with the bin edges that were used.

   Prior to calculating the densities all values are converted to numeric
   via `utils.cat_encode`.

   :param real: DataFrame containing the real data.
   :type real: pandas.DataFrame
   :param synth: DataFrame containing the sythetic data.
   :type synth: pandas.DataFrame
   :param feature: The feature that will be used to compute the density.
   :type feature: str
   :param bins: Bins to use for computing the density. This value is passed
                to `numpy.histogram_bin_edges` so can be any value accepted by
                that function. The default setting of 10 uses 10 bins.
   :type bins: str or int, optional

   :returns: * **hist_diff** (*np.ndarry*) -- The difference in feature density for each of the bins.
             * **bin_edges** (*np.ndarray*) -- The edges of the bins.


