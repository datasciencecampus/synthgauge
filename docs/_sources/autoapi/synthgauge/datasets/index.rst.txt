:py:mod:`synthgauge.datasets`
=============================

.. py:module:: synthgauge.datasets

.. autoapi-nested-parse::

   Functions for creating toy datasets.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.datasets._adjust_data_elements
   synthgauge.datasets.make_blood_types_df



.. py:function:: _adjust_data_elements(data, labels, noise, nan_prop, seed)

   Adjust the given data and put it into a dataframe.
   This function is not intended to be used directly by users.

   :param data: The data array to be adjusted.
   :type data: numpy.ndarray
   :param labels: A set of labels for classifying the rows of `data`.
   :type labels: numpy.ndarray
   :param noise: The amount of noise to inject into the data. Specifically,
                 this controls the `scale` parameter of a zero-centred normal
                 distribution.
   :type noise: float
   :param nan_prop: The proportion of elements to replace with missing values.
   :type nan_prop: float
   :param seed: A random seed used to choose missing element indices and sample
                noise.
   :type seed: int

   :returns: The adjusted, combined dataframe.
   :rtype: pandas.DataFrame


.. py:function:: make_blood_types_df(noise=0, nan_prop=0, seed=None)

   Create a toy dataset about blood types and physical atrtibutes.

   This function is used to create data for the package's examples and
   its tests. Its outputs are not intended to imply or be used for any
   meaningful data analysis.

   :param noise: Standard deviation of the Gaussian noise added to the data.
                 Default is zero (no noise) and must be non-negative.
   :type noise: float
   :param nan_prop: Proportion of dataset to replace with missing values.
   :type nan_prop: float, default 0
   :param seed: Seed used by all random samplers. Used for reproducibility.
   :type seed: int, optional

   :returns: **data** -- A toy "blood type" dataset.
   :rtype: pandas.DataFrame

   .. rubric:: Notes

   The amount of noise can be tuned to crudely simulate the creation of
   synthetic data.


