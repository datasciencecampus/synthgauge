:py:mod:`synthgauge.datasets`
=============================

.. py:module:: synthgauge.datasets


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.datasets.make_blood_types_df



.. py:function:: make_blood_types_df(noise=0, proportion_nan=0, random_seed=42)

   Create Dummy Data for examples

   This function creates dummy data for the synthgauge examples.

   :param noise: Standard deviation of the Gaussian noise to add to the data, default
                 zero.
   :type noise: int
   :param proportion_nan: Proportion of dataset to replace with nans.
   :type proportion_nan: float [0,1]
   :param random_seed: Use for reproducibility.
   :type random_seed: int

   :returns:
   :rtype: pd.DataFrame

   .. rubric:: Notes

   The amout of noise can be tuned to crudely simulate the creation of
   synthetic data.


