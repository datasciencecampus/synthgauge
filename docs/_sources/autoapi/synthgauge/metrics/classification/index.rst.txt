:py:mod:`synthgauge.metrics.classification`
===========================================

.. py:module:: synthgauge.metrics.classification


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.metrics.classification.classification_comparison



.. py:function:: classification_comparison(real, synth, key, target, sklearn_classifier, random_state=None, test_prop=0.2, **kwargs)

   Classification utility metric

   This metric fits two classification models to `real` and `synth`
   respectively, and then tests them both against withheld `real` data. We
   obtain utility scores by subtracting the precision, recall and f1 scores
   of the predictions obtained by the synth model from those obtained by the
   real model.

   :param real: Dataframe containing the real data.
   :type real: pandas dataframe
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas dataframe
   :param key: list of column names to use as the input in the classification.
   :type key: list of strings
   :param target: column to use as target in the classification.
   :type target: str
   :param sklearn_classifier: classifier with fit and predict methods.
   :type sklearn_classifier: scikit-learn estimator
   :param random_state: Controls the shuffling steps during the train-test split and the
                        classification algorithm itself. Pass an int for reproducible output
                        across multiple function calls.
   :type random_state: int, RandomState instance or None, default=42
   :param test_prop: If float, should be between 0.0 and 1.0 and represent the proportion
                     of the dataset to include in the test split. If int, represents the
                     absolute number of test samples.
   :type test_prop: float or int, default=0.2

   :returns: **ClassificationResult** --

             precision_difference : float
                 precision of model trained on real data subtracted by precision of
                 model trained on synthetic data.
             recall_difference : float
                 recall of model trained on real data subtracted by recall of
                 model trained on synthetic data.
             f1_difference : float
                 f1 score of model trained on real data subtracted by f1 score of
                 model trained on synthetic data.
   :rtype: namedtuple

   .. rubric:: Notes

   Some preprocessing is carried out before the models are trained. Numeric
   features are scaled and categorical features are one-hot-encoded.

   A score of zero tells us the synthetic data is just as good as the real at
   training classifier models. Increases in these scores indicate poorer
   utility.


