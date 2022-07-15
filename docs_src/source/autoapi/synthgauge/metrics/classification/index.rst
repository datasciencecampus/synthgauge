:py:mod:`synthgauge.metrics.classification`
===========================================

.. py:module:: synthgauge.metrics.classification

.. autoapi-nested-parse::

   Utility metrics using `scikit-learn`-style classifiers.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   synthgauge.metrics.classification._make_preprocessor
   synthgauge.metrics.classification._make_pipeline
   synthgauge.metrics.classification._get_scores
   synthgauge.metrics.classification.classification_comparison



.. py:function:: _make_preprocessor(data, feats)

   Make a pre-processing pipe for transforming numeric and
   categorical columns.

   :param data: The dataset containing at the least the columns in `feats`.
   :type data: pandas.DataFrame
   :param feats: A list of columns in `data` to be separated by data type.
   :type feats: list of str

   :returns: **preprocessor** -- The pre-processing pipeline.
   :rtype: sklearn.pipeline.Pipeline


.. py:function:: _make_pipeline(classifier, seed, preprocessor, **kwargs)

   Create the pipeline of data pre-processing and classification.

   :param classifier: The `scikit-learn`-style class to be used as the classifier.
   :type classifier: scikit-learn estimator
   :param seed: Random seed to use for reproducibility. Only used if
                `random_state` is a parameter of `classifier`.
   :type seed: int
   :param preprocessor: The pre-processing pipeline.
   :type preprocessor: sklearn.pipeline.Pipeline
   :param \*\*kwargs: featsword arguments for `classifier`.
   :type \*\*kwargs: dict, optinal

   :returns: **pipeline** -- A complete classification pipeline.
   :rtype: sklearn.pipeline.Pipeline


.. py:function:: _get_scores(test, pred)

   Calculate the precision, recall and f1 scores for a set of
   predicted values.

   :param test: Labels from the test set.
   :type test: array_like
   :param pred: Predicted labels.
   :type pred: array_like

   :returns: **scores** -- The precision, recall and f1 score given the test set and
             predicted labels.
   :rtype: list


.. py:function:: classification_comparison(real, synth, feats, target, classifier, test_prop=0.2, random_state=None, **kwargs)

   Classification utility metric.

   This metric fits two (identical) classification models to `real` and
   `synth`, and then tests them both against withheld `real` data. We
   obtain utility scores by subtracting the precision, recall and f1
   scores of the "synthetic" model predictions from the "real" model's.

   :param real: Dataframe containing the real data.
   :type real: pandas.DataFrame
   :param synth: Dataframe containing the synthetic data.
   :type synth: pandas.DataFrame
   :param feats: List of column names to use as the input in the classification.
   :type feats: list of str
   :param target: Column to use as target in the classification.
   :type target: str
   :param classifier: Classifier class with `fit` and `predict` methods.
   :type classifier: scikit-learn estimator
   :param test_prop: If `float`, should be between 0.0 and 1.0 and represent the
                     proportion of the dataset to include in the test split. If
                     `int`, represents the absolute number of test samples.
   :type test_prop: float or int, default 0.2
   :param random_state: Random seed for shuffling during the train-test split, and for
                        the classification algorithm itself.
   :type random_state: int, optional
   :param \*\*kwargs: featsword arguments passed to the classifier.
   :type \*\*kwargs: dict, optional

   :returns: * **precision_difference** (*float*) -- Precision of the real model subtracted by that of the
               synthetic model.
             * **recall_difference** (*float*) -- Recall of the real model subtracted by that of the synthetic
               model.
             * **f1_difference** (*float*) -- f1 score of the real model subtracted by that of the
               synthetic model.

   .. rubric:: Notes

   Some preprocessing is carried out before the models are trained.
   Numeric features are scaled and categorical features are
   one-hot-encoded.

   A score of zero tells us the synthetic data is just as good as the
   real at training the given classification model. Increases in these
   scores indicate poorer utility.


