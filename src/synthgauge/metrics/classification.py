"""Utility metrics using `scikit-learn`-style classifiers."""

import inspect
from collections import namedtuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_preprocessor(data, feats):
    """Make a pre-processing pipe for transforming numeric and
    categorical columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing at the least the columns in `feats`.
    feats : list of str
        A list of columns in `data` to be separated by data type.

    Returns
    -------
    preprocessor : sklearn.pipeline.Pipeline
        The pre-processing pipeline.
    """

    numeric = data[feats].select_dtypes(include="number").columns
    categorical = data[feats].select_dtypes(exclude="number").columns

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric),
            ("categorical", categorical_transformer, categorical),
        ]
    )

    return preprocessor


def _make_pipeline(classifier, seed, preprocessor, **kwargs):
    """Create the pipeline of data pre-processing and classification.

    Parameters
    ----------
    classifier : scikit-learn estimator
        The `scikit-learn`-style class to be used as the classifier.
    seed : int
        Random seed to use for reproducibility. Only used if
        `random_state` is a parameter of `classifier`.
    preprocessor : sklearn.pipeline.Pipeline
        The pre-processing pipeline.
    **kwargs : dict, optinal
        featsword arguments for `classifier`.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        A complete classification pipeline.
    """

    classifier_params = inspect.signature(classifier).parameters
    if classifier_params.get("random_state", None):
        kwargs["random_state"] = seed

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier(**kwargs)),
        ]
    )

    return pipeline


def _get_scores(test, pred):
    """Calculate the precision, recall and f1 scores for a set of
    predicted values.

    Parameters
    ----------
    test : array_like
        Labels from the test set.
    pred : array_like
        Predicted labels.

    Returns
    -------
    scores : list
        The precision, recall and f1 score given the test set and
        predicted labels.
    """

    scores = [
        score_func(test, pred, average="macro")
        for score_func in (precision_score, recall_score, f1_score)
    ]

    return scores


def classification_comparison(
    real,
    synth,
    feats,
    target,
    classifier,
    test_prop=0.2,
    random_state=None,
    **kwargs
):
    """Classification utility metric.

    This metric fits two (identical) classification models to `real` and
    `synth`, and then tests them both against withheld `real` data. We
    obtain utility scores by subtracting the precision, recall and f1
    scores of the "synthetic" model predictions from the "real" model's.

    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    feats : list of str
        List of column names to use as the input in the classification.
    target : str
        Column to use as target in the classification.
    classifier : scikit-learn estimator
        Classifier class with `fit` and `predict` methods.
    test_prop : float or int, default 0.2
        If `float`, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        `int`, represents the absolute number of test samples.
    random_state : int, optional
        Random seed for shuffling during the train-test split, and for
        the classification algorithm itself.
    **kwargs : dict, optional
        featsword arguments passed to the classifier.

    Returns
    -------
    precision_difference : float
        Precision of the real model subtracted by that of the
        synthetic model.
    recall_difference : float
        Recall of the real model subtracted by that of the synthetic
        model.
    f1_difference : float
        f1 score of the real model subtracted by that of the
        synthetic model.

    Notes
    -----
    Some preprocessing is carried out before the models are trained.
    Numeric features are scaled and categorical features are
    one-hot-encoded.

    A score of zero tells us the synthetic data is just as good as the
    real at training the given classification model. Increases in these
    scores indicate poorer utility.
    """

    real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(
        real[feats],
        real[target],
        test_size=test_prop,
        random_state=random_state,
    )

    synth_X_train, _, synth_y_train, _ = train_test_split(
        synth[feats],
        synth[target],
        test_size=test_prop,
        random_state=random_state,
    )

    # preprocessing
    preprocessor = _make_preprocessor(real, feats)

    # train real model, test on real
    real_pipeline = _make_pipeline(
        classifier, random_state, preprocessor, **kwargs
    ).fit(real_X_train, real_y_train.values.ravel())

    y_real_predicts_real = real_pipeline.predict(real_X_test)

    # train synth model, test on real
    synth_pipeline = _make_pipeline(
        classifier, random_state, preprocessor, **kwargs
    ).fit(synth_X_train, synth_y_train.values.ravel())

    y_synth_predicts_real = synth_pipeline.predict(real_X_test)

    # compare results
    real_scores = _get_scores(real_y_test, y_real_predicts_real)
    synth_scores = _get_scores(real_y_test, y_synth_predicts_real)
    score_differences = np.subtract(real_scores, synth_scores)

    ClassificationResult = namedtuple(
        "ClassificationResult",
        ("precision_difference", "recall_difference", "f1_difference"),
    )

    return ClassificationResult(*score_differences)
