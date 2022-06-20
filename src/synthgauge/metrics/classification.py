""" Generic ``scikit-learn``-style classification utility metrics. """

import inspect
from collections import namedtuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_preprocessor(data, key):
    """Make a pre-processing pipe for transforming numeric and
    categorical columns."""

    numeric_columns = data[key].select_dtypes(include="number").columns
    categorical_columns = data[key].select_dtypes(exclude="number").columns

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ]
    )

    return preprocessor


def _make_pipeline(classifier, random_state, preprocessor, **kwargs):
    """Create the pipeline of data pre-processing and classification."""

    classifier_params = inspect.signature(classifier).parameters
    if classifier_params.get("random_state", None):
        kwargs["random_state"] = random_state

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier(**kwargs)),
        ]
    )

    return pipeline


def _get_scores(test, pred):
    """Calculate the precision, recall and f1 scores for a set of
    predicted values."""

    scores = [
        score_func(test, pred, average="macro")
        for score_func in (precision_score, recall_score, f1_score)
    ]

    return scores


def classification_comparison(
    real,
    synth,
    key,
    target,
    sklearn_classifier,
    random_state=None,
    test_prop=0.2,
    **kwargs
):
    """Classification utility metric

    This metric fits two classification models to `real` and `synth`
    respectively, and then tests them both against withheld `real` data. We
    obtain utility scores by subtracting the precision, recall and f1 scores
    of the predictions obtained by the synth model from those obtained by the
    real model.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    key : list of strings
        list of column names to use as the input in the classification.
    target : str
        column to use as target in the classification.
    sklearn_classifier : scikit-learn estimator
        classifier with fit and predict methods.
    random_state : int, RandomState instance or None, default=42
        Controls the shuffling steps during the train-test split and the
        classification algorithm itself. Pass an int for reproducible output
        across multiple function calls.
    test_prop : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    Returns
    -------
    ClassificationResult : namedtuple
        precision_difference : float
            precision of model trained on real data subtracted by precision of
            model trained on synthetic data.
        recall_difference : float
            recall of model trained on real data subtracted by recall of
            model trained on synthetic data.
        f1_difference : float
            f1 score of model trained on real data subtracted by f1 score of
            model trained on synthetic data.

    Notes
    -----
    Some preprocessing is carried out before the models are trained. Numeric
    features are scaled and categorical features are one-hot-encoded.

    A score of zero tells us the synthetic data is just as good as the real at
    training classifier models. Increases in these scores indicate poorer
    utility.
    """

    real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(
        real[key], real[target], test_size=test_prop, random_state=random_state
    )

    synth_X_train, _, synth_y_train, _ = train_test_split(
        synth[key],
        synth[target],
        test_size=test_prop,
        random_state=random_state,
    )

    # preprocessing
    preprocessor = _make_preprocessor(real, key)

    # train real model, test on real
    real_pipeline = _make_pipeline(
        sklearn_classifier, random_state, preprocessor, **kwargs
    )

    real_pipeline.fit(real_X_train, real_y_train.values.ravel())
    y_real_predicts_real = real_pipeline.predict(real_X_test)

    # train synth model, test on real
    synth_pipeline = _make_pipeline(
        sklearn_classifier, random_state, preprocessor, **kwargs
    )

    synth_pipeline.fit(synth_X_train, synth_y_train.values.ravel())
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


if __name__ == "__main__":
    pass
