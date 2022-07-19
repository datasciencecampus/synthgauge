"""Property-based tests for the implemented metric functions."""

import inspect

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from synthgauge.metrics import classification

feats = ["age", "height", "weight", "eye_colour", "hair_colour"]
target = "blood_type"


@st.composite
def classifiers(
    draw, available_models=(SVC, DecisionTreeClassifier, KNeighborsClassifier)
):
    """Create a classifier-kwargs tuple for testing."""

    classifier = draw(st.sampled_from(available_models))

    if classifier is SVC:
        kwargs = {"kernel": "linear"}
    elif classifier is KNeighborsClassifier:
        kwargs = {"n_neighbors": 3}
    elif classifier is DecisionTreeClassifier:
        kwargs = {"max_depth": 3}
    else:
        kwargs = {}

    return classifier, kwargs


@st.composite
def toy_test_preds(draw, categories=("A", "B", "C", "D"), nrows=100):
    """Create toy sets of test and predicted classification labels."""

    strategy = st.lists(
        st.sampled_from(categories), min_size=nrows, max_size=nrows
    )
    test, pred = draw(strategy), draw(strategy)

    return test, pred


def test_make_preprocessor(real):
    """Test that a preprocessing pipeline can be created correctly."""

    preprocessor = classification._make_preprocessor(real, feats)

    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2

    for name, transformer, columns in preprocessor.transformers:
        assert isinstance(transformer, Pipeline)
        assert len(transformer.steps) == 1

        step_name, step_transformer = transformer.steps[0]
        assert name in ("numeric", "categorical")

        if name == "numeric":
            assert step_name == "scaler"
            assert isinstance(step_transformer, StandardScaler)
            assert list(columns) == ["age", "height", "weight"]
        if name == "categorical":
            assert step_name == "encoder"
            assert isinstance(step_transformer, OneHotEncoder)
            assert list(columns) == ["eye_colour", "hair_colour"]


@given(classifiers(), st.integers(0, 100))
def test_make_pipeline(classifier, seed):
    """Check that a full pipeline can be created correctly."""

    classifier, kwargs = classifier
    pipeline = classification._make_pipeline(classifier, seed, None, **kwargs)

    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0] == ("preprocessor", None)

    step_name, step_classifier = pipeline.steps[1]
    step_classifier_params = dict(inspect.getmembers(step_classifier))

    assert step_name == "classifier"
    assert isinstance(step_classifier, classifier)
    assert kwargs.items() <= step_classifier_params.items()

    instantiated_seed = step_classifier_params.get("random_state", None)
    assert instantiated_seed is None or instantiated_seed == seed


@given(toy_test_preds())
def test_get_scores(results):
    """Check that a valid array of scores can be created."""

    scores = classification._get_scores(*results)

    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(score >= 0 and score <= 1 for score in scores)


@given(classifier=classifiers(), seed=st.integers(0, 10))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_classification_comparison(real, synth, classifier, seed):
    """Check that a synthetic dataset can be evaluated by comparing the
    performance of a classifier (a linear SVM) on it and the 'real'
    data."""

    classifier, kwargs = classifier

    result = classification.classification_comparison(
        real,
        synth,
        feats,
        target,
        classifier=classifier,
        random_state=seed,
        **kwargs,
    )

    assert repr(result).startswith("ClassificationResult")
    assert result._fields == (
        "precision_difference",
        "recall_difference",
        "f1_difference",
    )
    assert all(res >= -1 and res <= 1 for res in result)


@given(classifier=classifiers(), seed=st.integers(0, 10))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_classification_identical_data(real, classifier, seed):
    """Check that any classification comparison does indeed return zero
    for all metrics when the real and synthetic data are the same."""

    classifier, kwargs = classifier

    result = classification.classification_comparison(
        real, real, feats, target, classifier, random_state=seed, **kwargs
    )

    assert all(res == 0 for res in result)
