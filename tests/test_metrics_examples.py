"""
Tests for metrics.

Note: many of these tests are just checking the output hasn't changed
from when the tests were written. Property-based tests for each type of
metric can be found in files matching `tests/test_metrics_*.py`
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import synthgauge as sg


@pytest.fixture
def evaluator():
    """An evaluator storing the datasets used to obtain the values in these
    tests."""
    return sg.Evaluator(
        sg.datasets.make_blood_types_df(0, 0, 314),
        sg.datasets.make_blood_types_df(1, 0, 314),
    )


def test_wrappers(evaluator):
    """Checks all the implemented wrapper metrics."""

    evaluator.add_metric(
        "jensen_shannon_distance", feature="hair_colour", bins=None
    )
    evaluator.add_metric(
        "jensen_shannon_divergence", feature="height", bins="auto"
    )
    evaluator.add_metric("kolmogorov_smirnov", feature="weight")
    evaluator.add_metric("kruskal_wallis", feature="weight")
    evaluator.add_metric("kullback_leibler", feature="weight")
    evaluator.add_metric("mann_whitney", feature="weight")
    evaluator.add_metric("wasserstein", feature="weight")
    evaluator.add_metric("wilcoxon", feature="weight")
    results = evaluator.evaluate()
    assert results["jensen_shannon_distance"] == pytest.approx(
        0.12163299183162701
    )
    assert results["jensen_shannon_divergence"] == pytest.approx(
        0.016975399683905556
    )
    assert results["kolmogorov_smirnov"].statistic == pytest.approx(0.075)
    assert results["kolmogorov_smirnov"].pvalue == pytest.approx(
        0.007195361443046065
    )
    assert np.isinf(results["kullback_leibler"])
    assert results["mann_whitney"].statistic == pytest.approx(500400.0)
    assert results["mann_whitney"].pvalue == pytest.approx(0.9752996559986855)
    assert results["wasserstein"] == pytest.approx(1.509)
    assert results["wilcoxon"].statistic == pytest.approx(216830.0)
    assert results["wilcoxon"].pvalue == pytest.approx(0.8121496973678162)
    assert results["kruskal_wallis"].statistic == pytest.approx(
        0.0009610612500442349
    )
    assert results["kruskal_wallis"].pvalue == pytest.approx(
        0.9752687518318524
    )


def test_classification_metric(evaluator):
    """Check classification metric."""
    evaluator.add_metric(
        "classification_comparison",
        key=["age", "height", "weight", "hair_colour", "eye_colour"],
        target="blood_type",
        sklearn_classifier=RandomForestClassifier,
        random_state=42,
    )
    result = evaluator.evaluate()
    assert result[
        "classification_comparison"
    ].precision_difference == pytest.approx(0.43868978507905143)
    assert result[
        "classification_comparison"
    ].recall_difference == pytest.approx(0.28062078823769654)
    assert result["classification_comparison"].f1_difference == pytest.approx(
        0.41882618117604753
    )


def test_cluster_metric(evaluator):
    """Check clustering metric."""
    evaluator.add_metric("multi_clustered_MSD", random_state=24)
    evaluator.evaluate()
    assert evaluator.metric_results == pytest.approx(
        {"multi_clustered_MSD": 0.004462729520817462}
    )


def test_correlation_metrics(evaluator):
    """Check correlation metrics.
    These include Pearson's correlation, Cramer's V and the correlation ratio
    """
    evaluator.add_metric("correlation_MSD")
    evaluator.add_metric("cramers_v_MSE")
    evaluator.add_metric("correlation_ratio_MSE")
    evaluator.evaluate()
    assert evaluator.metric_results == pytest.approx(
        {
            "correlation_MSD": 0.007424314497449256,
            "cramers_v_MSE": 0.00971817461064437,
            "correlation_ratio_MSE": 0.06174834836187228,
        }
    )


def test_propensity_metric_logistic(evaluator):
    """Check Propensity metric using Logistic Regression model."""
    evaluator.add_metric(
        "propensity_metrics",
        metric_alias="propensity_logrg",
        method="logr",
        max_iter=1e4,
        solver="sag",
        random_state=0,
    )
    results = evaluator.evaluate()
    assert results["propensity_logrg"].observed_p_MSE == pytest.approx(
        0.05493054447730439
    )
    assert results["propensity_logrg"].standardised_p_MSE == pytest.approx(
        65.6818850109618
    )
    assert results["propensity_logrg"].ratio_p_MSE == pytest.approx(
        6.7606823972066445
    )


def test_propensity_metric_CART(evaluator):
    """Check Propensity metric using CART model."""
    evaluator.add_metric(
        "propensity_metrics",
        method="cart",
        num_perms=1000,
        random_state=0,
    )
    results = evaluator.evaluate()
    assert results["propensity_metrics"].observed_p_MSE == pytest.approx(
        0.24683333333333332
    )
    assert results["propensity_metrics"].standardised_p_MSE == pytest.approx(
        0.32171687896285317
    )
    assert results["propensity_metrics"].ratio_p_MSE == pytest.approx(
        1.0008369591991682
    )


def test_TCAP(evaluator):
    """Check TCAP metric."""
    evaluator.add_metric(
        "TCAP",
        key=["age", "height", "weight", "hair_colour", "eye_colour"],
        target="blood_type",
    )
    evaluator.evaluate()
    assert evaluator.metric_results == pytest.approx({"TCAP": 0.007})


def test_NN_dist(evaluator):
    """Check minimum nearest neighbour distance metric.
    In this example, the real and synth datasets have some matching records,
    so the minimum distance is zero.
    """
    evaluator.add_metric("min_NN_dist")
    evaluator.evaluate()
    assert evaluator.metric_results == pytest.approx({"min_NN_dist": 0})


def sample_overlap_score(evaluator):
    """Check sample overlap score metric.
    This test doesn't sample and just considers the full dataset, and so can
    take some time.
    """
    evaluator.add_metric("sample_overlap_score", runs=1, sample_size=1)
    evaluator.evaluate()
    assert evaluator.metric_results == pytest.approx(
        {"samle_overlap_score": 0.9583333333333334}
    )
