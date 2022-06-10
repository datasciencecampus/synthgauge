"""
Tests for metrics.

Note: many of these tests are just checking the output hasn't changed from when
the tests were written. In some cases more work can be done to check that the
outputs are sensible.
"""
import pytest

import synthgauge as sg

from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def evaluator():
    return sg.Evaluator(
        sg.datasets.make_blood_types_df(0, 0, 314),
        sg.datasets.make_blood_types_df(1, 0, 314),
    )


def test_wrappers(evaluator):
    """Checks all scipy wrapper metrics. Futureproofs against breaking
    scipy changes.
    """
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
        0.0166117691741195
    )
    assert results["jensen_shannon_divergence"] == pytest.approx(
        0.01942803292479766
    )
    assert results["kolmogorov_smirnov"].statistic == pytest.approx(0.088)
    assert results["kolmogorov_smirnov"].pvalue == pytest.approx(
        0.0008613642727365059
    )
    assert results["kullback_leibler"] == pytest.approx(0.0944474889121178)
    assert results["mann_whitney"].statistic == pytest.approx(491047.0)
    assert results["mann_whitney"].pvalue == pytest.approx(0.48780212969583725)
    assert results["wasserstein"] == pytest.approx(1.8290000000000002)
    assert results["wilcoxon"].statistic == pytest.approx(203551.0)
    assert results["wilcoxon"].pvalue == pytest.approx(0.2546321509324163)
    assert results["kruskal_wallis"].statistic == pytest.approx(
        0.4814244074541294
    )
    assert results["kruskal_wallis"].pvalue == pytest.approx(
        0.48777782605216824
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
    ].precision_difference == pytest.approx(0.1959776822276822)
    assert result[
        "classification_comparison"
    ].recall_difference == pytest.approx(0.1530778893694964)
    assert result["classification_comparison"].f1_difference == pytest.approx(
        0.17732379554135785
    )


def test_cluster_metric(evaluator):
    """Check clustering metric."""
    evaluator.add_metric("multi_clustered_MSD", random_state=24)
    evaluator.evaluate()
    assert evaluator.metric_results == pytest.approx(
        {"multi_clustered_MSD": 0.00291989251686431}
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
            "correlation_MSD": 0.008427408567868748,
            "cramers_v_MSE": 0.005525979344857528,
            "correlation_ratio_MSE": 0.03648278105113976,
        }
    )


def test_propensity_metric_logistic(evaluator):
    """Check Propensity metric using Logistic Regression model."""
    evaluator.add_metric(
        "propensity_metrics",
        metric_alias="propensity_logrg",
        method="LogisticRegression",
        random_state=50,
        max_iter=10000,
        solver="sag",
    )
    results = evaluator.evaluate()
    assert results["propensity_logrg"].observed_p_MSE == pytest.approx(
        0.014081072437202892
    )
    assert results["propensity_logrg"].standardised_p_MSE == pytest.approx(
        31.523736020704618
    )
    assert results["propensity_logrg"].ratio_p_MSE == pytest.approx(
        10.728436142630775
    )


def test_propensity_metric_CART(evaluator):
    """Check Propensity metric using CART model."""
    evaluator.add_metric(
        "propensity_metrics", method="CART", random_state=50, num_perms=1000
    )
    results = evaluator.evaluate()
    assert results["propensity_metrics"].observed_p_MSE == pytest.approx(
        0.24841666666666665
    )
    assert results["propensity_metrics"].standardised_p_MSE < 3.0
    assert results["propensity_metrics"].standardised_p_MSE > 2.7
    assert results["propensity_metrics"].ratio_p_MSE > 1.007
    assert results["propensity_metrics"].ratio_p_MSE < 1.008


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
