"""
Tests for the Evaluator object and its metric methods.
"""
from copy import deepcopy
from statistics import mean
import pytest
import synthgauge as sg


@pytest.fixture
def evaluator():
    return sg.Evaluator(sg.datasets.make_blood_types_df(0, 0, 314),
                        sg.datasets.make_blood_types_df(1, 0, 314))


def test_add_metric(evaluator):
    """Check basic add_metric functionality adds metric to metric
    dictionary.
    """
    evaluator.add_metric('correlation_ratio_MSE')
    assert evaluator.metrics == {'correlation_ratio_MSE': {}}


def test_add_metric_alias(evaluator):
    """Check add_metric functionality with alias."""
    evaluator.add_metric('correlation_ratio_MSE', metric_alias='cor')
    assert evaluator.metrics == {
        'cor': {'metric_name': 'correlation_ratio_MSE'}}


def test_add_metric_arguments(evaluator):
    """Check add_metric functionality with arguments."""
    evaluator.add_metric('wasserstein', feature='age')
    assert evaluator.metrics == {'wasserstein': {'feature': 'age'}}


def test_add_metric_not_implemented(evaluator):
    """Check adding unknown metric returns error."""
    with pytest.raises(NotImplementedError):
        evaluator.add_metric('custom')


def test_custom_metric_def(evaluator):
    """Check adding custom metric with defined function."""
    def length_diff(real, synth):
        return abs(len(real)-len(synth))
    evaluator.add_custom_metric('length_diff', length_diff)
    assert evaluator.evaluate() == {'length_diff': 0}


def test_custom_metric_args(evaluator):
    """Check adding custom metric with arguments."""
    def mean_diff(real, synth, feat):
        return abs(mean(real[feat]) - mean(synth[feat]))
    evaluator.add_custom_metric('mean_diff', mean_diff, feat='weight')
    assert evaluator.evaluate() == {'mean_diff': 0.19900000000001228}


def test_copy_metrics(evaluator):
    """Check copying metrics from one Evaluator to another."""
    evaluator_2 = deepcopy(evaluator)
    evaluator.add_metric('wasserstein', feature='age')
    evaluator.add_metric('correlation_ratio_MSE', metric_alias='cor')
    evaluator.add_custom_metric('len_diff',
                                lambda real, synth: abs(len(real)-len(synth)))
    evaluator_2.copy_metrics(evaluator)
    assert evaluator.metrics == evaluator_2.metrics


def test_copy_metrics_empty(evaluator):
    """Check copying no metrics from one Evaluator to another."""
    evaluator_2 = deepcopy(evaluator)
    evaluator_2.copy_metrics(evaluator)
    assert evaluator_2.metrics == {}


def test_drop_metric(evaluator):
    """Check dropping metric functionality."""
    evaluator.add_metric('correlation_ratio_MSE')
    evaluator.drop_metric('correlation_ratio_MSE')
    assert evaluator.metrics == {}
