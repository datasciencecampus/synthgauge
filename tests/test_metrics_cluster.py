"""Property-based tests for clustering metrics."""

import string

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from synthgauge.metrics import cluster


def _combine_datasets(real_data, synth_data):
    """Concatenate the real and synthetic datasets."""

    combined = pd.concat((real_data, synth_data))
    indicator = np.array([0] * len(real_data) + [1] * len(synth_data))

    return combined, indicator


@given(
    method=st.sampled_from(("kmeans", "kprototypes")), seed=st.integers(0, 100)
)
@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_get_cluster_labels(real, synth, method, seed):
    """Check that the cluster labels can be created correctly. For the
    sake of time, only the two-cluster case is tested and only a few
    examples are required to pass."""

    combined, _ = _combine_datasets(real, synth)

    labels = cluster._get_cluster_labels(
        combined, method, k=2, random_state=seed
    )

    assert isinstance(labels, np.ndarray)
    assert labels.dtype == int
    assert len(labels) == len(combined)
    assert set(labels) == {0, 1}


@given(st.text(string.ascii_letters, min_size=1))
def test_get_cluster_labels_error(method):
    """Check that a ValueError is raised if the clustering method is not
    one of k-means or k-prototypes."""

    with pytest.raises(ValueError, match=f"not {method}.$"):
        _ = cluster._get_cluster_labels(None, method, None, None)


@given(st.lists(st.integers(0, 1), min_size=200, max_size=200))
def test_get_cluster_proportions(labels):
    """Check that a valid set of cluster proportions can be obtained
    from toy labels. We only look at the two-cluster case here."""

    labels = np.array(labels)
    indicator = np.array([0] * 100 + [1] * 100)
    n_clusters = max(labels) + 1

    proportions = cluster._get_cluster_proportions(labels, indicator)

    assert isinstance(proportions, np.ndarray)
    assert proportions.shape == (n_clusters,)
    assert all(val >= 0 and val <= 1 for val in proportions)


@given(n_clusters=st.integers(2, 10), seed=st.integers(0, 100))
@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_clustered_MSD(real, synth, n_clusters, seed):
    """Check that the clustered mean-squared difference can be found.
    For the sake of time, only k-means is tested. Again, only a few
    examples are required."""

    combined, indicator = _combine_datasets(real, synth)

    msd = cluster.clustered_MSD(
        combined, indicator, method="kmeans", k=n_clusters, random_state=seed
    )

    assert isinstance(msd, float)
    assert msd >= 0 and msd <= 1


@given(
    k_range=st.tuples(st.integers(2, 4), st.integers(2, 4)).map(sorted),
    seed=st.integers(0, 100),
)
@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_multi_clustered_MSD(real, synth, k_range, seed):
    """Check that the multiple clustered mean-squared difference can be
    found. Usual restrictions on search space to save time."""

    k_min, k_max = k_range
    multi_msd = cluster.multi_clustered_MSD(
        real, synth, k_min=k_min, k_max=k_max, random_state=seed
    )

    assert isinstance(multi_msd, float)
    assert multi_msd >= 0 and multi_msd <= 1
