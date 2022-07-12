"""Property-based tests for privacy metrics."""

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from synthgauge.metrics import privacy

from .utils import datasets


@given(datasets(column_spec={col: "object" for col in ("a", "b", "c")}))
def test_get_weap_scores(datasets):
    """Check that a WEAP score can be calculated for each row in a synthetic
    dataset."""

    _, synth = datasets
    assume(not synth.empty)

    *key, target = ["a", "b", "c"]

    weaps = privacy._get_weap_scores(synth, key, target)

    assert isinstance(weaps, pd.Series)
    assert pd.api.types.is_numeric_dtype(weaps)
    assert (weaps >= 0).all() and (weaps <= 1).all()


@given(datasets(column_spec={col: "object" for col in ("a", "b", "c")}))
def test_TCAP(datasets):
    """Check that a mean TCAP score can be calculated for a pair of
    datasets."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    *key, target = ["a", "b", "c"]

    score = privacy.TCAP(real, synth, key, target)

    assert isinstance(score, float)

    if real.equals(synth):
        assert score == 1
    else:
        assert score >= 0 and score < 1


@given(datasets(column_spec={col: "object" for col in ("a", "b", "c")}))
@settings(max_examples=20)
def test_TCAP_no_reduced_synthetic_data(datasets):
    """Check that the TCAP if-statement can be caught when no synthetic
    row's target variable is absolutely, uniquely identifiable from its
    key variables."""

    real, synth = datasets
    assume(not synth.empty and len(synth) > 1)

    *key, target = ["a", "b", "c"]

    synth.loc[:, key] = synth.loc[0, key].values
    synth.loc[:, target] = synth.index.astype(str)

    tcap = privacy.TCAP(real, synth, key, target)

    assert tcap == 0


@given(
    datasets(
        min_value=0,
        max_value=1000,
        allow_nan=False,
        available_dtypes=("float",),
    ),
    st.floats(min_value=0),
    st.integers(1, 5),
)
def test_find_outliers(datasets, threshold, neighbours):
    """Check that the local outlier detection mechanism works."""

    data, _ = datasets
    assume(not data.empty and len(data) > 5)

    outliers = privacy._find_outliers(data, threshold, neighbours)

    assert isinstance(outliers, np.ndarray)
    assert outliers.shape == (len(data),)
    assert pd.api.types.is_bool_dtype(outliers)

    if threshold == 0:
        assert all(np.equal(outliers, False))


@given(
    datasets(
        min_value=0,
        max_value=1000,
        allow_nan=False,
        available_dtypes=("float",),
    ),
    st.booleans(),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
def test_min_NN_dist(datasets, outliers_only):
    """Check that the minimum Nearest Neighbour distance can be
    calculated properly."""

    real, synth = datasets
    assume(
        not (real.empty or synth.empty) and (len(real) > 5 and len(synth) > 5)
    )

    distance = privacy.min_NN_dist(real, synth, None, outliers_only)

    assert isinstance(distance, float)

    real_unique_rows = {tuple(row) for _, row in real.iterrows()}
    synth_unique_rows = {tuple(row) for _, row in synth.iterrows()}
    if real_unique_rows.intersection(synth_unique_rows) and not outliers_only:
        assert distance == 0

    assert distance >= 0


@given(
    datasets(column_spec={"a": "object", "b": "object"}),
    st.one_of(st.none(), st.sampled_from(("a", ["a", "b"]))),
    st.one_of(st.floats(0, 1, allow_nan=False), st.integers(10, 100)),
    st.sampled_from(("unique", "sample")),
)
def test_sample_overlap_score(datasets, feats, size, score):
    """Check that sample overlap scores can be obtained."""

    real, synth = datasets
    assume(not (real.empty or synth.empty))

    score = privacy.sample_overlap_score(
        real, synth, feats, sample_size=size, score_type=score
    )

    assert isinstance(score, float)

    min_num_rows = min(len(real), len(synth))
    if int(size * min_num_rows) == 0:
        assert np.isnan(score)
    else:
        assert score >= 0 and score <= 1
