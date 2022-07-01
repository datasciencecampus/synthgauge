"""Fixtures used by several tests."""

import pytest

import synthgauge as sg


@pytest.fixture
def real():
    """Make some real (noiseless) data."""

    return sg.datasets.make_blood_types_df(0, 0)


@pytest.fixture
def synth():
    """Make some synthetic (noisy) data."""

    return sg.datasets.make_blood_types_df(1, 0)
