"""Fixtures used by several tests."""

import pytest
from hypothesis import settings

import synthgauge as sg

settings.register_profile("ci", deadline=60000)


@pytest.fixture
def real():
    """Make some real (noiseless) data."""

    return sg.datasets.make_blood_types_df(0, 0, 0)


@pytest.fixture
def synth():
    """Make some synthetic (noisy) data."""

    return sg.datasets.make_blood_types_df(1, 0, 0)
