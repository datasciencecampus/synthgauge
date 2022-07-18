"""Simple test to check the version number."""

import re

import synthgauge as sg


def test_version_regex():
    """Check that the version number is in semantic versioning form.
    Regex pattern lifted from https://semver.org.
    """

    pattern = (
        "^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)"
        "(?:-((?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        "(?:\\.(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
        "?(?:\\+([0-9a-zA-Z-]+(?:\\.[0-9a-zA-Z-]+)*))?$"
    )
    match = re.fullmatch(pattern, sg.__version__)

    assert match is not None
    assert match.group() == sg.__version__
