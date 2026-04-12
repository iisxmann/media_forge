"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def fixtures_dir():
    """Directory for test fixture files."""
    d = Path(__file__).parent / "fixtures"
    d.mkdir(exist_ok=True)
    return d
