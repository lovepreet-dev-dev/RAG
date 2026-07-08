"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the test fixtures directory."""
    d = PROJECT_ROOT / "tests" / "fixtures"
    d.mkdir(parents=True, exist_ok=True)
    return d
