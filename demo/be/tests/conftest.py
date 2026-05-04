"""Pytest fixtures for demo BE tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_BE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BE_DIR))

from server import app, _init_services
from config.resolve_demo_paths import load_demo_paths


@pytest.fixture(scope="session")
def demo_paths():
    """Load demo paths once per test session."""
    try:
        return load_demo_paths()
    except FileNotFoundError:
        pytest.skip("No result data available for testing")


@pytest.fixture(scope="session")
def client(demo_paths) -> TestClient:
    """Create test client with initialized services."""
    _init_services(app, demo_paths)
    return TestClient(app)
