"""Pytest fixtures for demo BE tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_BE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BE_DIR))

from config.resolve_demo_paths import MultiDatasetPaths, load_all_demo_paths
from routers.health_router import set_run_id
from server import app, init_demo_services


@pytest.fixture(scope="session")
def all_demo_paths() -> MultiDatasetPaths:
    """Load WebQA + optional MMQA paths once per test session."""
    try:
        return load_all_demo_paths()
    except FileNotFoundError:
        pytest.skip("No result data available for testing")


@pytest.fixture(scope="session")
def client(all_demo_paths) -> TestClient:
    """Test client with the same app state as production lifespan (WebQA + MMQA when present)."""
    init_demo_services(app, all_demo_paths)
    set_run_id(all_demo_paths.webqa.run_id)
    return TestClient(app)


@pytest.fixture(scope="session")
def mmqa_configured(all_demo_paths: MultiDatasetPaths) -> bool:
    """Whether MultimodalQA pipeline results are present for the demo."""
    return all_demo_paths.mmqa is not None
