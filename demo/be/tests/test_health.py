"""Test health endpoint."""

from __future__ import annotations


def test_health_check(client):
    """Health endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "run_id" in data


def test_root_endpoint(client):
    """Root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["docs"] == "/docs"


def test_datasets_list(client, all_demo_paths):
    """Dataset registry lists WebQA and MMQA; availability matches on-disk config."""
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert data.get("default") == "webqa"
    rows = {d["key"]: d for d in data["datasets"]}
    assert "webqa" in rows and "mmqa" in rows
    assert rows["webqa"]["label"] == "WebQA"
    assert rows["mmqa"]["label"] == "MultimodalQA"
    assert rows["webqa"]["available"] is True
    assert rows["webqa"]["run_id"] == all_demo_paths.webqa.run_id
    expect_mmqa = all_demo_paths.mmqa is not None
    assert rows["mmqa"]["available"] is expect_mmqa
    if expect_mmqa:
        assert rows["mmqa"]["run_id"] == all_demo_paths.mmqa.run_id
