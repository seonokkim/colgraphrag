"""Test run info and scores endpoints."""

from __future__ import annotations


def test_run_info(client):
    """Run info endpoint returns metadata."""
    response = client.get("/api/run/info")
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert "total_questions" in data
    assert "scored_questions" in data
    assert data["total_questions"] >= 0


def test_run_scores(client):
    """Run scores endpoint returns QA metrics."""
    response = client.get("/api/run/scores")
    assert response.status_code == 200
    data = response.json()
    assert "all" in data
    assert "unimodal" in data
    assert "multimodal" in data

    all_scores = data["all"]
    assert "qa_fl" in all_scores
    assert "qa_acc" in all_scores
    assert "qa" in all_scores
    assert 0 <= all_scores["qa_fl"] <= 100
    assert 0 <= all_scores["qa_acc"] <= 100
