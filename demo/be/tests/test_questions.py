"""Test questions endpoints."""

from __future__ import annotations

import pytest


def test_list_questions(client):
    """Questions list endpoint returns summaries."""
    response = client.get("/api/questions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        q = data[0]
        assert "qid" in q
        assert "question" in q
        assert "qcate" in q
        assert "gold_answer" in q


def test_get_question_detail(client):
    """Question detail endpoint returns full info."""
    list_response = client.get("/api/questions")
    questions = list_response.json()

    if not questions:
        return

    qid = questions[0]["qid"]
    response = client.get(f"/api/questions/{qid}")
    assert response.status_code == 200
    data = response.json()
    assert data["qid"] == qid
    assert "question" in data
    assert "gold_answers" in data
    assert "retrieval" in data
    assert "gold_facts" in data


def test_get_question_not_found(client):
    """Non-existent question returns 404."""
    response = client.get("/api/questions/nonexistent_qid_12345")
    assert response.status_code == 404


def test_list_questions_mmqa(client, mmqa_configured):
    """MMQA question list uses ``dataset=mmqa`` when a MMQA run exists."""
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run")

    response = client.get("/api/questions?dataset=mmqa")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    q = data[0]
    assert "qid" in q
    assert "question" in q
    assert "gold_answer" in q


def test_get_question_detail_mmqa(client, mmqa_configured):
    """MMQA detail endpoint returns the same schema as WebQA."""
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run")

    listing = client.get("/api/questions?dataset=mmqa").json()
    assert listing
    qid = listing[0]["qid"]
    response = client.get(f"/api/questions/{qid}?dataset=mmqa")
    assert response.status_code == 200
    data = response.json()
    assert data["qid"] == qid
    assert "question" in data
    assert "gold_answers" in data
    assert "gold_facts" in data


def test_get_question_not_found_mmqa(client, mmqa_configured):
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run")

    response = client.get("/api/questions/nonexistent_qid_12345?dataset=mmqa")
    assert response.status_code == 404
