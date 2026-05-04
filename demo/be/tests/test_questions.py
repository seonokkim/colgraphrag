"""Test questions endpoints."""

from __future__ import annotations


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
