"""Test graph endpoints."""

from __future__ import annotations


def test_get_graph(client):
    """Graph endpoint returns nodes and edges."""
    questions_response = client.get("/api/questions")
    questions = questions_response.json()

    question_with_graph = None
    for q in questions:
        if q.get("has_graph"):
            question_with_graph = q
            break

    if question_with_graph is None:
        return

    qid = question_with_graph["qid"]
    response = client.get(f"/api/graphs/{qid}")
    assert response.status_code == 200
    data = response.json()
    assert data["qid"] == qid
    assert "nodes" in data
    assert "edges" in data
    assert "node_count" in data
    assert "edge_count" in data
    assert data["node_count"] == len(data["nodes"])
    assert data["edge_count"] == len(data["edges"])


def test_get_graph_not_found(client):
    """Non-existent graph returns 404."""
    response = client.get("/api/graphs/nonexistent_qid_12345")
    assert response.status_code == 404


def test_get_graphml_file(client):
    """GraphML file endpoint returns XML."""
    questions_response = client.get("/api/questions")
    questions = questions_response.json()

    question_with_graph = None
    for q in questions:
        if q.get("has_graph"):
            question_with_graph = q
            break

    if question_with_graph is None:
        return

    qid = question_with_graph["qid"]
    response = client.get(f"/api/graphs/{qid}/graphml")
    assert response.status_code == 200
    assert "xml" in response.headers.get("content-type", "")
