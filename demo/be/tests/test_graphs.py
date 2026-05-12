"""Test graph endpoints (WebQA + MultimodalQA) and MMQA path resolution for visualization."""

from __future__ import annotations

import pytest


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


def test_get_graph_not_found_mmqa(client, mmqa_configured):
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run")

    response = client.get("/api/graphs/nonexistent_qid_12345?dataset=mmqa")
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


def _first_question_with_graph(client, dataset: str):
    questions = client.get(f"/api/questions?dataset={dataset}").json()
    for q in questions:
        if q.get("has_graph"):
            return q
    return None


def test_get_graph_mmqa(client, mmqa_configured):
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run")

    q = _first_question_with_graph(client, "mmqa")
    if q is None:
        pytest.skip("No MMQA question with graph in this run")

    qid = q["qid"]
    response = client.get(f"/api/graphs/{qid}?dataset=mmqa")
    assert response.status_code == 200
    data = response.json()
    assert data["qid"] == qid
    assert "nodes" in data and "edges" in data
    assert data["node_count"] == len(data["nodes"])


def test_get_graphml_file_mmqa(client, mmqa_configured):
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run")

    q = _first_question_with_graph(client, "mmqa")
    if q is None:
        pytest.skip("No MMQA question with graph in this run")

    qid = q["qid"]
    response = client.get(f"/api/graphs/{qid}/graphml?dataset=mmqa")
    assert response.status_code == 200
    assert "xml" in response.headers.get("content-type", "").lower()
    lowered = response.content[:800].lower()
    assert b"<graphml" in lowered or b"graphml" in lowered


def test_mmqa_run_has_phase5_predictions(mmqa_configured, all_demo_paths):
    """Resolved MMQA run must include finished inference (demo ``latest`` heuristic)."""
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run configured")
    mmqa = all_demo_paths.mmqa
    assert mmqa is not None
    predictions = mmqa.phase5_inference_dir / "predictions.json"
    assert predictions.is_file(), (
        "MMQA demo run missing phase5_inference/predictions.json — "
        "check resolve_demo_paths / run_id"
    )


def test_mmqa_phase4_dir_has_graphml(mmqa_configured, all_demo_paths):
    """Phase-4 folder must contain at least one GraphML for visualization to be possible."""
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run configured")
    mmqa = all_demo_paths.mmqa
    assert mmqa is not None
    p4 = mmqa.phase4_graphs_out
    assert p4.is_dir(), f"Phase-4 graph directory missing: {p4}"
    assert any(p4.glob("*_graph.graphml")), f"No *_graph.graphml under {p4}"


def test_mmqa_has_graph_flags_match_detail_and_api(client, mmqa_configured):
    """
    Every MMQA list item with has_graph: detail.graph_available True and
    GET /api/graphs returns a non-empty node list (FE visualization data).
    """
    if not mmqa_configured:
        pytest.skip("No MultimodalQA demo run configured")

    listing = client.get("/api/questions?dataset=mmqa")
    assert listing.status_code == 200
    flagged = [q for q in listing.json() if q.get("has_graph")]
    if not flagged:
        pytest.skip("No MMQA questions with has_graph in this resolved run")

    for q in flagged:
        qid = q["qid"]
        det = client.get(f"/api/questions/{qid}?dataset=mmqa")
        assert det.status_code == 200, f"detail failed for {qid}"
        assert det.json().get("graph_available") is True, (
            f"graph_available False for qid={qid} despite has_graph in list"
        )

        gr = client.get(f"/api/graphs/{qid}?dataset=mmqa")
        assert gr.status_code == 200, f"graph API failed for {qid}"
        data = gr.json()
        assert data["qid"] == qid
        assert data.get("node_count", 0) >= 1, f"empty graph for {qid}"
        assert len(data.get("nodes", [])) == data["node_count"]
