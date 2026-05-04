"""End-to-end API tests against a running server."""

from __future__ import annotations

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="module")
def client():
    """HTTP client for E2E tests."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: httpx.Client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["run_id"] is not None
        print(f"[E2E] Health OK, run_id={data['run_id']}")


class TestRunEndpoints:
    def test_run_info(self, client: httpx.Client):
        resp = client.get("/api/run/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "total_questions" in data
        assert data["total_questions"] >= 0
        print(f"[E2E] Run info: {data['run_id']}, {data['total_questions']} questions")

    def test_run_scores(self, client: httpx.Client):
        resp = client.get("/api/run/scores")
        assert resp.status_code == 200
        data = resp.json()
        assert "all" in data
        assert "qa_fl" in data["all"]
        assert "qa_acc" in data["all"]
        assert "qa" in data["all"]
        print(f"[E2E] Scores: QA-FL={data['all']['qa_fl']:.2f}, QA-Acc={data['all']['qa_acc']:.2f}, QA={data['all']['qa']:.2f}")


class TestQuestionsEndpoints:
    def test_list_questions(self, client: httpx.Client):
        resp = client.get("/api/questions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        print(f"[E2E] Questions list: {len(data)} items")
        return data

    def test_get_question_detail(self, client: httpx.Client):
        list_resp = client.get("/api/questions")
        questions = list_resp.json()
        assert len(questions) > 0

        qid = questions[0]["qid"]
        resp = client.get(f"/api/questions/{qid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["qid"] == qid
        assert "question" in data
        assert "gold_answers" in data
        assert "predicted_answer" in data
        print(f"[E2E] Question detail: qid={qid[:12]}..., qcate={data['qcate']}")

    def test_question_not_found(self, client: httpx.Client):
        resp = client.get("/api/questions/nonexistent_qid_12345")
        assert resp.status_code == 404


class TestGraphEndpoints:
    def test_get_graph(self, client: httpx.Client):
        list_resp = client.get("/api/questions")
        questions = list_resp.json()

        q_with_graph = next((q for q in questions if q.get("has_graph")), None)
        if q_with_graph is None:
            pytest.skip("No question with graph available")

        qid = q_with_graph["qid"]
        resp = client.get(f"/api/graphs/{qid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["qid"] == qid
        assert "nodes" in data
        assert "edges" in data
        print(f"[E2E] Graph: qid={qid[:12]}..., {data['node_count']} nodes, {data['edge_count']} edges")

    def test_get_graphml_file(self, client: httpx.Client):
        list_resp = client.get("/api/questions")
        questions = list_resp.json()

        q_with_graph = next((q for q in questions if q.get("has_graph")), None)
        if q_with_graph is None:
            pytest.skip("No question with graph available")

        qid = q_with_graph["qid"]
        resp = client.get(f"/api/graphs/{qid}/graphml")
        assert resp.status_code == 200
        assert "xml" in resp.headers.get("content-type", "")
        print(f"[E2E] GraphML file: {len(resp.content)} bytes")

    def test_graph_not_found(self, client: httpx.Client):
        resp = client.get("/api/graphs/nonexistent_qid_12345")
        assert resp.status_code == 404


class TestImagesEndpoints:
    def test_list_images(self, client: httpx.Client):
        resp = client.get("/api/images?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        print(f"[E2E] Images list: {len(data)} items")
        return data

    def test_get_image(self, client: httpx.Client):
        list_resp = client.get("/api/images?limit=10")
        images = list_resp.json()

        if not images:
            pytest.skip("No images available")

        image_id = images[0]
        resp = client.get(f"/api/images/{image_id}")
        if resp.status_code == 200:
            assert resp.headers.get("content-type") == "image/png"
            print(f"[E2E] Image: id={image_id}, {len(resp.content)} bytes")
        else:
            assert resp.status_code == 404

    def test_image_not_found(self, client: httpx.Client):
        resp = client.get("/api/images/nonexistent_image_999999")
        assert resp.status_code == 404


class TestFullFlow:
    """Test a complete user flow: browse -> select -> view detail -> view graph."""

    def test_full_qa_flow(self, client: httpx.Client):
        print("\n[E2E] === Full QA Flow Test ===")

        # 1. Check health
        health = client.get("/health").json()
        assert health["status"] == "ok"
        print(f"[E2E] 1. Health check passed (run_id={health['run_id']})")

        # 2. Get scores
        scores = client.get("/api/run/scores").json()
        print(f"[E2E] 2. Scores: QA={scores['all']['qa']:.2f}")

        # 3. List questions
        questions = client.get("/api/questions").json()
        assert len(questions) > 0
        print(f"[E2E] 3. Found {len(questions)} questions")

        # 4. Get first question detail
        q = questions[0]
        detail = client.get(f"/api/questions/{q['qid']}").json()
        print(f"[E2E] 4. Question: '{detail['question'][:50]}...'")
        print(f"[E2E]    Gold: '{detail['gold_answers'][0][:50]}...'")
        if detail['predicted_answer']:
            print(f"[E2E]    Pred: '{detail['predicted_answer'][:50]}...'")

        # 5. Get graph if available
        if detail["graph_available"]:
            graph = client.get(f"/api/graphs/{q['qid']}").json()
            print(f"[E2E] 5. Graph: {graph['node_count']} nodes, {graph['edge_count']} edges")
        else:
            print("[E2E] 5. No graph available for this question")

        print("[E2E] === Full Flow Complete ===\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
