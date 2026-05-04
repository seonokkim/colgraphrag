"""End-to-end tests for the /api/chat endpoint with Gemma LLM.

These tests verify that the live chat pipeline works:
1. Question similarity matching (with graph preference)
2. Gemma LLM answer generation
3. Graph data and retrieval sources in responses

Requirements:
- BE server must be running on http://127.0.0.1:8000
- Gemma 4 E4B IT model must be loaded (GEMMA4_E4B_IT_TORCH_DTYPE=bf16)
- Pipeline results with graphs must exist

Run:
    cd demo/be
    pytest tests/test_chat_e2e.py -v -s
"""

import time

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 300.0  # LLM first-load can be slow


@pytest.fixture(scope="module")
def client():
    """Create an HTTP client and verify the server is reachable."""
    c = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)
    try:
        resp = c.get("/health")
        if resp.status_code != 200:
            pytest.skip("BE server not running or not healthy")
    except httpx.ConnectError:
        pytest.skip("BE server not running at " + BASE_URL)
    yield c
    c.close()


class TestChatEndpoint:
    """Tests for POST /api/chat."""

    def test_chat_returns_200(self, client: httpx.Client):
        """Basic connectivity: endpoint exists and returns 200."""
        resp = client.post("/api/chat", json={"question": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "elapsed_ms" in data

    def test_chat_with_graph_question(self, client: httpx.Client):
        """Question matching a graph-bearing question should use graph context."""
        resp = client.post(
            "/api/chat",
            json={"question": "Does the Cincinnati Music Hall have columns?"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["answer"], "Answer should not be empty"
        assert "sorry" not in data["answer"].lower(), (
            f"LLM should not apologize with graph available: {data['answer']}"
        )
        assert "cannot answer" not in data["answer"].lower(), (
            f"LLM should answer with graph: {data['answer']}"
        )
        assert data["matched_qid"] == "d5c2d1760dba11ecb1e81171463288e9"
        assert data["graph"] is not None, "Graph data should be returned"
        assert data["graph"]["node_count"] > 0

        print(f"\n  Answer: {data['answer']}")
        print(f"  Matched QID: {data['matched_qid']}")
        print(f"  Graph: {data['graph']['node_count']} nodes, {data['graph']['edge_count']} edges")
        print(f"  Elapsed: {data['elapsed_ms']:.0f}ms")

    def test_chat_with_novel_question(self, client: httpx.Client):
        """Novel question should still get an answer (not a refusal)."""
        resp = client.post(
            "/api/chat",
            json={"question": "What color is the bridge in the photo?"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["answer"], "Answer should not be empty"
        assert "no knowledge graph" not in data["answer"].lower(), (
            f"Should not refuse due to missing graph: {data['answer']}"
        )
        assert "cannot answer" not in data["answer"].lower(), (
            f"Should provide an answer: {data['answer']}"
        )

        print(f"\n  Answer: {data['answer']}")
        print(f"  Matched QID: {data['matched_qid']}")
        print(f"  Has graph: {data['graph'] is not None}")
        print(f"  Elapsed: {data['elapsed_ms']:.0f}ms")

    def test_chat_yes_no_question(self, client: httpx.Client):
        """Yes/No question should get a Yes/No style answer."""
        resp = client.post(
            "/api/chat",
            json={"question": "Are the rear wheels on the wheelchairs straight?"},
        )
        assert resp.status_code == 200
        data = resp.json()

        answer_lower = data["answer"].lower()
        assert data["answer"], "Answer should not be empty"
        assert "yes" in answer_lower or "no" in answer_lower, (
            f"Yes/No question should get Yes/No answer: {data['answer']}"
        )

        print(f"\n  Answer: {data['answer']}")
        print(f"  Matched QID: {data['matched_qid']}")
        print(f"  Elapsed: {data['elapsed_ms']:.0f}ms")

    def test_chat_comparison_question(self, client: httpx.Client):
        """Comparison question should identify the answer."""
        resp = client.post(
            "/api/chat",
            json={"question": "Which building has more windows?"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["answer"], "Answer should not be empty"
        assert len(data["answer"]) > 10, (
            f"Answer should be a complete sentence: {data['answer']}"
        )

        print(f"\n  Answer: {data['answer']}")
        print(f"  Matched QID: {data['matched_qid']}")
        print(f"  Elapsed: {data['elapsed_ms']:.0f}ms")

    def test_chat_response_time(self, client: httpx.Client):
        """Second+ calls should be fast (model already loaded)."""
        # Warm up (first call loads model)
        client.post("/api/chat", json={"question": "warmup"})

        start = time.time()
        resp = client.post(
            "/api/chat",
            json={"question": "Are there buildings shorter than the flag pole?"},
        )
        elapsed = time.time() - start
        data = resp.json()

        assert resp.status_code == 200
        assert elapsed < 30.0, f"Response too slow: {elapsed:.1f}s (expected <30s)"

        print(f"\n  Answer: {data['answer']}")
        print(f"  Wall time: {elapsed:.1f}s")
        print(f"  Server elapsed: {data['elapsed_ms']:.0f}ms")

    def test_chat_returns_sources(self, client: httpx.Client):
        """Chat response should include retrieval sources when available."""
        resp = client.post(
            "/api/chat",
            json={"question": "Cincinnati Music Hall columns"},
        )
        assert resp.status_code == 200
        data = resp.json()

        if data["sources"]:
            print(f"\n  Sources: {len(data['sources'])} items")
            for s in data["sources"][:3]:
                print(f"    - {s['id']} (score={s['score']})")

    def test_chat_returns_gold_facts(self, client: httpx.Client):
        """Chat response should include gold facts (images/text)."""
        resp = client.post(
            "/api/chat",
            json={"question": "Does Cincinnati Music Hall have columns?"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["gold_facts"]) > 0, "Should have gold facts"
        fact_types = {f["fact_type"] for f in data["gold_facts"]}
        print(f"\n  Gold facts: {len(data['gold_facts'])} ({', '.join(fact_types)})")


class TestChatEdgeCases:
    """Edge case tests."""

    def test_empty_question(self, client: httpx.Client):
        """Empty question should still return 200 (validation at app level)."""
        resp = client.post("/api/chat", json={"question": ""})
        # FastAPI might return 422 for empty string if validated
        assert resp.status_code in (200, 422)

    def test_very_long_question(self, client: httpx.Client):
        """Very long question should not crash."""
        long_q = "What is the color of " * 100 + "the sky?"
        resp = client.post("/api/chat", json={"question": long_q})
        assert resp.status_code == 200

    def test_special_characters(self, client: httpx.Client):
        """Question with special characters should not crash."""
        resp = client.post(
            "/api/chat",
            json={"question": "What's the color? (red/blue) <test> & more"},
        )
        assert resp.status_code == 200
