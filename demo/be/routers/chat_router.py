"""Chat router: handles live question answering via Gemma LLM + knowledge graph."""

from __future__ import annotations

import time
from difflib import SequenceMatcher

from fastapi import APIRouter, Request
from pydantic import BaseModel

from schemas.question import RetrievalItem, GoldFact

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievalItem]
    graph: dict | None = None
    gold_facts: list[GoldFact]
    matched_qid: str | None = None
    elapsed_ms: float | None = None


def _similarity(a: str, b: str) -> float:
    """Compute normalized similarity between two strings."""
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if a_lower == b_lower:
        return 1.0
    tokens_a = set(a_lower.split())
    tokens_b = set(b_lower.split())
    if not tokens_a or not tokens_b:
        return 0.0
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    seq_ratio = SequenceMatcher(None, a_lower, b_lower).ratio()
    return 0.6 * jaccard + 0.4 * seq_ratio


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """
    Answer a question using the Gemma LLM with the best-matching knowledge graph.

    Strategy:
    1. Find the best matching question THAT HAS A GRAPH (preferred)
    2. If no graph-bearing question is similar enough, use best overall match
    3. Generate answer with Gemma using graph context (or direct QA if no graph)
    """
    start = time.perf_counter()
    question_service = getattr(request.app.state, "question_service", None)
    graph_service = getattr(request.app.state, "graph_service", None)
    llm_service = getattr(request.app.state, "llm_service", None)

    if question_service is None:
        return ChatResponse(
            answer="Backend services not initialized. Please ensure pipeline results exist.",
            sources=[],
            graph=None,
            gold_facts=[],
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

    questions = question_service.list_questions()
    if not questions:
        return ChatResponse(
            answer="No questions available in the dataset.",
            sources=[],
            graph=None,
            gold_facts=[],
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

    # Score all questions, separately track those with graphs
    best_score_all = 0.0
    best_qid_all = questions[0].qid
    best_score_graph = 0.0
    best_qid_graph: str | None = None

    for q in questions:
        score = _similarity(body.question, q.question)
        if score > best_score_all:
            best_score_all = score
            best_qid_all = q.qid
        if q.has_graph and score > best_score_graph:
            best_score_graph = score
            best_qid_graph = q.qid

    # Prefer graph-bearing question if similarity is high enough to be meaningful
    GRAPH_SIM_THRESHOLD = 0.25
    use_qid: str
    use_graph: bool
    if best_qid_graph and best_score_graph >= GRAPH_SIM_THRESHOLD:
        use_qid = best_qid_graph
        use_graph = True
    elif best_qid_graph and best_score_graph > 0.05:
        use_qid = best_qid_graph
        use_graph = False  # low similarity -> don't force irrelevant graph context
    else:
        use_qid = best_qid_all
        use_graph = False

    detail = question_service.get_question(use_qid)
    if detail is None:
        return ChatResponse(
            answer="Could not retrieve question details.",
            sources=[],
            graph=None,
            gold_facts=[],
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

    # Generate answer with Gemma
    answer_text: str | None = None
    if llm_service is not None:
        try:
            answer_text = llm_service.generate_answer(
                body.question, qid=use_qid, use_graph=use_graph
            )
        except Exception as exc:
            answer_text = None
            import logging
            logging.getLogger(__name__).warning("LLM generation failed: %s", exc)

    if not answer_text:
        if detail.predicted_answer:
            answer_text = detail.predicted_answer
        else:
            answer_text = (
                "The LLM model is not loaded or failed to generate. "
                "Ensure Gemma 4 E4B IT is available and restart the server with "
                "GEMMA4_E4B_IT_TORCH_DTYPE=bf16."
            )

    # Build graph response data (only when graph was used for answering)
    graph_data = None
    if use_graph and detail.graph_available and graph_service:
        try:
            gd = graph_service.get_graph(use_qid)
            if gd:
                graph_data = gd.model_dump()
        except Exception:
            pass

    elapsed = (time.perf_counter() - start) * 1000

    return ChatResponse(
        answer=answer_text,
        sources=detail.retrieval,
        graph=graph_data,
        gold_facts=detail.gold_facts,
        matched_qid=use_qid,
        elapsed_ms=elapsed,
    )
