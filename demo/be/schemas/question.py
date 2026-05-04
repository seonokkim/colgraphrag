"""Pydantic models for question-level data."""

from __future__ import annotations

from pydantic import BaseModel


class GoldFact(BaseModel):
    """Gold evidence fact (image or text)."""

    fact_type: str  # "image" or "text"
    id: str
    content: str  # url for image, snippet for text
    title: str | None = None
    caption: str | None = None


class RetrievalItem(BaseModel):
    """Retrieved source item from ColEmbed or graph."""

    id: str
    score: float
    rank: int | None = None
    source: str | None = None


class QuestionSummary(BaseModel):
    """Summary view for question list."""

    qid: str
    question: str
    qcate: str
    gold_answer: str
    predicted_answer: str | None = None
    has_graph: bool = False


class QuestionDetail(BaseModel):
    """Detailed view for a single question."""

    qid: str
    question: str
    qcate: str
    split: str
    gold_answers: list[str]
    keywords_answer: str | None = None
    predicted_answer: str | None = None
    retrieval: list[RetrievalItem]
    gold_facts: list[GoldFact]
    graph_available: bool = False
