"""Service for loading and joining question-level data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.question import (
    QuestionSummary,
    QuestionDetail,
    RetrievalItem,
    GoldFact,
)


class QuestionService:
    """Loads questions, predictions, and retrieval rankings."""

    def __init__(
        self,
        result_run_dir: Path,
        webqa_questions_jsonl: Path,
        phase4_graphs_out: Path,
    ) -> None:
        self._result_dir = result_run_dir
        self._questions_path = webqa_questions_jsonl
        self._graphs_dir = phase4_graphs_out
        self._phase5 = result_run_dir / "phase5_inference"

        self._questions: dict[str, dict[str, Any]] | None = None
        self._predictions: dict[str, str] | None = None
        self._retrieval: dict[str, list[dict[str, Any]]] | None = None

    def _load_questions(self) -> dict[str, dict[str, Any]]:
        if self._questions is None:
            self._questions = {}
            if self._questions_path.exists():
                with self._questions_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            qid = obj.get("Guid", "")
                            if qid:
                                self._questions[qid] = obj
        return self._questions

    def _load_predictions(self) -> dict[str, str]:
        if self._predictions is None:
            path = self._phase5 / "predictions.json"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "predictions" in data:
                        self._predictions = data["predictions"]
                    else:
                        self._predictions = data
            else:
                self._predictions = {}
        return self._predictions

    def _load_retrieval(self) -> dict[str, list[dict[str, Any]]]:
        if self._retrieval is None:
            path = self._phase5 / "predictions_retrieval.json"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    self._retrieval = json.load(f)
            else:
                self._retrieval = {}
        return self._retrieval

    def _has_graph(self, qid: str) -> bool:
        graph_path = self._graphs_dir / f"{qid}_graph.graphml"
        return graph_path.exists()

    def list_questions(self) -> list[QuestionSummary]:
        """Return summary list of all questions."""
        questions = self._load_questions()
        predictions = self._load_predictions()

        result: list[QuestionSummary] = []
        for qid, q in questions.items():
            gold_answers = q.get("A", [])
            gold_str = gold_answers[0] if gold_answers else ""
            result.append(
                QuestionSummary(
                    qid=qid,
                    question=q.get("Q", ""),
                    qcate=q.get("Qcate", ""),
                    gold_answer=gold_str,
                    predicted_answer=predictions.get(qid),
                    has_graph=self._has_graph(qid),
                )
            )
        return result

    def get_question(self, qid: str) -> QuestionDetail | None:
        """Return detailed info for a single question."""
        questions = self._load_questions()
        predictions = self._load_predictions()
        retrieval = self._load_retrieval()

        q = questions.get(qid)
        if q is None:
            return None

        gold_facts: list[GoldFact] = []
        for img in q.get("img_posFacts", []):
            gold_facts.append(
                GoldFact(
                    fact_type="image",
                    id=img.get("image_id", ""),
                    content=img.get("url", ""),
                    title=img.get("title"),
                    caption=img.get("caption"),
                )
            )
        for txt in q.get("txt_posFacts", []):
            gold_facts.append(
                GoldFact(
                    fact_type="text",
                    id=txt.get("snippet_id", ""),
                    content=txt.get("fact", ""),
                )
            )

        retrieval_items: list[RetrievalItem] = []
        for idx, item in enumerate(retrieval.get(qid, [])):
            retrieval_items.append(
                RetrievalItem(
                    id=item.get("id", ""),
                    score=item.get("score", 0.0),
                    rank=item.get("rank", idx + 1),
                    source=item.get("source"),
                )
            )

        return QuestionDetail(
            qid=qid,
            question=q.get("Q", ""),
            qcate=q.get("Qcate", ""),
            split=q.get("split", ""),
            gold_answers=q.get("A", []),
            keywords_answer=q.get("Keywords_A"),
            predicted_answer=predictions.get(qid),
            retrieval=retrieval_items,
            gold_facts=gold_facts,
            graph_available=self._has_graph(qid),
        )

    def get_question_ids(self) -> list[str]:
        """Return list of all question IDs."""
        return list(self._load_questions().keys())
