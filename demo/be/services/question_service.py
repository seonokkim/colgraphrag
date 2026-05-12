"""Service for loading and joining question-level data.

Supports both WebQA (Guid/Q/A fields) and MMQA (qid/question/answers fields).
Pass ``dataset="mmqa"`` to switch extraction logic.
"""

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
        dataset: str = "webqa",
    ) -> None:
        self._result_dir = result_run_dir
        self._questions_path = webqa_questions_jsonl
        self._graphs_dir = phase4_graphs_out
        self._phase5 = result_run_dir / "phase5_inference"
        self._dataset = dataset

        self._questions: dict[str, dict[str, Any]] | None = None
        self._predictions: dict[str, str] | None = None
        self._retrieval: dict[str, list[dict[str, Any]]] | None = None

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------

    def _extract_qid(self, obj: dict[str, Any]) -> str:
        if self._dataset == "mmqa":
            return obj.get("qid", "")
        return obj.get("Guid", "")

    def _extract_question(self, obj: dict[str, Any]) -> str:
        if self._dataset == "mmqa":
            return obj.get("question", "")
        return obj.get("Q", "")

    def _extract_qcate(self, obj: dict[str, Any]) -> str:
        if self._dataset == "mmqa":
            md = obj.get("metadata") or {}
            modalities = md.get("modalities") or []
            return "+".join(modalities) if modalities else md.get("type", "")
        return obj.get("Qcate", "")

    def _extract_gold_answers(self, obj: dict[str, Any]) -> list[str]:
        if self._dataset == "mmqa":
            raw = obj.get("answers") or []
            out: list[str] = []
            for a in raw:
                if isinstance(a, dict) and "answer" in a:
                    v = a["answer"]
                    out.append("" if v is None else str(v))
            return out
        return ["" if x is None else str(x) for x in obj.get("A", [])]

    def _extract_gold_facts(self, obj: dict[str, Any]) -> list[GoldFact]:
        if self._dataset == "mmqa":
            facts: list[GoldFact] = []
            md = obj.get("metadata") or {}
            for img_id in md.get("image_doc_ids") or []:
                facts.append(GoldFact(fact_type="image", id=str(img_id), content=""))
            for txt_id in md.get("text_doc_ids") or []:
                facts.append(GoldFact(fact_type="text", id=str(txt_id), content=""))
            return facts
        # WebQA
        facts = []
        for img in obj.get("img_posFacts", []):
            facts.append(
                GoldFact(
                    fact_type="image",
                    id=img.get("image_id", ""),
                    content=img.get("url", ""),
                    title=img.get("title"),
                    caption=img.get("caption"),
                )
            )
        for txt in obj.get("txt_posFacts", []):
            facts.append(
                GoldFact(
                    fact_type="text",
                    id=txt.get("snippet_id", ""),
                    content=txt.get("fact", ""),
                )
            )
        return facts

    # ------------------------------------------------------------------
    # Loaders (lazy, cached)
    # ------------------------------------------------------------------

    def _load_questions(self) -> dict[str, dict[str, Any]]:
        if self._questions is None:
            self._questions = {}
            if self._questions_path.exists():
                with self._questions_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            qid = self._extract_qid(obj)
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
        return (self._graphs_dir / f"{qid}_graph.graphml").exists()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_questions(self) -> list[QuestionSummary]:
        questions = self._load_questions()
        predictions = self._load_predictions()

        result: list[QuestionSummary] = []
        for qid, q in questions.items():
            gold_answers = self._extract_gold_answers(q)
            result.append(
                QuestionSummary(
                    qid=qid,
                    question=self._extract_question(q),
                    qcate=self._extract_qcate(q),
                    gold_answer=gold_answers[0] if gold_answers else "",
                    predicted_answer=predictions.get(qid),
                    has_graph=self._has_graph(qid),
                )
            )
        return result

    def get_question(self, qid: str) -> QuestionDetail | None:
        questions = self._load_questions()
        predictions = self._load_predictions()
        retrieval = self._load_retrieval()

        q = questions.get(qid)
        if q is None:
            return None

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

        gold_answers = self._extract_gold_answers(q)
        split = q.get("split", "dev" if self._dataset == "mmqa" else "")

        return QuestionDetail(
            qid=qid,
            question=self._extract_question(q),
            qcate=self._extract_qcate(q),
            split=split,
            gold_answers=gold_answers,
            keywords_answer=q.get("Keywords_A"),
            predicted_answer=predictions.get(qid),
            retrieval=retrieval_items,
            gold_facts=self._extract_gold_facts(q),
            graph_available=self._has_graph(qid),
        )

    def get_question_ids(self) -> list[str]:
        return list(self._load_questions().keys())
