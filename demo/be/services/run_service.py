"""Service for loading run-level evaluation reports and scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.run import RunInfo, LeaderboardScores, ScoreSet, QcateScores


class RunService:
    """Loads and serves run-level info and scores from result files."""

    def __init__(self, result_run_dir: Path) -> None:
        self._result_dir = result_run_dir
        self._phase5 = result_run_dir / "phase5_inference"
        self._eval_report: dict[str, Any] | None = None
        self._leaderboard: dict[str, Any] | None = None

    def _load_eval_report(self) -> dict[str, Any]:
        if self._eval_report is None:
            path = self._phase5 / "evaluation_report.json"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    self._eval_report = json.load(f)
            else:
                self._eval_report = {}
        return self._eval_report

    def _load_leaderboard(self) -> dict[str, Any]:
        if self._leaderboard is None:
            path = self._phase5 / "qa_leaderboard.json"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    self._leaderboard = json.load(f)
            else:
                self._leaderboard = {}
        return self._leaderboard

    def get_run_info(self) -> RunInfo:
        """Return run metadata."""
        report = self._load_eval_report()
        inputs = report.get("inputs") or {}
        counts = report.get("counts") or {}

        run_id = inputs.get("mmgraphrag_run_id")
        if not run_id:
            run_id = self._result_dir.name

        predictions_path = inputs.get("predictions_path") or ""
        gold_jsonl_path = inputs.get("gold_jsonl_path") or ""

        return RunInfo(
            run_id=str(run_id),
            generated_at=report.get("generated_at"),
            total_questions=int(counts.get("All", 0) or 0),
            scored_questions=int(counts.get("scored", 0) or 0),
            predictions_path=str(predictions_path),
            gold_jsonl_path=str(gold_jsonl_path),
            result_run_dir=str(self._result_dir),
            git_commit=inputs.get("git_commit"),
        )

    def get_scores(self) -> LeaderboardScores:
        """Return leaderboard scores including per-Qcate breakdown."""
        report = self._load_eval_report()
        lb = report.get("leaderboard_summary", {})
        scores = report.get("scores", {})

        def to_score_set(data: dict[str, Any]) -> ScoreSet:
            return ScoreSet(
                qa_fl=data.get("QA-FL", 0.0),
                qa_acc=data.get("QA-Acc", 0.0),
                qa=data.get("QA", 0.0),
            )

        qcate = None
        if scores:
            qcate = QcateScores(
                by_qcate_qa_fl=scores.get("by_Qcate_qa_fl", {}),
                by_qcate_qa_acc=scores.get("by_Qcate_qa_acc", {}),
                by_qcate_qa=scores.get("by_Qcate_qa", {}),
            )

        return LeaderboardScores(
            all=to_score_set(lb.get("All", {})),
            unimodal=to_score_set(lb.get("Unimodal", {})),
            multimodal=to_score_set(lb.get("Multimodal", {})),
            qcate=qcate,
            source=lb.get("source"),
        )
