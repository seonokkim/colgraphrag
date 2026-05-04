"""Pydantic models for run-level info and scores."""

from __future__ import annotations

from pydantic import BaseModel


class ScoreSet(BaseModel):
    """QA-FL / QA-Acc / QA triplet."""

    qa_fl: float
    qa_acc: float
    qa: float


class QcateScores(BaseModel):
    """Scores broken down by question category."""

    by_qcate_qa_fl: dict[str, float]
    by_qcate_qa_acc: dict[str, float]
    by_qcate_qa: dict[str, float]


class LeaderboardScores(BaseModel):
    """Leaderboard scores summary."""

    all: ScoreSet
    unimodal: ScoreSet
    multimodal: ScoreSet
    qcate: QcateScores | None = None
    source: str | None = None


class RunInfo(BaseModel):
    """Run metadata."""

    run_id: str
    generated_at: str | None = None
    total_questions: int
    scored_questions: int
    predictions_path: str
    gold_jsonl_path: str
    result_run_dir: str
    git_commit: str | None = None
