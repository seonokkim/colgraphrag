"""Run-level info and scores endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from schemas.run import RunInfo, LeaderboardScores


router = APIRouter(prefix="/api/run", tags=["run"])


@router.get("/info", response_model=RunInfo)
async def get_run_info(request: Request) -> RunInfo:
    """Return run metadata (run_id, counts, paths)."""
    run_service = request.app.state.run_service
    return run_service.get_run_info()


@router.get("/scores", response_model=LeaderboardScores)
async def get_run_scores(request: Request) -> LeaderboardScores:
    """Return QA-FL / QA-Acc / QA scores and Qcate breakdown."""
    run_service = request.app.state.run_service
    return run_service.get_scores()
