"""Run-level info and scores endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Query

from schemas.run import RunInfo, LeaderboardScores


router = APIRouter(prefix="/api/run", tags=["run"])


def _get_run_service(request: Request, dataset: str):
    if dataset == "mmqa":
        svc = getattr(request.app.state, "mmqa_run_service", None)
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="MultimodalQA demo not initialized (check demo/be/config).",
            )
        return svc
    svc = getattr(request.app.state, "run_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="WebQA demo not initialized (check demo/be/config).",
        )
    return svc


@router.get("/info", response_model=RunInfo)
async def get_run_info(
    request: Request,
    dataset: str = Query(default="webqa"),
) -> RunInfo:
    """Return run metadata (run_id, counts, paths)."""
    run_service = _get_run_service(request, dataset)
    return run_service.get_run_info()


@router.get("/scores", response_model=LeaderboardScores)
async def get_run_scores(
    request: Request,
    dataset: str = Query(default="webqa"),
) -> LeaderboardScores:
    """Return QA-FL / QA-Acc / QA scores and Qcate breakdown."""
    run_service = _get_run_service(request, dataset)
    return run_service.get_scores()
