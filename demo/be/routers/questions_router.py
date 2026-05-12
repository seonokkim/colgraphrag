"""Question-level endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException, Query

from schemas.question import QuestionSummary, QuestionDetail


router = APIRouter(prefix="/api/questions", tags=["questions"])


def _get_question_service(request: Request, dataset: str):
    if dataset == "mmqa":
        svc = getattr(request.app.state, "mmqa_question_service", None)
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "MultimodalQA demo not initialized "
                    "(no result run under result/multimodalqa — check demo/be/config)."
                ),
            )
        return svc
    svc = getattr(request.app.state, "question_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "WebQA demo not initialized "
                "(missing paths or no result run — see demo/be/config/paths.example.yaml)."
            ),
        )
    return svc


@router.get("", response_model=list[QuestionSummary])
async def list_questions(
    request: Request,
    dataset: str = Query(default="webqa"),
) -> list[QuestionSummary]:
    """Return summary list of all questions with predictions."""
    question_service = _get_question_service(request, dataset)
    return question_service.list_questions()


@router.get("/{qid}", response_model=QuestionDetail)
async def get_question(
    qid: str,
    request: Request,
    dataset: str = Query(default="webqa"),
) -> QuestionDetail:
    """Return detailed info for a single question."""
    question_service = _get_question_service(request, dataset)
    detail = question_service.get_question(qid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Question {qid} not found")
    return detail
