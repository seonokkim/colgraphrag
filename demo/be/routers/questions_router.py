"""Question-level endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

from schemas.question import QuestionSummary, QuestionDetail


router = APIRouter(prefix="/api/questions", tags=["questions"])


@router.get("", response_model=list[QuestionSummary])
async def list_questions(request: Request) -> list[QuestionSummary]:
    """Return summary list of all questions with predictions."""
    question_service = request.app.state.question_service
    return question_service.list_questions()


@router.get("/{qid}", response_model=QuestionDetail)
async def get_question(qid: str, request: Request) -> QuestionDetail:
    """Return detailed info for a single question."""
    question_service = request.app.state.question_service
    detail = question_service.get_question(qid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Question {qid} not found")
    return detail
