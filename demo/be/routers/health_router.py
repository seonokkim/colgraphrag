"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    run_id: str | None = None
    message: str | None = None


_run_id: str | None = None


def set_run_id(run_id: str) -> None:
    """Set the current run_id for health check response."""
    global _run_id
    _run_id = run_id


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return server health status."""
    return HealthResponse(
        status="ok",
        run_id=_run_id,
        message="ColGraphRAG WebQA Demo BE is running.",
    )
