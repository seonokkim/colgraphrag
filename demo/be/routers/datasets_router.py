"""Datasets discovery endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetInfo(BaseModel):
    key: str
    label: str
    available: bool
    run_id: str | None = None


class DatasetsResponse(BaseModel):
    datasets: list[DatasetInfo]
    default: str


@router.get("", response_model=DatasetsResponse)
async def list_datasets(request: Request) -> DatasetsResponse:
    """Return which datasets are loaded and available."""
    webqa_available = hasattr(request.app.state, "question_service")

    mmqa_svc = getattr(request.app.state, "mmqa_question_service", None)
    mmqa_available = mmqa_svc is not None

    mmqa_run_id: str | None = None
    mmqa_paths = getattr(request.app.state, "mmqa_paths", None)
    if mmqa_paths is not None:
        mmqa_run_id = mmqa_paths.run_id

    webqa_run_id: str | None = None
    demo_paths = getattr(request.app.state, "demo_paths", None)
    if demo_paths is not None:
        webqa_run_id = demo_paths.run_id

    return DatasetsResponse(
        datasets=[
            DatasetInfo(key="webqa", label="WebQA", available=webqa_available, run_id=webqa_run_id),
            DatasetInfo(key="mmqa", label="MultimodalQA", available=mmqa_available, run_id=mmqa_run_id),
        ],
        default="webqa",
    )
