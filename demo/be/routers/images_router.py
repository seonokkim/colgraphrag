"""Image serving endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import FileResponse


router = APIRouter(prefix="/api/images", tags=["images"])


def _get_image_service(request: Request, dataset: str):
    if dataset == "mmqa":
        svc = getattr(request.app.state, "mmqa_image_service", None)
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="MultimodalQA demo not initialized (check demo/be/config).",
            )
        return svc
    svc = getattr(request.app.state, "image_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="WebQA demo not initialized (check demo/be/config).",
        )
    return svc


@router.get("/{image_id}")
async def get_image(
    image_id: str,
    request: Request,
    dataset: str = Query(default="webqa"),
) -> FileResponse:
    """Serve an image by ID."""
    image_service = _get_image_service(request, dataset)
    result = image_service.find_image(image_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    path, media_type = result
    return FileResponse(path, media_type=media_type)


@router.get("")
async def list_images(
    request: Request,
    dataset: str = Query(default="webqa"),
    limit: int = 100,
) -> list[str]:
    """List available image IDs."""
    image_service = _get_image_service(request, dataset)
    return image_service.list_images(limit=limit)
