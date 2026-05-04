"""Image serving endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse


router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("/{image_id}")
async def get_image(image_id: str, request: Request) -> FileResponse:
    """Serve an image by ID."""
    image_service = request.app.state.image_service
    path = image_service.find_image(image_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    return FileResponse(path, media_type="image/png")


@router.get("")
async def list_images(
    request: Request,
    limit: int = 100,
) -> list[str]:
    """List available image IDs."""
    image_service = request.app.state.image_service
    return image_service.list_images(limit=limit)
