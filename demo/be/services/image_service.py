"""Service for serving images (WebQA and MMQA)."""

from __future__ import annotations

from pathlib import Path

_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")

_MEDIA_TYPE: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class ImageService:
    """Locates and serves images from an images directory.

    Supports:
    - MMQA pattern: ``{image_id}.jpg`` / ``{image_id}.png``
    - WebQA pattern: ``*_{image_id}.png``
    """

    def __init__(self, imgs_dir: Path) -> None:
        self._imgs_dir = imgs_dir

    def find_image(self, image_id: str) -> tuple[Path, str] | None:
        """Return ``(path, media_type)`` for the image, or ``None``."""
        if not self._imgs_dir.exists():
            return None

        # Direct filename match with any supported extension (MMQA style)
        for ext in _IMG_EXTENSIONS:
            p = self._imgs_dir / f"{image_id}{ext}"
            if p.exists():
                return p, _MEDIA_TYPE.get(ext, "image/jpeg")

        # WebQA pattern: something_{image_id}.png
        for p in self._imgs_dir.glob(f"*_{image_id}.png"):
            return p, "image/png"

        return None

    def list_images(self, limit: int = 100) -> list[str]:
        """List available image IDs (up to limit)."""
        if not self._imgs_dir.exists():
            return []

        ids: list[str] = []
        for p in self._imgs_dir.iterdir():
            if p.suffix.lower() not in _IMG_EXTENSIONS:
                continue
            name = p.stem
            if "_" in name:
                ids.append(name.split("_")[-1])
            else:
                ids.append(name)
            if len(ids) >= limit:
                break
        return ids
