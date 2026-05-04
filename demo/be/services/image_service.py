"""Service for serving WebQA images."""

from __future__ import annotations

from pathlib import Path


class ImageService:
    """Locates and serves images from WEBQA_IMGS_DIR."""

    def __init__(self, webqa_imgs_dir: Path) -> None:
        self._imgs_dir = webqa_imgs_dir

    def find_image(self, image_id: str) -> Path | None:
        """Find image file by ID (searches for *_{image_id}.png pattern)."""
        if not self._imgs_dir.exists():
            return None

        pattern = f"*_{image_id}.png"
        matches = list(self._imgs_dir.glob(pattern))
        if matches:
            return matches[0]

        exact = self._imgs_dir / f"{image_id}.png"
        if exact.exists():
            return exact

        return None

    def list_images(self, limit: int = 100) -> list[str]:
        """List available image IDs (up to limit)."""
        if not self._imgs_dir.exists():
            return []

        ids: list[str] = []
        for p in self._imgs_dir.glob("*.png"):
            name = p.stem
            if "_" in name:
                parts = name.split("_")
                ids.append(parts[-1])
            else:
                ids.append(name)
            if len(ids) >= limit:
                break
        return ids
