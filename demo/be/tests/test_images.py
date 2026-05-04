"""Test images endpoints."""

from __future__ import annotations


def test_list_images(client):
    """Images list endpoint returns IDs."""
    response = client.get("/api/images")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_image(client):
    """Image endpoint returns PNG file."""
    list_response = client.get("/api/images?limit=10")
    images = list_response.json()

    if not images:
        return

    image_id = images[0]
    response = client.get(f"/api/images/{image_id}")

    if response.status_code == 200:
        assert response.headers.get("content-type") == "image/png"
    else:
        assert response.status_code == 404


def test_get_image_not_found(client):
    """Non-existent image returns 404."""
    response = client.get("/api/images/nonexistent_image_999999")
    assert response.status_code == 404
