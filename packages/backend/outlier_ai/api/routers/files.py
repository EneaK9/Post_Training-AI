"""Serve stored objects (render images, brand assets) to authenticated users."""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response

from outlier_ai.api.deps import CurrentUser
from outlier_ai.core.storage import get_storage

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{key:path}")
async def get_file(key: str, _: CurrentUser) -> Response:
    storage = get_storage()
    if ".." in key or key.startswith("/") or not storage.exists(key):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "file not found")
    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return Response(content=storage.get(key), media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})
