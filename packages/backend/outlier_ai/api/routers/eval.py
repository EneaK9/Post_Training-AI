from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from outlier_ai.api.deps import DB, CurrentUser, require
from outlier_ai.models.auth import User

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/launch", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def launch(db: DB, _: User = Depends(require("eval:write"))) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "online eval harness lands in Phase 6")


@router.get("/{eval_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_eval(eval_id: UUID, db: DB, _: CurrentUser) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "online eval harness lands in Phase 6")
