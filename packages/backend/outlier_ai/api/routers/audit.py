from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select

from outlier_ai.api.deps import DB, CurrentUser
from outlier_ai.api.schemas import AuditOut
from outlier_ai.models.ops import AuditLog

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=list[AuditOut])
async def list_audit(
    db: DB,
    _: CurrentUser,
    object_type: str | None = None,
    object_id: str | None = None,
    actor: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> list[AuditOut]:
    stmt = select(AuditLog).order_by(AuditLog.at.desc()).limit(limit)
    if object_type:
        stmt = stmt.where(AuditLog.object_type == object_type)
    if object_id:
        stmt = stmt.where(AuditLog.object_id == object_id)
    if actor:
        stmt = stmt.where(AuditLog.actor_id == actor)
    return [AuditOut.model_validate(a) for a in (await db.execute(stmt)).scalars().all()]
