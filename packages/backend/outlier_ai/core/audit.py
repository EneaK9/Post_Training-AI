"""Audit log helper. Every mutation through the API records actor, action, object, before/after."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.models.ops import AuditLog


async def record_audit(
    session: AsyncSession,
    *,
    actor_id: str,
    action: str,
    object_type: str,
    object_id: Any,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> None:
    session.add(
        AuditLog(
            actor_id=actor_id,
            action=action,
            object_type=object_type,
            object_id=str(object_id),
            before=before,
            after=after,
        )
    )
