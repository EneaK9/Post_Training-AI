"""Job table helpers: enqueue is idempotent on (kind, key) while a job is pending or running."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.models.ops import Job
from outlier_schemas.enums import JobState


async def enqueue(
    session: AsyncSession,
    kind: str,
    *,
    key: str,
    payload: dict[str, Any] | None = None,
    scheduled_for: datetime | None = None,
) -> Job:
    existing = (
        await session.execute(select(Job).where(Job.kind == kind, Job.key == key))
    ).scalar_one_or_none()
    if existing is not None:
        if existing.state in (JobState.pending.value, JobState.running.value):
            return existing
        # re-arm a finished job with the same key
        existing.state = JobState.pending.value
        existing.payload = payload or {}
        existing.scheduled_for = scheduled_for or datetime.now(UTC)
        existing.attempts = 0
        existing.last_error = None
        existing.started_at = None
        existing.finished_at = None
        await session.flush()
        return existing
    job = Job(
        kind=kind,
        key=key,
        payload=payload or {},
        scheduled_for=scheduled_for or datetime.now(UTC),
        state=JobState.pending.value,
    )
    session.add(job)
    await session.flush()
    return job
