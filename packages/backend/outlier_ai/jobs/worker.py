"""Job worker: claims pending jobs with SKIP LOCKED, runs handlers, and schedules the daily
cadence (insights, comments, episode tick) with APScheduler-free cron logic kept simple:
a loop that enqueues due jobs by key `<kind>:<date>` so re-scheduling is idempotent."""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core import db as dbmod
from outlier_ai.core.config import ConfigStore
from outlier_ai.jobs.handlers import run_episode_tick, run_sync_comments, run_sync_insights
from outlier_ai.jobs.registry import enqueue
from outlier_ai.models.ops import Job
from outlier_ai.outlier.recompute import recompute_all
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import JobState

log = logging.getLogger("outlier.worker")

MAX_ATTEMPTS = 3


async def run_handler(session: AsyncSession, job: Job, cfg: AppConfig) -> dict[str, Any]:
    payload = job.payload or {}
    account_id = UUID(payload["account_id"]) if payload.get("account_id") else None
    if job.kind == "sync_insights":
        reports = await run_sync_insights(session, cfg, account_id=account_id)
        return {"accounts": [r.__dict__ for r in reports]}
    if job.kind == "sync_comments":
        from outlier_ai.generation.backends import get_judge

        reports = await run_sync_comments(
            session,
            cfg,
            account_id=account_id,
            judge=get_judge(cfg, model=cfg.signals.extraction_model),
        )
        return {"accounts": [r.__dict__ for r in reports]}
    if job.kind == "episode_tick":
        return {"episodes": await run_episode_tick(session, cfg)}
    if job.kind == "recompute_outcomes":
        return {
            "accounts": [
                s.__dict__ | {"account_id": str(s.account_id)}
                for s in await recompute_all(session, cfg)
            ]
        }
    raise ValueError(f"unknown job kind {job.kind}")


async def claim_next(session: AsyncSession) -> Job | None:
    row = (
        await session.execute(
            select(Job)
            .where(Job.state == JobState.pending.value, Job.scheduled_for <= datetime.now(UTC))
            .order_by(Job.scheduled_for)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    row.state = JobState.running.value
    row.started_at = datetime.now(UTC)
    row.attempts += 1
    await session.flush()
    return row


async def run_once(*, cfg: AppConfig | None = None) -> int:
    """Run every due job once. Returns the number of jobs processed."""
    processed = 0
    while True:
        async with dbmod.session_scope() as session:
            job = await claim_next(session)
            if job is None:
                return processed
            job_id, kind = job.id, job.kind
        try:
            async with dbmod.session_scope() as session:
                cfg_ = cfg or await ConfigStore(session).current()
                job = await session.get(Job, job_id)
                assert job is not None
                result = await run_handler(session, job, cfg_)
                job.state = JobState.done.value
                job.finished_at = datetime.now(UTC)
                job.payload = {**(job.payload or {}), "result": _trim(result)}
                log.info("job %s %s done", kind, job_id)
        except Exception as e:
            async with dbmod.session_scope() as session:
                job = await session.get(Job, job_id)
                if job is not None:
                    job.last_error = f"{e}\n{traceback.format_exc()[-2000:]}"
                    job.finished_at = datetime.now(UTC)
                    job.state = (
                        JobState.failed.value
                        if job.attempts >= MAX_ATTEMPTS
                        else JobState.pending.value
                    )
                    if job.state == JobState.pending.value:
                        job.scheduled_for = datetime.now(UTC) + timedelta(minutes=5 * job.attempts)
            log.exception("job %s %s failed", kind, job_id)
        processed += 1


def _trim(result: dict[str, Any]) -> dict[str, Any]:
    s = str(result)
    return result if len(s) < 20000 else {"truncated": s[:20000]}


async def schedule_daily(
    session: AsyncSession, cfg: AppConfig, now: datetime | None = None
) -> list[str]:
    """Enqueue today's cadence if due. Keys include the date so this is idempotent."""
    now = now or datetime.now(UTC)
    day = now.date().isoformat()
    keys: list[str] = []
    if now.hour >= cfg.meta.insights_sync_hour_utc:
        await enqueue(session, "sync_insights", key=f"daily:{day}", payload={})
        await enqueue(
            session,
            "episode_tick",
            key=f"after-insights:{day}",
            payload={},
            scheduled_for=now + timedelta(minutes=1),
        )
        keys += ["sync_insights", "episode_tick"]
    if now.hour >= cfg.meta.comments_sync_hour_utc:
        await enqueue(session, "sync_comments", key=f"daily:{day}", payload={})
        keys.append("sync_comments")
    # the controller also runs every 30 minutes for review polling
    slot = now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0).isoformat()
    await enqueue(session, "episode_tick", key=f"slot:{slot}", payload={})
    keys.append("episode_tick")
    return keys


async def serve(poll_seconds: float = 30.0) -> None:
    """Long-running worker loop."""
    logging.basicConfig(level=logging.INFO)
    log.info("worker started")
    while True:
        try:
            async with dbmod.session_scope() as session:
                cfg = await ConfigStore(session).current()
                await schedule_daily(session, cfg)
            await run_once()
        except Exception:
            log.exception("worker loop error")
        await asyncio.sleep(poll_seconds)


async def ensure_jobs_table_ready(session: AsyncSession) -> None:
    await session.execute(text("SELECT 1 FROM jobs LIMIT 1"))
