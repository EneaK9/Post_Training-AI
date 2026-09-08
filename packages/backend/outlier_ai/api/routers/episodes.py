from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select

from outlier_ai.api.deps import DB, Config, CurrentUser, require
from outlier_ai.api.schemas import BatchOut, EpisodeCreate, EpisodeOut
from outlier_ai.core.audit import record_audit
from outlier_ai.models.auth import User
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import Batch, SearchEpisode
from outlier_ai.models.meta import DailyInsight
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_schemas.enums import EpisodeStatus

router = APIRouter(prefix="/episodes", tags=["episodes"])


async def episode_out(db: DB, ep: SearchEpisode, *, include_trace: bool = False) -> EpisodeOut:
    batches = (
        (await db.execute(select(Batch).where(Batch.episode_id == ep.id).order_by(Batch.index)))
        .scalars()
        .all()
    )
    trajs = (
        await db.execute(
            select(Trajectory.id, Trajectory.batch_id, Trajectory.outlier_tier).where(
                Trajectory.episode_id == ep.id
            )
        )
    ).all()
    by_batch: dict[UUID, list[UUID]] = {}
    for tid, bid, _ in trajs:
        if bid:
            by_batch.setdefault(bid, []).append(tid)
    traj_ids = [t[0] for t in trajs]
    live = 0
    spent = 0.0
    if traj_ids:
        live = int(
            (
                await db.execute(
                    select(func.count())
                    .select_from(Render)
                    .where(
                        Render.trajectory_id.in_(traj_ids),
                        Render.status.in_(["screening", "scaled", "pending_review"]),
                    )
                )
            ).scalar_one()
        )
        spent = float(
            (
                await db.execute(
                    select(func.coalesce(func.sum(DailyInsight.spend), 0.0))
                    .join(Render, Render.id == DailyInsight.render_id)
                    .where(Render.trajectory_id.in_(traj_ids))
                )
            ).scalar_one()
        )
    new_signals = 0
    if traj_ids and batches:
        last = batches[-1].created_at
        new_signals = int(
            (
                await db.execute(
                    select(func.count())
                    .select_from(Signal)
                    .where(Signal.trajectory_id.in_(traj_ids), Signal.created_at >= last)
                )
            ).scalar_one()
        )
    tiers = [t[2] for t in trajs if t[2] is not None]
    return EpisodeOut(
        id=ep.id,
        brief_id=ep.brief_id,
        ad_account_id=ep.ad_account_id,
        backend=ep.backend,
        budget_cap=ep.budget_cap,
        spent=spent if spent > 0 else ep.spent,
        status=ep.status,
        keep_running_after_outlier=ep.keep_running_after_outlier,
        campaign_id=ep.campaign_id,
        config_hash=ep.config_hash,
        created_by=ep.created_by,
        created_at=ep.created_at,
        ended_at=ep.ended_at,
        stop_reason=ep.stop_reason,
        batches=[
            BatchOut(
                id=b.id,
                index=b.index,
                state=b.state,
                created_at=b.created_at,
                trajectory_ids=by_batch.get(b.id, []),
                prompt_trace=b.prompt_trace if include_trace else None,
            )
            for b in batches
        ],
        live_ads=live,
        best_tier=max(tiers) if tiers else None,
        new_signals=new_signals,
    )


async def _get(db: DB, episode_id: UUID) -> SearchEpisode:
    ep = await db.get(SearchEpisode, episode_id)
    if ep is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "episode not found")
    return ep


@router.get("", response_model=list[EpisodeOut])
async def list_episodes(
    db: DB, _: CurrentUser, brief_id: UUID | None = None, status_: str | None = None
) -> list[EpisodeOut]:
    stmt = select(SearchEpisode).order_by(SearchEpisode.created_at.desc())
    if brief_id:
        stmt = stmt.where(SearchEpisode.brief_id == brief_id)
    if status_:
        stmt = stmt.where(SearchEpisode.status == status_)
    return [await episode_out(db, e) for e in (await db.execute(stmt)).scalars().all()]


@router.get("/{episode_id}", response_model=EpisodeOut)
async def get_episode(
    episode_id: UUID, db: DB, _: CurrentUser, include_trace: bool = False
) -> EpisodeOut:
    return await episode_out(db, await _get(db, episode_id), include_trace=include_trace)


@router.post("", response_model=EpisodeOut, status_code=status.HTTP_201_CREATED)
async def create_episode(
    body: EpisodeCreate, db: DB, cfg: Config, user: User = Depends(require("episodes:write"))
) -> EpisodeOut:
    brief = await db.get(Brief, body.brief_id)
    if brief is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "brief not found")
    ep = SearchEpisode(
        brief_id=brief.id,
        ad_account_id=body.ad_account_id or brief.ad_account_id,
        backend=body.backend.value,
        budget_cap=body.budget_cap or cfg.episode.default_budget_cap_usd,
        status=EpisodeStatus.searching.value,
        keep_running_after_outlier=body.keep_running_after_outlier
        or cfg.episode.keep_running_after_outlier,
        config_hash=cfg.hash,
        created_by=user.email,
    )
    db.add(ep)
    await db.flush()
    await record_audit(
        db,
        actor_id=user.email,
        action="episode.create",
        object_type="episode",
        object_id=ep.id,
        after={"brief_id": str(brief.id), "budget_cap": ep.budget_cap, "backend": ep.backend},
    )
    return await episode_out(db, ep)


@router.post("/{episode_id}/stop", response_model=EpisodeOut)
async def stop_episode(
    episode_id: UUID,
    db: DB,
    user: User = Depends(require("episodes:write")),
    reason: str = "operator stop",
) -> EpisodeOut:
    ep = await _get(db, episode_id)
    if ep.status == EpisodeStatus.searching.value:
        ep.status = EpisodeStatus.stopped.value
        ep.ended_at = datetime.now(UTC)
        ep.stop_reason = reason
        await record_audit(
            db,
            actor_id=user.email,
            action="episode.stop",
            object_type="episode",
            object_id=ep.id,
            after={"reason": reason},
        )
    return await episode_out(db, ep)


@router.post("/{episode_id}/keep_running", response_model=EpisodeOut)
async def keep_running(
    episode_id: UUID,
    db: DB,
    user: User = Depends(require("episodes:write")),
    value: bool = True,
) -> EpisodeOut:
    ep = await _get(db, episode_id)
    ep.keep_running_after_outlier = value
    await record_audit(
        db,
        actor_id=user.email,
        action="episode.keep_running",
        object_type="episode",
        object_id=ep.id,
        after={"keep_running_after_outlier": value},
    )
    return await episode_out(db, ep)
