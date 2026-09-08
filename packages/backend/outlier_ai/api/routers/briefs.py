from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select

from outlier_ai.api.deps import DB, Config, CurrentUser, require
from outlier_ai.api.routers.episodes import episode_out
from outlier_ai.api.schemas import BriefCreate, BriefOut, BriefUpdate, EpisodeOut
from outlier_ai.core.audit import record_audit
from outlier_ai.core.embeddings import get_embedder
from outlier_ai.models.auth import User
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.trajectories import Trajectory
from outlier_schemas.enums import Channel, GoalMetric

router = APIRouter(prefix="/briefs", tags=["briefs"])

VIDEO_MESSAGE = "Video ads are out of scope for v1. Only meta_feed_image is supported."


def _validate_channel(channel: str) -> None:
    if channel != Channel.meta_feed_image.value:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            VIDEO_MESSAGE if "video" in channel.lower() else f"unsupported channel {channel}",
        )


async def _counts(db: DB, brief_ids: list[UUID]) -> tuple[dict[UUID, int], dict[UUID, int]]:
    if not brief_ids:
        return {}, {}
    ep = await db.execute(
        select(SearchEpisode.brief_id, func.count())
        .where(SearchEpisode.brief_id.in_(brief_ids))
        .group_by(SearchEpisode.brief_id)
    )
    tr = await db.execute(
        select(Trajectory.brief_id, func.count())
        .where(Trajectory.brief_id.in_(brief_ids))
        .group_by(Trajectory.brief_id)
    )
    return {bid: int(n) for bid, n in ep.all()}, {bid: int(n) for bid, n in tr.all()}


def _out(brief: Brief, ep: dict[UUID, int], tr: dict[UUID, int]) -> BriefOut:
    out = BriefOut.model_validate(brief)
    out.episode_count = ep.get(brief.id, 0)
    out.trajectory_count = tr.get(brief.id, 0)
    return out


@router.get("", response_model=list[BriefOut])
async def list_briefs(
    db: DB, _: CurrentUser, category: str | None = None, ad_account_id: UUID | None = None
) -> list[BriefOut]:
    stmt = select(Brief).order_by(Brief.created_at.desc())
    if category:
        stmt = stmt.where(Brief.category == category)
    if ad_account_id:
        stmt = stmt.where(Brief.ad_account_id == ad_account_id)
    briefs = (await db.execute(stmt)).scalars().all()
    ep, tr = await _counts(db, [b.id for b in briefs])
    return [_out(b, ep, tr) for b in briefs]


@router.get("/{brief_id}", response_model=BriefOut)
async def get_brief(brief_id: UUID, db: DB, _: CurrentUser) -> BriefOut:
    brief = await db.get(Brief, brief_id)
    if brief is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "brief not found")
    ep, tr = await _counts(db, [brief.id])
    return _out(brief, ep, tr)


@router.get("/{brief_id}/episodes", response_model=list[EpisodeOut])
async def brief_episodes(brief_id: UUID, db: DB, _: CurrentUser) -> list[EpisodeOut]:
    rows = (
        (
            await db.execute(
                select(SearchEpisode)
                .where(SearchEpisode.brief_id == brief_id)
                .order_by(SearchEpisode.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return [await episode_out(db, e) for e in rows]


@router.post("", response_model=BriefOut, status_code=status.HTTP_201_CREATED)
async def create_brief(
    body: BriefCreate, db: DB, cfg: Config, user: User = Depends(require("briefs:write"))
) -> BriefOut:
    _validate_channel(body.channel)
    if body.goal_metric not in {g.value for g in GoalMetric}:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"unsupported goal_metric {body.goal_metric}"
        )
    embedder = get_embedder(dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model)
    text = (
        body.raw_text
        or f"{body.company} sells {body.product}. Offer: {body.offer}. Audience: {body.audience}."
    )
    brief = Brief(
        created_by=user.email,
        ad_account_id=body.ad_account_id,
        company=body.company,
        product=body.product,
        offer=body.offer,
        audience=body.audience,
        goal_metric=body.goal_metric,
        channel=body.channel,
        category=body.category,
        world_state=body.world_state,
        world_state_at=datetime.now(UTC) if body.world_state else None,
        constraints=body.constraints,
        brand_assets=body.brand_assets,
        raw_text=text,
        meta=body.meta,
        embedding=embedder.embed([text])[0].tolist(),
    )
    db.add(brief)
    await db.flush()
    await record_audit(
        db,
        actor_id=user.email,
        action="brief.create",
        object_type="brief",
        object_id=brief.id,
        after={"company": brief.company, "product": brief.product},
    )
    return _out(brief, {}, {})


@router.put("/{brief_id}", response_model=BriefOut)
async def update_brief(
    brief_id: UUID,
    body: BriefUpdate,
    db: DB,
    cfg: Config,
    user: User = Depends(require("briefs:write")),
) -> BriefOut:
    brief = await db.get(Brief, brief_id)
    if brief is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "brief not found")
    before = {"world_state": brief.world_state, "constraints": list(brief.constraints or [])}
    data = body.model_dump(exclude_none=True)
    for k, v in data.items():
        setattr(brief, k, v)
    if "world_state" in data:
        brief.world_state_at = datetime.now(UTC)
    if {"raw_text", "company", "product", "offer", "audience"} & set(data):
        embedder = get_embedder(
            dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
        )
        brief.embedding = embedder.embed(
            [brief.raw_text or f"{brief.company} {brief.product} {brief.offer}"]
        )[0].tolist()
    await record_audit(
        db,
        actor_id=user.email,
        action="brief.update",
        object_type="brief",
        object_id=brief.id,
        before=before,
        after={"world_state": brief.world_state, "constraints": list(brief.constraints or [])},
    )
    ep, tr = await _counts(db, [brief.id])
    return _out(brief, ep, tr)
