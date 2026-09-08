from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from outlier_ai.api.deps import DB, CurrentUser, require
from outlier_ai.api.schemas import (
    CopyEdit,
    NoteIn,
    Paged,
    ReviewIn,
    TrajectoryDetail,
    TrajectoryOut,
)
from outlier_ai.api.views import cards_map, trajectory_detail, trajectory_outs
from outlier_ai.core.audit import record_audit
from outlier_ai.models.auth import User
from outlier_ai.models.cards import Card
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Note, Render, Review, Trajectory
from outlier_schemas.enums import ReviewLabel, TagSource

router = APIRouter(prefix="/trajectories", tags=["trajectories"])


async def _get(db: DB, trajectory_id: UUID) -> Trajectory:
    t = await db.get(Trajectory, trajectory_id)
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "trajectory not found")
    return t


@router.get("", response_model=Paged)
async def list_trajectories(
    db: DB,
    _: CurrentUser,
    brief_id: UUID | None = None,
    episode_id: UUID | None = None,
    tier: int | None = Query(None, ge=0, le=3),
    min_tier: int | None = Query(None, ge=0, le=3),
    card: str | None = None,
    author: str | None = None,
    mismatch: bool | None = None,
    signal_kind: str | None = None,
    typicality: str | None = None,
    episode_status: str | None = None,
    review_label: str | None = None,
    unlabeled: bool | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Paged:
    stmt = select(Trajectory)
    if brief_id:
        stmt = stmt.where(Trajectory.brief_id == brief_id)
    if episode_id:
        stmt = stmt.where(Trajectory.episode_id == episode_id)
    if tier is not None:
        stmt = stmt.where(Trajectory.outlier_tier == tier)
    if min_tier is not None:
        stmt = stmt.where(Trajectory.outlier_tier >= min_tier)
    if author:
        stmt = stmt.where(Trajectory.author_id == author)
    if mismatch is not None:
        stmt = stmt.where(Trajectory.tag_match.is_(not mismatch))
    if typicality:
        stmt = stmt.where(Trajectory.typicality == typicality)
    if card:
        card_row = (await db.execute(select(Card).where(Card.slug == card))).scalar_one_or_none()
        if card_row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"card {card} not found")
        stmt = stmt.where(
            Trajectory.card_ids.contains([card_row.id])
            | Trajectory.verified_card_ids.contains([card_row.id])
        )
    if signal_kind:
        sub = select(Signal.trajectory_id).where(
            Signal.kind == signal_kind, Signal.status != "rejected"
        )
        stmt = stmt.where(Trajectory.id.in_(sub))
    if episode_status:
        sub = select(SearchEpisode.id).where(SearchEpisode.status == episode_status)
        stmt = stmt.where(Trajectory.episode_id.in_(sub))
    if review_label:
        sub = select(Review.trajectory_id).where(Review.label == review_label)
        stmt = stmt.where(Trajectory.id.in_(sub))
    if unlabeled:
        stmt = stmt.where(~Trajectory.id.in_(select(Review.trajectory_id)))
    total = int((await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one())
    rows = (
        (await db.execute(stmt.order_by(Trajectory.created_at.desc()).limit(limit).offset(offset)))
        .scalars()
        .all()
    )
    outs = await trajectory_outs(db, rows, cards=await cards_map(db))
    return Paged(total=total, items=[o.model_dump(mode="json") for o in outs])


@router.get("/{trajectory_id}", response_model=TrajectoryDetail)
async def get_trajectory(trajectory_id: UUID, db: DB, _: CurrentUser) -> TrajectoryDetail:
    return await trajectory_detail(db, await _get(db, trajectory_id))


@router.post("/{trajectory_id}/review", response_model=TrajectoryOut)
async def review_trajectory(
    trajectory_id: UUID,
    body: ReviewIn,
    db: DB,
    user: User = Depends(require("reviews:write")),
) -> TrajectoryOut:
    t = await _get(db, trajectory_id)
    if body.label == ReviewLabel.wrong_cards and not body.corrected_card_ids:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "wrong_cards needs corrected_card_ids"
        )
    if body.label == ReviewLabel.run and not t.format_ok:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "malformed ideas cannot be approved"
        )
    if (
        body.label == ReviewLabel.run
        and t.preship
        and not (t.preship.get("policy_ok", True) and t.preship.get("brand_ok", True))
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "pre-ship checks flagged this idea; fix copy first",
        )
    existing = (
        await db.execute(select(Review).where(Review.trajectory_id == t.id))
    ).scalar_one_or_none()
    before = (
        {"label": existing.label, "corrected": [str(i) for i in existing.corrected_card_ids or []]}
        if existing
        else None
    )
    if existing is None:
        existing = Review(
            trajectory_id=t.id, label=body.label.value, reviewer_id=user.email, note=body.note
        )
        db.add(existing)
    existing.label = body.label.value
    existing.corrected_card_ids = body.corrected_card_ids
    existing.reviewer_id = user.email
    existing.note = body.note
    existing.reviewed_at = datetime.now(UTC)
    if body.label == ReviewLabel.wrong_cards:
        t.tag_source = TagSource.expert.value
    await record_audit(
        db,
        actor_id=user.email,
        action="trajectory.review",
        object_type="trajectory",
        object_id=t.id,
        before=before,
        after={
            "label": body.label.value,
            "corrected": [str(i) for i in body.corrected_card_ids or []],
        },
    )
    await db.flush()
    [out] = await trajectory_outs(db, [t])
    return out


@router.put("/{trajectory_id}/copy", response_model=TrajectoryOut)
async def edit_copy(
    trajectory_id: UUID,
    body: CopyEdit,
    db: DB,
    user: User = Depends(require("reviews:write")),
) -> TrajectoryOut:
    t = await _get(db, trajectory_id)
    shipped_rows = (
        await db.execute(
            select(Render.id).where(Render.trajectory_id == t.id, Render.shipped_at.is_not(None))
        )
    ).first()
    shipped = shipped_rows is not None
    if shipped:
        raise HTTPException(status.HTTP_409_CONFLICT, "copy cannot change after shipping")
    before = dict(t.ad_copy or {})
    t.ad_copy = body.model_dump()
    await record_audit(
        db,
        actor_id=user.email,
        action="trajectory.edit_copy",
        object_type="trajectory",
        object_id=t.id,
        before=before,
        after=t.ad_copy,
    )
    [out] = await trajectory_outs(db, [t])
    return out


@router.post(
    "/{trajectory_id}/notes", response_model=TrajectoryOut, status_code=status.HTTP_201_CREATED
)
async def add_note(trajectory_id: UUID, body: NoteIn, db: DB, user: CurrentUser) -> TrajectoryOut:
    t = await _get(db, trajectory_id)
    db.add(Note(trajectory_id=t.id, author_id=user.email, text=body.text))
    await db.flush()
    [out] = await trajectory_outs(db, [t])
    return out
