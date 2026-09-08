from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import String, cast, or_, select

from outlier_ai.api.deps import DB, CurrentUser
from outlier_ai.api.schemas import SearchHit, SearchOut
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import Card
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Trajectory

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchOut)
async def search(
    db: DB,
    _: CurrentUser,
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(8, ge=1, le=50),
) -> SearchOut:
    like = f"%{q}%"
    hits: list[SearchHit] = []
    for c in (
        await db.execute(
            select(Card)
            .where(or_(Card.slug.ilike(like), Card.name.ilike(like), Card.definition.ilike(like)))
            .limit(limit)
        )
    ).scalars():
        hits.append(SearchHit(type="card", id=c.id, label=f"{c.name} ({c.slug})", sub=c.kind))
    for b in (
        await db.execute(
            select(Brief)
            .where(
                or_(
                    Brief.company.ilike(like), Brief.product.ilike(like), Brief.raw_text.ilike(like)
                )
            )
            .limit(limit)
        )
    ).scalars():
        hits.append(
            SearchHit(type="brief", id=b.id, label=f"{b.company}: {b.product}", sub=b.category)
        )
    for t in (
        await db.execute(
            select(Trajectory)
            .where(
                or_(
                    Trajectory.angle.ilike(like),
                    cast(Trajectory.ad_copy, String).ilike(like),
                    cast(Trajectory.id, String).ilike(like),
                )
            )
            .limit(limit)
        )
    ).scalars():
        hits.append(
            SearchHit(
                type="trajectory",
                id=t.id,
                label=(t.ad_copy or {}).get("headline", t.angle[:60]),
                sub=f"tier {t.outlier_tier}" if t.outlier_tier is not None else "pending",
            )
        )
    for s in (
        await db.execute(select(Signal).where(Signal.text.ilike(like)).limit(limit))
    ).scalars():
        hits.append(SearchHit(type="signal", id=s.id, label=s.text, sub=f"{s.kind} x{s.count}"))
    for e in (
        await db.execute(
            select(SearchEpisode).where(cast(SearchEpisode.id, String).ilike(like)).limit(limit)
        )
    ).scalars():
        hits.append(
            SearchHit(type="episode", id=e.id, label=f"episode {str(e.id)[:8]}", sub=e.status)
        )
    return SearchOut(q=q, hits=hits)
