from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import select

from outlier_ai.api.deps import DB, CurrentUser
from outlier_ai.api.schemas import TrajectoryOut
from outlier_ai.api.views import trajectory_outs
from outlier_ai.models.trajectories import Review, Trajectory

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/queue", response_model=list[TrajectoryOut])
async def review_queue(
    db: DB, _: CurrentUser, brief_id: UUID | None = None, limit: int = Query(50, ge=1, le=500)
) -> list[TrajectoryOut]:
    """Unlabeled, well-formed ideas sorted by rm_score (nulls last)."""
    stmt = (
        select(Trajectory)
        .where(~Trajectory.id.in_(select(Review.trajectory_id)), Trajectory.format_ok.is_(True))
        .order_by(Trajectory.rm_score.desc().nulls_last(), Trajectory.created_at.desc())
        .limit(limit)
    )
    if brief_id:
        stmt = stmt.where(Trajectory.brief_id == brief_id)
    return await trajectory_outs(db, (await db.execute(stmt)).scalars().all())
