from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from outlier_ai.api.deps import DB, CurrentUser, require
from outlier_ai.api.schemas import SignalCreate, SignalOut
from outlier_ai.core.audit import record_audit
from outlier_ai.models.auth import User
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Trajectory
from outlier_schemas.enums import AuthorKind, SignalStatus

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("", response_model=list[SignalOut])
async def list_signals(
    db: DB,
    _: CurrentUser,
    status_: str | None = Query(None, alias="status"),
    kind: str | None = None,
    trajectory_id: UUID | None = None,
    limit: int = Query(200, ge=1, le=1000),
) -> list[SignalOut]:
    stmt = select(Signal).order_by(Signal.created_at.desc()).limit(limit)
    if status_:
        stmt = stmt.where(Signal.status == status_)
    if kind:
        stmt = stmt.where(Signal.kind == kind)
    if trajectory_id:
        stmt = stmt.where(Signal.trajectory_id == trajectory_id)
    return [SignalOut.model_validate(s) for s in (await db.execute(stmt)).scalars().all()]


@router.get("/queue", response_model=list[SignalOut])
async def signals_queue(
    db: DB, _: CurrentUser, limit: int = Query(200, ge=1, le=1000)
) -> list[SignalOut]:
    stmt = (
        select(Signal)
        .where(Signal.status == SignalStatus.proposed.value)
        .order_by(Signal.trajectory_id, Signal.count.desc())
        .limit(limit)
    )
    return [SignalOut.model_validate(s) for s in (await db.execute(stmt)).scalars().all()]


@router.get("/{signal_id}", response_model=SignalOut)
async def get_signal(signal_id: UUID, db: DB, _: CurrentUser) -> SignalOut:
    s = await db.get(Signal, signal_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "signal not found")
    return SignalOut.model_validate(s)


@router.post("", response_model=SignalOut, status_code=status.HTTP_201_CREATED)
async def create_signal(
    body: SignalCreate, db: DB, user: User = Depends(require("signals:write"))
) -> SignalOut:
    if await db.get(Trajectory, body.trajectory_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "trajectory not found")
    s = Signal(
        trajectory_id=body.trajectory_id,
        render_id=body.render_id,
        kind=body.kind.value,
        text=body.text,
        evidence=[{"source": "manual", "ref": user.email, "excerpt": body.text[:200]}],
        sentiment=body.sentiment.value,
        count=body.count,
        extracted_by=AuthorKind.human.value,
        status=SignalStatus.confirmed.value,
        decided_by=user.email,
        decided_at=datetime.now(UTC),
    )
    db.add(s)
    await db.flush()
    await record_audit(
        db,
        actor_id=user.email,
        action="signal.create",
        object_type="signal",
        object_id=s.id,
        after={"kind": s.kind, "text": s.text},
    )
    return SignalOut.model_validate(s)


async def _decide(db: DB, signal_id: UUID, user, decision: SignalStatus) -> SignalOut:
    s = await db.get(Signal, signal_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "signal not found")
    before = s.status
    s.status = decision.value
    s.decided_by = user.email
    s.decided_at = datetime.now(UTC)
    await record_audit(
        db,
        actor_id=user.email,
        action=f"signal.{decision.value}",
        object_type="signal",
        object_id=s.id,
        before={"status": before},
        after={"status": s.status},
    )
    return SignalOut.model_validate(s)


@router.put("/{signal_id}/confirm", response_model=SignalOut)
async def confirm_signal(
    signal_id: UUID, db: DB, user: User = Depends(require("signals:write"))
) -> SignalOut:
    return await _decide(db, signal_id, user, SignalStatus.confirmed)


@router.put("/{signal_id}/reject", response_model=SignalOut)
async def reject_signal(
    signal_id: UUID, db: DB, user: User = Depends(require("signals:write"))
) -> SignalOut:
    return await _decide(db, signal_id, user, SignalStatus.rejected)


@router.post("/extract", response_model=list[SignalOut], status_code=status.HTTP_201_CREATED)
async def extract_signals(
    trajectory_id: UUID, db: DB, user: User = Depends(require("signals:write"))
) -> list[SignalOut]:
    """Group this trajectory's stored comments into proposed signals (theme, count, evidence)."""
    from outlier_ai.signals.extraction import extract_for_trajectory

    if await db.get(Trajectory, trajectory_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "trajectory not found")
    created = await extract_for_trajectory(db, trajectory_id, actor=user.email)
    return [SignalOut.model_validate(s) for s in created]
