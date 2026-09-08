"""Loop B training runs: launch a stage against a fresh archive snapshot, list, inspect, stop."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from outlier_ai.api.deps import DB, Config, CurrentUser, require
from outlier_ai.api.schemas import TrainingRunIn, TrainingRunOut
from outlier_ai.core.audit import record_audit
from outlier_ai.jobs.training import queue_training_run
from outlier_ai.models.auth import User
from outlier_ai.models.ml import TrainingRun
from outlier_schemas.enums import TrainingRunStatus, TrainingStage

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/runs", response_model=list[TrainingRunOut])
async def list_runs(db: DB, _: CurrentUser, limit: int = 50) -> list[TrainingRunOut]:
    rows = (
        (await db.execute(select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(limit)))
        .scalars()
        .all()
    )
    return [TrainingRunOut.model_validate(r) for r in rows]


@router.post("/runs", response_model=TrainingRunOut, status_code=status.HTTP_201_CREATED)
async def launch(
    body: TrainingRunIn, db: DB, cfg: Config, user: User = Depends(require("training:write"))
) -> TrainingRunOut:
    if body.stage == TrainingStage.grpo_onpolicy.value and not (
        body.simulator or body.allow_rm_reward
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "grpo_onpolicy trains against the reward model, not real outcomes; "
            "set simulator=true or allow_rm_reward=true on purpose",
        )
    try:
        run, job = await queue_training_run(
            db,
            cfg=cfg,
            stage=TrainingStage(body.stage),
            base_model=body.base_model,
            created_by=user.email,
            smoke=body.smoke,
            dry=body.dry,
            simulator=body.simulator,
            allow_rm_reward=body.allow_rm_reward,
            adapter_from=body.adapter_from,
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    await record_audit(
        db,
        actor_id=user.email,
        action="training.launched",
        object_type="training_run",
        object_id=run.id,
        after={"stage": run.stage, "job_id": str(job.id), "smoke": body.smoke, "dry": body.dry},
    )
    return TrainingRunOut.model_validate(run)


@router.get("/runs/{run_id}", response_model=TrainingRunOut)
async def get_run(run_id: UUID, db: DB, _: CurrentUser) -> TrainingRunOut:
    run = await db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "training run not found")
    return TrainingRunOut.model_validate(run)


@router.post("/runs/{run_id}/stop", response_model=TrainingRunOut)
async def stop_run(
    run_id: UUID, db: DB, user: User = Depends(require("training:write"))
) -> TrainingRunOut:
    run = await db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "training run not found")
    if run.status in (TrainingRunStatus.queued.value, TrainingRunStatus.running.value):
        run.status = TrainingRunStatus.stopped.value
        run.stop_reason = f"stopped by {user.email}"
        run.finished_at = datetime.now(UTC)
        await record_audit(
            db,
            actor_id=user.email,
            action="training.stopped",
            object_type="training_run",
            object_id=run.id,
        )
    return TrainingRunOut.model_validate(run)
