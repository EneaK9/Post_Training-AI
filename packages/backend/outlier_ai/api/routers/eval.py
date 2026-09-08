"""Online eval: launch arms on held-out briefs, read results, refresh live runs."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from outlier_ai.api.deps import DB, Config, CurrentUser, require
from outlier_ai.api.schemas import EvalArmOut, EvalLaunchIn, EvalRunOut
from outlier_ai.core.audit import record_audit
from outlier_ai.core.storage import get_storage
from outlier_ai.eval.harness import refresh_live_eval, run_eval
from outlier_ai.eval.systems import SYSTEMS
from outlier_ai.jobs.registry import enqueue
from outlier_ai.models.auth import User
from outlier_ai.models.ml import EvalArm, EvalRun
from outlier_schemas.enums import EvalKind

router = APIRouter(prefix="/eval", tags=["eval"])


async def _out(db: DB, run: EvalRun, job_id: UUID | None = None) -> EvalRunOut:
    arms = (
        (
            await db.execute(
                select(EvalArm).where(EvalArm.eval_run_id == run.id).order_by(EvalArm.blind_label)
            )
        )
        .scalars()
        .all()
    )
    out = EvalRunOut.model_validate(run)
    out.arms = [EvalArmOut.model_validate(a) for a in arms]
    out.job_id = job_id
    return out


@router.get("/systems")
async def systems(_: CurrentUser) -> dict[str, dict[str, str]]:
    return {
        k: {"backend": v.backend.value, "approval": v.approval, "description": v.description}
        for k, v in SYSTEMS.items()
    }


@router.get("", response_model=list[EvalRunOut])
async def list_evals(db: DB, _: CurrentUser, limit: int = 50) -> list[EvalRunOut]:
    runs = (
        (await db.execute(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)))
        .scalars()
        .all()
    )
    return [await _out(db, r) for r in runs]


@router.post("/launch", response_model=EvalRunOut, status_code=status.HTTP_201_CREATED)
async def launch(
    body: EvalLaunchIn, db: DB, cfg: Config, user: User = Depends(require("eval:write"))
) -> EvalRunOut:
    unknown = [s for s in body.systems if s not in SYSTEMS]
    if unknown:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"unknown systems {unknown}")
    if body.inline:
        try:
            result = await run_eval(
                db,
                cfg=cfg,
                storage=get_storage(),
                systems=body.systems,
                n_briefs=body.n_briefs,
                budget_cap=body.budget_cap,
                created_by=user.email,
                kind=EvalKind(body.kind),
                seed=body.seed,
                max_days=body.max_days,
            )
        except ValueError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
        run = await db.get(EvalRun, result.eval_id)
        assert run is not None
        await record_audit(
            db,
            actor_id=user.email,
            action="eval.launched",
            object_type="eval_run",
            object_id=run.id,
            after=result.summary(),
        )
        return await _out(db, run)
    run = EvalRun(
        kind=body.kind,
        status="queued",
        holdout_brief_ids=[],
        config_hash=cfg.hash,
        created_by=user.email,
        summary={"requested": body.model_dump(mode="json")},
    )
    db.add(run)
    await db.flush()
    job = await enqueue(
        db,
        "eval",
        key=f"eval:{run.id}",
        payload={"eval_id": str(run.id), **body.model_dump(mode="json"), "created_by": user.email},
    )
    await record_audit(
        db,
        actor_id=user.email,
        action="eval.queued",
        object_type="eval_run",
        object_id=run.id,
        after=body.model_dump(mode="json"),
    )
    return await _out(db, run, job.id)


@router.get("/{eval_id}", response_model=EvalRunOut)
async def get_eval(eval_id: UUID, db: DB, _: CurrentUser) -> EvalRunOut:
    run = await db.get(EvalRun, eval_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "eval not found")
    return await _out(db, run)


@router.post("/{eval_id}/refresh", response_model=EvalRunOut)
async def refresh(eval_id: UUID, db: DB, _: User = Depends(require("eval:write"))) -> EvalRunOut:
    run = await db.get(EvalRun, eval_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "eval not found")
    await refresh_live_eval(db, eval_id)
    return await _out(db, run)
