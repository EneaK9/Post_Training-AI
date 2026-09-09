"""Feedback loops on demand: suggested relations, verifier retraining, RM retraining."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from outlier_ai.api.deps import DB, Config, require
from outlier_ai.api.schemas import FeedbackOut, RMTrainIn, RMTrainOut
from outlier_ai.core.embeddings import get_embedder
from outlier_ai.core.settings import get_settings
from outlier_ai.core.storage import get_storage
from outlier_ai.jobs.feedback import retrain_verifier, suggest_relations
from outlier_ai.models.auth import User
from outlier_ai.reward.train import train_reward_model
from outlier_schemas.enums import RewardModelKind

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/suggest_relations", response_model=FeedbackOut)
async def suggest(db: DB, user: User = Depends(require("feedback:write"))) -> FeedbackOut:
    out = await suggest_relations(db, actor=user.email)
    return FeedbackOut(action="suggest_relations", count=len(out))


@router.post("/retrain_verifier", response_model=FeedbackOut)
async def retrain_verifier_now(
    db: DB,
    cfg: Config,
    min_labels: int | None = None,
    user: User = Depends(require("feedback:write")),
) -> FeedbackOut:
    s = get_settings()
    embedder = get_embedder(
        s, dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
    )
    vv = await retrain_verifier(
        db,
        cfg=cfg,
        embedder=embedder,
        storage=get_storage(s),
        min_labels=min_labels,
        actor=user.email,
    )
    if vv is None:
        need = cfg.verifier.classifier_min_labels if min_labels is None else min_labels
        return FeedbackOut(
            action="retrain_verifier", count=0, reason=f"fewer than {need} expert labels available"
        )
    return FeedbackOut(
        action="retrain_verifier",
        count=vv.trained_on_labels,
        version=vv.version,
        metrics=vv.metrics,
    )


@router.post("/retrain_rm", response_model=RMTrainOut)
async def retrain_rm(
    body: RMTrainIn, db: DB, cfg: Config, _: User = Depends(require("training:write"))
) -> RMTrainOut:
    s = get_settings()
    embedder = get_embedder(
        s, dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
    )
    res = await train_reward_model(
        db,
        cfg=cfg,
        embedder=embedder,
        storage=get_storage(s),
        kind=RewardModelKind(body.kind) if body.kind else None,
        activate=body.activate,
        seed=body.seed,
    )
    if res is None:
        g = cfg.reward_model
        return RMTrainOut(
            trained=False,
            reason=f"not enough labeled rows: need {g.min_rows} rows with {g.min_per_class}+ of "
            "each class; the cold reward model stays in use",
        )
    return RMTrainOut(
        trained=True,
        version=res.version,
        kind=res.kind.value,
        n_rows=res.n_rows,
        n_positive=res.n_positive,
        metrics={
            "train": res.report.__dict__,
            "calibration": res.calibration.__dict__,
            "val_auc": res.val_auc,
            "activated": res.activated,
            "note": res.note,
        },
        reason=res.note or None,
    )
