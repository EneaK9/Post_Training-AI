from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from outlier_ai.api.deps import DB, Config, require
from outlier_ai.api.schemas import ImportResult
from outlier_ai.core.audit import record_audit
from outlier_ai.core.embeddings import get_embedder
from outlier_ai.generation.backends import get_judge
from outlier_ai.generation.verifier import (
    HeuristicVerifier,
    LLMVerifier,
    load_card_refs,
    load_few_shot,
)
from outlier_ai.imports.csv_import import (
    import_cards,
    import_comments,
    import_outcomes,
    import_trajectories,
)
from outlier_ai.models.auth import User
from outlier_schemas.enums import VerifierKind

router = APIRouter(prefix="/import", tags=["import"])


@router.post("/cards", response_model=ImportResult)
async def cards(
    db: DB, file: UploadFile = File(...), user: User = Depends(require("import:write"))
) -> ImportResult:
    res = await import_cards(db, await file.read(), actor=user.email)
    await record_audit(
        db,
        actor_id=user.email,
        action="import.cards",
        object_type="import",
        object_id=file.filename or "cards.csv",
        after=res.model_dump(),
    )
    return res


@router.post("/trajectories", response_model=ImportResult)
async def trajectories(
    db: DB,
    cfg: Config,
    file: UploadFile = File(...),
    user: User = Depends(require("import:write")),
) -> ImportResult:
    card_refs = await load_card_refs(db)
    judge = get_judge(cfg)
    if judge is not None and cfg.verifier.kind == VerifierKind.llm:
        verifier = LLMVerifier(
            judge, card_refs, await load_few_shot(db, cfg.verifier.few_shot_n, card_refs)
        )
    else:
        verifier = HeuristicVerifier(card_refs)
    res = await import_trajectories(
        db,
        await file.read(),
        actor=user.email,
        cfg=cfg,
        verifier=verifier,
        embedder=get_embedder(
            dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
        ),
    )
    await record_audit(
        db,
        actor_id=user.email,
        action="import.trajectories",
        object_type="import",
        object_id=file.filename or "trajectories.csv",
        after=res.model_dump(),
    )
    return res


@router.post("/outcomes", response_model=ImportResult)
async def outcomes(
    db: DB,
    cfg: Config,
    file: UploadFile = File(...),
    user: User = Depends(require("import:write")),
) -> ImportResult:
    res = await import_outcomes(db, await file.read(), cfg=cfg)
    await record_audit(
        db,
        actor_id=user.email,
        action="import.outcomes",
        object_type="import",
        object_id=file.filename or "outcomes.csv",
        after=res.model_dump(),
    )
    return res


@router.post("/comments", response_model=ImportResult)
async def comments(
    db: DB, file: UploadFile = File(...), user: User = Depends(require("import:write"))
) -> ImportResult:
    res = await import_comments(db, await file.read())
    await record_audit(
        db,
        actor_id=user.email,
        action="import.comments",
        object_type="import",
        object_id=file.filename or "comments.csv",
        after=res.model_dump(),
    )
    return res
