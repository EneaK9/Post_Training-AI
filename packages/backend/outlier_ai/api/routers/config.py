from __future__ import annotations

from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError as PydanticValidationError

from outlier_ai.api.deps import DB, CurrentUser, require
from outlier_ai.api.schemas import ConfigIn, ConfigOut, ConfigValidateOut, ConfigVersionOut
from outlier_ai.core.audit import record_audit
from outlier_ai.core.config import ConfigStore, config_to_yaml, load_file_config
from outlier_ai.models.auth import User
from outlier_ai.outlier.recompute import recompute_all
from outlier_schemas.config import AppConfig

router = APIRouter(prefix="/config", tags=["config"])


def _parse(body: ConfigIn) -> tuple[AppConfig | None, list[str]]:
    raw: dict[str, Any] | None = body.config
    if body.yaml:
        try:
            raw = yaml.safe_load(body.yaml) or {}
        except yaml.YAMLError as e:
            return None, [f"yaml: {e}"]
    if raw is None:
        return None, ["provide yaml or config"]
    try:
        return AppConfig.from_dict(raw), []
    except PydanticValidationError as e:
        return None, [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]


def _recompute_required(diff: dict[str, Any]) -> bool:
    return any(k.startswith("outlier.") or k.startswith("category_medians") for k in diff)


@router.get("", response_model=ConfigOut)
async def get_config(db: DB, _: CurrentUser) -> ConfigOut:
    store = ConfigStore(db)
    row = await store.latest()
    cfg = await store.current()
    return ConfigOut(
        hash=cfg.hash,
        applied_by=row.applied_by if row else None,
        applied_at=row.applied_at if row else None,
        note=row.note if row else None,
        config=cfg.model_dump(mode="json"),
        yaml=config_to_yaml(cfg),
        file_hash=load_file_config().hash,
    )


@router.post("/validate", response_model=ConfigValidateOut)
async def validate_config(body: ConfigIn, db: DB, _: CurrentUser) -> ConfigValidateOut:
    cfg, errors = _parse(body)
    current = await ConfigStore(db).current()
    if cfg is None:
        return ConfigValidateOut(
            ok=False, errors=errors, hash=None, diff={}, recompute_required=False
        )
    diff = {k: {"current": a, "proposed": b} for k, (a, b) in current.diff(cfg).items()}
    return ConfigValidateOut(
        ok=True, errors=[], hash=cfg.hash, diff=diff, recompute_required=_recompute_required(diff)
    )


@router.post("/apply", response_model=ConfigOut)
async def apply_config(
    body: ConfigIn, db: DB, user: User = Depends(require("config:write"))
) -> ConfigOut:
    cfg, errors = _parse(body)
    if cfg is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "; ".join(errors))
    store = ConfigStore(db)
    current = await store.current()
    diff = current.diff(cfg)
    row = await store.apply(cfg, applied_by=user.email, note=body.note)
    await record_audit(
        db,
        actor_id=user.email,
        action="config.apply",
        object_type="config",
        object_id=cfg.hash,
        before={"hash": current.hash},
        after={"hash": cfg.hash, "changed": sorted(diff)},
    )
    if _recompute_required({k: v for k, v in diff.items()}):
        await recompute_all(db, cfg)
    return ConfigOut(
        hash=cfg.hash,
        applied_by=row.applied_by,
        applied_at=row.applied_at,
        note=row.note,
        config=cfg.model_dump(mode="json"),
        yaml=config_to_yaml(cfg),
        file_hash=load_file_config().hash,
    )


@router.get("/history", response_model=list[ConfigVersionOut])
async def config_history(db: DB, _: CurrentUser, limit: int = 50) -> list[ConfigVersionOut]:
    return [ConfigVersionOut.model_validate(r) for r in await ConfigStore(db).history(limit)]
