from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from outlier_ai.api.deps import DB, Config, CurrentUser, require
from outlier_ai.api.schemas import (
    AccountCreate,
    AccountOut,
    AccountUpdate,
    KillSwitchIn,
    KillSwitchOut,
)
from outlier_ai.core.audit import record_audit
from outlier_ai.core.crypto import get_cipher
from outlier_ai.models.auth import User
from outlier_ai.models.meta import AdAccount
from outlier_ai.models.ops import KillSwitch

router = APIRouter(prefix="/meta", tags=["meta"])


def _out(a: AdAccount) -> AccountOut:
    out = AccountOut.model_validate(a)
    out.has_access_token = bool(a.access_token_enc)
    out.has_page_token = bool(a.page_token_enc)
    return out


@router.get("/accounts", response_model=list[AccountOut])
async def list_accounts(db: DB, _: CurrentUser) -> list[AccountOut]:
    return [
        _out(a)
        for a in (await db.execute(select(AdAccount).order_by(AdAccount.created_at)))
        .scalars()
        .all()
    ]


@router.post("/accounts", response_model=AccountOut, status_code=status.HTTP_201_CREATED)
async def create_account(
    body: AccountCreate, db: DB, cfg: Config, user: User = Depends(require("accounts:write"))
) -> AccountOut:
    if (
        await db.execute(select(AdAccount).where(AdAccount.meta_account_id == body.meta_account_id))
    ).scalar_one_or_none():
        raise HTTPException(status.HTTP_409_CONFLICT, "account already registered")
    cipher = get_cipher() if (body.access_token or body.page_token) else None
    acc = AdAccount(
        name=body.name,
        meta_account_id=body.meta_account_id,
        page_id=body.page_id,
        pixel_id=body.pixel_id,
        status="active",
        attribution_setting=body.attribution_setting,
        api_version=cfg.meta.api_version,
        category=body.category,
        daily_cap_usd=body.daily_cap_usd
        if body.daily_cap_usd is not None
        else cfg.meta.daily_account_cap_usd,
        access_token_enc=cipher.encrypt(body.access_token)
        if cipher and body.access_token
        else None,
        page_token_enc=cipher.encrypt(body.page_token) if cipher and body.page_token else None,
        is_fake=body.is_fake,
    )
    db.add(acc)
    await db.flush()
    await record_audit(
        db,
        actor_id=user.email,
        action="account.create",
        object_type="ad_account",
        object_id=acc.id,
        after={"meta_account_id": acc.meta_account_id, "is_fake": acc.is_fake},
    )
    return _out(acc)


@router.put("/accounts/{account_id}", response_model=AccountOut)
async def update_account(
    account_id: UUID,
    body: AccountUpdate,
    db: DB,
    user: User = Depends(require("accounts:write")),
) -> AccountOut:
    acc = await db.get(AdAccount, account_id)
    if acc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "account not found")
    before = {"status": acc.status, "daily_cap_usd": acc.daily_cap_usd}
    data = body.model_dump(exclude_none=True)
    tokens = {k: data.pop(k) for k in ("access_token", "page_token") if k in data}
    for k, v in data.items():
        setattr(acc, k, v)
    if tokens:
        cipher = get_cipher()
        if "access_token" in tokens:
            acc.access_token_enc = cipher.encrypt(tokens["access_token"])
            acc.status = "active"
            acc.last_error = None
        if "page_token" in tokens:
            acc.page_token_enc = cipher.encrypt(tokens["page_token"])
    await record_audit(
        db,
        actor_id=user.email,
        action="account.update",
        object_type="ad_account",
        object_id=acc.id,
        before=before,
        after={
            "status": acc.status,
            "daily_cap_usd": acc.daily_cap_usd,
            "tokens_rotated": sorted(tokens),
        },
    )
    return _out(acc)


@router.post("/sync", status_code=status.HTTP_202_ACCEPTED)
async def sync_insights(
    db: DB, user: User = Depends(require("accounts:write")), account_id: UUID | None = None
) -> dict:
    from outlier_ai.jobs.registry import enqueue

    job = await enqueue(
        db,
        "sync_insights",
        key=str(account_id or "all"),
        payload={"account_id": str(account_id) if account_id else None},
    )
    return {"job_id": str(job.id), "kind": job.kind, "state": job.state}


@router.post("/sync_comments", status_code=status.HTTP_202_ACCEPTED)
async def sync_comments(
    db: DB, user: User = Depends(require("accounts:write")), account_id: UUID | None = None
) -> dict:
    from outlier_ai.jobs.registry import enqueue

    job = await enqueue(
        db,
        "sync_comments",
        key=str(account_id or "all"),
        payload={"account_id": str(account_id) if account_id else None},
    )
    return {"job_id": str(job.id), "kind": job.kind, "state": job.state}


@router.get("/kill_switch", response_model=KillSwitchOut)
async def get_kill_switch(db: DB, _: CurrentUser) -> KillSwitchOut:
    ks = await db.get(KillSwitch, 1)
    if ks is None:
        ks = KillSwitch(id=1, shipping_enabled=True, reason="", changed_by="system")
        db.add(ks)
        await db.flush()
    return KillSwitchOut(
        shipping_enabled=ks.shipping_enabled,
        reason=ks.reason,
        changed_by=ks.changed_by,
        changed_at=ks.changed_at,
    )


@router.put("/kill_switch", response_model=KillSwitchOut)
async def set_kill_switch(
    body: KillSwitchIn, db: DB, user: User = Depends(require("kill_switch:write"))
) -> KillSwitchOut:
    from datetime import UTC, datetime

    ks = await db.get(KillSwitch, 1)
    if ks is None:
        ks = KillSwitch(id=1)
        db.add(ks)
    before = ks.shipping_enabled
    ks.shipping_enabled = body.shipping_enabled
    ks.reason = body.reason
    ks.changed_by = user.email
    ks.changed_at = datetime.now(UTC)
    await db.flush()
    await record_audit(
        db,
        actor_id=user.email,
        action="kill_switch.set",
        object_type="kill_switch",
        object_id=1,
        before={"shipping_enabled": before},
        after={"shipping_enabled": ks.shipping_enabled, "reason": ks.reason},
    )
    return KillSwitchOut(
        shipping_enabled=ks.shipping_enabled,
        reason=ks.reason,
        changed_by=ks.changed_by,
        changed_at=ks.changed_at,
    )
