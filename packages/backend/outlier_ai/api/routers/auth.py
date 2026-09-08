from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from outlier_ai.api.deps import DB, CurrentUser
from outlier_ai.api.schemas import LoginIn, UserOut
from outlier_ai.core.auth import (
    SESSION_COOKIE,
    SESSION_TTL,
    authenticate,
    create_session,
    revoke_session,
)
from outlier_ai.core.settings import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=UserOut)
async def login(body: LoginIn, response: Response, db: DB) -> UserOut:
    user = await authenticate(db, body.email, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    token = await create_session(db, user)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=get_settings().is_prod,
    )
    return UserOut.model_validate(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, response: Response, db: DB) -> None:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        await revoke_session(db, token)
    response.delete_cookie(SESSION_COOKIE)


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
