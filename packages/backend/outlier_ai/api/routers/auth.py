from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import func, select

from outlier_ai.api.deps import DB, CurrentUser
from outlier_ai.api.schemas import AuthOptionsOut, LoginIn, SignupIn, UserOut
from outlier_ai.core.audit import record_audit
from outlier_ai.core.auth import (
    SESSION_COOKIE,
    SESSION_TTL,
    authenticate,
    create_session,
    create_user,
    revoke_session,
)
from outlier_ai.core.settings import get_settings
from outlier_ai.models.auth import User
from outlier_schemas.enums import Role

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=get_settings().is_prod,
    )


@router.get("/options", response_model=AuthOptionsOut)
async def options() -> AuthOptionsOut:
    """What the login and sign-up pages may offer. Public: read before anyone is signed in."""
    return AuthOptionsOut(
        signup_enabled=get_settings().signup_enabled, roles=[r.value for r in Role]
    )


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupIn, response: Response, db: DB) -> UserOut:
    """Self-service account creation, then signed in. Off when SIGNUP_ENABLED=false."""
    if not get_settings().signup_enabled:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "sign-up is disabled; ask an operator")
    email = body.email.strip().lower()
    exists = (await db.execute(select(User.id).where(func.lower(User.email) == email))).first()
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with this email already exists")
    user = await create_user(db, email, body.password, body.role, body.display_name.strip())
    await record_audit(
        db,
        actor_id=user.email,
        action="user.signup",
        object_type="user",
        object_id=user.id,
        after={"role": user.role},
    )
    _set_session_cookie(response, await create_session(db, user))
    return UserOut.model_validate(user)


@router.post("/login", response_model=UserOut)
async def login(body: LoginIn, response: Response, db: DB) -> UserOut:
    user = await authenticate(db, body.email, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    _set_session_cookie(response, await create_session(db, user))
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
