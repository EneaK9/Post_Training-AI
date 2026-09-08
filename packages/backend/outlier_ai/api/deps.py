"""Request-scoped dependencies: database session, config, current user, role gates."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core import db as dbmod
from outlier_ai.core.auth import SESSION_COOKIE, user_from_token
from outlier_ai.core.config import ConfigStore
from outlier_ai.models.auth import User
from outlier_schemas.config import AppConfig

# Which roles may perform which class of write. Reads are open to every authenticated user.
ROLE_MATRIX: dict[str, set[str]] = {
    "cards:write": {"expert", "researcher"},
    "reviews:write": {"expert", "operator"},
    "signals:write": {"expert", "operator"},
    "briefs:write": {"operator", "expert"},
    "episodes:write": {"operator"},
    "generate": {"operator", "researcher", "expert"},
    "config:write": {"researcher"},
    "accounts:write": {"operator"},
    "eval:write": {"researcher"},
    "import:write": {"expert", "operator", "researcher"},
    "kill_switch:write": {"operator", "researcher"},
}


async def get_db() -> AsyncIterator[AsyncSession]:
    factory = dbmod.get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


DB = Annotated[AsyncSession, Depends(get_db)]


async def get_config(db: DB) -> AppConfig:
    return await ConfigStore(db).current()


Config = Annotated[AppConfig, Depends(get_config)]


def _token_from_request(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.cookies.get(SESSION_COOKIE)


async def current_user(request: Request, db: DB) -> User:
    token = _token_from_request(request)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")
    user = await user_from_token(db, token)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "session invalid or expired")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


def require(permission: str) -> Callable[..., User]:
    allowed = ROLE_MATRIX[permission]

    async def _dep(user: CurrentUser) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"role {user.role} may not {permission}; needs one of {sorted(allowed)}",
            )
        return user

    return _dep  # type: ignore[return-value]
