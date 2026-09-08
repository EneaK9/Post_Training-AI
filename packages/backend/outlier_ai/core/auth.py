"""Built-in email + password auth with server-side sessions and three roles."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.errors import ValidationError
from outlier_ai.models.auth import Session, User
from outlier_schemas.enums import Role

SESSION_COOKIE = "oai_session"
SESSION_TTL = timedelta(days=14)
_ph = PasswordHasher()


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValidationError("password must be at least 8 characters")
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    if not password_hash or password_hash.startswith("$unset$"):
        return False
    try:
        return _ph.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def create_user(
    session: AsyncSession, email: str, password: str, role: Role, display_name: str = ""
) -> User:
    email = email.strip().lower()
    existing = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if existing is not None:
        raise ValidationError(f"user {email} already exists")
    user = User(
        email=email,
        display_name=display_name or email.split("@")[0],
        password_hash=hash_password(password),
        role=role.value,
    )
    session.add(user)
    await session.flush()
    return user


async def set_password(session: AsyncSession, email: str, password: str) -> User:
    user = (
        await session.execute(select(User).where(User.email == email.lower()))
    ).scalar_one_or_none()
    if user is None:
        raise ValidationError(f"no user {email}")
    user.password_hash = hash_password(password)
    await session.flush()
    return user


async def authenticate(session: AsyncSession, email: str, password: str) -> User | None:
    user = (
        await session.execute(select(User).where(User.email == email.strip().lower()))
    ).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(user.password_hash, password):
        return None
    return user


async def create_session(session: AsyncSession, user: User) -> str:
    token = secrets.token_urlsafe(32)
    session.add(
        Session(
            user_id=user.id,
            token_hash=token_hash(token),
            expires_at=datetime.now(UTC) + SESSION_TTL,
        )
    )
    await session.flush()
    return token


async def user_from_token(session: AsyncSession, token: str) -> User | None:
    row = (
        await session.execute(
            select(Session, User)
            .join(User, User.id == Session.user_id)
            .where(Session.token_hash == token_hash(token))
        )
    ).first()
    if row is None:
        return None
    sess, user = row
    if sess.expires_at < datetime.now(UTC) or not user.is_active:
        return None
    return user


async def revoke_session(session: AsyncSession, token: str) -> None:
    row = (
        await session.execute(select(Session).where(Session.token_hash == token_hash(token)))
    ).scalar_one_or_none()
    if row is not None:
        await session.delete(row)
        await session.flush()
