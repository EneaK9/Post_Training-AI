"""Async SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from outlier_ai.core.settings import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def make_engine(url: str | None = None, **kwargs: object) -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        url or settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        pool_pre_ping=True,
        # Neon's pooler is PgBouncer in transaction mode. We use the direct endpoint, but
        # disabling asyncpg's statement cache keeps the code safe if someone points it at
        # the pooler. Cost is negligible for this workload.
        connect_args={"statement_cache_size": 0},
        **kwargs,
    )


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


def configure(engine: AsyncEngine) -> None:
    """Point the process at a specific engine (tests, workers)."""
    global _engine, _session_factory
    _engine = engine
    _session_factory = async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Unit of work: commit on success, roll back on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


async def dispose() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
