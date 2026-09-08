"""Shared fixtures.

Integration tests need Postgres at TEST_DATABASE_URL (docker compose provides one). When it
is unreachable they are skipped, not failed, so unit tests run anywhere.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from outlier_ai.core import db as dbmod
from outlier_ai.models import Base
from outlier_schemas.config import AppConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://outlier:outlier@127.0.0.1:5433/outlier_test"
)

os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("PII_HMAC_KEY", "test-key")
os.environ.setdefault("CONFIG_PATH", str(REPO_ROOT / "config" / "config.yaml"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return AppConfig.from_yaml(REPO_ROOT / "config" / "config.yaml")


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(
        TEST_DATABASE_URL, poolclass=NullPool, connect_args={"statement_cache_size": 0}
    )
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        await engine.dispose()
        pytest.skip(f"test database unavailable at {TEST_DATABASE_URL}: {e!r}")
    dbmod.configure(engine)
    try:
        yield engine
    finally:
        await engine.dispose()
        await dbmod.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
