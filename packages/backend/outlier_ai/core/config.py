"""Load, hash, version, and apply the typed config tree.

The YAML file is the boot default. Applying a config through the API stores it in
`config_versions`; the newest stored version is authoritative at runtime. Pure modules
receive an `AppConfig` explicitly and never read the database.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.settings import get_settings
from outlier_ai.models.ops import ConfigVersion
from outlier_schemas.config import AppConfig


@lru_cache(maxsize=4)
def load_file_config(path: str | Path | None = None) -> AppConfig:
    p = Path(path) if path else get_settings().config_path
    return AppConfig.from_yaml(p)


def config_to_yaml(cfg: AppConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)


class ConfigStore:
    """Database-backed history of applied configs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def latest(self) -> ConfigVersion | None:
        res = await self.session.execute(
            select(ConfigVersion).order_by(ConfigVersion.applied_at.desc()).limit(1)
        )
        return res.scalar_one_or_none()

    async def current(self) -> AppConfig:
        row = await self.latest()
        if row is None:
            return load_file_config()
        return AppConfig.from_dict(row.config_json)

    async def get(self, config_hash: str) -> AppConfig | None:
        res = await self.session.execute(
            select(ConfigVersion).where(ConfigVersion.hash == config_hash)
        )
        row = res.scalar_one_or_none()
        return AppConfig.from_dict(row.config_json) if row else None

    async def apply(self, cfg: AppConfig, applied_by: str, note: str = "") -> ConfigVersion:
        """Store the config if its hash is new; return the stored row either way."""
        res = await self.session.execute(
            select(ConfigVersion).where(ConfigVersion.hash == cfg.hash)
        )
        existing = res.scalar_one_or_none()
        if existing is not None:
            return existing
        row = ConfigVersion(
            hash=cfg.hash,
            yaml_text=config_to_yaml(cfg),
            config_json=cfg.model_dump(mode="json"),
            applied_by=applied_by,
            note=note,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def ensure_bootstrapped(self, applied_by: str = "system") -> ConfigVersion:
        """Make sure the file config exists as version 1 so hashes can be pinned."""
        row = await self.latest()
        if row is not None:
            return row
        return await self.apply(
            load_file_config(), applied_by=applied_by, note="bootstrap from file"
        )

    async def history(self, limit: int = 50) -> list[ConfigVersion]:
        res = await self.session.execute(
            select(ConfigVersion).order_by(ConfigVersion.applied_at.desc()).limit(limit)
        )
        return list(res.scalars().all())
