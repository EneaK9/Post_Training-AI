"""Process settings from environment variables and .env.

Settings are secrets and deployment facts. Tunable system behavior lives in
config/config.yaml (see core.config), not here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "test", "staging", "prod"] = "dev"

    database_url: str = "postgresql+asyncpg://outlier:outlier@127.0.0.1:5433/outlier"
    test_database_url: str | None = None
    db_echo: bool = False
    db_pool_size: int = 10

    # Anything other than the literal string "false" keeps dry run on.
    dry_run: bool = True

    config_path: Path = Path("config/config.yaml")

    fernet_key: str | None = None
    pii_hmac_key: str = "dev-only-not-secret"

    storage_backend: Literal["local", "s3"] = "local"
    local_storage_dir: Path = Path(".storage")
    s3_endpoint: str | None = None
    s3_bucket: str = "outlier-ai"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"

    anthropic_api_key: str | None = None

    meta_app_id: str | None = None
    meta_app_secret: str | None = None
    meta_sandbox_token: str | None = None

    @field_validator("dry_run", mode="before")
    @classmethod
    def _strict_dry_run(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() != "false"

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """For tests that mutate the environment."""
    get_settings.cache_clear()
