"""Operational tables: config versions, jobs, audit log, kill switch, holdout blocklist."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from outlier_ai.models.base import Base, created_at_col, enum_col, uuid_pk


class ConfigVersion(Base):
    __tablename__ = "config_versions"

    id: Mapped[uuid.UUID] = uuid_pk()
    hash: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    yaml_text: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    applied_by: Mapped[str] = mapped_column(String(120), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    applied_at: Mapped[datetime] = created_at_col()


class Job(Base):
    """Idempotent scheduled work. (kind, key) is unique so re-scheduling is a no-op."""

    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("kind", "key"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    scheduled_for: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    state: Mapped[str] = enum_col(default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at_col()


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    object_type: Mapped[str] = mapped_column(String(64), nullable=False)
    object_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    at: Mapped[datetime] = created_at_col()


class KillSwitch(Base):
    """Single row (id = 1). shipping_enabled = false blocks every ship path."""

    __tablename__ = "kill_switch"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    shipping_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    changed_by: Mapped[str] = mapped_column(String(120), nullable=False, default="system")
    changed_at: Mapped[datetime] = created_at_col()


class HoldoutCampaign(Base):
    """Leakage blocklist by campaign id (spec section 11 and scenario table last row)."""

    __tablename__ = "holdout_campaigns"

    campaign_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    added_by: Mapped[str] = mapped_column(String(120), nullable=False)
    added_at: Mapped[datetime] = created_at_col()
