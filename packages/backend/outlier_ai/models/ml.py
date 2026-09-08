"""Model registry and run bookkeeping: reward models, verifiers, snapshots, training, eval."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from outlier_ai.models.base import Base, created_at_col, enum_col, uuid_pk


class RewardModelVersion(Base):
    __tablename__ = "rm_models"

    id: Mapped[uuid.UUID] = uuid_pk()
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    kind: Mapped[str] = enum_col()  # rm_cold | rm_outcome
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    trained_on_snapshot: Mapped[str | None] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = created_at_col()


class VerifierVersion(Base):
    __tablename__ = "verifier_models"

    id: Mapped[uuid.UUID] = uuid_pk()
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    kind: Mapped[str] = enum_col()  # llm | classifier
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    trained_on_labels: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = created_at_col()


class ArchiveSnapshot(Base):
    __tablename__ = "archive_snapshots"

    id: Mapped[uuid.UUID] = uuid_pk()
    hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    n_trajectories: Mapped[int] = mapped_column(Integer, nullable=False)
    n_tier2: Mapped[int] = mapped_column(Integer, nullable=False)
    config_hash: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = created_at_col()


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = uuid_pk()
    stage: Mapped[str] = enum_col()
    status: Mapped[str] = enum_col(default="queued", index=True)
    config_hash: Mapped[str | None] = mapped_column(String(16))
    snapshot_hash: Mapped[str | None] = mapped_column(String(64))
    rm_version: Mapped[str | None] = mapped_column(String(64))
    verifier_version: Mapped[str | None] = mapped_column(String(64))
    base_model: Mapped[str | None] = mapped_column(String(128))
    checkpoint_uri: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    stop_reason: Mapped[str | None] = mapped_column(String(255))
    logs_uri: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = uuid_pk()
    kind: Mapped[str] = enum_col()
    status: Mapped[str] = enum_col(default="queued", index=True)
    holdout_brief_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    attribution_setting: Mapped[str | None] = mapped_column(String(32))
    config_hash: Mapped[str | None] = mapped_column(String(16))
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_by: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalArm(Base):
    __tablename__ = "eval_arms"

    id: Mapped[uuid.UUID] = uuid_pk()
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    system: Mapped[str] = mapped_column(String(64), nullable=False)
    blind_label: Mapped[str] = mapped_column(String(16), nullable=False)
    n_briefs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tier2_rate: Mapped[float | None] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
