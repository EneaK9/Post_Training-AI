"""Search episodes and batches. [v3-assumed] fields reconstructed in docs/assumptions.md."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from outlier_ai.models.base import Base, created_at_col, enum_col, uuid_pk

if TYPE_CHECKING:
    from outlier_ai.models.briefs import Brief
    from outlier_ai.models.meta import AdAccount


class SearchEpisode(Base):
    __tablename__ = "search_episodes"

    id: Mapped[uuid.UUID] = uuid_pk()
    brief_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("briefs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    ad_account_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("ad_accounts.id", ondelete="SET NULL"), index=True
    )
    backend: Mapped[str] = enum_col()
    budget_cap: Mapped[float] = mapped_column(Float, nullable=False)
    spent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = enum_col(default="searching", index=True)
    keep_running_after_outlier: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    campaign_id: Mapped[str | None] = mapped_column(String(64), index=True)
    config_hash: Mapped[str | None] = mapped_column(String(16))
    created_by: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stop_reason: Mapped[str | None] = mapped_column(String(255))

    brief: Mapped[Brief] = relationship("Brief")
    ad_account: Mapped[AdAccount | None] = relationship("AdAccount")
    batches: Mapped[list[Batch]] = relationship(
        back_populates="episode", cascade="all, delete-orphan", order_by="Batch.index"
    )


class Batch(Base):
    __tablename__ = "batches"
    __table_args__ = (UniqueConstraint("episode_id", "index"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    episode_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("search_episodes.id", ondelete="CASCADE"), index=True, nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = enum_col(default="proposed", index=True)
    prompt_trace: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = created_at_col()
    state_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    episode: Mapped[SearchEpisode] = relationship(back_populates="batches")
