"""Evaluator side: ad accounts, raw daily insights, raw comments, and fake-client state."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from outlier_ai.models.base import Base, created_at_col, enum_col, updated_at_col, uuid_pk

if TYPE_CHECKING:
    from outlier_ai.models.trajectories import Render


class AdAccount(Base):
    __tablename__ = "ad_accounts"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    meta_account_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    page_id: Mapped[str | None] = mapped_column(String(64))
    pixel_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = enum_col(default="active", index=True)
    attribution_setting: Mapped[str] = mapped_column(String(32), nullable=False)
    api_version: Mapped[str] = mapped_column(String(16), nullable=False)
    category: Mapped[str | None] = mapped_column(String(64))
    daily_cap_usd: Mapped[float | None] = mapped_column(Float)
    access_token_enc: Mapped[str | None] = mapped_column(Text)
    page_token_enc: Mapped[str | None] = mapped_column(Text)
    is_fake: Mapped[bool] = mapped_column(nullable=False, default=False)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()


class DailyInsight(Base):
    """RAW. One row per render per day. Everything derived is recomputed from here."""

    __tablename__ = "daily_insights"
    __table_args__ = (UniqueConstraint("render_id", "day"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    render_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("renders.id", ondelete="CASCADE"), index=True, nullable=False
    )
    day: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    phase: Mapped[str] = enum_col(default="screening")  # screening | scale
    impressions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    link_clicks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    spend: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    purchases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    revenue: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frequency: Mapped[float | None] = mapped_column(Float)
    reactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    comments: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    shares: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    saves: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    daily_budget: Mapped[float | None] = mapped_column(Float)
    attribution_setting: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    source: Mapped[str] = enum_col(default="meta")  # meta | fake | import
    ingested_at: Mapped[datetime] = created_at_col()

    render: Mapped[Render] = relationship("Render")


class Comment(Base):
    """RAW but PII-stripped before it gets here. Commenter identity is an HMAC hash."""

    __tablename__ = "comments"

    id: Mapped[uuid.UUID] = uuid_pk()
    render_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("renders.id", ondelete="CASCADE"), index=True, nullable=False
    )
    external_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    commenter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    like_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    filtered_reason: Mapped[str | None] = mapped_column(String(64))
    pii_removed: Mapped[bool] = mapped_column(nullable=False, default=False)
    ingested_at: Mapped[datetime] = created_at_col()

    render: Mapped[Render] = relationship("Render")


class FakeMetaObject(Base):
    """State store for the fake Meta client. Never populated in production."""

    __tablename__ = "fake_meta_objects"
    __table_args__ = (UniqueConstraint("kind", "external_id"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    account_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()
