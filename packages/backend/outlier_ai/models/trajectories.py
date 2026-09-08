"""Column 3: trajectories, renders, derived stats, reviews, notes."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from outlier_ai.models.base import Base, created_at_col, enum_col, uuid_pk
from outlier_ai.models.briefs import EMBEDDING_DIMS

if TYPE_CHECKING:
    from outlier_ai.models.briefs import Brief
    from outlier_ai.models.episodes import Batch, SearchEpisode


class Trajectory(Base):
    __tablename__ = "trajectories"

    id: Mapped[uuid.UUID] = uuid_pk()
    episode_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("search_episodes.id", ondelete="SET NULL"), index=True
    )
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("batches.id", ondelete="SET NULL"), index=True
    )
    brief_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("briefs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    campaign_id: Mapped[str | None] = mapped_column(String(64), index=True)
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    author_id: Mapped[str] = mapped_column(String(120), nullable=False)
    author_kind: Mapped[str] = enum_col()
    backend: Mapped[str | None] = mapped_column(String(32))

    card_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    verified_card_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    tag_source: Mapped[str | None] = mapped_column(String(32))
    tag_match: Mapped[bool | None] = mapped_column(Boolean)
    tag_jaccard: Mapped[float | None] = mapped_column(Float)
    typicality: Mapped[str | None] = mapped_column(String(16), index=True)
    format_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    format_errors: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)

    reasoning: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cited_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    angle: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ad_copy: Mapped[dict[str, Any]] = mapped_column("copy", JSONB, nullable=False, default=dict)
    visual_brief: Mapped[str] = mapped_column(Text, nullable=False, default="")
    preship: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Denormalized idea-level outcome (max across renders), maintained by outlier.recompute.
    outlier_tier: Mapped[int | None] = mapped_column(Integer, index=True)
    outcome_render_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    outcome_measured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    rm_score: Mapped[float | None] = mapped_column(Float)
    rm_version: Mapped[str | None] = mapped_column(String(64))
    library_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    config_hash: Mapped[str | None] = mapped_column(String(16))
    combo_angle_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMS))
    created_at: Mapped[datetime] = created_at_col()

    brief: Mapped[Brief] = relationship("Brief")
    episode: Mapped[SearchEpisode | None] = relationship("SearchEpisode")
    batch: Mapped[Batch | None] = relationship("Batch")
    renders: Mapped[list[Render]] = relationship(
        back_populates="trajectory", cascade="all, delete-orphan", order_by="Render.seed"
    )
    review: Mapped[Review | None] = relationship(
        back_populates="trajectory", cascade="all, delete-orphan", uselist=False
    )
    notes: Mapped[list[Note]] = relationship(
        back_populates="trajectory", cascade="all, delete-orphan", order_by="Note.created_at"
    )


class Render(Base):
    __tablename__ = "renders"

    id: Mapped[uuid.UUID] = uuid_pk()
    trajectory_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("trajectories.id", ondelete="CASCADE"), index=True, nullable=False
    )
    image_uri: Mapped[str | None] = mapped_column(Text)
    image_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    meta_ad_id: Mapped[str | None] = mapped_column(String(64), index=True)
    meta_adset_id: Mapped[str | None] = mapped_column(String(64))
    meta_creative_id: Mapped[str | None] = mapped_column(String(64))
    effective_object_story_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = enum_col(default="draft", index=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    # Controller bookkeeping: which stage the ad is in and what it is allowed to spend per day.
    phase: Mapped[str] = enum_col(default="screening")  # screening | scale
    daily_budget_usd: Mapped[float | None] = mapped_column(Float)
    scale_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    shipped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at_col()

    trajectory: Mapped[Trajectory] = relationship(back_populates="renders")
    screening: Mapped[ScreeningStats | None] = relationship(
        back_populates="render", cascade="all, delete-orphan", uselist=False
    )
    outcome: Mapped[Outcome | None] = relationship(
        back_populates="render", cascade="all, delete-orphan", uselist=False
    )


class ScreeningStats(Base):
    """DERIVED from daily_insights over the screening window."""

    __tablename__ = "screening_stats"

    render_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("renders.id", ondelete="CASCADE"), primary_key=True
    )
    impressions: Mapped[int] = mapped_column(Integer, nullable=False)
    link_clicks: Mapped[int] = mapped_column(Integer, nullable=False)
    spend: Mapped[float] = mapped_column(Float, nullable=False)
    ctr: Mapped[float] = mapped_column(Float, nullable=False)
    cpc: Mapped[float | None] = mapped_column(Float)
    ctr_lower_bound: Mapped[float] = mapped_column(Float, nullable=False)
    account_median_ctr_90d: Mapped[float] = mapped_column(Float, nullable=False)
    screening_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    window_complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    engagement: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    config_hash: Mapped[str | None] = mapped_column(String(16))
    computed_at: Mapped[datetime] = created_at_col()

    render: Mapped[Render] = relationship(back_populates="screening")


class Outcome(Base):
    """DERIVED from daily_insights at scale. The only training reward."""

    __tablename__ = "outcomes"

    render_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("renders.id", ondelete="CASCADE"), primary_key=True
    )
    metric: Mapped[str] = enum_col()
    value: Mapped[float] = mapped_column(Float, nullable=False)
    impressions: Mapped[int] = mapped_column(Integer, nullable=False)
    conversions: Mapped[int] = mapped_column(Integer, nullable=False)
    spend: Mapped[float] = mapped_column(Float, nullable=False)
    revenue: Mapped[float] = mapped_column(Float, nullable=False)
    days_at_scale: Mapped[int] = mapped_column(Integer, nullable=False)
    account_median_90d: Mapped[float] = mapped_column(Float, nullable=False)
    account_median_human_90d: Mapped[float | None] = mapped_column(Float)
    category_median: Mapped[float] = mapped_column(Float, nullable=False)
    baseline: Mapped[str] = enum_col()
    ratio: Mapped[float] = mapped_column(Float, nullable=False)
    ratio_lower_bound: Mapped[float] = mapped_column(Float, nullable=False)
    outlier_tier: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    gates: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    attribution_setting: Mapped[str] = mapped_column(String(32), nullable=False)
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    config_hash: Mapped[str | None] = mapped_column(String(16))
    computed_at: Mapped[datetime] = created_at_col()

    render: Mapped[Render] = relationship(back_populates="outcome")


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[uuid.UUID] = uuid_pk()
    trajectory_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("trajectories.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    label: Mapped[str] = enum_col(index=True)
    corrected_card_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    reviewer_id: Mapped[str] = mapped_column(String(120), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reviewed_at: Mapped[datetime] = created_at_col()

    trajectory: Mapped[Trajectory] = relationship(back_populates="review")


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[uuid.UUID] = uuid_pk()
    trajectory_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("trajectories.id", ondelete="CASCADE"), index=True, nullable=False
    )
    author_id: Mapped[str] = mapped_column(String(120), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = created_at_col()

    trajectory: Mapped[Trajectory] = relationship(back_populates="notes")
