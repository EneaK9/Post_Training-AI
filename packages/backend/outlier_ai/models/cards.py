"""Column 1: cards, versions, relations, and derived combinations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from outlier_ai.models.base import Base, created_at_col, enum_col, updated_at_col, uuid_pk


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[uuid.UUID] = uuid_pk()
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    kind: Mapped[str] = enum_col(index=True)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    qualifying_condition: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source: Mapped[str | None] = mapped_column(Text)
    contributed_by: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = enum_col(default="draft", index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    example_trajectory_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    created_at: Mapped[datetime] = created_at_col()
    updated_at: Mapped[datetime] = updated_at_col()

    versions: Mapped[list[CardVersion]] = relationship(
        back_populates="card", cascade="all, delete-orphan", order_by="CardVersion.version"
    )


class CardVersion(Base):
    """Immutable snapshot per edit. Runs pin `library_version`; diffs read from here."""

    __tablename__ = "card_versions"
    __table_args__ = (UniqueConstraint("card_id", "version"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    card_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cards.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    kind: Mapped[str] = enum_col()
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    qualifying_condition: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = enum_col()
    edited_by: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_col()

    card: Mapped[Card] = relationship(back_populates="versions")


class CardRelation(Base):
    __tablename__ = "card_relations"
    __table_args__ = (UniqueConstraint("from_id", "to_id", "kind"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    from_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cards.id", ondelete="CASCADE"), index=True
    )
    to_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cards.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = enum_col()
    source: Mapped[str] = enum_col()
    status: Mapped[str] = enum_col(default="pending")
    created_by: Mapped[str] = mapped_column(String(120), nullable=False)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_at_col()


class Combination(Base):
    """Derived. One row per distinct sorted set of card ids ever used."""

    __tablename__ = "combinations"

    id: Mapped[uuid.UUID] = uuid_pk()
    card_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, unique=True
    )
    niche_key: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    uses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tier_counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=lambda: {"0": 0, "1": 0, "2": 0, "3": 0}
    )
    tier2_rate: Mapped[float | None] = mapped_column(Float)
    first_used: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    named_as_card_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("cards.id", ondelete="SET NULL")
    )
