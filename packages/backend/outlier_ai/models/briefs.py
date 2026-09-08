"""Column 2: briefs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from outlier_ai.models.base import Base, created_at_col, enum_col, uuid_pk

if TYPE_CHECKING:
    from outlier_ai.models.meta import AdAccount

EMBEDDING_DIMS = 384


class Brief(Base):
    __tablename__ = "briefs"

    id: Mapped[uuid.UUID] = uuid_pk()
    created_by: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_col()
    ad_account_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("ad_accounts.id", ondelete="SET NULL"), index=True
    )
    company: Mapped[str] = mapped_column(String(200), nullable=False)
    product: Mapped[str] = mapped_column(Text, nullable=False)
    offer: Mapped[str] = mapped_column(Text, nullable=False)
    audience: Mapped[str] = mapped_column(Text, nullable=False)
    goal_metric: Mapped[str] = enum_col(default="purchases")
    channel: Mapped[str] = enum_col(default="meta_feed_image")
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    world_state: Mapped[str] = mapped_column(Text, nullable=False, default="")
    world_state_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    constraints: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    brand_assets: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMS))

    # Relationships exist so the unit of work orders inserts by dependency.
    ad_account: Mapped[AdAccount | None] = relationship("AdAccount")
