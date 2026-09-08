"""Column 3: signals, what the campaign threw off."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from outlier_ai.models.base import Base, created_at_col, enum_col, uuid_pk

if TYPE_CHECKING:
    from outlier_ai.models.trajectories import Render, Trajectory


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = uuid_pk()
    trajectory_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("trajectories.id", ondelete="CASCADE"), index=True, nullable=False
    )
    render_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("renders.id", ondelete="SET NULL"), index=True
    )
    kind: Mapped[str] = enum_col(index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    sentiment: Mapped[str] = enum_col(default="neu")
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    extracted_by: Mapped[str] = enum_col()
    status: Mapped[str] = enum_col(default="proposed", index=True)
    decided_by: Mapped[str | None] = mapped_column(String(120))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at_col()

    trajectory: Mapped[Trajectory] = relationship("Trajectory")
    render: Mapped[Render | None] = relationship("Render")
