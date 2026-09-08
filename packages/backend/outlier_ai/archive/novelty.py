"""Novelty rejection (spec section 5.6).

Before an idea reaches the review queue: reject if its verified combination plus angle
embedding is within `novelty_threshold` of any trajectory already shipped for this brief,
or of any tier 0 trajectory in the account with the same combination in the last
`history_days`. Rejected ideas are stored with `review.label = skip` and a `novelty_reject`
note and never shipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.combinations import combination_key
from outlier_ai.core.embeddings import Embedder
from outlier_ai.models.briefs import Brief
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_schemas.config import AppConfig


@dataclass(frozen=True)
class NoveltyDecision:
    rejected: bool
    nearest_trajectory_id: UUID | None
    distance: float | None
    reason: str  # novel | duplicate_of_shipped | duplicate_of_tier0_same_combination

    @property
    def note(self) -> str:
        if not self.rejected:
            return ""
        return f"novelty_reject:{self.reason}:{self.nearest_trajectory_id}:{self.distance:.3f}"


def combo_angle_text(card_slugs: Sequence[str], angle: str) -> str:
    """The text that is embedded for novelty: combination first, then the angle."""
    return f"cards: {' + '.join(sorted(card_slugs))}\nangle: {angle.strip()}"


def embed_combo_angle(embedder: Embedder, card_slugs: Sequence[str], angle: str) -> list[float]:
    vec = embedder.embed([combo_angle_text(card_slugs, angle)])[0]
    return [float(x) for x in np.asarray(vec, dtype=np.float32)]


async def check_novelty(
    session: AsyncSession,
    *,
    brief_id: UUID,
    verified_card_ids: Sequence[UUID],
    embedding: Sequence[float],
    cfg: AppConfig,
    as_of: datetime | None = None,
    exclude_trajectory_id: UUID | None = None,
) -> NoveltyDecision:
    now = as_of or datetime.now(UTC)
    threshold = cfg.archive.novelty_threshold
    vec = list(embedding)
    distance = Trajectory.combo_angle_embedding.cosine_distance(vec)

    shipped_ids = (
        select(Render.trajectory_id).where(Render.shipped_at.is_not(None)).distinct().subquery()
    )

    # (a) anything already shipped for this brief
    stmt_a = (
        select(Trajectory.id, distance.label("d"))
        .join(shipped_ids, shipped_ids.c.trajectory_id == Trajectory.id)
        .where(Trajectory.brief_id == brief_id, Trajectory.combo_angle_embedding.is_not(None))
        .order_by(distance)
        .limit(1)
    )
    if exclude_trajectory_id is not None:
        stmt_a = stmt_a.where(Trajectory.id != exclude_trajectory_id)
    row = (await session.execute(stmt_a)).first()
    if row is not None and row.d is not None and row.d <= threshold:
        return NoveltyDecision(True, row.id, float(row.d), "duplicate_of_shipped")

    # (b) tier 0 in the account with the same combination within the window
    brief = await session.get(Brief, brief_id)
    if brief is not None and verified_card_ids:
        account_briefs = select(Brief.id).where(Brief.ad_account_id == brief.ad_account_id)
        key = list(combination_key(verified_card_ids))
        stmt_b = (
            select(Trajectory.id, distance.label("d"))
            .where(
                Trajectory.brief_id.in_(account_briefs),
                Trajectory.outlier_tier == 0,
                Trajectory.verified_card_ids == key,
                Trajectory.created_at >= now - timedelta(days=cfg.archive.history_days),
                Trajectory.combo_angle_embedding.is_not(None),
            )
            .order_by(distance)
            .limit(1)
        )
        if exclude_trajectory_id is not None:
            stmt_b = stmt_b.where(Trajectory.id != exclude_trajectory_id)
        row = (await session.execute(stmt_b)).first()
        if row is not None and row.d is not None and row.d <= threshold:
            return NoveltyDecision(
                True, row.id, float(row.d), "duplicate_of_tier0_same_combination"
            )

    nearest_id = row.id if row is not None else None
    nearest_d = float(row.d) if row is not None and row.d is not None else None
    return NoveltyDecision(False, nearest_id, nearest_d, "novel")
