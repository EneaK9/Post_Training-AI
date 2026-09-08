"""Held-out briefs and the campaign blocklist.

Leakage is by campaign id only (scenario table last row): every campaign a held-out brief has
run or will run is added to `holdout_campaigns`, which the archive sampler, reward-model dataset,
verifier few-shot, and baselines all consult."""

from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.models.briefs import Brief
from outlier_ai.models.ops import HoldoutCampaign
from outlier_ai.models.trajectories import Trajectory


async def select_holdout_briefs(
    session: AsyncSession, *, n: int, seed: int = 0, account_id: UUID | None = None
) -> list[Brief]:
    stmt = select(Brief).order_by(Brief.created_at)
    if account_id:
        stmt = stmt.where(Brief.ad_account_id == account_id)
    briefs = list((await session.execute(stmt)).scalars().all())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(briefs))[: min(n, len(briefs))]
    return [briefs[int(i)] for i in idx]


async def blocklist_campaigns(
    session: AsyncSession,
    brief_ids: list[UUID],
    *,
    campaign_ids: list[str] | None = None,
    added_by: str = "eval",
    reason: str = "online eval holdout",
) -> int:
    ids = set(campaign_ids or [])
    rows = await session.execute(
        select(Trajectory.campaign_id)
        .where(Trajectory.brief_id.in_(brief_ids), Trajectory.campaign_id.is_not(None))
        .distinct()
    )
    ids |= {c for (c,) in rows.all() if c}
    existing = set((await session.execute(select(HoldoutCampaign.campaign_id))).scalars().all())
    n = 0
    for cid in sorted(ids - existing):
        session.add(HoldoutCampaign(campaign_id=cid, added_by=added_by, reason=reason))
        n += 1
    await session.flush()
    return n
