"""Archive sampling: what history goes into the prompt (spec section 5.3).

From history for this brief and similar briefs: the best trajectory per niche for the top
`n_elite` niches, plus `n_rare` trajectories from rarely used niches, plus every tier 2+
trajectory in the account within `history_days`. Each carries its cards, angle, copy, tier,
and confirmed signals. Held-out campaigns are excluded. The sample records why each item is
there so the "what the model saw" panel can show it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.combinations import load_cards_by_id, niche_key
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import Card
from outlier_ai.models.ops import HoldoutCampaign
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Outcome, Render, ScreeningStats, Trajectory
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import SignalStatus
from outlier_schemas.models import short_ref


@dataclass(frozen=True)
class SignalItem:
    id: UUID
    ref: str
    kind: str
    text: str
    sentiment: str
    count: int
    status: str


@dataclass
class ArchiveItem:
    trajectory_id: UUID
    ref: str
    brief_id: UUID
    brief_company: str
    same_brief: bool
    card_slugs: list[str]
    verified_card_slugs: list[str]
    niche: str
    typicality: str | None
    angle: str
    copy: dict[str, Any]
    tier: int | None
    ratio: float | None
    screening_ratio: float | None
    created_at: datetime
    reason: str  # elite | rare | tier2_recent
    signals: list[SignalItem] = field(default_factory=list)


@dataclass
class ArchiveSample:
    brief_id: UUID
    items: list[ArchiveItem]
    similar_brief_ids: list[UUID]
    excluded_holdout: int
    niche_uses: dict[str, int]
    ref_map: dict[str, UUID]

    @property
    def elites(self) -> list[ArchiveItem]:
        return [i for i in self.items if i.reason == "elite"]

    @property
    def rares(self) -> list[ArchiveItem]:
        return [i for i in self.items if i.reason == "rare"]

    @property
    def tier2_recent(self) -> list[ArchiveItem]:
        return [i for i in self.items if i.reason == "tier2_recent"]


async def find_similar_briefs(
    session: AsyncSession, brief: Brief, cfg: AppConfig, limit: int = 10
) -> list[UUID]:
    """Other briefs on the same account above the cosine threshold. Falls back to same
    category when the brief has no embedding."""
    if brief.embedding is None:
        stmt = select(Brief.id).where(
            Brief.ad_account_id == brief.ad_account_id,
            Brief.category == brief.category,
            Brief.id != brief.id,
        )
        return list((await session.execute(stmt.limit(limit))).scalars().all())
    distance = Brief.embedding.cosine_distance(brief.embedding)
    stmt = (
        select(Brief.id, distance.label("d"))
        .where(
            Brief.ad_account_id == brief.ad_account_id,
            Brief.id != brief.id,
            Brief.embedding.is_not(None),
        )
        .order_by(distance)
        .limit(limit)
    )
    max_d = 1.0 - cfg.archive.similar_brief_min_cosine
    return [bid for bid, d in (await session.execute(stmt)).all() if d is not None and d <= max_d]


def _rank_key(t: Trajectory, ratio: float | None) -> tuple[int, float, float]:
    tier = t.outlier_tier if t.outlier_tier is not None else -1
    created = t.created_at.timestamp() if t.created_at else 0.0
    return (tier, ratio or 0.0, created)


async def sample_archive(
    session: AsyncSession,
    brief_id: UUID,
    cfg: AppConfig,
    *,
    as_of: datetime | None = None,
) -> ArchiveSample:
    brief = await session.get(Brief, brief_id)
    if brief is None:
        raise ValueError(f"brief {brief_id} not found")
    now = as_of or datetime.now(UTC)
    since = now - timedelta(days=cfg.archive.history_days)
    cards_by_id = await load_cards_by_id(session)
    holdout = frozenset((await session.execute(select(HoldoutCampaign.campaign_id))).scalars())
    similar = await find_similar_briefs(session, brief, cfg)

    # Shipped trajectories for this brief and similar briefs, within the history window.
    shipped_render = (
        select(Render.trajectory_id).where(Render.shipped_at.is_not(None)).distinct().subquery()
    )
    base = (
        select(Trajectory)
        .join(shipped_render, shipped_render.c.trajectory_id == Trajectory.id)
        .where(Trajectory.created_at >= since, Trajectory.created_at <= now)
    )
    local_stmt = base.where(Trajectory.brief_id.in_([brief.id, *similar]))
    local = list((await session.execute(local_stmt)).scalars().all())

    account_brief_ids = select(Brief.id).where(Brief.ad_account_id == brief.ad_account_id)
    tier2_stmt = base.where(
        Trajectory.brief_id.in_(account_brief_ids), Trajectory.outlier_tier >= 2
    )
    tier2 = list((await session.execute(tier2_stmt)).scalars().all())

    excluded = 0

    def keep(t: Trajectory) -> bool:
        nonlocal excluded
        if t.campaign_id and t.campaign_id in holdout:
            excluded += 1
            return False
        return True

    local = [t for t in local if keep(t)]
    tier2 = [t for t in tier2 if keep(t)]

    # ratios and screening ratios for ranking and display
    all_ids = {t.id for t in local} | {t.id for t in tier2}
    ratio_by_traj: dict[UUID, float] = {}
    screening_by_traj: dict[UUID, float] = {}
    if all_ids:
        rows = await session.execute(
            select(Render.trajectory_id, Outcome.ratio, ScreeningStats.screening_ratio)
            .join(Outcome, Outcome.render_id == Render.id, isouter=True)
            .join(ScreeningStats, ScreeningStats.render_id == Render.id, isouter=True)
            .where(Render.trajectory_id.in_(all_ids))
        )
        for tid, ratio, sr in rows.all():
            if ratio is not None:
                ratio_by_traj[tid] = max(ratio_by_traj.get(tid, 0.0), ratio)
            if sr is not None:
                screening_by_traj[tid] = max(screening_by_traj.get(tid, 0.0), sr)

    def niche_of(t: Trajectory) -> str:
        ids = t.verified_card_ids or t.card_ids
        return niche_key(ids, cards_by_id, cfg.archive.niche_projection)

    # niche usage across the account window (rarity is account-wide)
    niche_uses = Counter(niche_of(t) for t in local)

    by_niche: dict[str, list[Trajectory]] = {}
    for t in local:
        by_niche.setdefault(niche_of(t), []).append(t)
    best_per_niche = {
        n: max(ts, key=lambda t: _rank_key(t, ratio_by_traj.get(t.id)))
        for n, ts in by_niche.items()
    }
    elite_niches = sorted(
        best_per_niche,
        key=lambda n: _rank_key(best_per_niche[n], ratio_by_traj.get(best_per_niche[n].id)),
        reverse=True,
    )[: cfg.archive.n_elite]
    rare_niches = [
        n
        for n, _ in sorted(niche_uses.items(), key=lambda kv: (kv[1], kv[0]))
        if n not in elite_niches
    ][: cfg.archive.n_rare]

    chosen: list[tuple[Trajectory, str]] = []
    seen: set[UUID] = set()
    for n in elite_niches:
        t = best_per_niche[n]
        chosen.append((t, "elite"))
        seen.add(t.id)
    for n in rare_niches:
        t = best_per_niche[n]
        if t.id not in seen:
            chosen.append((t, "rare"))
            seen.add(t.id)
    for t in sorted(tier2, key=lambda t: _rank_key(t, ratio_by_traj.get(t.id)), reverse=True):
        if t.id not in seen:
            chosen.append((t, "tier2_recent"))
            seen.add(t.id)

    # confirmed signals for chosen trajectories
    signals_by_traj: dict[UUID, list[SignalItem]] = {}
    if seen:
        sig_rows = await session.execute(
            select(Signal).where(
                Signal.trajectory_id.in_(seen), Signal.status == SignalStatus.confirmed.value
            )
        )
        for s in sig_rows.scalars().all():
            signals_by_traj.setdefault(s.trajectory_id, []).append(
                SignalItem(s.id, short_ref(s.id), s.kind, s.text, s.sentiment, s.count, s.status)
            )

    brief_company = {brief.id: brief.company}
    other_ids = {t.brief_id for t, _ in chosen} - set(brief_company)
    if other_ids:
        for bid, company in (
            await session.execute(select(Brief.id, Brief.company).where(Brief.id.in_(other_ids)))
        ).all():
            brief_company[bid] = company

    def slugs(ids: list[UUID]) -> list[str]:
        return [str(cards_by_id[i].slug) for i in ids if i in cards_by_id]

    items: list[ArchiveItem] = []
    ref_map: dict[str, UUID] = {}
    for t, reason in chosen:
        ref = short_ref(t.id)
        ref_map[ref] = t.id
        sigs = sorted(signals_by_traj.get(t.id, []), key=lambda s: -s.count)
        for s in sigs:
            ref_map[s.ref] = s.id
        items.append(
            ArchiveItem(
                trajectory_id=t.id,
                ref=ref,
                brief_id=t.brief_id,
                brief_company=brief_company.get(t.brief_id, ""),
                same_brief=t.brief_id == brief.id,
                card_slugs=slugs(t.card_ids),
                verified_card_slugs=slugs(t.verified_card_ids),
                niche=niche_of(t),
                typicality=t.typicality,
                angle=t.angle,
                copy=dict(t.ad_copy or {}),
                tier=t.outlier_tier,
                ratio=ratio_by_traj.get(t.id),
                screening_ratio=screening_by_traj.get(t.id),
                created_at=t.created_at,
                reason=reason,
                signals=sigs,
            )
        )

    return ArchiveSample(
        brief_id=brief.id,
        items=items,
        similar_brief_ids=similar,
        excluded_holdout=excluded,
        niche_uses=dict(niche_uses),
        ref_map=ref_map,
    )


def active_cards(cards_by_id: dict[UUID, Card]) -> list[Card]:
    return [c for c in cards_by_id.values() if c.status == "active"]
