"""Niche keys and combination bookkeeping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.models.cards import Card, Combination
from outlier_ai.models.trajectories import Trajectory
from outlier_schemas.enums import CardKind, NicheProjection


class CardLike(Protocol):
    """Anything with a slug and a kind: ORM Card, pydantic Card, or a CardSpec.

    Attributes are typed Any so SQLAlchemy `Mapped[str]` descriptors satisfy the protocol.
    """

    slug: Any
    kind: Any


def niche_key(
    card_ids: Iterable[UUID],
    cards_by_id: Mapping[UUID, CardLike],
    projection: NicheProjection = NicheProjection.strategy_only,
) -> str:
    """Project a combination onto its niche.

    `full_combination` reproduces the literal spec (one niche per distinct card set).
    `strategy_only` collapses to the strategy cards, which keeps niches populated when the
    library is large. `strategy_plus_style` sits in between. If the projection would be
    empty (an idea with no strategy card) the full combination is used so nothing is lost.
    """
    cards = [cards_by_id[cid] for cid in card_ids if cid in cards_by_id]
    if projection == NicheProjection.full_combination:
        chosen = cards
    elif projection == NicheProjection.strategy_only:
        chosen = [c for c in cards if c.kind == CardKind.strategy]
    else:
        chosen = [c for c in cards if c.kind in (CardKind.strategy, CardKind.style)]
    if not chosen:
        chosen = cards
    return "+".join(sorted(str(c.slug) for c in chosen))


def combination_key(card_ids: Iterable[UUID]) -> tuple[UUID, ...]:
    return tuple(sorted(set(card_ids)))


async def load_cards_by_id(session: AsyncSession) -> dict[UUID, Card]:
    res = await session.execute(select(Card))
    return {c.id: c for c in res.scalars().all()}


async def upsert_combination(
    session: AsyncSession,
    card_ids: Iterable[UUID],
    niche: str,
    used_at: datetime,
) -> Combination:
    """Record one more use of a combination. Tier counts are maintained by refresh."""
    key = list(combination_key(card_ids))
    res = await session.execute(select(Combination).where(Combination.card_ids == key))
    combo = res.scalar_one_or_none()
    if combo is None:
        combo = Combination(
            card_ids=key, niche_key=niche, uses=0, first_used=used_at, last_used=used_at
        )
        session.add(combo)
    combo.uses += 1
    combo.niche_key = niche
    if combo.first_used is None or used_at < combo.first_used:
        combo.first_used = used_at
    if combo.last_used is None or used_at > combo.last_used:
        combo.last_used = used_at
    return combo


async def refresh_tier_counts(
    session: AsyncSession, projection: NicheProjection | None = None
) -> int:
    """Rebuild uses, tier_counts, tier2_rate, first/last used for every combination.

    Uses `verified_card_ids` when present, else the written cards, matching how reward and
    the uniqueness bonus attribute results to combinations. Returns the number of rows.
    """
    cards_by_id = await load_cards_by_id(session)
    res = await session.execute(
        select(
            Trajectory.card_ids,
            Trajectory.verified_card_ids,
            Trajectory.outlier_tier,
            Trajectory.created_at,
        )
    )
    stats: dict[tuple[UUID, ...], dict[str, Any]] = {}
    for written, verified, tier, created in res.all():
        ids = verified or written
        if not ids:
            continue
        key = combination_key(ids)
        s = stats.setdefault(
            key,
            {
                "uses": 0,
                "tiers": {"0": 0, "1": 0, "2": 0, "3": 0},
                "first": created,
                "last": created,
            },
        )
        s["uses"] += 1
        if tier is not None:
            s["tiers"][str(int(tier))] += 1
        s["first"] = min(s["first"], created)
        s["last"] = max(s["last"], created)

    existing = {
        tuple(c.card_ids): c for c in (await session.execute(select(Combination))).scalars().all()
    }
    for key, s in stats.items():
        measured = sum(s["tiers"].values())
        tier2 = s["tiers"]["2"] + s["tiers"]["3"]
        combo = existing.get(key)
        if combo is None:
            combo = Combination(card_ids=list(key))
            session.add(combo)
        combo.uses = s["uses"]
        combo.tier_counts = s["tiers"]
        combo.tier2_rate = (tier2 / measured) if measured else None
        combo.first_used = s["first"]
        combo.last_used = s["last"]
        if projection is not None or not combo.niche_key:
            combo.niche_key = niche_key(
                key, cards_by_id, projection or NicheProjection.strategy_only
            )
    await session.flush()
    return len(stats)
