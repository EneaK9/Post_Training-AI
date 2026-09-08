"""Niche keys and combination bookkeeping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.models.cards import Combination
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
    return "+".join(sorted(c.slug for c in chosen))


def combination_key(card_ids: Iterable[UUID]) -> tuple[UUID, ...]:
    return tuple(sorted(set(card_ids)))


async def upsert_combination(
    session: AsyncSession,
    card_ids: Iterable[UUID],
    niche: str,
    used_at: datetime,
) -> Combination:
    """Record one more use of a combination. Tier counts are maintained by recompute."""
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
