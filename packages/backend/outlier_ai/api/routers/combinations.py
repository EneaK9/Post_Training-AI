from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from outlier_ai.api.deps import DB, CurrentUser, require
from outlier_ai.api.schemas import CardOut, CombinationOut, NameAsCardIn
from outlier_ai.api.views import card_out, cards_map, combination_out
from outlier_ai.core.audit import record_audit
from outlier_ai.models.auth import User
from outlier_ai.models.cards import Card, CardVersion, Combination

router = APIRouter(prefix="/combinations", tags=["combinations"])


@router.get("", response_model=list[CombinationOut])
async def list_combinations(
    db: DB, _: CurrentUser, sort: str = "tier2_rate", min_uses: int = 0, limit: int = 200
) -> list[CombinationOut]:
    cards = await cards_map(db)
    rows = (
        (await db.execute(select(Combination).where(Combination.uses >= min_uses))).scalars().all()
    )
    if sort == "uses":
        rows = sorted(rows, key=lambda c: -c.uses)
    elif sort == "recent":
        rows = sorted(
            rows, key=lambda c: c.last_used.timestamp() if c.last_used else 0, reverse=True
        )
    else:
        rows = sorted(rows, key=lambda c: ((c.tier2_rate or 0.0), c.uses), reverse=True)
    return [combination_out(c, cards) for c in rows[:limit]]


@router.post(
    "/{combination_id}/name_as_card", response_model=CardOut, status_code=status.HTTP_201_CREATED
)
async def name_as_card(
    combination_id: UUID,
    body: NameAsCardIn,
    db: DB,
    user: User = Depends(require("cards:write")),
) -> CardOut:
    combo = await db.get(Combination, combination_id)
    if combo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "combination not found")
    if (await db.execute(select(Card).where(Card.slug == body.slug))).scalar_one_or_none():
        raise HTTPException(status.HTTP_409_CONFLICT, f"slug {body.slug} already exists")
    cards = await cards_map(db)
    parts = [cards[i].slug for i in combo.card_ids if i in cards]
    source = (
        f"named from combination {' + '.join(parts)} (uses={combo.uses}, tiers={combo.tier_counts})"
    )
    card = Card(
        slug=body.slug,
        name=body.name,
        kind=body.kind.value,
        definition=body.definition,
        qualifying_condition=body.qualifying_condition,
        source=source,
        contributed_by=user.email,
        status="draft",
        version=1,
    )
    db.add(card)
    await db.flush()
    db.add(
        CardVersion(
            card_id=card.id,
            version=1,
            name=card.name,
            kind=card.kind,
            definition=card.definition,
            qualifying_condition=card.qualifying_condition,
            source=card.source,
            status="draft",
            edited_by=user.email,
        )
    )
    combo.named_as_card_id = card.id
    await record_audit(
        db,
        actor_id=user.email,
        action="combination.name_as_card",
        object_type="combination",
        object_id=combo.id,
        after={"card_id": str(card.id), "slug": card.slug},
    )
    await db.flush()
    return card_out(card)
