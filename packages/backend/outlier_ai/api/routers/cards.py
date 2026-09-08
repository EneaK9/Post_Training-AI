from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from outlier_ai.api.deps import DB, CurrentUser, require
from outlier_ai.api.schemas import (
    CardCreate,
    CardOut,
    CardUpdate,
    CardVersionOut,
    RelationCreate,
    RelationDecision,
    RelationOut,
)
from outlier_ai.api.views import card_out, card_stats
from outlier_ai.core.audit import record_audit
from outlier_ai.models.auth import User
from outlier_ai.models.cards import Card, CardRelation, CardVersion

router = APIRouter(prefix="/cards", tags=["cards"])


def _snapshot(card: Card) -> dict:
    return {
        "name": card.name,
        "kind": card.kind,
        "definition": card.definition,
        "qualifying_condition": card.qualifying_condition,
        "source": card.source,
        "status": card.status,
        "version": card.version,
    }


async def _new_version(db: DB, card: Card, editor: str) -> None:
    card.version += 1
    db.add(
        CardVersion(
            card_id=card.id,
            version=card.version,
            name=card.name,
            kind=card.kind,
            definition=card.definition,
            qualifying_condition=card.qualifying_condition,
            source=card.source,
            status=card.status,
            edited_by=editor,
        )
    )


async def _get(db: DB, card_id: UUID) -> Card:
    card = await db.get(Card, card_id)
    if card is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "card not found")
    return card


# relations first so the literal path wins over /{card_id}
@router.get("/relations", response_model=list[RelationOut])
async def list_relations(
    db: DB, _: CurrentUser, status_: str | None = Query(None, alias="status")
) -> list[RelationOut]:
    stmt = select(CardRelation)
    if status_:
        stmt = stmt.where(CardRelation.status == status_)
    return [RelationOut.model_validate(r) for r in (await db.execute(stmt)).scalars().all()]


@router.post("/relations", response_model=RelationOut, status_code=status.HTTP_201_CREATED)
async def create_relation(
    body: RelationCreate, db: DB, user: User = Depends(require("cards:write"))
) -> RelationOut:
    if body.from_id == body.to_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "a card cannot relate to itself")
    await _get(db, body.from_id)
    await _get(db, body.to_id)
    rel = CardRelation(
        from_id=body.from_id,
        to_id=body.to_id,
        kind=body.kind.value,
        source="expert",
        status="accepted",
        created_by=user.email,
    )
    db.add(rel)
    await db.flush()
    await record_audit(
        db,
        actor_id=user.email,
        action="relation.create",
        object_type="card_relation",
        object_id=rel.id,
        after={"from": str(body.from_id), "to": str(body.to_id), "kind": body.kind.value},
    )
    return RelationOut.model_validate(rel)


@router.put("/relations/{relation_id}", response_model=RelationOut)
async def decide_relation(
    relation_id: UUID,
    body: RelationDecision,
    db: DB,
    user: User = Depends(require("cards:write")),
) -> RelationOut:
    rel = await db.get(CardRelation, relation_id)
    if rel is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "relation not found")
    before = rel.status
    rel.status = body.status
    await record_audit(
        db,
        actor_id=user.email,
        action="relation.decide",
        object_type="card_relation",
        object_id=rel.id,
        before={"status": before},
        after={"status": rel.status},
    )
    return RelationOut.model_validate(rel)


@router.get("", response_model=list[CardOut])
async def list_cards(
    db: DB,
    _: CurrentUser,
    kind: str | None = None,
    status_: str | None = Query(None, alias="status"),
    contributor: str | None = None,
) -> list[CardOut]:
    stmt = select(Card).order_by(Card.kind, Card.slug)
    if kind:
        stmt = stmt.where(Card.kind == kind)
    if status_:
        stmt = stmt.where(Card.status == status_)
    if contributor:
        stmt = stmt.where(Card.contributed_by == contributor)
    cards = (await db.execute(stmt)).scalars().all()
    stats = await card_stats(db)
    return [card_out(c, stats) for c in cards]


@router.get("/{card_id}", response_model=CardOut)
async def get_card(card_id: UUID, db: DB, _: CurrentUser) -> CardOut:
    card = await _get(db, card_id)
    return card_out(card, await card_stats(db))


@router.post("", response_model=CardOut, status_code=status.HTTP_201_CREATED)
async def create_card(
    body: CardCreate, db: DB, user: User = Depends(require("cards:write"))
) -> CardOut:
    if (await db.execute(select(Card).where(Card.slug == body.slug))).scalar_one_or_none():
        raise HTTPException(status.HTTP_409_CONFLICT, f"slug {body.slug} already exists")
    card = Card(
        slug=body.slug,
        name=body.name,
        kind=body.kind.value,
        definition=body.definition,
        qualifying_condition=body.qualifying_condition,
        source=body.source,
        contributed_by=user.email,
        status=body.status,
        version=0,
    )
    db.add(card)
    await db.flush()
    await _new_version(db, card, user.email)
    await record_audit(
        db,
        actor_id=user.email,
        action="card.create",
        object_type="card",
        object_id=card.id,
        after=_snapshot(card),
    )
    await db.flush()
    return card_out(card)


@router.put("/{card_id}", response_model=CardOut)
async def update_card(
    card_id: UUID, body: CardUpdate, db: DB, user: User = Depends(require("cards:write"))
) -> CardOut:
    card = await _get(db, card_id)
    before = _snapshot(card)
    changed = False
    for field_name, value in body.model_dump(exclude_none=True).items():
        current = getattr(card, field_name)
        new = value.value if hasattr(value, "value") else value
        if current != new:
            setattr(card, field_name, new)
            changed = True
    if not changed:
        return card_out(card, await card_stats(db))
    await _new_version(db, card, user.email)
    await record_audit(
        db,
        actor_id=user.email,
        action="card.update",
        object_type="card",
        object_id=card.id,
        before=before,
        after=_snapshot(card),
    )
    await db.flush()
    return card_out(card, await card_stats(db))


async def _set_status(db: DB, card_id: UUID, user, new_status: str) -> CardOut:
    card = await _get(db, card_id)
    if card.status != new_status:
        before = _snapshot(card)
        card.status = new_status
        await _new_version(db, card, user.email)
        await record_audit(
            db,
            actor_id=user.email,
            action=f"card.{new_status}",
            object_type="card",
            object_id=card.id,
            before=before,
            after=_snapshot(card),
        )
        await db.flush()
    return card_out(card, await card_stats(db))


@router.post("/{card_id}/retire", response_model=CardOut)
async def retire_card(
    card_id: UUID, db: DB, user: User = Depends(require("cards:write"))
) -> CardOut:
    return await _set_status(db, card_id, user, "retired")


@router.post("/{card_id}/activate", response_model=CardOut)
async def activate_card(
    card_id: UUID, db: DB, user: User = Depends(require("cards:write"))
) -> CardOut:
    return await _set_status(db, card_id, user, "active")


@router.get("/{card_id}/versions", response_model=list[CardVersionOut])
async def card_versions(card_id: UUID, db: DB, _: CurrentUser) -> list[CardVersionOut]:
    await _get(db, card_id)
    rows = (
        (
            await db.execute(
                select(CardVersion)
                .where(CardVersion.card_id == card_id)
                .order_by(CardVersion.version)
            )
        )
        .scalars()
        .all()
    )
    return [CardVersionOut.model_validate(v) for v in rows]
