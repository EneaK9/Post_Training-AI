"""CSV import (spec section 9.3).

- cards: slug,name,kind,definition,qualifying_condition,source,status
- trajectories: brief_id,angle,primary_text,headline,description,cta,visual_brief,cards,author_id,
  created_at,campaign_id. `cards` is `slug|slug`. When present the tags are expert tags; when
  absent the verifier proposes and the expert confirms later (untagged rows stay out of training).
- outcomes: trajectory_id or render_id, day, phase, impressions, link_clicks, spend, purchases,
  revenue[, frequency, reactions, comments, shares, saves]. Creates one render per imported
  trajectory when no render_id is given. Derived tiers come from recompute.
- comments: render_id, external_id, commenter_id, text, created_time[, like_count]. PII stripped.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, date, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.api.schemas import ImportResult
from outlier_ai.archive.combinations import niche_key, upsert_combination
from outlier_ai.archive.novelty import embed_combo_angle
from outlier_ai.core.crypto import hash_identity
from outlier_ai.core.embeddings import Embedder
from outlier_ai.generation.verifier import CardVerifier
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import Card, CardVersion
from outlier_ai.models.meta import Comment, DailyInsight
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_ai.signals.pii import strip_pii
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import AuthorKind, CardKind, RenderStatus, TagSource
from outlier_schemas.models import AdCopy


def _rows(data: bytes | str) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    return [
        {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        for r in csv.DictReader(io.StringIO(text))
    ]


def _dt(value: str) -> datetime:
    if not value:
        return datetime.now(UTC)
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=UTC)


async def import_cards(session: AsyncSession, data: bytes | str, *, actor: str) -> ImportResult:
    res = ImportResult()
    for i, r in enumerate(_rows(data), start=2):
        try:
            slug, name, kind = r["slug"], r["name"], CardKind(r["kind"])
        except (KeyError, ValueError) as e:
            res.errors.append(f"row {i}: {e}")
            continue
        existing = (
            await session.execute(select(Card).where(Card.slug == slug))
        ).scalar_one_or_none()
        status_ = r.get("status") or "draft"
        if status_ not in ("draft", "active"):
            status_ = "draft"
        if existing is None:
            card = Card(
                slug=slug,
                name=name,
                kind=kind.value,
                definition=r.get("definition", ""),
                qualifying_condition=r.get("qualifying_condition", ""),
                source=r.get("source") or None,
                contributed_by=r.get("contributed_by") or actor,
                status=status_,
                version=1,
            )
            session.add(card)
            await session.flush()
            session.add(
                CardVersion(
                    card_id=card.id,
                    version=1,
                    name=name,
                    kind=kind.value,
                    definition=card.definition,
                    qualifying_condition=card.qualifying_condition,
                    source=card.source,
                    status=status_,
                    edited_by=actor,
                )
            )
            res.created += 1
        else:
            changed = False
            for field_name in ("name", "definition", "qualifying_condition", "source"):
                new = r.get(field_name) or (None if field_name == "source" else "")
                if field_name in r and getattr(existing, field_name) != new:
                    setattr(existing, field_name, new)
                    changed = True
            if changed:
                existing.version += 1
                session.add(
                    CardVersion(
                        card_id=existing.id,
                        version=existing.version,
                        name=existing.name,
                        kind=existing.kind,
                        definition=existing.definition,
                        qualifying_condition=existing.qualifying_condition,
                        source=existing.source,
                        status=existing.status,
                        edited_by=actor,
                    )
                )
                res.updated += 1
            else:
                res.skipped += 1
    await session.flush()
    return res


async def import_trajectories(
    session: AsyncSession,
    data: bytes | str,
    *,
    actor: str,
    cfg: AppConfig,
    verifier: CardVerifier,
    embedder: Embedder,
) -> ImportResult:
    res = ImportResult()
    cards = {c.slug: c for c in (await session.execute(select(Card))).scalars().all()}
    cards_by_id = {c.id: c for c in cards.values()}
    for i, r in enumerate(_rows(data), start=2):
        try:
            brief_id = UUID(r["brief_id"])
        except (KeyError, ValueError):
            res.errors.append(f"row {i}: brief_id missing or invalid")
            continue
        brief = await session.get(Brief, brief_id)
        if brief is None:
            res.errors.append(f"row {i}: brief {brief_id} not found")
            continue
        copy = AdCopy(
            primary_text=r.get("primary_text", ""),
            headline=r.get("headline", ""),
            description=r.get("description", ""),
            cta=r.get("cta", ""),
        )
        angle = r.get("angle", "")
        visual = r.get("visual_brief", "")
        given = [s for s in (r.get("cards") or "").split("|") if s]
        unknown = [s for s in given if s not in cards]
        if unknown:
            res.errors.append(f"row {i}: unknown cards {unknown}")
            continue
        verified = await verifier.tag(angle, copy, visual)
        written_ids = [cards[s].id for s in given]
        if given:
            tag_source = TagSource.expert
            verified_ids = written_ids
        else:
            tag_source = TagSource.verifier
            verified_ids = verified.card_ids
        created_at = _dt(r.get("created_at", ""))
        slugs = [cards_by_id[c].slug for c in (verified_ids or written_ids) if c in cards_by_id]
        traj = Trajectory(
            id=uuid.uuid4(),
            brief_id=brief.id,
            campaign_id=r.get("campaign_id") or None,
            attempt_index=0,
            author_id=r.get("author_id") or actor,
            author_kind=AuthorKind.human.value,
            backend=None,
            card_ids=written_ids,
            verified_card_ids=verified_ids,
            tag_source=tag_source.value,
            tag_match=(set(written_ids) == set(verified.card_ids)) if given else None,
            tag_jaccard=None,
            typicality=None,
            format_ok=True,
            format_errors=[],
            reasoning=r.get("reasoning", "imported practitioner trajectory"),
            cited_ids=[],
            angle=angle,
            ad_copy=copy.model_dump(),
            visual_brief=visual,
            library_version=0,
            config_hash=cfg.hash,
            combo_angle_embedding=embed_combo_angle(embedder, slugs, angle) if slugs else None,
            created_at=created_at,
        )
        session.add(traj)
        if verified_ids or written_ids:
            ids = verified_ids or written_ids
            await upsert_combination(
                session, ids, niche_key(ids, cards_by_id, cfg.archive.niche_projection), created_at
            )
        res.created += 1
    await session.flush()
    return res


async def import_outcomes(
    session: AsyncSession, data: bytes | str, *, cfg: AppConfig
) -> ImportResult:
    res = ImportResult()
    render_for_traj: dict[UUID, Render] = {}
    for i, r in enumerate(_rows(data), start=2):
        try:
            day = date.fromisoformat(r["day"])
        except (KeyError, ValueError):
            res.errors.append(f"row {i}: day missing or invalid")
            continue
        render: Render | None = None
        if r.get("render_id"):
            try:
                render = await session.get(Render, UUID(r["render_id"]))
            except ValueError:
                render = None
            if render is None:
                res.errors.append(f"row {i}: render not found")
                continue
        elif r.get("trajectory_id"):
            try:
                tid = UUID(r["trajectory_id"])
            except ValueError:
                res.errors.append(f"row {i}: trajectory_id invalid")
                continue
            render = render_for_traj.get(tid)
            if render is None:
                render = (
                    (
                        await session.execute(
                            select(Render)
                            .where(Render.trajectory_id == tid)
                            .order_by(Render.seed)
                            .limit(1)
                        )
                    )
                    .scalars()
                    .first()
                )
            if render is None:
                traj = await session.get(Trajectory, tid)
                if traj is None:
                    res.errors.append(f"row {i}: trajectory not found")
                    continue
                render = Render(
                    id=uuid.uuid4(),
                    trajectory_id=tid,
                    image_backend="import",
                    seed=0,
                    width=1080,
                    height=1080,
                    status=RenderStatus.stopped.value,
                    shipped_at=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
                )
                session.add(render)
                await session.flush()
            render_for_traj[tid] = render
        else:
            res.errors.append(f"row {i}: render_id or trajectory_id required")
            continue
        phase = r.get("phase") or "scale"
        if phase not in ("screening", "scale"):
            res.errors.append(f"row {i}: phase must be screening or scale")
            continue
        existing = (
            await session.execute(
                select(DailyInsight).where(
                    DailyInsight.render_id == render.id, DailyInsight.day == day
                )
            )
        ).scalar_one_or_none()
        values = {
            "phase": phase,
            "impressions": int(float(r.get("impressions") or 0)),
            "link_clicks": int(float(r.get("link_clicks") or 0)),
            "spend": float(r.get("spend") or 0),
            "purchases": int(float(r.get("purchases") or 0)),
            "revenue": float(r.get("revenue") or 0),
            "frequency": float(r["frequency"]) if r.get("frequency") else None,
            "reactions": int(float(r.get("reactions") or 0)),
            "comments": int(float(r.get("comments") or 0)),
            "shares": int(float(r.get("shares") or 0)),
            "saves": int(float(r.get("saves") or 0)),
            "attribution_setting": r.get("attribution_setting") or cfg.meta.attribution_setting,
            "source": "import",
        }
        if existing is None:
            session.add(DailyInsight(render_id=render.id, day=day, **values))
            res.created += 1
        else:
            for k, v in values.items():
                setattr(existing, k, v)
            res.updated += 1
        if render.shipped_at is None:
            render.shipped_at = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    await session.flush()
    return res


async def import_comments(session: AsyncSession, data: bytes | str) -> ImportResult:
    res = ImportResult()
    for i, r in enumerate(_rows(data), start=2):
        try:
            render_id = UUID(r["render_id"])
        except (KeyError, ValueError):
            res.errors.append(f"row {i}: render_id missing or invalid")
            continue
        if await session.get(Render, render_id) is None:
            res.errors.append(f"row {i}: render not found")
            continue
        ext = r.get("external_id") or None
        if (
            ext
            and (
                await session.execute(select(Comment).where(Comment.external_id == ext))
            ).scalar_one_or_none()
        ):
            res.skipped += 1
            continue
        pii = strip_pii(r.get("text", ""))
        session.add(
            Comment(
                render_id=render_id,
                external_id=ext,
                commenter_hash=hash_identity(r.get("commenter_id") or f"import-{i}"),
                text=pii.text,
                created_time=_dt(r.get("created_time", "")),
                like_count=int(float(r.get("like_count") or 0)),
                pii_removed=pii.removed,
            )
        )
        res.created += 1
    await session.flush()
    return res
