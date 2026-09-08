"""ORM rows -> API shapes, with the bulk lookups the Data screen needs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.api.schemas import (
    CardOut,
    CardStats,
    CombinationOut,
    CommentOut,
    NoteOut,
    OutcomeOut,
    RenderOut,
    ReviewOut,
    ScreeningOut,
    SignalOut,
    TrajectoryDetail,
    TrajectoryOut,
)
from outlier_ai.models.cards import Card, Combination
from outlier_ai.models.meta import Comment
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Note, Outcome, Render, Review, ScreeningStats, Trajectory
from outlier_schemas.models import AdCopy, PreshipReport


async def cards_map(session: AsyncSession) -> dict[UUID, Card]:
    return {c.id: c for c in (await session.execute(select(Card))).scalars().all()}


def slugs(ids: Sequence[UUID], cards: dict[UUID, Card]) -> list[str]:
    return [cards[i].slug for i in ids if i in cards]


async def card_stats(session: AsyncSession) -> dict[UUID, CardStats]:
    """Per-card uses, measured, tier 2+ from combinations; verifier agreement from trajectories."""
    acc: dict[UUID, dict[str, float]] = defaultdict(
        lambda: {"uses": 0, "measured": 0, "tier2": 0, "agree": 0, "tagged": 0}
    )
    for combo in (await session.execute(select(Combination))).scalars().all():
        tiers = combo.tier_counts or {}
        measured = sum(int(v) for v in tiers.values())
        tier2 = int(tiers.get("2", 0)) + int(tiers.get("3", 0))
        for cid in combo.card_ids:
            a = acc[cid]
            a["uses"] += combo.uses
            a["measured"] += measured
            a["tier2"] += tier2
    rows = await session.execute(
        select(Trajectory.card_ids, Trajectory.tag_match).where(Trajectory.tag_match.is_not(None))
    )
    for ids, match in rows.all():
        for cid in ids:
            acc[cid]["tagged"] += 1
            acc[cid]["agree"] += 1 if match else 0
    out: dict[UUID, CardStats] = {}
    for cid, a in acc.items():
        out[cid] = CardStats(
            uses=int(a["uses"]),
            measured=int(a["measured"]),
            tier2_count=int(a["tier2"]),
            tier2_rate=(a["tier2"] / a["measured"]) if a["measured"] else None,
            verifier_agreement=(a["agree"] / a["tagged"]) if a["tagged"] else None,
        )
    return out


def card_out(card: Card, stats: dict[UUID, CardStats] | None = None) -> CardOut:
    out = CardOut.model_validate(card)
    if stats and card.id in stats:
        out.stats = stats[card.id]
    return out


def combination_out(combo: Combination, cards: dict[UUID, Card]) -> CombinationOut:
    out = CombinationOut.model_validate(combo)
    out.card_slugs = slugs(combo.card_ids, cards)
    return out


async def trajectory_outs(
    session: AsyncSession,
    trajs: Sequence[Trajectory],
    *,
    cards: dict[UUID, Card] | None = None,
    include_signals: bool = True,
) -> list[TrajectoryOut]:
    if not trajs:
        return []
    cards = cards or await cards_map(session)
    ids = [t.id for t in trajs]

    renders_by: dict[UUID, list[RenderOut]] = defaultdict(list)
    ratio_by: dict[UUID, float] = {}
    screening_by: dict[UUID, float] = {}
    rows = await session.execute(
        select(Render, ScreeningStats, Outcome)
        .join(ScreeningStats, ScreeningStats.render_id == Render.id, isouter=True)
        .join(Outcome, Outcome.render_id == Render.id, isouter=True)
        .where(Render.trajectory_id.in_(ids))
        .order_by(Render.seed)
    )
    for render, scr, out in rows.all():
        ro = RenderOut(
            id=render.id,
            seed=render.seed,
            width=render.width,
            height=render.height,
            image_uri=render.image_uri,
            image_backend=render.image_backend,
            status=render.status,
            rejection_reason=render.rejection_reason,
            meta_ad_id=render.meta_ad_id,
            shipped_at=render.shipped_at,
            screening=ScreeningOut.model_validate(scr) if scr else None,
            outcome=OutcomeOut.model_validate(out) if out else None,
        )
        renders_by[render.trajectory_id].append(ro)
        if scr is not None:
            screening_by[render.trajectory_id] = max(
                screening_by.get(render.trajectory_id, 0.0), scr.screening_ratio
            )
        if out is not None:
            ratio_by[render.trajectory_id] = max(ratio_by.get(render.trajectory_id, 0.0), out.ratio)

    reviews = {
        r.trajectory_id: ReviewOut.model_validate(r)
        for r in (await session.execute(select(Review).where(Review.trajectory_id.in_(ids))))
        .scalars()
        .all()
    }
    notes_by: dict[UUID, list[NoteOut]] = defaultdict(list)
    for n in (
        (
            await session.execute(
                select(Note).where(Note.trajectory_id.in_(ids)).order_by(Note.created_at)
            )
        )
        .scalars()
        .all()
    ):
        notes_by[n.trajectory_id].append(NoteOut.model_validate(n))
    signals_by: dict[UUID, list[SignalOut]] = defaultdict(list)
    if include_signals:
        for s in (
            (
                await session.execute(
                    select(Signal)
                    .where(Signal.trajectory_id.in_(ids))
                    .order_by(Signal.count.desc())
                )
            )
            .scalars()
            .all()
        ):
            signals_by[s.trajectory_id].append(SignalOut.model_validate(s))

    outs: list[TrajectoryOut] = []
    for t in trajs:
        out = TrajectoryOut(
            id=t.id,
            brief_id=t.brief_id,
            episode_id=t.episode_id,
            batch_id=t.batch_id,
            campaign_id=t.campaign_id,
            attempt_index=t.attempt_index,
            author_id=t.author_id,
            author_kind=t.author_kind,
            backend=t.backend,
            card_ids=list(t.card_ids),
            verified_card_ids=list(t.verified_card_ids),
            card_slugs=slugs(t.card_ids, cards),
            verified_card_slugs=slugs(t.verified_card_ids, cards),
            tag_source=t.tag_source,
            tag_match=t.tag_match,
            tag_jaccard=t.tag_jaccard,
            typicality=t.typicality,
            format_ok=t.format_ok,
            format_errors=list(t.format_errors or []),
            reasoning=t.reasoning,
            cited_ids=list(t.cited_ids or []),
            angle=t.angle,
            ad_copy=AdCopy(**(t.ad_copy or {})),
            visual_brief=t.visual_brief,
            preship=PreshipReport(**t.preship) if t.preship else None,
            outlier_tier=t.outlier_tier,
            screening_ratio=screening_by.get(t.id),
            ratio=ratio_by.get(t.id),
            rm_score=t.rm_score,
            rm_version=t.rm_version,
            library_version=t.library_version,
            config_hash=t.config_hash,
            created_at=t.created_at,
            review=reviews.get(t.id),
            renders=renders_by.get(t.id, []),
            signals=signals_by.get(t.id, []),
            notes=notes_by.get(t.id, []),
        )
        outs.append(out)
    return outs


async def trajectory_detail(session: AsyncSession, traj: Trajectory) -> TrajectoryDetail:
    [base] = await trajectory_outs(session, [traj])
    render_ids = [r.id for r in base.renders]
    comments: list[CommentOut] = []
    if render_ids:
        comments = [
            CommentOut.model_validate(c)
            for c in (
                await session.execute(
                    select(Comment)
                    .where(Comment.render_id.in_(render_ids))
                    .order_by(Comment.created_time)
                )
            )
            .scalars()
            .all()
        ]
    return TrajectoryDetail(**base.model_dump(), comments=comments)
