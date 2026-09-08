"""Build renderer views from the database: brief, active playbook, archive sample, episode."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.combinations import load_cards_by_id, niche_key
from outlier_ai.archive.sampling import ArchiveSample, sample_archive
from outlier_ai.generation.renderer import PromptTrace, render_prompt
from outlier_ai.generation.views import (
    BatchView,
    BriefView,
    CardView,
    HistoryItemView,
    SignalView,
)
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import Batch
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Outcome, Render, ScreeningStats, Trajectory
from outlier_schemas.config import AppConfig
from outlier_schemas.models import short_ref


def brief_view(brief: Brief) -> BriefView:
    return BriefView(
        id=brief.id,
        company=brief.company,
        product=brief.product,
        offer=brief.offer,
        audience=brief.audience,
        category=brief.category,
        goal_metric=brief.goal_metric,
        channel=brief.channel,
        world_state=brief.world_state,
        world_state_at=brief.world_state_at,
        constraints=list(brief.constraints or []),
        raw_text=brief.raw_text,
    )


async def playbook_views(session: AsyncSession) -> tuple[list[CardView], dict[str, UUID]]:
    cards = await load_cards_by_id(session)
    active = [c for c in cards.values() if c.status == "active"]
    views = [CardView(c.slug, c.name, c.kind, c.definition, c.qualifying_condition) for c in active]
    return views, {c.slug: c.id for c in active}


def archive_views(sample: ArchiveSample, now: datetime) -> list[HistoryItemView]:
    out: list[HistoryItemView] = []
    for i in sample.items:
        days = (now - i.created_at).days if i.created_at else None
        out.append(
            HistoryItemView(
                ref=i.ref,
                reason=i.reason,
                same_brief=i.same_brief,
                brief_company=i.brief_company,
                card_slugs=i.card_slugs,
                verified_card_slugs=i.verified_card_slugs,
                niche=i.niche,
                typicality=i.typicality,
                angle=i.angle,
                copy=i.copy,
                tier=i.tier,
                ratio=i.ratio,
                screening_ratio=i.screening_ratio,
                days_ago=days,
                signals=[
                    SignalView(s.ref, s.kind, s.text, s.sentiment, s.count, s.status)
                    for s in i.signals
                ],
            )
        )
    return out


async def episode_views(
    session: AsyncSession, episode_id: UUID, cfg: AppConfig, ref_map: dict[str, UUID]
) -> list[BatchView]:
    """Every prior batch in the episode with screening, tiers, and proposed + confirmed signals."""
    cards_by_id = await load_cards_by_id(session)
    batches = (
        (
            await session.execute(
                select(Batch).where(Batch.episode_id == episode_id).order_by(Batch.index)
            )
        )
        .scalars()
        .all()
    )
    if not batches:
        return []
    trajs = (
        (
            await session.execute(
                select(Trajectory).where(Trajectory.batch_id.in_([b.id for b in batches]))
            )
        )
        .scalars()
        .all()
    )
    traj_ids = [t.id for t in trajs]
    ratio_by: dict[UUID, float] = {}
    screening_by: dict[UUID, float] = {}
    if traj_ids:
        rows = await session.execute(
            select(Render.trajectory_id, Outcome.ratio, ScreeningStats.screening_ratio)
            .join(Outcome, Outcome.render_id == Render.id, isouter=True)
            .join(ScreeningStats, ScreeningStats.render_id == Render.id, isouter=True)
            .where(Render.trajectory_id.in_(traj_ids))
        )
        for tid, ratio, sr in rows.all():
            if ratio is not None:
                ratio_by[tid] = max(ratio_by.get(tid, 0.0), ratio)
            if sr is not None:
                screening_by[tid] = max(screening_by.get(tid, 0.0), sr)
    signals_by: dict[UUID, list[SignalView]] = {}
    if traj_ids:
        for s in (
            (
                await session.execute(
                    select(Signal).where(
                        Signal.trajectory_id.in_(traj_ids), Signal.status != "rejected"
                    )
                )
            )
            .scalars()
            .all()
        ):
            ref = short_ref(s.id)
            ref_map[ref] = s.id
            signals_by.setdefault(s.trajectory_id, []).append(
                SignalView(ref, s.kind, s.text, s.sentiment, s.count, s.status)
            )

    def slugs(ids: list[UUID]) -> list[str]:
        return [str(cards_by_id[i].slug) for i in ids if i in cards_by_id]

    by_batch: dict[UUID, list[HistoryItemView]] = {}
    for t in trajs:
        ref = short_ref(t.id)
        ref_map[ref] = t.id
        item = HistoryItemView(
            ref=ref,
            reason="episode",
            same_brief=True,
            brief_company="",
            card_slugs=slugs(t.card_ids),
            verified_card_slugs=slugs(t.verified_card_ids),
            niche=niche_key(
                t.verified_card_ids or t.card_ids, cards_by_id, cfg.archive.niche_projection
            ),
            typicality=t.typicality,
            angle=t.angle,
            copy=dict(t.ad_copy or {}),
            tier=t.outlier_tier,
            ratio=ratio_by.get(t.id),
            screening_ratio=screening_by.get(t.id),
            days_ago=None,
            signals=sorted(signals_by.get(t.id, []), key=lambda s: -s.count),
        )
        by_batch.setdefault(t.batch_id, []).append(item) if t.batch_id else None
    return [BatchView(b.index, b.state, by_batch.get(b.id, [])) for b in batches]


async def build_prompt(
    session: AsyncSession,
    brief_id: UUID,
    cfg: AppConfig,
    *,
    episode_id: UUID | None = None,
    k: int | None = None,
    now: datetime | None = None,
) -> tuple[str, PromptTrace, ArchiveSample, dict[str, UUID]]:
    """Everything the generation service needs: prompt, trace, sample, and slug -> id map."""
    now = now or datetime.now(UTC)
    brief = await session.get(Brief, brief_id)
    if brief is None:
        raise ValueError(f"brief {brief_id} not found")
    playbook, slug_map = await playbook_views(session)
    sample = await sample_archive(session, brief_id, cfg, as_of=now)
    ref_map = dict(sample.ref_map)
    batches = await episode_views(session, episode_id, cfg, ref_map) if episode_id else []
    prompt, trace = render_prompt(
        brief=brief_view(brief),
        playbook=playbook,
        archive_items=archive_views(sample, now),
        episode_batches=batches,
        niche_uses=sample.niche_uses,
        ref_map=ref_map,
        cfg=cfg,
        k=k,
    )
    return prompt, trace, sample, slug_map
