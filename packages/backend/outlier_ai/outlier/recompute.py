"""Derive ScreeningStats and Outcome rows from raw daily insights.

The only place derived outlier data is written. Runs after every insights sync and whenever
`outlier:` config changes (tiers are recomputed from raw rows, spec section 4).

Tier assignment: the tier is the one the bootstrap lower bound of the ROAS ratio clears,
which is how "the lower bound clears the tier" is read here. Gates then apply; a failed gate
records tier 0 with the flags kept.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from uuid import UUID

import numpy as np
from sqlalchemy import false, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.combinations import refresh_tier_counts
from outlier_ai.models.briefs import Brief
from outlier_ai.models.meta import AdAccount, DailyInsight
from outlier_ai.models.ops import HoldoutCampaign
from outlier_ai.models.trajectories import Outcome, Render, ScreeningStats, Trajectory
from outlier_ai.outlier.baselines import (
    Baselines,
    DayRow,
    RenderAgg,
    category_median_for,
    compute_baselines,
)
from outlier_ai.outlier.gates import GateResult, check_gates
from outlier_ai.outlier.screening import compute_screening
from outlier_ai.outlier.stats import bootstrap_ratio_lower_bound
from outlier_ai.outlier.tiers import tier_for_ratio
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BaselineKind, OutcomeMetric, RenderStatus, Tier


@dataclass(frozen=True)
class OutcomeResult:
    metric: OutcomeMetric
    value: float
    impressions: int
    conversions: int
    spend: float
    revenue: float
    days_at_scale: int
    account_median_90d: float
    account_median_human_90d: float | None
    category_median: float
    baseline: BaselineKind
    ratio: float
    ratio_lower_bound: float
    outlier_tier: Tier
    gates: GateResult
    attribution_setting: str
    measured_at: datetime


def compute_outcome(
    agg: RenderAgg,
    baselines: Baselines,
    category_median: float,
    cfg: AppConfig,
    *,
    measured_at: datetime,
    rng: np.random.Generator | None = None,
) -> OutcomeResult | None:
    """Pure outcome computation for one render. None when there is no scale data."""
    scale = agg.scale_rows
    if not scale or agg.scale_spend <= 0:
        return None
    value = agg.scale_value
    if value is None:
        return None
    baseline_value = baselines.roas.value
    if baseline_value <= 0:
        return None
    metric = OutcomeMetric.leads_per_dollar if agg.goal_metric == "leads" else OutcomeMetric.roas
    revenue = np.array([r.revenue for r in scale], dtype=float)
    spend = np.array([r.spend for r in scale], dtype=float)
    if metric == OutcomeMetric.leads_per_dollar:
        revenue = np.array([r.purchases for r in scale], dtype=float)
    ratio = value / baseline_value
    lb = bootstrap_ratio_lower_bound(
        revenue,
        spend,
        baseline_value,
        n_samples=cfg.outlier.bootstrap_samples,
        confidence=cfg.outlier.bootstrap_confidence,
        rng=rng or np.random.default_rng(int(agg.render_id.int % (2**32))),
    )
    candidate = tier_for_ratio(lb, cfg.outlier.tier_multiples)
    gates = check_gates(
        conversions=agg.scale_conversions,
        value=value,
        category_median=category_median,
        revenue_by_day=revenue.tolist(),
        spend_by_day=spend.tolist(),
        baseline=baseline_value,
        tier=candidate,
        cfg=cfg.outlier,
    )
    final = candidate if gates.all_ok else Tier.zero
    return OutcomeResult(
        metric=metric,
        value=value,
        impressions=sum(r.impressions for r in scale),
        conversions=agg.scale_conversions,
        spend=agg.scale_spend,
        revenue=agg.scale_revenue,
        days_at_scale=len(scale),
        account_median_90d=baseline_value,
        account_median_human_90d=baselines.human_roas,
        category_median=category_median,
        baseline=baselines.roas.kind,
        ratio=ratio,
        ratio_lower_bound=lb,
        outlier_tier=final,
        gates=gates,
        attribution_setting=agg.attribution_setting,
        measured_at=measured_at,
    )


# --------------------------------------------------------------------------------------
# Database side
# --------------------------------------------------------------------------------------


async def load_render_aggs(session: AsyncSession, account_id: UUID) -> list[RenderAgg]:
    stmt = (
        select(DailyInsight, Render.id, Trajectory, Brief)
        .join(Render, DailyInsight.render_id == Render.id)
        .join(Trajectory, Render.trajectory_id == Trajectory.id)
        .join(Brief, Trajectory.brief_id == Brief.id)
        .where(Brief.ad_account_id == account_id)
        .order_by(DailyInsight.day)
    )
    aggs: dict[UUID, RenderAgg] = {}
    for insight, render_id, traj, brief in (await session.execute(stmt)).all():
        agg = aggs.get(render_id)
        if agg is None:
            agg = RenderAgg(
                render_id=render_id,
                trajectory_id=traj.id,
                brief_id=brief.id,
                campaign_id=traj.campaign_id,
                author_kind=traj.author_kind,
                category=brief.category,
                goal_metric=brief.goal_metric,
                attribution_setting=insight.attribution_setting,
            )
            aggs[render_id] = agg
        agg.rows.append(
            DayRow(
                day=insight.day,
                phase=insight.phase,
                impressions=insight.impressions,
                link_clicks=insight.link_clicks,
                spend=insight.spend,
                purchases=insight.purchases,
                revenue=insight.revenue,
                reactions=insight.reactions,
                comments=insight.comments,
                shares=insight.shares,
                saves=insight.saves,
                frequency=insight.frequency,
            )
        )
    return list(aggs.values())


async def load_holdout(session: AsyncSession) -> frozenset[str]:
    res = await session.execute(select(HoldoutCampaign.campaign_id))
    return frozenset(res.scalars().all())


@dataclass
class RecomputeSummary:
    account_id: UUID
    renders: int = 0
    screening_rows: int = 0
    outcome_rows: int = 0
    trajectories_tiered: int = 0
    tier_counts: dict[int, int] | None = None
    baseline_fallbacks: int = 0


async def recompute_account(
    session: AsyncSession,
    account_id: UUID,
    cfg: AppConfig,
    *,
    as_of: date | None = None,
) -> RecomputeSummary:
    """Recompute every derived row for one account from raw daily insights."""
    summary = RecomputeSummary(account_id=account_id)
    aggs = await load_render_aggs(session, account_id)
    holdout = await load_holdout(session)
    now = datetime.now(UTC)
    tier_counts: dict[int, int] = defaultdict(int)
    best_by_traj: dict[UUID, tuple[int, float, UUID, datetime]] = {}
    trajectories_seen: set[UUID] = set()
    # shipped ideas whose screening window completed without a pass: tier 0 (spec 7.2: the
    # outcome reward is null only for ideas never shipped)
    screen_fail_by_traj: dict[UUID, tuple[UUID, datetime]] = {}

    for agg in aggs:
        summary.renders += 1
        trajectories_seen.add(agg.trajectory_id)
        measured_day = as_of or agg.last_day or now.date()
        baselines = compute_baselines(
            aggs,
            as_of=measured_day,
            category=agg.category,
            goal_metric=agg.goal_metric,
            attribution_setting=agg.attribution_setting,
            cfg=cfg.outlier,
            category_medians=cfg.category_medians,
            exclude_trajectory_id=agg.trajectory_id,
            holdout_campaigns=holdout,
        )
        if baselines.roas.kind == BaselineKind.category_fallback:
            summary.baseline_fallbacks += 1

        screening = compute_screening(
            agg.screening_rows, baselines.ctr.value, cfg.outlier.screening
        )
        if screening is not None:
            await session.merge(
                ScreeningStats(
                    render_id=agg.render_id,
                    impressions=screening.impressions,
                    link_clicks=screening.link_clicks,
                    spend=screening.spend,
                    ctr=screening.ctr,
                    cpc=screening.cpc,
                    ctr_lower_bound=screening.ctr_lower_bound,
                    account_median_ctr_90d=screening.account_median_ctr_90d,
                    screening_ratio=screening.screening_ratio,
                    passed=screening.passed,
                    window_complete=screening.window_complete,
                    engagement=screening.engagement,
                    config_hash=cfg.hash,
                )
            )
            summary.screening_rows += 1
            if screening.window_complete and not screening.passed:
                fail_at = datetime.combine(
                    agg.last_day or (as_of or now.date()), datetime.min.time(), tzinfo=UTC
                )
                prev = screen_fail_by_traj.get(agg.trajectory_id)
                if prev is None or fail_at > prev[1]:
                    screen_fail_by_traj[agg.trajectory_id] = (agg.render_id, fail_at)

        cat = category_median_for(agg.category, cfg.category_medians)
        measured_at = datetime.combine(measured_day, datetime.min.time(), tzinfo=UTC)
        outcome = compute_outcome(
            agg, baselines, cat.roas if cat else 0.0, cfg, measured_at=measured_at
        )
        if outcome is not None:
            await session.merge(
                Outcome(
                    render_id=agg.render_id,
                    metric=outcome.metric.value,
                    value=outcome.value,
                    impressions=outcome.impressions,
                    conversions=outcome.conversions,
                    spend=outcome.spend,
                    revenue=outcome.revenue,
                    days_at_scale=outcome.days_at_scale,
                    account_median_90d=outcome.account_median_90d,
                    account_median_human_90d=outcome.account_median_human_90d,
                    category_median=outcome.category_median,
                    baseline=outcome.baseline.value,
                    ratio=outcome.ratio,
                    ratio_lower_bound=outcome.ratio_lower_bound,
                    outlier_tier=int(outcome.outlier_tier),
                    gates=outcome.gates.as_dict(),
                    attribution_setting=outcome.attribution_setting,
                    measured_at=outcome.measured_at,
                    config_hash=cfg.hash,
                )
            )
            summary.outcome_rows += 1
            key = (int(outcome.outlier_tier), outcome.ratio, agg.render_id, outcome.measured_at)
            best = best_by_traj.get(agg.trajectory_id)
            if best is None or key[:2] > best[:2]:
                best_by_traj[agg.trajectory_id] = key

    # Idea-level outcome: max across renders. Ideas with no scale data are tier 0 once every
    # shipped render has finished screening without a pass and nothing is still live; ideas
    # still pending or at scale without a full window stay None ("no outcome yet").
    live_rows = await session.execute(
        select(Render.trajectory_id)
        .where(
            Render.trajectory_id.in_(trajectories_seen) if trajectories_seen else false(),
            Render.status.in_(
                (
                    RenderStatus.pending_review.value,
                    RenderStatus.screening.value,
                    RenderStatus.scaled.value,
                )
            ),
        )
        .distinct()
    )
    still_live = {tid for (tid,) in live_rows.all()}
    for traj_id in trajectories_seen:
        traj = await session.get(Trajectory, traj_id)
        if traj is None:
            continue
        best = best_by_traj.get(traj_id)
        if best is None and traj_id in screen_fail_by_traj and traj_id not in still_live:
            render_id, fail_at = screen_fail_by_traj[traj_id]
            best = (0, 0.0, render_id, fail_at)
        if best is None:
            traj.outlier_tier = None
            traj.outcome_render_id = None
            traj.outcome_measured_at = None
        else:
            traj.outlier_tier, _, traj.outcome_render_id, traj.outcome_measured_at = best
            tier_counts[best[0]] += 1
            summary.trajectories_tiered += 1

    await session.flush()
    await refresh_tier_counts(session)
    summary.tier_counts = dict(tier_counts)
    return summary


async def recompute_all(session: AsyncSession, cfg: AppConfig) -> list[RecomputeSummary]:
    res = await session.execute(select(AdAccount.id))
    return [await recompute_account(session, aid, cfg) for aid in res.scalars().all()]
