"""Budget governor: the last line of defense before money moves.

Independent of the controller so a controller bug cannot overspend. Every ship path calls
`assert_can_ship`, which checks, in order: the global kill switch, dry-run mode (only fake
accounts may ship while DRY_RUN is on), account status, the per-account daily hard cap
across all live ads, and the episode's own budget cap including the projected screening
spend of what is about to ship. The episode row is locked for the duration so concurrent
ship calls serialize.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.errors import SafetyError
from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.meta import AdAccount, DailyInsight
from outlier_ai.models.ops import KillSwitch
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import AccountStatus, EpisodeStatus

LIVE_STATUSES = ("pending_review", "screening", "scaled")


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str
    account_live_daily_usd: float
    account_cap_daily_usd: float
    episode_spent_usd: float
    episode_projected_usd: float
    episode_cap_usd: float


async def kill_switch_open(session: AsyncSession) -> bool:
    ks = await session.get(KillSwitch, 1)
    return ks is None or bool(ks.shipping_enabled)


async def account_live_daily(session: AsyncSession, account_id) -> float:
    """Sum of daily budgets of ads that are live on this account right now."""
    stmt = (
        select(func.coalesce(func.sum(Render.daily_budget_usd), 0.0))
        .join(Trajectory, Trajectory.id == Render.trajectory_id)
        .join(Brief, Brief.id == Trajectory.brief_id)
        .where(Brief.ad_account_id == account_id, Render.status.in_(LIVE_STATUSES))
    )
    return float((await session.execute(stmt)).scalar_one() or 0.0)


async def episode_spent(session: AsyncSession, episode_id) -> float:
    stmt = (
        select(func.coalesce(func.sum(DailyInsight.spend), 0.0))
        .join(Render, Render.id == DailyInsight.render_id)
        .join(Trajectory, Trajectory.id == Render.trajectory_id)
        .where(Trajectory.episode_id == episode_id)
    )
    return float((await session.execute(stmt)).scalar_one() or 0.0)


async def episode_committed(session: AsyncSession, episode_id, cfg: AppConfig) -> float:
    """Spend the live ads of this episode will still make before their windows end (upper bound)."""
    stmt = (
        select(Render.phase, Render.daily_budget_usd)
        .join(Trajectory, Trajectory.id == Render.trajectory_id)
        .where(Trajectory.episode_id == episode_id, Render.status.in_(LIVE_STATUSES))
    )
    total = 0.0
    for phase, budget in (await session.execute(stmt)).all():
        days = cfg.episode.scale_days if phase == "scale" else cfg.episode.screening_days
        total += float(budget or 0.0) * days
    return total


async def check_budget(
    session: AsyncSession,
    *,
    episode: SearchEpisode,
    account: AdAccount | None,
    add_daily_usd: float,
    cfg: AppConfig,
    settings: Settings | None = None,
) -> BudgetDecision:
    s = settings or get_settings()
    cap_daily = float(
        account.daily_cap_usd
        if account and account.daily_cap_usd
        else cfg.meta.daily_account_cap_usd
    )
    live = await account_live_daily(session, account.id) if account else 0.0
    spent = await episode_spent(session, episode.id)
    committed = await episode_committed(session, episode.id, cfg)
    projected = spent + committed + add_daily_usd * cfg.episode.screening_days

    def deny(reason: str) -> BudgetDecision:
        return BudgetDecision(False, reason, live, cap_daily, spent, projected, episode.budget_cap)

    if not await kill_switch_open(session):
        return deny("kill switch: shipping is disabled")
    is_fake = bool(account and account.is_fake)
    if s.dry_run and not is_fake:
        return deny("dry run: real accounts cannot ship while DRY_RUN is on")
    if account is None:
        return deny("episode has no ad account")
    if account.status != AccountStatus.active.value:
        return deny(f"account is {account.status}")
    if episode.status != EpisodeStatus.searching.value:
        return deny(f"episode is {episode.status}")
    if live + add_daily_usd > cap_daily + 1e-9:
        return deny(
            f"account daily cap: {live:.0f} live + {add_daily_usd:.0f} new > {cap_daily:.0f}"
        )
    if projected > episode.budget_cap + 1e-9:
        new_spend = add_daily_usd * cfg.episode.screening_days
        return deny(
            f"episode cap: {spent:.0f} spent + {committed:.0f} committed + "
            f"{new_spend:.0f} projected > {episode.budget_cap:.0f}"
        )
    return BudgetDecision(True, "ok", live, cap_daily, spent, projected, episode.budget_cap)


async def assert_can_ship(
    session: AsyncSession,
    *,
    episode: SearchEpisode,
    account: AdAccount | None,
    add_daily_usd: float,
    cfg: AppConfig,
    settings: Settings | None = None,
) -> BudgetDecision:
    """Lock the episode row, evaluate, raise SafetyError if not allowed."""
    await session.execute(
        select(SearchEpisode).where(SearchEpisode.id == episode.id).with_for_update()
    )
    decision = await check_budget(
        session,
        episode=episode,
        account=account,
        add_daily_usd=add_daily_usd,
        cfg=cfg,
        settings=settings,
    )
    if not decision.allowed:
        raise SafetyError(f"refusing to ship: {decision.reason}")
    return decision
