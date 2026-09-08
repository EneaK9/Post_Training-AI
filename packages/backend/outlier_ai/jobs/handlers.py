"""Job handlers: insights sync, comment sync + signal extraction, recompute, episode tick.

Every handler is idempotent: re-running for the same day upserts the same rows and re-derives
the same stats. Auth failures mark the account `needs_reauth` and never stop other accounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.audit import record_audit
from outlier_ai.core.crypto import hash_identity
from outlier_ai.core.settings import Settings
from outlier_ai.core.storage import get_storage
from outlier_ai.episodes.controller import tick_episode
from outlier_ai.generation.backends.base import JudgeBackend
from outlier_ai.generation.preship import check_policy_keywords
from outlier_ai.meta.base import MetaClient
from outlier_ai.meta.errors import MetaApiVersionError, MetaAuthError, MetaError
from outlier_ai.meta.factory import client_for_account
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.meta import AdAccount, Comment, DailyInsight
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_ai.outlier.recompute import recompute_account
from outlier_ai.signals.extraction import extract_for_trajectory
from outlier_ai.signals.pii import strip_pii
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import AccountStatus, EpisodeStatus, RenderStatus

LIVE = (RenderStatus.pending_review.value, RenderStatus.screening.value, RenderStatus.scaled.value)


@dataclass
class SyncReport:
    account_id: str
    renders: int = 0
    rows_upserted: int = 0
    comments_added: int = 0
    signals_proposed: int = 0
    errors: list[str] = field(default_factory=list)
    marked_needs_reauth: bool = False


async def _accounts(session: AsyncSession, account_id: UUID | None) -> list[AdAccount]:
    stmt = select(AdAccount).where(AdAccount.status == AccountStatus.active.value)
    if account_id:
        stmt = select(AdAccount).where(AdAccount.id == account_id)
    return list((await session.execute(stmt)).scalars().all())


async def _mark_needs_reauth(session: AsyncSession, account: AdAccount, err: MetaError) -> None:
    account.status = AccountStatus.needs_reauth.value
    account.last_error = str(err)[:1000]
    await record_audit(
        session,
        actor_id="sync",
        action="account.needs_reauth",
        object_type="ad_account",
        object_id=account.id,
        after={"error": str(err)[:300]},
    )


async def _live_renders(
    session: AsyncSession, account: AdAccount, statuses: tuple[str, ...]
) -> list[Render]:
    stmt = (
        select(Render)
        .join(Trajectory, Trajectory.id == Render.trajectory_id)
        .join(Brief, Brief.id == Trajectory.brief_id)
        .where(
            Brief.ad_account_id == account.id,
            Render.meta_ad_id.is_not(None),
            Render.status.in_(statuses),
        )
    )
    return list((await session.execute(stmt)).scalars().all())


async def _last_day(session: AsyncSession, render_id: UUID) -> date | None:
    return (
        await session.execute(
            select(func.max(DailyInsight.day)).where(DailyInsight.render_id == render_id)
        )
    ).scalar_one()


async def sync_insights_for_account(
    session: AsyncSession,
    account: AdAccount,
    cfg: AppConfig,
    *,
    client: MetaClient,
    today: date | None = None,
    lookback_days: int = 3,
) -> SyncReport:
    """Pull daily rows for every live ad from the day after the last ingested day (minus a
    lookback for attribution updates) to yesterday, then recompute derived stats."""
    report = SyncReport(account_id=account.meta_account_id)
    today = today or datetime.now(UTC).date()
    renders = await _live_renders(session, account, (*LIVE, RenderStatus.stopped.value))
    renders = [
        r
        for r in renders
        if r.shipped_at
        and (r.stopped_at is None or (today - r.stopped_at.date()).days <= lookback_days)
    ]
    report.renders = len(renders)
    try:
        for r in renders:
            last = await _last_day(session, r.id)
            start = (last - timedelta(days=lookback_days - 1)) if last else r.shipped_at.date()  # type: ignore[union-attr]
            day = max(start, r.shipped_at.date())  # type: ignore[union-attr]
            while day < today:
                for row in await client.get_daily_insights([r.meta_ad_id or ""], day):
                    existing = (
                        await session.execute(
                            select(DailyInsight).where(
                                DailyInsight.render_id == r.id, DailyInsight.day == day
                            )
                        )
                    ).scalar_one_or_none()
                    values: dict[str, Any] = {
                        "phase": r.phase
                        if not (
                            r.scale_started_at
                            and datetime.combine(day, datetime.min.time(), tzinfo=UTC)
                            < r.scale_started_at
                        )
                        else "screening",
                        "impressions": row.impressions,
                        "link_clicks": row.link_clicks,
                        "spend": row.spend,
                        "purchases": row.purchases,
                        "revenue": row.revenue,
                        "frequency": row.frequency,
                        "reactions": row.reactions,
                        "comments": row.comments,
                        "shares": row.shares,
                        "saves": row.saves,
                        "daily_budget": r.daily_budget_usd,
                        "attribution_setting": account.attribution_setting,
                        "source": "fake" if account.is_fake else "meta",
                    }
                    if existing is None:
                        session.add(DailyInsight(render_id=r.id, day=day, **values))
                    else:
                        for k, v in values.items():
                            setattr(existing, k, v)
                    report.rows_upserted += 1
                day += timedelta(days=1)
        await session.flush()
        await recompute_account(session, account.id, cfg)
        account.last_sync_at = datetime.now(UTC)
        account.last_error = None
    except (MetaAuthError, MetaApiVersionError) as e:
        await _mark_needs_reauth(session, account, e)
        report.marked_needs_reauth = True
        report.errors.append(str(e))
    except MetaError as e:
        account.last_error = str(e)[:1000]
        report.errors.append(str(e))
    return report


async def sync_comments_for_account(
    session: AsyncSession,
    account: AdAccount,
    cfg: AppConfig,
    *,
    client: MetaClient,
    judge: JudgeBackend | None = None,
) -> SyncReport:
    report = SyncReport(account_id=account.meta_account_id)
    renders = [
        r
        for r in await _live_renders(session, account, (*LIVE, RenderStatus.stopped.value))
        if r.effective_object_story_id
    ]
    report.renders = len(renders)
    touched: set[UUID] = set()
    try:
        for r in renders:
            since = (
                await session.execute(
                    select(func.max(Comment.created_time)).where(Comment.render_id == r.id)
                )
            ).scalar_one()
            for c in await client.get_post_comments(r.effective_object_story_id or "", since):
                if (
                    await session.execute(select(Comment.id).where(Comment.external_id == c.id))
                ).first():
                    continue
                pii = strip_pii(c.message)
                flags = check_policy_keywords(pii.text.lower())
                session.add(
                    Comment(
                        render_id=r.id,
                        external_id=c.id,
                        commenter_hash=hash_identity(c.from_id or c.id),
                        text=pii.text,
                        created_time=c.created_time,
                        like_count=c.like_count,
                        filtered_reason="policy:" + flags[0][:40] if flags else None,
                        pii_removed=pii.removed,
                    )
                )
                report.comments_added += 1
                touched.add(r.trajectory_id)
        await session.flush()
        for tid in touched:
            created = await extract_for_trajectory(
                session,
                tid,
                actor="sync",
                judge=judge,
                min_count=cfg.signals.min_comments_for_theme,
            )
            report.signals_proposed += len(created)
    except (MetaAuthError, MetaApiVersionError) as e:
        await _mark_needs_reauth(session, account, e)
        report.marked_needs_reauth = True
        report.errors.append(str(e))
    except MetaError as e:
        account.last_error = str(e)[:1000]
        report.errors.append(str(e))
    return report


async def run_sync_insights(
    session: AsyncSession,
    cfg: AppConfig,
    *,
    account_id: UUID | None = None,
    settings: Settings | None = None,
    today: date | None = None,
    clients: dict[UUID, MetaClient] | None = None,
) -> list[SyncReport]:
    reports: list[SyncReport] = []
    for account in await _accounts(session, account_id):
        try:
            client = (clients or {}).get(account.id) or await client_for_account(
                session, account, cfg, settings=settings
            )
        except Exception as e:
            reports.append(SyncReport(account_id=account.meta_account_id, errors=[str(e)]))
            continue
        reports.append(
            await sync_insights_for_account(session, account, cfg, client=client, today=today)
        )
    return reports


async def run_sync_comments(
    session: AsyncSession,
    cfg: AppConfig,
    *,
    account_id: UUID | None = None,
    settings: Settings | None = None,
    judge: JudgeBackend | None = None,
    clients: dict[UUID, MetaClient] | None = None,
) -> list[SyncReport]:
    reports: list[SyncReport] = []
    for account in await _accounts(session, account_id):
        try:
            client = (clients or {}).get(account.id) or await client_for_account(
                session, account, cfg, settings=settings
            )
        except Exception as e:
            reports.append(SyncReport(account_id=account.meta_account_id, errors=[str(e)]))
            continue
        reports.append(
            await sync_comments_for_account(session, account, cfg, client=client, judge=judge)
        )
    return reports


async def run_episode_tick(
    session: AsyncSession,
    cfg: AppConfig,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
    clients: dict[UUID, MetaClient] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    episodes = (
        (
            await session.execute(
                select(SearchEpisode).where(SearchEpisode.status == EpisodeStatus.searching.value)
            )
        )
        .scalars()
        .all()
    )
    for ep in episodes:
        if ep.ad_account_id is None:
            continue
        account = await session.get(AdAccount, ep.ad_account_id)
        if account is None:
            continue
        try:
            client = (clients or {}).get(account.id) or await client_for_account(
                session, account, cfg, settings=settings, require_shippable=False
            )
        except Exception as e:
            out.append({"episode_id": str(ep.id), "error": str(e)})
            continue
        report = await tick_episode(
            session, ep, client=client, cfg=cfg, now=now, storage=get_storage(), settings=settings
        )
        out.append({"episode_id": str(ep.id), "status": report.status, "events": report.events})
    return out
