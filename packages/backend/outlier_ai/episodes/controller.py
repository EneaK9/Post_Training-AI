"""Episode controller: the batch state machine over screening and scale.

`tick_episode` is idempotent and runs after every insights sync:

- in_review  -> screening    once every shipped render is approved or rejected
- screening  -> scaling      when the screening window is complete: passing renders get the
                             scale budget, failing renders are paused
- scaling    -> measured     when the scale window is complete: ads are paused
- episode    -> outlier_found when any trajectory reaches tier 2 (remaining ads stopped unless
                             keep_running_after_outlier); budget_exhausted when the cap is hit.

Overlapping batches: the next batch may start as soon as the previous batch's screening has
resolved (its signals are already in history). All three renders rejected -> the trajectory
keeps `outcome = null`, the reviewer is notified via the audit log, and the spend is not counted
because rejected ads never ran (scenario row 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.audit import record_audit
from outlier_ai.episodes.budget import episode_spent
from outlier_ai.meta import campaign as campaigns
from outlier_ai.meta.base import MetaClient
from outlier_ai.models.episodes import Batch, SearchEpisode
from outlier_ai.models.meta import DailyInsight
from outlier_ai.models.trajectories import Render, ScreeningStats, Trajectory
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BatchState, EpisodeStatus, RenderStatus

TERMINAL_BATCH = {BatchState.measured.value, BatchState.stopped.value}
LIVE = {RenderStatus.pending_review.value, RenderStatus.screening.value, RenderStatus.scaled.value}


@dataclass
class TickReport:
    episode_id: UUID
    events: list[str] = field(default_factory=list)
    status: str = ""

    def log(self, msg: str) -> None:
        self.events.append(msg)


async def poll_reviews(
    session: AsyncSession,
    episode: SearchEpisode,
    client: MetaClient,
    report: TickReport,
    now: datetime,
) -> None:
    pending = (
        (
            await session.execute(
                select(Render)
                .join(Trajectory, Trajectory.id == Render.trajectory_id)
                .where(
                    Trajectory.episode_id == episode.id,
                    Render.status == RenderStatus.pending_review.value,
                    Render.meta_ad_id.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    )
    if not pending:
        return
    statuses = {
        s.ad_id: s
        for s in await client.get_ad_review_status([r.meta_ad_id for r in pending if r.meta_ad_id])
    }
    for r in pending:
        st = statuses.get(r.meta_ad_id or "")
        if st is None:
            continue
        if st.approved:
            r.status = RenderStatus.screening.value
            report.log(f"render {r.id} approved")
        elif st.rejected:
            r.status = RenderStatus.rejected.value
            r.rejection_reason = st.review_feedback or "disapproved by Meta review"
            r.stopped_at = now
            r.daily_budget_usd = 0.0
            report.log(f"render {r.id} rejected: {r.rejection_reason}")


async def _render_days(session: AsyncSession, render_id: UUID, phase: str) -> int:
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(DailyInsight)
                .where(DailyInsight.render_id == render_id, DailyInsight.phase == phase)
            )
        ).scalar_one()
    )


async def advance_screening(
    session: AsyncSession,
    episode: SearchEpisode,
    client: MetaClient,
    cfg: AppConfig,
    report: TickReport,
    now: datetime,
) -> None:
    rows = await session.execute(
        select(Render, ScreeningStats)
        .join(Trajectory, Trajectory.id == Render.trajectory_id)
        .join(ScreeningStats, ScreeningStats.render_id == Render.id, isouter=True)
        .where(Trajectory.episode_id == episode.id, Render.status == RenderStatus.screening.value)
    )
    for render, stats in rows.all():
        days = await _render_days(session, render.id, "screening")
        if days < cfg.outlier.screening.window_days:
            continue
        if stats is not None and stats.passed:
            if render.meta_adset_id:
                await client.update_adset_budget(
                    render.meta_adset_id, round(campaigns.scale_budget(cfg) * 100)
                )
            render.status = RenderStatus.scaled.value
            render.phase = "scale"
            render.daily_budget_usd = campaigns.scale_budget(cfg)
            render.scale_started_at = now
            report.log(
                f"render {render.id} passed screening ({stats.screening_ratio:.2f}x), scaling"
            )
        else:
            if render.meta_ad_id:
                await client.pause_ad(render.meta_ad_id)
            render.status = RenderStatus.stopped.value
            render.stopped_at = now
            render.daily_budget_usd = 0.0
            ratio = f"{stats.screening_ratio:.2f}x" if stats else "no stats"
            report.log(f"render {render.id} failed screening ({ratio}), stopped")


async def advance_scale(
    session: AsyncSession,
    episode: SearchEpisode,
    client: MetaClient,
    cfg: AppConfig,
    report: TickReport,
    now: datetime,
) -> None:
    scaled = (
        (
            await session.execute(
                select(Render)
                .join(Trajectory, Trajectory.id == Render.trajectory_id)
                .where(
                    Trajectory.episode_id == episode.id,
                    Render.status == RenderStatus.scaled.value,
                    Render.stopped_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    for render in scaled:
        days = await _render_days(session, render.id, "scale")
        if days >= cfg.episode.scale_days:
            if render.meta_ad_id:
                await client.pause_ad(render.meta_ad_id)
            render.stopped_at = now
            render.daily_budget_usd = 0.0
            report.log(f"render {render.id} finished its scale window, paused")


async def update_batch_states(
    session: AsyncSession, episode: SearchEpisode, report: TickReport, now: datetime
) -> None:
    batches = (
        (
            await session.execute(
                select(Batch).where(Batch.episode_id == episode.id).order_by(Batch.index)
            )
        )
        .scalars()
        .all()
    )
    for batch in batches:
        if batch.state in TERMINAL_BATCH or batch.state == BatchState.proposed.value:
            continue
        renders = (
            (
                await session.execute(
                    select(Render)
                    .join(Trajectory, Trajectory.id == Render.trajectory_id)
                    .where(Trajectory.batch_id == batch.id, Render.shipped_at.is_not(None))
                )
            )
            .scalars()
            .all()
        )
        if not renders:
            continue
        statuses = {r.status for r in renders}
        # all renders of a trajectory rejected -> reviewer notified once
        by_traj: dict[UUID, list[Render]] = {}
        for r in renders:
            by_traj.setdefault(r.trajectory_id, []).append(r)
        for tid, rs in by_traj.items():
            if all(r.status == RenderStatus.rejected.value for r in rs) and not any(
                "rejected_all" in e and str(tid) in e for e in report.events
            ):
                await record_audit(
                    session,
                    actor_id="controller",
                    action="trajectory.rejected_all",
                    object_type="trajectory",
                    object_id=tid,
                    after={"reasons": [r.rejection_reason for r in rs]},
                )
                report.log(
                    f"trajectory {tid} rejected_all: every render disapproved; outcome stays null"
                )
        new_state = batch.state
        if RenderStatus.pending_review.value in statuses:
            new_state = BatchState.in_review.value
        elif RenderStatus.screening.value in statuses:
            new_state = BatchState.screening.value
        elif any(r.status == RenderStatus.scaled.value and r.stopped_at is None for r in renders):
            new_state = BatchState.scaling.value
        else:
            new_state = BatchState.measured.value
        if new_state != batch.state:
            report.log(f"batch {batch.index}: {batch.state} -> {new_state}")
            batch.state = new_state
            batch.state_changed_at = now


async def stop_live_ads(
    session: AsyncSession,
    episode: SearchEpisode,
    client: MetaClient,
    report: TickReport,
    now: datetime,
    *,
    except_trajectory: UUID | None = None,
) -> int:
    live = (
        (
            await session.execute(
                select(Render)
                .join(Trajectory, Trajectory.id == Render.trajectory_id)
                .where(
                    Trajectory.episode_id == episode.id,
                    Render.status.in_(LIVE),
                    Render.stopped_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    n = 0
    for r in live:
        if except_trajectory and r.trajectory_id == except_trajectory:
            continue
        if r.meta_ad_id:
            await client.pause_ad(r.meta_ad_id)
        if r.status != RenderStatus.scaled.value:
            r.status = RenderStatus.stopped.value
        r.stopped_at = now
        r.daily_budget_usd = 0.0
        n += 1
    if n:
        report.log(f"stopped {n} live ads")
    return n


async def can_start_next_batch(
    session: AsyncSession, episode: SearchEpisode, cfg: AppConfig
) -> tuple[bool, str]:
    if episode.status != EpisodeStatus.searching.value:
        return False, f"episode is {episode.status}"
    batches = (
        (
            await session.execute(
                select(Batch).where(Batch.episode_id == episode.id).order_by(Batch.index)
            )
        )
        .scalars()
        .all()
    )
    if len(batches) >= cfg.episode.max_batches:
        return False, f"max batches ({cfg.episode.max_batches}) reached"
    if not batches:
        return True, "first batch"
    last = batches[-1]
    if last.state in (
        BatchState.proposed.value,
        BatchState.approved.value,
        BatchState.shipping.value,
        BatchState.in_review.value,
    ):
        return False, f"batch {last.index} is {last.state}"
    if last.state == BatchState.screening.value and not cfg.episode.overlap_batches:
        return False, f"batch {last.index} still screening and overlap is off"
    if last.state == BatchState.screening.value:
        return False, f"batch {last.index} screening has not resolved"
    spent = await episode_spent(session, episode.id)
    next_cost = (
        cfg.episode.ideas_per_batch
        * cfg.episode.renders_per_idea
        * cfg.episode.screening_budget_per_ad_usd
        * cfg.episode.screening_days
    )
    if spent + next_cost > episode.budget_cap:
        return (
            False,
            f"budget: {spent:.0f} spent + {next_cost:.0f} for a batch > {episode.budget_cap:.0f} cap",
        )
    return True, "ok"


async def tick_episode(
    session: AsyncSession,
    episode: SearchEpisode,
    *,
    client: MetaClient,
    cfg: AppConfig,
    now: datetime | None = None,
) -> TickReport:
    """One controller step. Call after insights have been synced and outcomes recomputed."""
    now = now or datetime.now(UTC)
    report = TickReport(episode_id=episode.id)
    if episode.status != EpisodeStatus.searching.value:
        report.status = episode.status
        return report

    await poll_reviews(session, episode, client, report, now)
    await advance_screening(session, episode, client, cfg, report, now)
    await advance_scale(session, episode, client, cfg, report, now)
    await update_batch_states(session, episode, report, now)

    # outlier found?
    best = (
        (
            await session.execute(
                select(Trajectory)
                .where(Trajectory.episode_id == episode.id, Trajectory.outlier_tier >= 2)
                .order_by(Trajectory.outlier_tier.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if best is not None:
        episode.status = EpisodeStatus.outlier_found.value
        episode.ended_at = now
        episode.stop_reason = f"tier {best.outlier_tier} on trajectory {best.id}"
        report.log(f"OUTLIER FOUND: trajectory {best.id} tier {best.outlier_tier}")
        if not episode.keep_running_after_outlier:
            await stop_live_ads(session, episode, client, report, now, except_trajectory=best.id)
        await record_audit(
            session,
            actor_id="controller",
            action="episode.outlier_found",
            object_type="episode",
            object_id=episode.id,
            after={"trajectory_id": str(best.id), "tier": best.outlier_tier},
        )
    else:
        spent = await episode_spent(session, episode.id)
        episode.spent = spent
        live = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Render)
                    .join(Trajectory, Trajectory.id == Render.trajectory_id)
                    .where(
                        Trajectory.episode_id == episode.id,
                        Render.status.in_(LIVE),
                        Render.stopped_at.is_(None),
                    )
                )
            ).scalar_one()
        )
        ok, reason = await can_start_next_batch(session, episode, cfg)
        if (
            live == 0
            and not ok
            and (reason.startswith("budget") or reason.startswith("max batches"))
        ):
            episode.status = EpisodeStatus.budget_exhausted.value
            episode.ended_at = now
            episode.stop_reason = reason
            report.log(f"episode budget_exhausted: {reason}")
            await record_audit(
                session,
                actor_id="controller",
                action="episode.budget_exhausted",
                object_type="episode",
                object_id=episode.id,
                after={"spent": spent, "reason": reason},
            )
    report.status = episode.status
    await session.flush()
    return report
