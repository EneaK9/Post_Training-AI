"""Loop A runner: generate -> approve -> ship -> (advance) -> sync -> tick, until the episode ends.

Two modes:
- `step`: one decision cycle for a live episode (real or fake account). The worker's daily
  cadence does the rest. Used by the Generate screen's "next batch" and by the API.
- `simulate`: for fake accounts only. Drives the fake clock day by day until the episode reaches
  a terminal state or `max_days` passes. This is the step-12 gate of the spec and the engine of
  the loop-A-vs-random experiment.

Approval policies: `top_rm` (delegate selection to the reward model, spec section 6),
`random` (baseline), or `manual` (label nothing; humans approve on the Generate screen).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from uuid import UUID

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.audit import record_audit
from outlier_ai.core.errors import NotFoundError, ValidationError
from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.core.storage import Storage
from outlier_ai.episodes.controller import can_start_next_batch, tick_episode
from outlier_ai.generation.service import GenerationService
from outlier_ai.jobs.handlers import sync_insights_for_account
from outlier_ai.meta.base import MetaClient
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.meta.ship import ship_batch
from outlier_ai.models.episodes import Batch, SearchEpisode
from outlier_ai.models.meta import AdAccount
from outlier_ai.models.trajectories import Review, Trajectory
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BackendKind, BatchState, EpisodeStatus, ReviewLabel
from outlier_schemas.models import GenerationRequest

Approval = Literal["top_rm", "random", "manual"]


@dataclass
class StepReport:
    episode_id: UUID
    generated: int = 0
    queued: int = 0
    approved: int = 0
    shipped_renders: int = 0
    skipped_reason: str = ""
    batch_id: UUID | None = None
    events: list[str] = field(default_factory=list)


@dataclass
class LoopAResult:
    episode_id: UUID
    brief_id: UUID
    status: str
    batches: int
    spent: float
    best_tier: int | None
    outlier_trajectory_id: UUID | None
    days: int
    ideas_generated: int
    ideas_shipped: int
    events: list[str] = field(default_factory=list)


async def approve_batch(
    session: AsyncSession,
    batch_id: UUID,
    *,
    n: int,
    policy: str,
    actor: str,
    rng: np.random.Generator | None = None,
) -> list[UUID]:
    """Label the chosen ideas `run` and the rest `skip`. Manual policy labels nothing."""
    if policy not in ("top_rm", "random", "manual"):
        raise ValidationError(f"unknown approval policy {policy!r}")
    if policy == "manual":
        return []
    rows = (
        (
            await session.execute(
                select(Trajectory).where(
                    Trajectory.batch_id == batch_id,
                    Trajectory.format_ok.is_(True),
                    ~Trajectory.id.in_(select(Review.trajectory_id)),
                )
            )
        )
        .scalars()
        .all()
    )
    # only shippable ideas may be auto-approved: well-formed and clean on pre-ship checks
    rows = [
        t
        for t in rows
        if not t.preship or (t.preship.get("policy_ok", True) and t.preship.get("brand_ok", True))
    ]
    if not rows:
        return []
    if policy == "top_rm":
        ordered = sorted(
            rows, key=lambda t: (-(t.rm_score if t.rm_score is not None else -1.0), t.attempt_index)
        )
    else:
        rng = rng or np.random.default_rng(0)
        ordered = list(rows)
        rng.shuffle(ordered)  # type: ignore[arg-type]
    chosen = ordered[:n]
    chosen_ids = {t.id for t in chosen}
    for t in rows:
        label = ReviewLabel.run if t.id in chosen_ids else ReviewLabel.skip
        session.add(
            Review(trajectory_id=t.id, label=label.value, reviewer_id=actor, note=f"auto:{policy}")
        )
    await session.flush()
    return [t.id for t in chosen]


class LoopARunner:
    def __init__(
        self,
        *,
        session: AsyncSession,
        cfg: AppConfig,
        service: GenerationService,
        client: MetaClient,
        storage: Storage,
        settings: Settings | None = None,
        approval: str = "top_rm",
        actor: str = "loop-a",
        seed: int = 0,
    ) -> None:
        self.session = session
        self.cfg = cfg
        self.service = service
        self.client = client
        self.storage = storage
        self.settings = settings or get_settings()
        self.approval = approval
        self.actor = actor
        self.rng = np.random.default_rng(seed)

    async def step(
        self, episode: SearchEpisode, *, now: datetime | None = None, k: int | None = None
    ) -> StepReport:
        """If the episode can take a batch: generate, approve, ship. Otherwise say why not."""
        now = now or datetime.now(UTC)
        report = StepReport(episode_id=episode.id)
        ok, reason = await can_start_next_batch(self.session, episode, self.cfg)
        if not ok:
            report.skipped_reason = reason
            return report
        req = GenerationRequest(
            brief_id=episode.brief_id,
            episode_id=episode.id,
            backend=BackendKind(episode.backend),
            k=k or self.cfg.generation.k,
            renders_per_idea=self.cfg.episode.renders_per_idea,
        )
        result = await self.service.generate_batch(self.session, req, now=now)
        report.generated = len(result.ideas)
        report.queued = len(result.queued)
        report.batch_id = result.batch_id
        report.events.append(
            f"batch {result.batch_id}: {report.generated} ideas, {report.queued} queued, "
            f"{len(result.rejected)} rejected"
        )
        if result.batch_id is None:
            return report
        chosen = await approve_batch(
            self.session,
            result.batch_id,
            n=self.cfg.episode.ideas_per_batch,
            policy=self.approval,
            actor=self.actor,
            rng=self.rng,
        )
        report.approved = len(chosen)
        if chosen:
            results = await ship_batch(
                self.session,
                result.batch_id,
                client=self.client,
                storage=self.storage,
                cfg=self.cfg,
                actor=self.actor,
                settings=self.settings,
                now=now,
            )
            report.shipped_renders = sum(len(r.shipped_render_ids) for r in results)
            refused = [r for r in results if r.skipped and not r.shipped_render_ids]
            report.events.append(
                f"shipped {report.shipped_renders} renders from {len(results)} ideas"
                + (f", {len(refused)} refused" if refused else "")
            )
            if results and report.shipped_renders == 0:
                # nothing could ship (budget or safety): close the batch so the episode can conclude
                batch = await self.session.get(Batch, result.batch_id)
                if batch is not None:
                    batch.state = BatchState.stopped.value
                    batch.state_changed_at = now
                reasons = "; ".join(r.skipped[0] for r in refused[:2])
                report.events.append(f"batch stopped: no idea could ship: {reasons}")
        if self.approval != "manual" and report.approved == 0 and result.batch_id is not None:
            # nothing shippable came out of this batch (all rejected or flagged); close it so the
            # controller can start the next one or conclude the episode
            batch = await self.session.get(Batch, result.batch_id)
            if batch is not None and batch.state == BatchState.proposed.value:
                batch.state = BatchState.stopped.value
                batch.state_changed_at = now
                report.events.append("batch stopped: nothing to approve")
        await record_audit(
            self.session,
            actor_id=self.actor,
            action="loop_a.step",
            object_type="episode",
            object_id=episode.id,
            after={
                "generated": report.generated,
                "approved": report.approved,
                "shipped_renders": report.shipped_renders,
            },
        )
        await self.session.flush()
        return report

    async def simulate(
        self, episode: SearchEpisode, *, max_days: int = 120, k: int | None = None
    ) -> LoopAResult:
        """Fake accounts only: run the whole episode against the fake clock."""
        if not isinstance(self.client, FakeMetaClient):
            raise ValidationError("simulate needs a fake Meta client; use step() for live accounts")
        account = (
            await self.session.get(AdAccount, episode.ad_account_id)
            if episode.ad_account_id
            else None
        )
        if account is None:
            raise NotFoundError("episode has no ad account")
        events: list[str] = []
        generated = shipped = 0
        today: date = await self.client.today()
        day = 0
        while day < max_days and episode.status == EpisodeStatus.searching.value:
            step = await self.step(episode, now=_dt(today), k=k)
            generated += step.generated
            shipped += step.shipped_renders
            events.extend(f"day {day}: {e}" for e in step.events)
            today = await self.client.advance_days(1)
            day += 1
            await sync_insights_for_account(
                self.session, account, self.cfg, client=self.client, today=today + timedelta(days=1)
            )
            tick = await tick_episode(
                self.session,
                episode,
                client=self.client,
                cfg=self.cfg,
                now=_dt(today),
                storage=self.storage,
                settings=self.settings,
            )
            events.extend(f"day {day}: {e}" for e in tick.events)
            await self.session.flush()
        best = await session_best(self.session, episode.id)
        n_batches = int(
            (
                await self.session.execute(
                    select(func.count(func.distinct(Trajectory.batch_id))).where(
                        Trajectory.episode_id == episode.id
                    )
                )
            ).scalar_one()
        )
        return LoopAResult(
            episode_id=episode.id,
            brief_id=episode.brief_id,
            status=episode.status,
            batches=n_batches,
            spent=float(episode.spent),
            best_tier=best[0] if best else None,
            outlier_trajectory_id=best[1]
            if best and best[0] is not None and best[0] >= 2
            else None,
            days=day,
            ideas_generated=generated,
            ideas_shipped=shipped,
            events=events,
        )


def _dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 12, tzinfo=UTC)


async def session_best(
    session: AsyncSession, episode_id: UUID
) -> tuple[int | None, UUID | None] | None:
    row = (
        await session.execute(
            select(Trajectory.outlier_tier, Trajectory.id)
            .where(Trajectory.episode_id == episode_id, Trajectory.outlier_tier.is_not(None))
            .order_by(Trajectory.outlier_tier.desc())
            .limit(1)
        )
    ).first()
    return (row[0], row[1]) if row else None
