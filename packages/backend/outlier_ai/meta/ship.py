"""Ship approved trajectories: image -> creative -> ad set -> ad, one ad set per ad.

Guard chain before any call: format ok, human `run` label, pre-ship clean, then the budget
governor (kill switch, dry run, account, daily cap, episode cap). Renders enter
`pending_review`; the review poll flips them to `screening` or `rejected`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.audit import record_audit
from outlier_ai.core.errors import NotFoundError, SafetyError, ValidationError
from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.core.storage import Storage, key_from_uri
from outlier_ai.episodes.budget import assert_can_ship
from outlier_ai.meta import campaign as campaigns
from outlier_ai.meta.base import MetaClient
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import Batch, SearchEpisode
from outlier_ai.models.meta import AdAccount
from outlier_ai.models.trajectories import Render, Review, Trajectory
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BatchState, RenderStatus, ReviewLabel


@dataclass
class ShipResult:
    trajectory_id: UUID
    shipped_render_ids: list[UUID] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


async def _episode_campaign(
    session: AsyncSession, episode: SearchEpisode, brief: Brief, client: MetaClient
) -> str:
    if episode.campaign_id:
        return episode.campaign_id
    campaign_id = await client.create_campaign(campaigns.campaign_spec(episode, brief))
    episode.campaign_id = campaign_id
    await session.flush()
    return campaign_id


async def ship_trajectory(
    session: AsyncSession,
    trajectory_id: UUID,
    *,
    client: MetaClient,
    storage: Storage,
    cfg: AppConfig,
    actor: str,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> ShipResult:
    s = settings or get_settings()
    now = now or datetime.now(UTC)
    # lock the trajectory so concurrent ship calls (tick + button) serialize; the second one
    # then finds no draft renders and skips
    traj = (
        await session.execute(
            select(Trajectory).where(Trajectory.id == trajectory_id).with_for_update()
        )
    ).scalar_one_or_none()
    if traj is None:
        raise NotFoundError(f"trajectory {trajectory_id} not found")
    if traj.episode_id is None:
        raise ValidationError("only trajectories inside an episode can ship")
    episode = await session.get(SearchEpisode, traj.episode_id)
    brief = await session.get(Brief, traj.brief_id)
    if episode is None or brief is None:
        raise NotFoundError("episode or brief missing")
    account = await session.get(AdAccount, episode.ad_account_id) if episode.ad_account_id else None
    if account is None:
        raise SafetyError("episode has no ad account")

    review = (
        await session.execute(select(Review).where(Review.trajectory_id == traj.id))
    ).scalar_one_or_none()
    if review is None or review.label != ReviewLabel.run.value:
        raise SafetyError("trajectory has no `run` review label; the human label is the approval")
    if not traj.format_ok:
        raise SafetyError("malformed ideas never ship")
    if traj.preship and not (
        traj.preship.get("policy_ok", True) and traj.preship.get("brand_ok", True)
    ):
        raise SafetyError("pre-ship checks flagged this idea")

    renders = [
        r
        for r in (
            await session.execute(
                select(Render).where(Render.trajectory_id == traj.id).order_by(Render.seed)
            )
        )
        .scalars()
        .all()
        if r.status == RenderStatus.draft.value and r.image_uri
    ]
    result = ShipResult(trajectory_id=traj.id)
    if not renders:
        result.skipped.append("no draft renders with images")
        return result

    per_ad = campaigns.screening_budget(cfg)
    await assert_can_ship(
        session,
        episode=episode,
        account=account,
        add_daily_usd=per_ad * len(renders),
        cfg=cfg,
        settings=s,
    )

    campaign_id = await _episode_campaign(session, episode, brief, client)
    traj.campaign_id = campaign_id
    for render in renders:
        image_bytes = storage.get(key_from_uri(render.image_uri or ""))
        image = await client.upload_image(image_bytes, f"{traj.id}-{render.seed}.png")
        creative = await client.create_creative(
            campaigns.creative_spec(
                trajectory=traj,
                render=render,
                brief=brief,
                account=account,
                image_hash=image.image_hash,
            )
        )
        adset_id = await client.create_adset(
            campaigns.adset_spec(
                episode=episode,
                trajectory=traj,
                render=render,
                brief=brief,
                account=account,
                campaign_id=campaign_id,
                daily_budget_usd=per_ad,
            )
        )
        ad_id = await client.create_ad(
            campaigns.ad_spec(
                episode=episode,
                trajectory=traj,
                render=render,
                adset_id=adset_id,
                creative_id=creative.creative_id,
            )
        )
        render.meta_ad_id = ad_id
        render.meta_adset_id = adset_id
        render.meta_creative_id = creative.creative_id
        render.effective_object_story_id = creative.effective_object_story_id
        render.status = RenderStatus.pending_review.value
        render.phase = "screening"
        render.daily_budget_usd = per_ad
        render.shipped_at = now
        result.shipped_render_ids.append(render.id)

    if traj.batch_id:
        batch = await session.get(Batch, traj.batch_id)
        if batch is not None and batch.state in (
            BatchState.proposed.value,
            BatchState.approved.value,
            BatchState.shipping.value,
        ):
            batch.state = BatchState.in_review.value
            batch.state_changed_at = now
    await record_audit(
        session,
        actor_id=actor,
        action="trajectory.ship",
        object_type="trajectory",
        object_id=traj.id,
        after={
            "renders": [str(r) for r in result.shipped_render_ids],
            "campaign_id": campaign_id,
            "daily_budget_usd": per_ad,
        },
    )
    await session.flush()
    return result


async def ship_batch(
    session: AsyncSession,
    batch_id: UUID,
    *,
    client: MetaClient,
    storage: Storage,
    cfg: AppConfig,
    actor: str,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> list[ShipResult]:
    """Ship every `run`-labeled trajectory in a batch. Others are left alone."""
    batch = await session.get(Batch, batch_id)
    if batch is None:
        raise NotFoundError(f"batch {batch_id} not found")
    rows = await session.execute(
        select(Trajectory.id)
        .join(Review, Review.trajectory_id == Trajectory.id)
        .where(Trajectory.batch_id == batch_id, Review.label == ReviewLabel.run.value)
        .order_by(Trajectory.attempt_index)
    )
    results: list[ShipResult] = []
    for (tid,) in rows.all():
        try:
            results.append(
                await ship_trajectory(
                    session,
                    tid,
                    client=client,
                    storage=storage,
                    cfg=cfg,
                    actor=actor,
                    settings=settings,
                    now=now,
                )
            )
        except SafetyError as e:
            # one refused idea must not abort the batch; the refusal is recorded and visible
            results.append(ShipResult(trajectory_id=tid, skipped=[str(e)]))
            await record_audit(
                session,
                actor_id=actor,
                action="trajectory.ship_refused",
                object_type="trajectory",
                object_id=tid,
                after={"reason": str(e)},
            )
    if results and batch.state in (BatchState.proposed.value, BatchState.approved.value):
        batch.state = BatchState.in_review.value
    return results
