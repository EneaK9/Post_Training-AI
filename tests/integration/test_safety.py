"""Safety tests (plan Phase 7): the kill switch blocks shipping at the API and the job level, and
concurrent ship calls cannot overshoot the episode cap because the governor locks the episode row.
The dry-run spy test lives in test_episode_lifecycle."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from helpers.episode_stack import START, Stack, dt
from outlier_ai.api.app import create_app
from outlier_ai.core.auth import create_user
from outlier_ai.core.errors import SafetyError
from outlier_ai.episodes.controller import tick_episode
from outlier_ai.meta.factory import default_latent
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.meta.ship import ship_batch, ship_trajectory
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.ops import AuditLog
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_ai.synthetic.seed import FAKE_ACCOUNT_ID
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import Role

pytestmark = pytest.mark.integration


async def _operator(db_session: AsyncSession) -> AsyncClient:
    email = f"operator-{uuid4().hex[:6]}@test.local"
    await create_user(db_session, email, "password123", Role.operator)
    await db_session.commit()
    client = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    r = await client.post("/api/auth/login", json={"email": email, "password": "password123"})
    assert r.status_code == 200, r.text
    return client


async def _non_draft(session: AsyncSession, episode_id) -> int:
    return (
        await session.execute(
            select(func.count(Render.id))
            .join(Trajectory, Trajectory.id == Render.trajectory_id)
            .where(Trajectory.episode_id == episode_id, Render.status != "draft")
        )
    ).scalar_one()


async def test_kill_switch_blocks_api_and_job_level(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    st = await Stack(db_session, app_config, tmp_path).setup(n=20, s=7)
    result = await st.generate_and_approve(k=2, renders=1, now=dt(START))
    assert result.batch_id is not None and len(result.queued) >= 1
    operator = await _operator(db_session)

    off = await operator.put(
        "/api/meta/kill_switch", json={"shipping_enabled": False, "reason": "safety test"}
    )
    assert off.status_code == 200 and off.json()["shipping_enabled"] is False

    # API level: the ship endpoint is refused and nothing moves
    r = await operator.post(f"/api/episodes/{st.episode.id}/ship")
    assert r.status_code == 403, r.text
    assert "kill switch" in r.json()["detail"]
    assert await _non_draft(db_session, st.episode.id) == 0

    # job level: ship_batch records refusals, tick_episode ships nothing
    results = await ship_batch(
        db_session,
        result.batch_id,
        client=st.client,
        storage=st.storage,
        cfg=app_config,
        actor="worker",
        now=dt(START),
    )
    await db_session.commit()
    assert results and all(not r.shipped_render_ids for r in results)
    assert await _non_draft(db_session, st.episode.id) == 0
    refused = (
        await db_session.execute(
            select(func.count(AuditLog.id)).where(AuditLog.action == "trajectory.ship_refused")
        )
    ).scalar_one()
    assert refused >= 1
    report = await tick_episode(
        db_session,
        st.episode,
        client=st.client,
        cfg=app_config,
        now=dt(START),
        storage=st.storage,
    )
    await db_session.commit()
    assert report is not None
    assert await _non_draft(db_session, st.episode.id) == 0
    with pytest.raises(SafetyError, match="kill switch"):
        await ship_trajectory(
            db_session,
            result.queued[0].trajectory.id,
            client=st.client,
            storage=st.storage,
            cfg=app_config,
            actor="test",
            now=dt(START),
        )

    # flipping it back on lets the same batch ship
    on = await operator.put("/api/meta/kill_switch", json={"shipping_enabled": True})
    assert on.status_code == 200
    r = await operator.post(f"/api/episodes/{st.episode.id}/ship")
    assert r.status_code == 200, r.text
    assert await _non_draft(db_session, st.episode.id) >= 1


async def test_concurrent_ship_calls_respect_the_episode_cap(
    db_engine: AsyncEngine, db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    st = await Stack(db_session, app_config, tmp_path).setup(n=20, s=8)
    result = await st.generate_and_approve(k=3, renders=1, now=dt(START))
    ids = [i.trajectory.id for i in result.queued]
    assert len(ids) >= 2
    per_idea = app_config.episode.screening_budget_per_ad_usd * app_config.episode.screening_days
    episode = await db_session.get(SearchEpisode, st.episode.id)
    assert episode is not None
    episode.budget_cap = per_idea * 1.5  # exactly one idea fits
    await db_session.commit()

    factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async def ship(tid):
        async with factory() as session:
            client = FakeMetaClient(
                session, FAKE_ACCOUNT_ID, default_latent(), seed=1, start_day=START
            )
            try:
                out = await ship_trajectory(
                    session,
                    tid,
                    client=client,
                    storage=st.storage,
                    cfg=app_config,
                    actor="race",
                    now=dt(START),
                )
                await session.commit()
                return out
            except SafetyError as e:
                await session.rollback()
                return e

    outcomes = await asyncio.gather(ship(ids[0]), ship(ids[1]))
    errors = [o for o in outcomes if isinstance(o, SafetyError)]
    successes = [o for o in outcomes if not isinstance(o, SafetyError)]
    assert len(successes) == 1 and len(errors) == 1, outcomes
    assert "episode cap" in str(errors[0])
    assert await _non_draft(db_session, episode.id) == 1
