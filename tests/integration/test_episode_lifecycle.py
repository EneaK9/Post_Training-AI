"""Whole-episode walk through the fake evaluator plus the section 12 scenario rows that Phase 4 owns."""

from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from helpers.episode_stack import START, Stack, _count, dt
from outlier_ai.core.errors import SafetyError
from outlier_ai.core.settings import Settings
from outlier_ai.episodes.budget import assert_can_ship, check_budget
from outlier_ai.episodes.controller import can_start_next_batch
from outlier_ai.jobs.handlers import run_sync_comments, run_sync_insights
from outlier_ai.meta.errors import MetaAuthError
from outlier_ai.meta.factory import client_for_account
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.meta.ship import ship_trajectory
from outlier_ai.models.episodes import Batch, SearchEpisode
from outlier_ai.models.meta import AdAccount, Comment, DailyInsight
from outlier_ai.models.ops import AuditLog, KillSwitch
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Outcome, Render, Review, Trajectory
from outlier_ai.synthetic.latent import AdTruth
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BackendKind
from outlier_schemas.models import GenerationRequest

pytestmark = pytest.mark.integration


async def test_full_episode_screening_to_measured(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    st = await Stack(db_session, app_config, tmp_path).setup()
    result = await st.generate_and_approve(k=4, renders=1, now=dt(START))
    assert len(result.queued) == 4
    shipped = await st.ship(result.batch_id, dt(START))
    assert sum(len(r.shipped_render_ids) for r in shipped) == 4
    await db_session.refresh(st.episode)
    assert st.episode.campaign_id and st.episode.campaign_id.startswith("camp_")
    renders = (
        (
            await db_session.execute(
                select(Render).join(Trajectory).where(Trajectory.episode_id == st.episode.id)
            )
        )
        .scalars()
        .all()
    )
    assert all(
        r.status == "pending_review"
        and r.meta_ad_id
        and r.effective_object_story_id
        and r.daily_budget_usd == 20
        for r in renders
    )
    batch = await db_session.get(Batch, result.batch_id)
    assert batch is not None
    assert batch.state == "in_review"
    assert await _count(db_session, AuditLog, AuditLog.action == "trajectory.ship") == 4
    ok, reason = await can_start_next_batch(db_session, st.episode, app_config)
    assert not ok and "in_review" in reason

    # day 2: reviews resolve
    today = await st.advance(1)
    report = await st.sync_and_tick(today)
    assert any("approved" in e for e in report.events)
    await db_session.refresh(batch)
    assert batch.state == "screening"

    # screening window
    today = await st.advance(7)
    report = await st.sync_and_tick(today)
    rows = await _count(
        db_session, DailyInsight, DailyInsight.render_id.in_([r.id for r in renders])
    )
    assert rows == 4 * 7
    for r in renders:
        await db_session.refresh(r)
    statuses = {r.status for r in renders}
    assert statuses <= {"scaled", "stopped"} and "pending_review" not in statuses
    assert any("screening" in e for e in report.events)
    await db_session.refresh(batch)
    assert batch.state in ("scaling", "measured")

    # sync is idempotent
    before = await _count(db_session, DailyInsight)
    await run_sync_insights(
        db_session,
        app_config,
        account_id=st.account.id,
        today=today + timedelta(days=1),
        clients={st.account.id: st.client},
    )
    assert await _count(db_session, DailyInsight) == before

    if batch.state == "scaling":
        ok, reason = await can_start_next_batch(db_session, st.episode, app_config)
        assert ok, reason  # overlapping batches: screening resolved
        today = await st.advance(7)
        report = await st.sync_and_tick(today)
        await db_session.refresh(batch)
        assert batch.state == "measured", report.events
        scaled = [r for r in renders if r.status == "scaled"]
        assert await _count(
            db_session, Outcome, Outcome.render_id.in_([r.id for r in scaled])
        ) == len(scaled)
        assert all(r.stopped_at is not None for r in scaled)


async def test_all_renders_rejected_keeps_outcome_null_and_spend_zero(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    st = await Stack(db_session, app_config, tmp_path, rejection_rate=1.0).setup(n=30, s=6)
    result = await st.generate_and_approve(k=2, renders=2, now=dt(START))
    await st.ship(result.batch_id, dt(START))
    today = await st.advance(1)
    report = await st.sync_and_tick(today)
    renders = (
        (
            await db_session.execute(
                select(Render).join(Trajectory).where(Trajectory.episode_id == st.episode.id)
            )
        )
        .scalars()
        .all()
    )
    assert all(r.status == "rejected" and r.rejection_reason for r in renders)
    assert any("rejected_all" in e for e in report.events)
    assert await _count(db_session, AuditLog, AuditLog.action == "trajectory.rejected_all") == 2
    trajs = (
        (await db_session.execute(select(Trajectory).where(Trajectory.episode_id == st.episode.id)))
        .scalars()
        .all()
    )
    assert all(t.outlier_tier is None for t in trajs)
    await db_session.refresh(st.episode)
    assert st.episode.spent == 0
    batch = await db_session.get(Batch, result.batch_id)
    assert batch is not None
    assert batch.state == "measured"
    ok, _ = await can_start_next_batch(db_session, st.episode, app_config)
    assert ok  # the episode requests the next batch without counting the spend


async def test_outlier_found_stops_other_ads(
    db_session: AsyncSession, app_config: AppConfig, tmp_path, monkeypatch
):
    st = await Stack(db_session, app_config, tmp_path).setup(n=80, s=9)
    strong = AdTruth(roas_multiple=5.0, ctr_multiple=2.6, polarity=0.3)
    weak = AdTruth(roas_multiple=0.6, ctr_multiple=0.7, polarity=0.5)
    calls = {"n": 0}

    async def truth(name):  # first ad strong, the rest weak
        calls["n"] += 1
        return strong if calls["n"] == 1 else weak

    monkeypatch.setattr(st.client, "_truth_for_ad_name", truth)
    result = await st.generate_and_approve(k=3, renders=1, now=dt(START))
    await st.ship(result.batch_id, dt(START))
    today = await st.advance(1)
    await st.sync_and_tick(today)
    today = await st.advance(7)
    report = await st.sync_and_tick(today)
    assert any("passed screening" in e for e in report.events)
    today = await st.advance(7)
    report = await st.sync_and_tick(today)
    await db_session.refresh(st.episode)
    assert st.episode.status == "outlier_found", report.events
    winner = (
        (
            await db_session.execute(
                select(Trajectory).where(
                    Trajectory.episode_id == st.episode.id, Trajectory.outlier_tier >= 2
                )
            )
        )
        .scalars()
        .first()
    )
    assert winner is not None
    others = (
        (
            await db_session.execute(
                select(Render)
                .join(Trajectory)
                .where(Trajectory.episode_id == st.episode.id, Trajectory.id != winner.id)
            )
        )
        .scalars()
        .all()
    )
    assert all(r.stopped_at is not None for r in others)
    assert await _count(db_session, AuditLog, AuditLog.action == "episode.outlier_found") == 1
    with pytest.raises(SafetyError):
        await assert_can_ship(
            db_session, episode=st.episode, account=st.account, add_daily_usd=20, cfg=app_config
        )


async def test_budget_exhausted_and_governor_caps(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    st = await Stack(db_session, app_config, tmp_path).setup(n=30, s=2)
    st.episode.budget_cap = 700.0  # exactly one batch of 4 ads x $20 x 7 days = 560
    await db_session.commit()
    result = await st.generate_and_approve(k=4, renders=1, now=dt(START))
    await st.ship(result.batch_id, dt(START))
    ok, _reason = await can_start_next_batch(db_session, st.episode, app_config)
    assert not ok
    today = await st.advance(1)
    await st.sync_and_tick(today)
    today = await st.advance(7)
    report = await st.sync_and_tick(today)
    await db_session.refresh(st.episode)
    if st.episode.status == "searching":  # a render scaled; finish its window
        today = await st.advance(7)
        report = await st.sync_and_tick(today)
        await db_session.refresh(st.episode)
    assert st.episode.status == "budget_exhausted", report.events
    assert st.episode.spent > 0

    # governor: daily cap
    ep2 = SearchEpisode(
        brief_id=st.brief.id,
        ad_account_id=st.account.id,
        backend="fake",
        budget_cap=100000.0,
        created_by="test",
    )
    db_session.add(ep2)
    await db_session.commit()
    d = await check_budget(
        db_session,
        episode=ep2,
        account=st.account,
        add_daily_usd=app_config.meta.daily_account_cap_usd + 1,
        cfg=app_config,
    )
    assert not d.allowed and "daily cap" in d.reason
    # governor: kill switch
    ks = await db_session.get(KillSwitch, 1)
    assert ks is not None
    ks.shipping_enabled = False
    await db_session.commit()
    d = await check_budget(
        db_session, episode=ep2, account=st.account, add_daily_usd=20, cfg=app_config
    )
    assert not d.allowed and "kill switch" in d.reason
    ks.shipping_enabled = True
    await db_session.commit()


async def test_dry_run_blocks_real_accounts_and_never_builds_the_client(
    db_session: AsyncSession, app_config: AppConfig, tmp_path, monkeypatch
):
    st = await Stack(db_session, app_config, tmp_path).setup(n=20, s=1)
    real = AdAccount(
        name="Real",
        meta_account_id="act_real_1",
        status="active",
        attribution_setting="7d_click_1d_view",
        api_version="v26.0",
        is_fake=False,
        access_token_enc="enc",
    )
    db_session.add(real)
    await db_session.commit()

    import outlier_ai.meta.client as client_mod

    def boom(*a, **k):
        raise AssertionError("real client must not be constructed")

    monkeypatch.setattr(client_mod.MetaGraphClient, "__init__", boom)
    settings = Settings(_env_file=None, dry_run=True)  # type: ignore[call-arg]
    with pytest.raises(SafetyError, match="DRY_RUN"):
        await client_for_account(db_session, real, app_config, settings=settings)
    ep = SearchEpisode(
        brief_id=st.brief.id,
        ad_account_id=real.id,
        backend="fake",
        budget_cap=1000.0,
        created_by="test",
    )
    db_session.add(ep)
    await db_session.commit()
    d = await check_budget(
        db_session, episode=ep, account=real, add_daily_usd=20, cfg=app_config, settings=settings
    )
    assert not d.allowed and "dry run" in d.reason
    # fake accounts are fine in dry run
    fake_client = await client_for_account(db_session, st.account, app_config, settings=settings)
    assert isinstance(fake_client, FakeMetaClient)


async def test_ship_requires_run_label_and_clean_preship(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    st = await Stack(db_session, app_config, tmp_path).setup(n=20, s=5)
    result = await st.service.generate_batch(
        db_session,
        GenerationRequest(
            brief_id=st.brief.id,
            episode_id=st.episode.id,
            backend=BackendKind.fake,
            k=2,
            renders_per_idea=1,
        ),
        now=dt(START),
    )
    await db_session.commit()
    tid = result.queued[0].trajectory.id
    with pytest.raises(SafetyError, match="run"):
        await ship_trajectory(
            db_session, tid, client=st.client, storage=st.storage, cfg=app_config, actor="test"
        )
    traj = await db_session.get(Trajectory, tid)
    assert traj is not None
    traj.preship = {
        **(traj.preship or {}),
        "policy_ok": False,
        "policy_flags": ["unrealistic_outcomes: miracle"],
    }
    db_session.add(Review(trajectory_id=tid, label="run", reviewer_id="e", note=""))
    await db_session.commit()
    with pytest.raises(SafetyError, match="pre-ship"):
        await ship_trajectory(
            db_session, tid, client=st.client, storage=st.storage, cfg=app_config, actor="test"
        )


async def test_auth_error_marks_account_needs_reauth(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    st = await Stack(db_session, app_config, tmp_path).setup(n=20, s=7)
    result = await st.generate_and_approve(k=1, renders=1, now=dt(START))
    await st.ship(result.batch_id, dt(START))
    today = await st.advance(2)
    st.client.fail_next(MetaAuthError)
    reports = await run_sync_insights(
        db_session,
        app_config,
        account_id=st.account.id,
        today=today + timedelta(days=1),
        clients={st.account.id: st.client},
    )
    assert reports[0].marked_needs_reauth
    await db_session.commit()
    await db_session.refresh(st.account)
    assert st.account.status == "needs_reauth" and st.account.last_error
    assert await _count(db_session, AuditLog, AuditLog.action == "account.needs_reauth") == 1
    with pytest.raises(SafetyError):
        await assert_can_ship(
            db_session, episode=st.episode, account=st.account, add_daily_usd=20, cfg=app_config
        )


async def test_comment_sync_strips_pii_and_proposes_signals(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    st = await Stack(db_session, app_config, tmp_path, fake_seed=11).setup(n=30, s=3)
    result = await st.generate_and_approve(k=3, renders=1, now=dt(START))
    await st.ship(result.batch_id, dt(START))
    today = await st.advance(1)
    await st.sync_and_tick(today)
    today = await st.advance(6)
    reports = await run_sync_comments(
        db_session, app_config, account_id=st.account.id, clients={st.account.id: st.client}
    )
    await db_session.commit()
    assert reports[0].comments_added > 0, reports[0]
    render_ids = [
        r.id
        for r in (
            await db_session.execute(
                select(Render).join(Trajectory).where(Trajectory.episode_id == st.episode.id)
            )
        )
        .scalars()
        .all()
    ]
    comments = (
        (await db_session.execute(select(Comment).where(Comment.render_id.in_(render_ids))))
        .scalars()
        .all()
    )
    assert comments and not any("@example.com" in c.text or "555-" in c.text for c in comments)
    assert all(len(c.commenter_hash) == 64 for c in comments)
    proposed = await _count(
        db_session,
        Signal,
        Signal.trajectory_id.in_(
            select(Trajectory.id).where(Trajectory.episode_id == st.episode.id)
        ),
        Signal.status == "proposed",
    )
    assert proposed == reports[0].signals_proposed >= 1
    # second sync adds nothing new
    again = await run_sync_comments(
        db_session, app_config, account_id=st.account.id, clients={st.account.id: st.client}
    )
    assert again[0].comments_added == 0
