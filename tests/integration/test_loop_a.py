from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.api.app import create_app
from outlier_ai.core.auth import create_user
from outlier_ai.core.embeddings import HashEmbedder
from outlier_ai.core.storage import LocalStorage
from outlier_ai.episodes.loop_a import LoopARunner
from outlier_ai.generation.backends.fake_backend import FakeBackend
from outlier_ai.generation.service import GenerationService
from outlier_ai.generation.verifier import HeuristicVerifier, load_card_refs
from outlier_ai.images import get_image_backend
from outlier_ai.meta.factory import default_latent
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import Batch, SearchEpisode
from outlier_ai.models.meta import AdAccount
from outlier_ai.models.trajectories import Render, Review, Trajectory
from outlier_ai.outlier.recompute import recompute_all
from outlier_ai.reward.cold import HeuristicColdRewardModel
from outlier_ai.synthetic.seed import FAKE_ACCOUNT_ID, seed
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import ImageMode, Role

pytestmark = pytest.mark.integration


async def _prep(session: AsyncSession, cfg: AppConfig, tmp_path, *, n=60, s=4):
    await seed(session, n_briefs=3, n_trajectories=n, seed=s, days_back=90, config=cfg)
    await recompute_all(session, cfg)
    await session.commit()
    account = (
        await session.execute(select(AdAccount).where(AdAccount.meta_account_id == FAKE_ACCOUNT_ID))
    ).scalar_one()
    brief = (await session.execute(select(Brief).order_by(Brief.created_at))).scalars().first()
    assert brief is not None
    cards = await load_card_refs(session)
    service = GenerationService(
        backend=FakeBackend(seed=s, policy="archive"),
        verifier=HeuristicVerifier(cards),
        reward_model=HeuristicColdRewardModel(),
        image_backend=get_image_backend(ImageMode.brand_assets),
        storage=LocalStorage(tmp_path),
        embedder=HashEmbedder(cfg.archive.embedding_dims),
        cfg=cfg,
    )
    client = FakeMetaClient(
        session,
        FAKE_ACCOUNT_ID,
        default_latent(),
        seed=s,
        rejection_rate=0.0,
        start_day=date(2026, 9, 1),
    )
    return account, brief, service, client


def test_archive_policy_scores_history():
    prompt = (
        "--- history:aaaa0001 [elite] 5 days ago\nCards: contrarian + specific-numbers | labeled common\nResult: tier 2 (3.40x median)\n"
        "--- history:bbbb0002 [rare] 9 days ago\nCards: human-desires\nResult: screen fail (0.90x CTR median)\n"
        "--- history:cccc0003 [elite] 2 days ago\nCards: objection-flip + meme-format\nResult: tier 0 (0.70x median)\n"
    )
    scores = FakeBackend.strategy_scores(
        prompt, ["contrarian", "human-desires", "objection-flip", "process-moats"]
    )
    assert scores["contrarian"] > scores["process-moats"] > scores["human-desires"]
    assert scores["objection-flip"] < scores["process-moats"]


async def test_simulate_runs_an_episode_to_a_terminal_state(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    cfg = app_config.model_copy(deep=True)
    cfg.episode.renders_per_idea = 1
    cfg.episode.max_batches = 2
    cfg.generation.k = 4
    cfg.episode.ideas_per_batch = 2
    account, brief, service, client = await _prep(db_session, cfg, tmp_path)
    episode = SearchEpisode(
        brief_id=brief.id,
        ad_account_id=account.id,
        backend="fake",
        budget_cap=1500.0,
        created_by="test",
    )
    db_session.add(episode)
    await db_session.commit()
    runner = LoopARunner(
        session=db_session,
        cfg=cfg,
        service=service,
        client=client,
        storage=LocalStorage(tmp_path),
        approval="top_rm",
        actor="test",
    )
    result = await runner.simulate(episode, max_days=60)
    await db_session.commit()
    assert result.status in ("outlier_found", "budget_exhausted"), result.events[-5:]
    assert result.batches >= 1 and result.ideas_shipped >= 2 and result.days >= 15
    assert result.spent > 0
    n_batches = int(
        (
            await db_session.execute(
                select(func.count()).select_from(Batch).where(Batch.episode_id == episode.id)
            )
        ).scalar_one()
    )
    assert n_batches == result.batches
    approved = int(
        (
            await db_session.execute(
                select(func.count())
                .select_from(Review)
                .join(Trajectory, Trajectory.id == Review.trajectory_id)
                .where(Trajectory.episode_id == episode.id, Review.label == "run")
            )
        ).scalar_one()
    )
    assert approved == 2 * n_batches
    # every shipped render has a terminal state and spend was recorded
    live = (
        (
            await db_session.execute(
                select(Render)
                .join(Trajectory)
                .where(
                    Trajectory.episode_id == episode.id,
                    Render.shipped_at.is_not(None),
                    Render.stopped_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    if result.status == "budget_exhausted":
        assert live == []


async def test_manual_step_generates_without_shipping(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    account, brief, service, client = await _prep(db_session, app_config, tmp_path, n=30, s=2)
    episode = SearchEpisode(
        brief_id=brief.id,
        ad_account_id=account.id,
        backend="fake",
        budget_cap=5000.0,
        created_by="test",
    )
    db_session.add(episode)
    await db_session.commit()
    runner = LoopARunner(
        session=db_session,
        cfg=app_config,
        service=service,
        client=client,
        storage=LocalStorage(tmp_path),
        approval="manual",
        actor="test",
    )
    report = await runner.step(episode)
    assert (
        report.generated == app_config.generation.k
        and report.approved == 0
        and report.shipped_renders == 0
    )
    again = await runner.step(episode)
    assert again.skipped_reason.startswith("batch 0 is proposed")


async def test_api_step_and_simulate(
    db_session: AsyncSession, app_config: AppConfig, tmp_path, monkeypatch
):
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    from outlier_ai.core.settings import reset_settings_cache

    reset_settings_cache()
    account, brief, _, _ = await _prep(db_session, app_config, tmp_path, n=30, s=6)
    await create_user(db_session, "op@test.local", "password123", Role.operator)
    await db_session.commit()
    client = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    assert (
        await client.post(
            "/api/auth/login", json={"email": "op@test.local", "password": "password123"}
        )
    ).status_code == 200
    ep = (
        await client.post(
            "/api/episodes", json={"brief_id": str(brief.id), "backend": "fake", "budget_cap": 3000}
        )
    ).json()
    stepped = await client.post(f"/api/episodes/{ep['id']}/step?approval=top_rm&k=4&no_llm=true")
    assert stepped.status_code == 200, stepped.text
    body = stepped.json()
    assert (
        len(body["batches"]) == 1 and body["live_ads"] >= 1 and body["batches"][0]["prompt_trace"]
    )
    blocked = await client.post(f"/api/episodes/{ep['id']}/step?approval=top_rm&no_llm=true")
    assert blocked.status_code == 409
    sim = await client.post(f"/api/episodes/{ep['id']}/simulate?days=9")
    assert sim.status_code == 200, sim.text
    assert sim.json()["spent"] > 0
    stats_resp = await client.get("/api/stats/loop_a")
    assert stats_resp.status_code == 200, stats_resp.text
    stats = stats_resp.json()
    assert "totals" in stats, list(stats)
    assert (
        stats["totals"]["generated"] > 0 and "by_typicality" in stats and "by_combination" in stats
    )
    arch = (await client.get("/api/stats/architecture")).json()
    assert {n["id"] for n in arch["nodes"]} >= {"archive", "backend", "verifier", "meta", "trainer"}
    assert (await client.get("/api/stats/verifier")).json()["tagged"] > 0
    assert "loop_b_ready" in (await client.get("/api/stats/reward_model")).json()
    # needs_reauth blocks Run
    account.status = "needs_reauth"
    await db_session.commit()
    denied = await client.post(f"/api/episodes/{ep['id']}/step?approval=top_rm&no_llm=true")
    assert denied.status_code == 409 and "reconnect" in denied.json()["detail"]
