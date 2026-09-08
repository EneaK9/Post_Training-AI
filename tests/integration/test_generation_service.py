import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.embeddings import HashEmbedder
from outlier_ai.core.storage import LocalStorage
from outlier_ai.generation.backends.fake_backend import FakeBackend
from outlier_ai.generation.service import GenerationService
from outlier_ai.generation.verifier import HeuristicVerifier, load_card_refs
from outlier_ai.images import get_image_backend
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import Combination
from outlier_ai.models.episodes import Batch, SearchEpisode
from outlier_ai.models.trajectories import Render, Review, Trajectory
from outlier_ai.outlier.recompute import recompute_all
from outlier_ai.reward.cold import HeuristicColdRewardModel
from outlier_ai.synthetic.seed import seed
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BackendKind, ImageMode
from outlier_schemas.models import GenerationRequest

pytestmark = pytest.mark.integration


async def _service(
    session: AsyncSession, cfg: AppConfig, tmp_path, **backend_kw
) -> GenerationService:
    cards = await load_card_refs(session)
    return GenerationService(
        backend=FakeBackend(**backend_kw),
        verifier=HeuristicVerifier(cards),
        reward_model=HeuristicColdRewardModel(),
        image_backend=get_image_backend(ImageMode.brand_assets),
        storage=LocalStorage(tmp_path),
        embedder=HashEmbedder(cfg.archive.embedding_dims),
        cfg=cfg,
    )


async def _count(session, model, *where) -> int:
    return int(
        (await session.execute(select(func.count()).select_from(model).where(*where))).scalar_one()
    )


async def test_generate_batch_persists_ideas_renders_and_queue(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    await seed(db_session, n_briefs=3, n_trajectories=60, seed=5, days_back=90, config=app_config)
    await recompute_all(db_session, app_config)
    await db_session.commit()
    brief = (await db_session.execute(select(Brief).order_by(Brief.created_at))).scalars().first()
    assert brief is not None
    before_traj = await _count(db_session, Trajectory)
    service = await _service(db_session, app_config, tmp_path, seed=2)
    result = await service.generate_batch(
        db_session,
        GenerationRequest(brief_id=brief.id, backend=BackendKind.fake, k=6, renders_per_idea=3),
    )
    await db_session.commit()

    assert len(result.ideas) == 6
    assert result.batch_id is None and result.episode_id is None
    assert await _count(db_session, Trajectory) == before_traj + 6
    for idea in result.queued:
        t = idea.trajectory
        assert t.format_ok and t.rm_score is not None
        assert t.rm_version is not None and t.rm_version.startswith("rm_cold")
        assert t.verified_card_ids and t.tag_source == "verifier"
        assert t.combo_angle_embedding is not None and t.config_hash == app_config.hash
        assert t.preship is not None and idea.preship is not None
        assert len(idea.renders) == 3 and all(
            r.image_uri and r.status == "draft" for r in idea.renders
        )
    assert result.queued, "the fake backend's ideas should reach the queue"
    for idea in result.rejected:
        review = (
            await db_session.execute(
                select(Review).where(Review.trajectory_id == idea.trajectory.id)
            )
        ).scalar_one()
        assert review.label == "skip" and review.reviewer_id == "system"
        assert not idea.renders
    renders = await _count(
        db_session, Render, Render.trajectory_id.in_([i.trajectory.id for i in result.queued])
    )
    assert renders == 3 * len(result.queued)
    combos = (await db_session.execute(select(Combination))).scalars().all()
    assert any(c.uses > 0 for c in combos)
    assert "Propose exactly 6 ideas" in result.prompt and result.trace.k == 6


async def test_generate_batch_attaches_to_episode_and_rejects_duplicates(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    await seed(db_session, n_briefs=2, n_trajectories=40, seed=8, days_back=90, config=app_config)
    await recompute_all(db_session, app_config)
    brief = (await db_session.execute(select(Brief).order_by(Brief.created_at))).scalars().first()
    assert brief is not None
    episode = SearchEpisode(
        brief_id=brief.id,
        ad_account_id=brief.ad_account_id,
        backend="fake",
        budget_cap=5000.0,
        created_by="test",
    )
    db_session.add(episode)
    await db_session.commit()

    service = await _service(db_session, app_config, tmp_path, seed=1)
    req = GenerationRequest(
        brief_id=brief.id, episode_id=episode.id, backend=BackendKind.fake, k=4, renders_per_idea=1
    )
    r1 = await service.generate_batch(db_session, req)
    await db_session.commit()
    assert r1.batch_id is not None
    batch = await db_session.get(Batch, r1.batch_id)
    assert batch is not None and batch.index == 0 and batch.prompt_trace["k"] == 4
    assert all(
        i.trajectory.episode_id == episode.id and i.trajectory.batch_id == batch.id
        for i in r1.ideas
    )

    # A permissive novelty threshold makes everything a duplicate of shipped history.
    loose = app_config.model_copy(deep=True)
    loose.archive.novelty_threshold = 2.0
    service2 = await _service(db_session, loose, tmp_path, seed=1)
    r2 = await service2.generate_batch(db_session, req)
    await db_session.commit()
    assert r2.batch_id is not None
    batch2 = await db_session.get(Batch, r2.batch_id)
    assert batch2 is not None and batch2.index == 1
    assert r2.queued == [] and all(i.status == "novelty_rejected" for i in r2.ideas)
    notes = [
        (await db_session.execute(select(Review).where(Review.trajectory_id == i.trajectory.id)))
        .scalar_one()
        .note
        for i in r2.ideas
    ]
    assert all(n.startswith("novelty_reject:duplicate_of_shipped") for n in notes)


async def test_malformed_output_is_recorded_not_queued(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    await seed(db_session, n_briefs=2, n_trajectories=20, seed=2, days_back=90, config=app_config)
    await db_session.commit()
    brief = (await db_session.execute(select(Brief).order_by(Brief.created_at))).scalars().first()
    assert brief is not None
    service = await _service(db_session, app_config, tmp_path, seed=4, malformed_rate=1.0)
    result = await service.generate_batch(
        db_session,
        GenerationRequest(brief_id=brief.id, backend=BackendKind.fake, k=3, renders_per_idea=1),
    )
    await db_session.commit()
    assert result.queued == []
    assert all(
        i.status == "format_rejected" and not i.trajectory.format_ok and i.trajectory.format_errors
        for i in result.ideas
    )
    assert all(
        i.trajectory.rm_score is not None and i.trajectory.rm_score < 0.3 for i in result.ideas
    )
