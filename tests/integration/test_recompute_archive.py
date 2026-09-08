import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.novelty import check_novelty
from outlier_ai.archive.sampling import sample_archive
from outlier_ai.generation.assemble import build_prompt
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import Combination
from outlier_ai.models.ops import HoldoutCampaign
from outlier_ai.models.trajectories import Outcome, Render, ScreeningStats, Trajectory
from outlier_ai.outlier.recompute import recompute_all
from outlier_ai.synthetic.seed import seed
from outlier_schemas.config import AppConfig

pytestmark = pytest.mark.integration


async def _seeded(session: AsyncSession, cfg: AppConfig, n: int = 120, s: int = 4):
    await seed(session, n_briefs=4, n_trajectories=n, seed=s, days_back=100, config=cfg)
    summaries = await recompute_all(session, cfg)
    await session.commit()
    return summaries


async def test_recompute_derives_stats_outcomes_and_tiers(
    db_session: AsyncSession, app_config: AppConfig
):
    [summary] = await _seeded(db_session, app_config)
    scaled = (
        await db_session.execute(
            select(func.count()).select_from(Render).where(Render.status == "scaled")
        )
    ).scalar_one()
    assert summary.outcome_rows == scaled > 0
    assert summary.screening_rows == summary.renders
    stats = (await db_session.execute(select(ScreeningStats))).scalars().all()
    assert all(s.config_hash == app_config.hash for s in stats)
    outcomes = (await db_session.execute(select(Outcome))).scalars().all()
    assert all(0 <= o.outlier_tier <= 3 for o in outcomes)
    assert all(o.ratio_lower_bound <= o.ratio + 1e-9 for o in outcomes)
    tiered = (
        (await db_session.execute(select(Trajectory).where(Trajectory.outlier_tier.is_not(None))))
        .scalars()
        .all()
    )
    assert len(tiered) == summary.trajectories_tiered > 0
    assert all(t.outcome_render_id is not None for t in tiered)
    # passing screening must be what produced scale rows
    passed = (
        await db_session.execute(
            select(func.count()).select_from(ScreeningStats).where(ScreeningStats.passed.is_(True))
        )
    ).scalar_one()
    # Historical seed decided scaling against a fixed profile median; the system's data-driven
    # median is lower here, so everything that scaled must pass and possibly more.
    assert passed >= scaled
    scaled_pass = (
        await db_session.execute(
            select(func.count())
            .select_from(ScreeningStats)
            .join(Render, Render.id == ScreeningStats.render_id)
            .where(Render.status == "scaled", ScreeningStats.passed.is_(True))
        )
    ).scalar_one()
    assert scaled_pass >= 0.9 * scaled
    # combination tier counts add up to tiered trajectories
    combos = (await db_session.execute(select(Combination))).scalars().all()
    assert sum(sum(c.tier_counts.values()) for c in combos) == len(tiered)
    assert all(c.niche_key for c in combos)


async def test_threshold_change_recomputes_tiers(db_session: AsyncSession, app_config: AppConfig):
    await _seeded(db_session, app_config, n=150, s=9)
    before = (
        await db_session.execute(
            select(func.count()).select_from(Trajectory).where(Trajectory.outlier_tier >= 2)
        )
    ).scalar_one()
    strict = app_config.model_copy(deep=True)
    strict.outlier.tier_multiples.tier2 = 50.0
    strict.outlier.tier_multiples.tier3 = 100.0
    await recompute_all(db_session, strict)
    await db_session.commit()
    after = (
        await db_session.execute(
            select(func.count()).select_from(Trajectory).where(Trajectory.outlier_tier >= 2)
        )
    ).scalar_one()
    assert after == 0
    assert after <= before
    hashes = {o.config_hash for o in (await db_session.execute(select(Outcome))).scalars().all()}
    assert hashes == {strict.hash}


async def test_archive_sample_shape_and_holdout(db_session: AsyncSession, app_config: AppConfig):
    await _seeded(db_session, app_config, n=160, s=6)
    brief = (await db_session.execute(select(Brief).order_by(Brief.created_at))).scalars().first()
    assert brief is not None and brief.embedding is not None
    sample = await sample_archive(db_session, brief.id, app_config)
    assert sample.items, "seeded history should produce a sample"
    elites = sample.elites
    assert 0 < len(elites) <= app_config.archive.n_elite
    assert len({i.niche for i in elites}) == len(elites), "one elite per niche"
    assert len(sample.rares) <= app_config.archive.n_rare
    assert all((i.tier or 0) >= 2 for i in sample.tier2_recent)
    assert len({i.trajectory_id for i in sample.items}) == len(sample.items)
    assert set(sample.ref_map) >= {i.ref for i in sample.items}
    assert all(s.status == "confirmed" for i in sample.items for s in i.signals)
    # ranking: elites come tier-desc
    tiers = [(i.tier if i.tier is not None else -1) for i in elites]
    assert tiers == sorted(tiers, reverse=True)

    # hold out one campaign and confirm it disappears
    victim = sample.items[0]
    traj = await db_session.get(Trajectory, victim.trajectory_id)
    assert traj is not None and traj.campaign_id
    db_session.add(HoldoutCampaign(campaign_id=traj.campaign_id, added_by="test", reason="eval"))
    await db_session.commit()
    again = await sample_archive(db_session, brief.id, app_config)
    assert again.excluded_holdout > 0
    held_ids = {
        t.id
        for t in (
            await db_session.execute(
                select(Trajectory).where(Trajectory.campaign_id == traj.campaign_id)
            )
        )
        .scalars()
        .all()
    }
    assert not ({i.trajectory_id for i in again.items} & held_ids)


async def test_novelty_rejects_duplicates_and_accepts_new(
    db_session: AsyncSession, app_config: AppConfig
):
    await _seeded(db_session, app_config, n=80, s=3)
    shipped = (
        (
            await db_session.execute(
                select(Trajectory)
                .join(Render, Render.trajectory_id == Trajectory.id)
                .where(
                    Render.shipped_at.is_not(None), Trajectory.combo_angle_embedding.is_not(None)
                )
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    assert shipped is not None and shipped.combo_angle_embedding is not None
    dup = await check_novelty(
        db_session,
        brief_id=shipped.brief_id,
        verified_card_ids=shipped.verified_card_ids,
        embedding=list(shipped.combo_angle_embedding),
        cfg=app_config,
    )
    assert (
        dup.rejected
        and dup.reason == "duplicate_of_shipped"
        and dup.nearest_trajectory_id == shipped.id
    )
    assert dup.note.startswith("novelty_reject:duplicate_of_shipped")
    rng = np.random.default_rng(0)
    random_vec = rng.normal(size=app_config.archive.embedding_dims)
    random_vec /= np.linalg.norm(random_vec)
    novel = await check_novelty(
        db_session,
        brief_id=shipped.brief_id,
        verified_card_ids=shipped.verified_card_ids,
        embedding=random_vec.tolist(),
        cfg=app_config,
    )
    assert not novel.rejected and novel.reason == "novel" and novel.note == ""


async def test_build_prompt_end_to_end(db_session: AsyncSession, app_config: AppConfig):
    await _seeded(db_session, app_config, n=60, s=2)
    brief = (await db_session.execute(select(Brief).order_by(Brief.created_at))).scalars().first()
    assert brief is not None
    prompt, trace, sample, slug_map = await build_prompt(db_session, brief.id, app_config, k=4)
    assert "COLUMN 3: HISTORY" in prompt and brief.company in prompt
    assert trace.k == 4 and trace.token_estimate <= app_config.generation.max_prompt_tokens
    assert len(slug_map) == 28
    assert all(
        f"history:{i.ref}" in prompt
        for i in sample.items
        if i.ref in {r["ref"] for r in trace.archive_refs}
    )
