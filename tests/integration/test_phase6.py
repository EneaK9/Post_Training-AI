"""Phase 6: snapshot export, reward-model training, eval harness on fake arms, feedback loops,
and the training/eval/feedback API surface."""

from __future__ import annotations

import io
import uuid

import pyarrow.parquet as pq
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.api.app import create_app
from outlier_ai.archive.export import export_snapshot
from outlier_ai.core.auth import create_user
from outlier_ai.core.embeddings import HashEmbedder
from outlier_ai.core.storage import LocalStorage, key_from_uri
from outlier_ai.eval.harness import run_eval
from outlier_ai.generation.verifier import ClassifierVerifier, load_card_refs
from outlier_ai.jobs.feedback import (
    load_active_classifier,
    retrain_verifier,
    suggest_relations,
)
from outlier_ai.jobs.worker import run_handler
from outlier_ai.models.cards import Card, CardRelation
from outlier_ai.models.ml import ArchiveSnapshot, EvalArm, EvalRun, RewardModelVersion, TrainingRun
from outlier_ai.models.ops import HoldoutCampaign, Job
from outlier_ai.models.trajectories import Review, Trajectory
from outlier_ai.outlier.recompute import recompute_all
from outlier_ai.reward.train import load_active_reward_model, train_reward_model
from outlier_ai.synthetic.seed import seed
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import Role
from outlier_schemas.models import AdCopy

pytestmark = pytest.mark.integration


async def _seeded(session: AsyncSession, cfg: AppConfig, n: int = 120, s: int = 5) -> None:
    await seed(session, n_briefs=4, n_trajectories=n, seed=s, days_back=90, config=cfg)
    await recompute_all(session, cfg)
    await session.commit()


async def _client(db_session: AsyncSession, role: Role) -> AsyncClient:
    email = f"{role.value}-{uuid.uuid4().hex[:6]}@test.local"
    await create_user(db_session, email, "password123", role)
    await db_session.commit()
    client = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    r = await client.post("/api/auth/login", json={"email": email, "password": "password123"})
    assert r.status_code == 200, r.text
    return client


async def test_export_snapshot_excludes_holdout_and_records_row(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    await _seeded(db_session, app_config)
    storage = LocalStorage(tmp_path)
    measured = (
        (
            await db_session.execute(
                select(Trajectory).where(Trajectory.outlier_tier.is_not(None)).limit(3)
            )
        )
        .scalars()
        .all()
    )
    blocked = measured[0]
    db_session.add(HoldoutCampaign(campaign_id=blocked.campaign_id, added_by="t", reason="t"))
    await db_session.commit()
    snap = await export_snapshot(db_session, cfg=app_config, storage=storage)
    await db_session.commit()
    assert snap.n_trajectories > 10 and snap.n_with_prompt == snap.n_trajectories
    table = pq.read_table(io.BytesIO(storage.get(key_from_uri(snap.uri))))
    ids = set(table.column("trajectory_id").to_pylist())
    assert str(blocked.id) not in ids
    assert "prompt" in table.column_names and "completion" in table.column_names
    row = (await db_session.execute(select(ArchiveSnapshot))).scalar_one()
    assert row.hash == snap.hash and row.n_tier2 == snap.n_tier2
    # same data -> same hash (idempotent)
    again = await export_snapshot(db_session, cfg=app_config, storage=storage)
    assert again.hash == snap.hash


async def test_train_reward_model_and_load(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    await _seeded(db_session, app_config)
    storage = LocalStorage(tmp_path)
    embedder = HashEmbedder(app_config.archive.embedding_dims)
    # seeded expert labels carry no learnable signal: either the data guard refuses to train, or
    # the held-out AUC guard refuses to activate; in both cases the cold reward model stays in use
    strict = await train_reward_model(
        db_session, cfg=app_config, embedder=embedder, storage=storage
    )
    assert strict is None or (not strict.activated and strict.note)
    assert await load_active_reward_model(db_session, storage, embedder, app_config) is None
    relaxed = app_config.model_copy(
        update={
            "reward_model": app_config.reward_model.model_copy(
                update={"min_rows": 20, "min_per_class": 3, "min_val_auc": 0.5}
            )
        }
    )
    res = await train_reward_model(
        db_session, cfg=relaxed, embedder=embedder, storage=storage, force=True
    )
    await db_session.commit()
    assert res is not None and res.n_rows >= 20 and res.n_positive >= 3
    assert res.calibration.expected_calibration_error is not None
    assert res.activated  # forced: the note still records the guard verdict
    assert res.val_auc is not None
    active = (
        await db_session.execute(select(RewardModelVersion).where(RewardModelVersion.is_active))
    ).scalar_one()
    assert active.version == res.version and active.artifact_uri
    rm = await load_active_reward_model(db_session, storage, embedder, relaxed)
    assert rm is not None and rm.version == res.version
    from outlier_ai.reward.base import IdeaFeatures

    score = await rm.score(
        IdeaFeatures(
            brief_text="A protein bar for climbers",
            world_state="",
            angle="Fuel for the crux",
            copy=AdCopy(primary_text="p", headline="h", description="d", cta="SHOP_NOW"),
            visual_brief="v",
            verified_card_slugs=["contrarian", "ugc-testimonial"],
            typicality="rare",
            novelty_distance=0.4,
            cited_signal_count=0,
            format_ok=True,
            tag_match=True,
        )
    )
    assert 0.0 <= score.mean <= 1.0 and score.std >= 0.0
    assert score.version == res.version


async def test_eval_harness_runs_fake_arms(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    await _seeded(db_session, app_config, n=80)
    result = await run_eval(
        db_session,
        cfg=app_config,
        storage=LocalStorage(tmp_path),
        systems=["loop_a_fake", "random_fake"],
        n_briefs=2,
        budget_cap=900.0,
        created_by="test",
        seed=1,
        max_days=40,
    )
    await db_session.commit()
    assert set(result.arms) == {"loop_a_fake", "random_fake"}
    labels = {a.blind_label for a in result.arms.values()}
    assert len(labels) == 2  # distinct blind labels
    for arm in result.arms.values():
        assert len(arm.successes) == 2 and arm.ci is not None
        assert all(x in (0, 1) for x in arm.successes)
    assert result.success is None  # < 20 briefs: no verdict
    run = await db_session.get(EvalRun, result.eval_id)
    assert run is not None and run.status == "completed" and len(run.holdout_brief_ids) == 2
    arms = (
        (await db_session.execute(select(EvalArm).where(EvalArm.eval_run_id == run.id)))
        .scalars()
        .all()
    )
    assert len(arms) == 2 and all(a.tier2_rate is not None for a in arms)
    # held-out briefs' campaigns are blocklisted
    n_holdout = (
        await db_session.execute(select(func.count(HoldoutCampaign.campaign_id)))
    ).scalar_one()
    assert n_holdout > 0


async def test_suggest_relations_from_tier2_cooccurrence(
    db_session: AsyncSession, app_config: AppConfig
):
    await _seeded(db_session, app_config, n=40)
    cards = (await db_session.execute(select(Card).where(Card.status == "active"))).scalars().all()
    a, b, c, d = cards[0], cards[1], cards[2], cards[3]
    # plant a pair (a, b) that co-occurs far above what its members' frequencies predict
    tier2 = (
        (await db_session.execute(select(Trajectory).order_by(Trajectory.created_at).limit(6)))
        .scalars()
        .all()
    )
    for i, t in enumerate(tier2):
        t.outlier_tier = 2
        t.verified_card_ids = [a.id, b.id] if i < 4 else [c.id, d.id]
    await db_session.commit()
    out = await suggest_relations(db_session, actor="test")
    await db_session.commit()
    pairs = {tuple(sorted((r.a, r.b))) for r in out}
    assert tuple(sorted((a.id, b.id))) in pairs
    rel = (
        (await db_session.execute(select(CardRelation).where(CardRelation.source == "suggested")))
        .scalars()
        .all()
    )
    assert rel and all(r.status == "pending" for r in rel)
    # idempotent: a second pass adds nothing
    assert await suggest_relations(db_session, actor="test") == []


async def test_retrain_verifier_classifier(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    await _seeded(db_session, app_config, n=60)
    storage = LocalStorage(tmp_path)
    embedder = HashEmbedder(app_config.archive.embedding_dims)
    # not enough labels at the configured threshold
    assert (
        await retrain_verifier(db_session, cfg=app_config, embedder=embedder, storage=storage)
        is None
    )
    trajs = (
        (
            await db_session.execute(
                select(Trajectory).where(Trajectory.format_ok.is_(True)).limit(30)
            )
        )
        .scalars()
        .all()
    )
    for t in trajs:
        review = (
            await db_session.execute(select(Review).where(Review.trajectory_id == t.id))
        ).scalar_one_or_none()
        if review is None:
            db_session.add(
                Review(
                    trajectory_id=t.id,
                    label="wrong_cards",
                    corrected_card_ids=list(t.card_ids),
                    reviewer_id="expert@test",
                )
            )
        else:
            review.label = "wrong_cards"
            review.corrected_card_ids = list(t.card_ids)
    await db_session.commit()
    vv = await retrain_verifier(
        db_session, cfg=app_config, embedder=embedder, storage=storage, min_labels=20
    )
    await db_session.commit()
    assert vv is not None and vv.kind == "classifier" and vv.is_active
    assert vv.metrics["cards"] >= 2 and 0 <= vv.metrics["train_jaccard"] <= 1
    model = await load_active_classifier(db_session, storage)
    assert model is not None
    verifier = ClassifierVerifier(model, embedder, await load_card_refs(db_session), vv.version)
    t = trajs[0]
    res = await verifier.tag(t.angle, AdCopy(**t.ad_copy), t.visual_brief)
    assert res.version == vv.version and isinstance(res.card_slugs, list)


async def test_training_eval_feedback_api(
    db_session: AsyncSession, app_config: AppConfig, tmp_path, monkeypatch
):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    from outlier_ai.core.settings import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    await _seeded(db_session, app_config, n=60)
    operator = await _client(db_session, Role.operator)
    researcher = await _client(db_session, Role.researcher)

    # role gate
    assert (await operator.post("/api/training/runs", json={"stage": "rft"})).status_code == 403
    # on-policy is gated
    r = await researcher.post("/api/training/runs", json={"stage": "grpo_onpolicy"})
    assert r.status_code == 422
    # real run refused without enough positives
    r = await researcher.post("/api/training/runs", json={"stage": "rft"})
    assert r.status_code in (201, 409)
    # dry run: snapshot + queued job + trainer subprocess reports counts
    r = await researcher.post("/api/training/runs", json={"stage": "rft", "dry": True})
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]
    assert r.json()["status"] == "queued" and r.json()["snapshot_hash"]
    job = (
        await db_session.execute(
            select(Job).where(Job.kind == "train", Job.key == f"train:{run_id}")
        )
    ).scalar_one()
    result = await run_handler(db_session, job, app_config)
    await db_session.commit()
    assert result["status"] == "completed", result
    run = await db_session.get(TrainingRun, uuid.UUID(run_id))
    assert run is not None and run.status == "completed"
    assert run.metrics["trainer"]["rows"] > 0 and run.metrics["trainer"]["dry"] is True
    assert run.logs_uri
    listed = await researcher.get("/api/training/runs")
    assert listed.status_code == 200 and any(x["id"] == run_id for x in listed.json())

    # eval: systems + queued launch + inline launch on tiny fake arms
    systems = await researcher.get("/api/eval/systems")
    assert "loop_a_fake" in systems.json()
    queued = await researcher.post(
        "/api/eval/launch", json={"systems": ["loop_a_fake"], "n_briefs": 1, "inline": False}
    )
    assert queued.status_code == 201 and queued.json()["status"] == "queued"
    assert queued.json()["job_id"]
    bad = await researcher.post("/api/eval/launch", json={"systems": ["nope"]})
    assert bad.status_code == 422
    inline = await researcher.post(
        "/api/eval/launch",
        json={
            "systems": ["loop_a_fake", "random_fake"],
            "n_briefs": 1,
            "budget_cap": 600,
            "max_days": 25,
        },
    )
    assert inline.status_code == 201, inline.text
    body = inline.json()
    assert body["status"] == "completed" and len(body["arms"]) == 2
    got = await researcher.get(f"/api/eval/{body['id']}")
    assert got.status_code == 200 and got.json()["summary"]["n_briefs"] == 1
    assert (await operator.post("/api/eval/launch", json={})).status_code == 403

    # feedback endpoints
    fb = await researcher.post("/api/feedback/suggest_relations")
    assert fb.status_code == 200 and fb.json()["action"] == "suggest_relations"
    rv = await researcher.post("/api/feedback/retrain_verifier")
    assert rv.status_code == 200 and rv.json()["count"] == 0 and rv.json()["reason"]
    rm = await researcher.post("/api/feedback/retrain_rm", json={})
    assert rm.status_code == 200 and rm.json()["trained"] is False
    assert "cold reward model stays" in rm.json()["reason"]
    gg = await researcher.get("/api/stats/gold_gap")
    assert gg.status_code == 200 and "tripped" in gg.json()
    stats = await researcher.get("/api/stats/reward_model")
    assert stats.status_code == 200
