import io
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.api.app import create_app
from outlier_ai.core.auth import create_user
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import Combination
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_ai.outlier.recompute import recompute_all
from outlier_ai.synthetic.seed import seed
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import Role

pytestmark = pytest.mark.integration


async def _client(db_session: AsyncSession, role: Role = Role.expert) -> AsyncClient:
    email = f"{role.value}-{uuid4().hex[:6]}@test.local"
    await create_user(db_session, email, "password123", role)
    await db_session.commit()
    client = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    r = await client.post("/api/auth/login", json={"email": email, "password": "password123"})
    assert r.status_code == 200, r.text
    return client


async def _seeded(db_session: AsyncSession, cfg: AppConfig, n: int = 60, s: int = 4) -> Brief:
    await seed(db_session, n_briefs=3, n_trajectories=n, seed=s, days_back=90, config=cfg)
    await recompute_all(db_session, cfg)
    await db_session.commit()
    brief = (await db_session.execute(select(Brief).order_by(Brief.created_at))).scalars().first()
    assert brief is not None
    return brief


async def test_auth_and_roles(db_session: AsyncSession, app_config: AppConfig):
    await _seeded(db_session, app_config, n=10)
    anon = AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")
    assert (await anon.get("/api/cards")).status_code == 401
    assert (await anon.get("/api/health")).status_code == 200
    bad = await anon.post("/api/auth/login", json={"email": "nobody@test.local", "password": "x"})
    assert bad.status_code == 401

    operator = await _client(db_session, Role.operator)
    me = await operator.get("/api/auth/me")
    assert me.status_code == 200 and me.json()["role"] == "operator"
    forbidden = await operator.post(
        "/api/cards", json={"slug": "new-card", "name": "New", "kind": "style", "definition": "d"}
    )
    assert forbidden.status_code == 403
    assert (await operator.post("/api/auth/logout")).status_code == 204
    assert (await operator.get("/api/auth/me")).status_code == 401


async def test_cards_versions_relations_and_combinations(
    db_session: AsyncSession, app_config: AppConfig
):
    await _seeded(db_session, app_config, n=40)
    expert = await _client(db_session, Role.expert)
    cards = (await expert.get("/api/cards")).json()
    assert len(cards) == 28 and all("stats" in c for c in cards)
    strategies = [c for c in cards if c["kind"] == "strategy"]
    assert any(c["stats"]["uses"] > 0 for c in strategies)

    created = await expert.post(
        "/api/cards",
        json={
            "slug": "ogilvy-long-copy",
            "name": "Ogilvy Long Copy",
            "kind": "style",
            "definition": "Long copy sells.",
            "qualifying_condition": "Considered purchases",
            "source": "Ogilvy on Advertising, ch. 7",
        },
    )
    assert created.status_code == 201, created.text
    card = created.json()
    assert card["status"] == "draft" and card["version"] == 1
    updated = await expert.put(
        f"/api/cards/{card['id']}",
        json={"definition": "Long copy sells when the reader is interested."},
    )
    assert updated.json()["version"] == 2
    versions = (await expert.get(f"/api/cards/{card['id']}/versions")).json()
    assert [v["version"] for v in versions] == [1, 2] and versions[0][
        "definition"
    ] == "Long copy sells."
    activated = await expert.post(f"/api/cards/{card['id']}/activate")
    assert activated.json()["status"] == "active" and activated.json()["version"] == 3
    dup = await expert.post(
        "/api/cards",
        json={"slug": "ogilvy-long-copy", "name": "x", "kind": "style", "definition": "d"},
    )
    assert dup.status_code == 409

    rel = await expert.post(
        "/api/cards/relations",
        json={"from_id": card["id"], "to_id": strategies[0]["id"], "kind": "complements"},
    )
    assert rel.status_code == 201 and rel.json()["status"] == "accepted"
    decided = await expert.put(
        f"/api/cards/relations/{rel.json()['id']}", json={"status": "rejected"}
    )
    assert decided.json()["status"] == "rejected"

    combos = (await expert.get("/api/combinations?sort=uses")).json()
    assert combos and combos[0]["uses"] >= combos[-1]["uses"] and combos[0]["card_slugs"]
    named = await expert.post(
        f"/api/combinations/{combos[0]['id']}/name_as_card",
        json={"slug": "named-combo", "name": "Named Combo", "definition": "A proven pairing."},
    )
    assert named.status_code == 201 and named.json()["status"] == "draft"
    combo_row = await db_session.get(Combination, combos[0]["id"])
    assert combo_row is not None
    await db_session.refresh(combo_row)
    assert str(combo_row.named_as_card_id) == named.json()["id"]

    audit = (await expert.get("/api/audit?object_type=card")).json()
    assert any(a["action"] == "card.create" for a in audit) and any(
        a["action"] == "card.update" for a in audit
    )


async def test_briefs_trajectories_reviews_signals(db_session: AsyncSession, app_config: AppConfig):
    brief = await _seeded(db_session, app_config, n=60)
    expert = await _client(db_session, Role.expert)

    briefs = (await expert.get("/api/briefs")).json()
    assert len(briefs) == 3 and all(b["trajectory_count"] > 0 for b in briefs)
    video = await expert.post(
        "/api/briefs",
        json={
            "company": "X",
            "product": "p",
            "offer": "o",
            "audience": "a",
            "category": "skincare",
            "channel": "meta_video",
        },
    )
    assert video.status_code == 422 and "Video ads are out of scope" in video.json()["detail"]
    ok = await expert.post(
        "/api/briefs",
        json={
            "company": "Newco",
            "product": "a thing",
            "offer": "10% off",
            "audience": "people",
            "category": "coffee",
            "world_state": "quiet week",
            "constraints": ["no health claims"],
        },
    )
    assert ok.status_code == 201 and ok.json()["world_state_at"] is not None
    new_brief = await db_session.get(Brief, ok.json()["id"])
    assert new_brief is not None and new_brief.embedding is not None

    page = (await expert.get(f"/api/trajectories?brief_id={brief.id}&limit=10")).json()
    assert page["total"] > 0 and len(page["items"]) <= 10
    item = page["items"][0]
    assert "card_slugs" in item and "renders" in item and "copy" in item
    detail = (await expert.get(f"/api/trajectories/{item['id']}")).json()
    assert "comments" in detail and detail["id"] == item["id"]
    tiered = (await expert.get("/api/trajectories?min_tier=0&limit=5")).json()
    assert all(t["outlier_tier"] is not None for t in tiered["items"])
    mismatched = (await expert.get("/api/trajectories?mismatch=true&limit=5")).json()
    assert all(t["tag_match"] is False for t in mismatched["items"])

    queue = (await expert.get("/api/reviews/queue?limit=5")).json()
    assert isinstance(queue, list)

    traj = (
        (
            await db_session.execute(
                select(Trajectory).where(Trajectory.format_ok.is_(True)).limit(1)
            )
        )
        .scalars()
        .first()
    )
    assert traj is not None
    wrong = await expert.post(f"/api/trajectories/{traj.id}/review", json={"label": "wrong_cards"})
    assert wrong.status_code == 422
    reviewed = await expert.post(
        f"/api/trajectories/{traj.id}/review",
        json={
            "label": "wrong_cards",
            "corrected_card_ids": [str(traj.card_ids[0])],
            "note": "only the first card applies",
        },
    )
    assert (
        reviewed.status_code == 200
        and reviewed.json()["review"]["label"] == "wrong_cards"
        and reviewed.json()["tag_source"] == "expert"
    )
    noted = await expert.post(
        f"/api/trajectories/{traj.id}/notes", json={"text": "watch the price objections"}
    )
    assert (
        noted.status_code == 201
        and noted.json()["notes"][-1]["text"] == "watch the price objections"
    )
    shipped = any(
        r.shipped_at
        for r in (await db_session.execute(select(Render).where(Render.trajectory_id == traj.id)))
        .scalars()
        .all()
    )
    edit = await expert.put(
        f"/api/trajectories/{traj.id}/copy",
        json={"primary_text": "p", "headline": "h", "description": "d", "cta": "Shop"},
    )
    assert edit.status_code == (409 if shipped else 200)

    proposed = (await expert.get("/api/signals/queue")).json()
    assert proposed and all(s["status"] == "proposed" for s in proposed)
    confirmed = await expert.put(f"/api/signals/{proposed[0]['id']}/confirm")
    assert confirmed.json()["status"] == "confirmed" and confirmed.json()["decided_by"].endswith(
        "@test.local"
    )
    manual = await expert.post(
        "/api/signals",
        json={
            "trajectory_id": str(traj.id),
            "text": "Sales team says returns spiked",
            "kind": "note",
        },
    )
    assert manual.status_code == 201 and manual.json()["status"] == "confirmed"
    extracted = await expert.post(f"/api/signals/extract?trajectory_id={traj.id}")
    assert extracted.status_code == 201
    total_signals = (
        (await db_session.execute(select(Signal).where(Signal.trajectory_id == traj.id)))
        .scalars()
        .all()
    )
    assert len(total_signals) >= 1

    hits = (await expert.get("/api/search?q=price")).json()["hits"]
    assert any(h["type"] == "signal" for h in hits)
    sample = (await expert.get(f"/api/archive/sample?brief_id={brief.id}")).json()
    assert sample["brief_id"] == str(brief.id) and isinstance(sample["items"], list)
    prompt = (await expert.get(f"/api/archive/prompt?brief_id={brief.id}&k=3")).json()
    assert "COLUMN 3: HISTORY" in prompt["prompt"] and prompt["trace"]["k"] == 3


async def test_episodes_generate_config_and_import(
    db_session: AsyncSession, app_config: AppConfig, tmp_path, monkeypatch
):
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    from outlier_ai.core.settings import reset_settings_cache

    reset_settings_cache()
    brief = await _seeded(db_session, app_config, n=40)
    operator = await _client(db_session, Role.operator)
    researcher = await _client(db_session, Role.researcher)

    ep = await operator.post(
        "/api/episodes", json={"brief_id": str(brief.id), "backend": "fake", "budget_cap": 3000}
    )
    assert (
        ep.status_code == 201
        and ep.json()["status"] == "searching"
        and ep.json()["budget_cap"] == 3000
    )
    gen = await operator.post(
        "/api/generate",
        json={
            "brief_id": str(brief.id),
            "episode_id": ep.json()["id"],
            "backend": "fake",
            "k": 4,
            "renders_per_idea": 1,
            "no_llm": True,
        },
    )
    assert gen.status_code == 200, gen.text
    body = gen.json()
    assert len(body["ideas"]) == 4 and body["batch_id"] and body["trace"]["k"] == 4
    assert all(i["trajectory"]["episode_id"] == ep.json()["id"] for i in body["ideas"])
    ep_view = (await operator.get(f"/api/episodes/{ep.json()['id']}?include_trace=true")).json()
    assert (
        len(ep_view["batches"]) == 1
        and len(ep_view["batches"][0]["trajectory_ids"]) == 4
        and ep_view["batches"][0]["prompt_trace"]
    )
    by_brief = (await operator.get(f"/api/briefs/{brief.id}/episodes")).json()
    assert len(by_brief) == 1
    stopped = await operator.post(f"/api/episodes/{ep.json()['id']}/stop")
    assert stopped.json()["status"] == "stopped"
    blocked = await operator.post(
        "/api/generate",
        json={
            "brief_id": str(brief.id),
            "episode_id": ep.json()["id"],
            "backend": "fake",
            "k": 2,
            "renders_per_idea": 1,
            "no_llm": True,
        },
    )
    assert blocked.status_code == 422

    cfg_now = (await researcher.get("/api/config")).json()
    assert cfg_now["hash"] == app_config.hash
    proposed = dict(cfg_now["config"])
    proposed["outlier"]["tier_multiples"]["tier2"] = 3.5
    validated = await researcher.post("/api/config/validate", json={"config": proposed})
    assert (
        validated.json()["ok"]
        and validated.json()["recompute_required"]
        and "outlier.tier_multiples.tier2" in validated.json()["diff"]
    )
    bad = await researcher.post("/api/config/validate", json={"yaml": "outlier: {nope: 1}"})
    assert not bad.json()["ok"] and bad.json()["errors"]
    denied = await operator.post("/api/config/apply", json={"config": proposed})
    assert denied.status_code == 403
    applied = await researcher.post(
        "/api/config/apply", json={"config": proposed, "note": "tighter tier 2"}
    )
    assert applied.status_code == 200 and applied.json()["hash"] != app_config.hash
    history = (await researcher.get("/api/config/history")).json()
    assert len(history) == 2 and history[0]["note"] == "tighter tier 2"
    current = (await researcher.get("/api/config")).json()
    assert current["config"]["outlier"]["tier_multiples"]["tier2"] == 3.5

    csv_cards = "slug,name,kind,definition,qualifying_condition,source\nogilvy-headline,Ogilvy Headline,principle,Five times as many people read the headline.,Always,Ogilvy\n"
    imp = await researcher.post(
        "/api/import/cards",
        files={"file": ("cards.csv", io.BytesIO(csv_cards.encode()), "text/csv")},
    )
    assert imp.status_code == 200 and imp.json()["created"] == 1
    again = await researcher.post(
        "/api/import/cards",
        files={"file": ("cards.csv", io.BytesIO(csv_cards.encode()), "text/csv")},
    )
    assert again.json()["skipped"] == 1
    csv_traj = f"brief_id,angle,primary_text,headline,description,cta,visual_brief,cards,author_id,created_at,campaign_id\n{brief.id},Hook: a / Enemy: b,Body copy here,Head,Desc,Shop Now,A bottle,contrarian|specific-numbers,pract@agency.com,2026-06-01T00:00:00Z,camp_import_1\n{brief.id},Hook: c,Untagged body,Head2,Desc,Shop,A cup,,pract@agency.com,2026-06-02T00:00:00Z,camp_import_1\n"
    imp_t = await researcher.post(
        "/api/import/trajectories",
        files={"file": ("t.csv", io.BytesIO(csv_traj.encode()), "text/csv")},
    )
    assert imp_t.status_code == 200 and imp_t.json()["created"] == 2, imp_t.text
    imported = (
        (
            await db_session.execute(
                select(Trajectory)
                .where(Trajectory.campaign_id == "camp_import_1")
                .order_by(Trajectory.created_at)
            )
        )
        .scalars()
        .all()
    )
    assert imported[0].tag_source == "expert" and len(imported[0].card_ids) == 2
    assert imported[1].tag_source == "verifier" and imported[1].card_ids == []
    csv_out = f"trajectory_id,day,phase,impressions,link_clicks,spend,purchases,revenue\n{imported[0].id},2026-06-05,scale,8000,100,100,9,700\n{imported[0].id},2026-06-06,scale,8000,90,100,8,650\n"
    imp_o = await researcher.post(
        "/api/import/outcomes", files={"file": ("o.csv", io.BytesIO(csv_out.encode()), "text/csv")}
    )
    assert imp_o.json()["created"] == 2 and imp_o.json()["errors"] == []
    render = (
        (await db_session.execute(select(Render).where(Render.trajectory_id == imported[0].id)))
        .scalars()
        .first()
    )
    assert render is not None and render.image_backend == "import" and render.shipped_at is not None
    csv_c = f"render_id,external_id,commenter_id,text,created_time\n{render.id},c1,u1,Way overpriced. email me at jane@example.com,2026-06-05T10:00:00Z\n{render.id},c2,u2,love it @bestie,2026-06-05T11:00:00Z\n"
    imp_c = await researcher.post(
        "/api/import/comments", files={"file": ("c.csv", io.BytesIO(csv_c.encode()), "text/csv")}
    )
    assert imp_c.json()["created"] == 2
    from outlier_ai.models.meta import Comment

    stored = (
        (
            await db_session.execute(
                select(Comment).where(Comment.render_id == render.id).order_by(Comment.created_time)
            )
        )
        .scalars()
        .all()
    )
    assert "[email]" in stored[0].text and "jane@" not in stored[0].text and stored[0].pii_removed
    assert "[handle]" in stored[1].text

    ks = (await operator.get("/api/meta/kill_switch")).json()
    assert ks["shipping_enabled"] is True
    off = await operator.put(
        "/api/meta/kill_switch", json={"shipping_enabled": False, "reason": "drill"}
    )
    assert off.json()["shipping_enabled"] is False
    acct = await operator.post(
        "/api/meta/accounts",
        json={"name": "Fake 2", "meta_account_id": "act_fake_2", "is_fake": True},
    )
    assert (
        acct.status_code == 201
        and acct.json()["daily_cap_usd"] == app_config.meta.daily_account_cap_usd
    )
    sync = await operator.post("/api/meta/sync")
    assert sync.status_code == 202 and sync.json()["state"] == "pending"
