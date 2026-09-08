from datetime import UTC, datetime
from uuid import uuid4

from outlier_ai.generation.renderer import estimate_tokens, render_prompt
from outlier_ai.generation.views import BatchView, BriefView, CardView, HistoryItemView, SignalView
from outlier_schemas.config import AppConfig


def cards() -> list[CardView]:
    return [
        CardView(
            "contrarian", "Contrarian", "strategy", "State the opposite.", "Strong received wisdom."
        ),
        CardView(
            "objection-flip",
            "Objection Flip",
            "strategy",
            "Make the objection the ad.",
            "Recurring objection.",
        ),
        CardView(
            "specific-numbers", "Specific Numbers", "principle", "Exact figures.", "Any claim."
        ),
        CardView("meme-format", "Meme Format", "style", "Meme frame.", "Young audiences."),
    ]


def brief() -> BriefView:
    return BriefView(
        id=uuid4(),
        company="Lumen Labs",
        product="a retinol serum",
        offer="20% off",
        audience="women 28-45",
        category="skincare",
        goal_metric="purchases",
        channel="meta_feed_image",
        world_state="Black Friday is three weeks out.",
        world_state_at=datetime(2026, 9, 1, tzinfo=UTC),
        constraints=["no medical claims"],
        raw_text="Lumen Labs sells a retinol serum.",
    )


def item(ref: str, reason: str, tier: int | None = 0, signals: int = 1) -> HistoryItemView:
    return HistoryItemView(
        ref=ref,
        reason=reason,
        same_brief=True,
        brief_company="Lumen Labs",
        card_slugs=["contrarian", "specific-numbers"],
        verified_card_slugs=["contrarian", "specific-numbers"],
        niche="contrarian",
        typicality="common",
        angle="Hook: a / Enemy: b\nPromise: c",
        copy={"headline": "H", "primary_text": "P " * 300},
        tier=tier,
        ratio=0.8 if tier == 0 else 3.4,
        screening_ratio=1.7,
        days_ago=12,
        signals=[
            SignalView(
                f"s{ref}{i}", "objection", "Price is seen as too high", "neg", 14, "confirmed"
            )
            for i in range(signals)
        ],
    )


def test_render_contains_three_columns_refs_and_ask(app_config: AppConfig):
    items = [item("aaaa0001", "elite", tier=2), item("bbbb0002", "rare", tier=0)]
    batch = BatchView(0, "screening", [item("cccc0003", "episode", tier=None, signals=0)])
    ref_map = {
        "aaaa0001": uuid4(),
        "bbbb0002": uuid4(),
        "cccc0003": uuid4(),
        "saaaa00010": uuid4(),
        "sbbbb00020": uuid4(),
    }
    prompt, trace = render_prompt(
        brief=brief(),
        playbook=cards(),
        archive_items=items,
        episode_batches=[batch],
        niche_uses={"contrarian": 5, "objection-flip": 1},
        ref_map=ref_map,
        cfg=app_config,
    )
    for needle in ("COLUMN 1: PLAYBOOK", "COLUMN 2: BRIEF", "COLUMN 3: HISTORY", "THE ASK"):
        assert needle in prompt
    assert "[STRATEGY]" in prompt and prompt.index("[STRATEGY]") < prompt.index("[STYLE]")
    assert "card:objection-flip" in prompt
    assert "history:aaaa0001" in prompt and "history:cccc0003" in prompt
    assert "signal:saaaa00010" in prompt
    assert "Black Friday" in prompt and "no medical claims" in prompt
    assert f"Propose exactly {app_config.generation.k} ideas" in prompt
    assert "common | uncommon | rare" in prompt
    assert trace.k == app_config.generation.k
    assert trace.config_hash == app_config.hash
    assert set(trace.ref_map) == {"aaaa0001", "bbbb0002", "cccc0003", "saaaa00010", "sbbbb00020"}
    assert [r["ref"] for r in trace.archive_refs] == ["aaaa0001", "bbbb0002"]
    assert trace.episode_refs[0]["batch"] == 0
    assert trace.trimmed == []
    assert trace.token_estimate == estimate_tokens(prompt)


def test_trimming_drops_rare_then_elite_and_records_it(app_config: AppConfig):
    cfg = app_config.model_copy(deep=True)
    cfg.generation.max_prompt_tokens = 2000
    items = [item(f"e{i:07d}", "elite", tier=1, signals=3) for i in range(14)] + [
        item(f"r{i:07d}", "rare", signals=3) for i in range(3)
    ]
    ref_map = {i.ref: uuid4() for i in items}
    ref_map.update({s.ref: uuid4() for i in items for s in i.signals})
    prompt, trace = render_prompt(
        brief=brief(),
        playbook=cards(),
        archive_items=items,
        episode_batches=[],
        niche_uses={},
        ref_map=ref_map,
        cfg=cfg,
    )
    assert trace.trimmed, "an oversize prompt must be trimmed"
    assert trace.trimmed[0].endswith("(rare)")
    assert estimate_tokens(prompt) <= 2000 or "copy shortened" in trace.trimmed
    kept = {r["ref"] for r in trace.archive_refs}
    assert all(ref in prompt for ref in kept)
    assert set(trace.ref_map) == kept | {s.ref for i in items if i.ref in kept for s in i.signals}


def test_empty_history_says_so(app_config: AppConfig):
    prompt, trace = render_prompt(
        brief=brief(),
        playbook=cards(),
        archive_items=[],
        episode_batches=[],
        niche_uses={},
        ref_map={},
        cfg=app_config,
        k=3,
    )
    assert "No prior attempts are on record" in prompt
    assert "Propose exactly 3 ideas" in prompt and trace.k == 3
