import asyncio
from uuid import uuid4

from outlier_ai.generation.backends.fake_backend import FakeBackend
from outlier_ai.generation.parser import parse_ideas, set_uses_distinct_combinations
from outlier_ai.generation.renderer import render_prompt
from outlier_ai.generation.views import BriefView, CardView, HistoryItemView, SignalView
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import Typicality

CARDS = [
    CardView("contrarian", "Contrarian", "strategy", "Opposite.", "Received wisdom."),
    CardView("objection-flip", "Objection Flip", "strategy", "Objection is the ad.", "Objection."),
    CardView("human-desires", "Human Desires", "strategy", "Primal desire.", "Always."),
    CardView("specific-numbers", "Specific Numbers", "principle", "Exact figures.", "Any claim."),
    CardView("meme-format", "Meme Format", "style", "Meme frame.", "Young."),
    CardView("founder-face", "Founder Face", "mechanic", "Founder on camera.", "Founder."),
]


def _prompt(cfg: AppConfig, k: int) -> tuple[str, dict]:
    brief = BriefView(
        id=uuid4(),
        company="Lumen Labs",
        product="a retinol serum",
        offer="20% off",
        audience="women",
        category="skincare",
        goal_metric="purchases",
        channel="meta_feed_image",
        world_state="",
        world_state_at=None,
        constraints=[],
        raw_text="Lumen Labs sells a retinol serum.",
    )
    item = HistoryItemView(
        ref="abcd1234",
        reason="elite",
        same_brief=True,
        brief_company="Lumen Labs",
        card_slugs=["contrarian"],
        verified_card_slugs=["contrarian"],
        niche="contrarian",
        typicality="common",
        angle="Hook: x / Enemy: y\nPromise: z",
        copy={"headline": "H", "primary_text": "P"},
        tier=0,
        ratio=0.7,
        screening_ratio=1.1,
        days_ago=5,
        signals=[
            SignalView("beefcafe", "objection", "Price is seen as too high", "neg", 12, "confirmed")
        ],
    )
    ref_map = {"abcd1234": uuid4(), "beefcafe": uuid4()}
    prompt, _ = render_prompt(
        brief=brief,
        playbook=CARDS,
        archive_items=[item],
        episode_batches=[],
        niche_uses={"contrarian": 3},
        ref_map=ref_map,
        cfg=cfg,
        k=k,
    )
    return prompt, ref_map


def test_fake_backend_reads_prompt_and_emits_valid_set(app_config: AppConfig):
    prompt, ref_map = _prompt(app_config, k=6)
    playbook = FakeBackend.parse_playbook(prompt)
    assert [s for s, _ in playbook["strategy"]] == ["contrarian", "human-desires", "objection-flip"]
    assert playbook["principle"][0] == ("specific-numbers", "Specific Numbers")

    out = asyncio.run(FakeBackend(seed=3).generate(prompt))[0]
    ideas = parse_ideas(
        out.text, cards_by_slug={c.slug: uuid4() for c in CARDS}, ref_map=ref_map, expected_k=6
    )
    assert len(ideas) == 6
    assert all(i.format_ok for i in ideas), [i.errors for i in ideas]
    assert set_uses_distinct_combinations(ideas) == []
    assert any(i.typicality == Typicality.rare for i in ideas)
    assert all(len(i.card_ids) >= 2 for i in ideas)
    assert any(i.citations.history for i in ideas)
    assert all("Lumen Labs" in i.copy.primary_text for i in ideas)


def test_fake_backend_is_deterministic_and_can_break_format(app_config: AppConfig):
    prompt, ref_map = _prompt(app_config, k=4)
    a = asyncio.run(FakeBackend(seed=1).generate(prompt))[0].text
    b = asyncio.run(FakeBackend(seed=1).generate(prompt))[0].text
    assert a == b
    bad = asyncio.run(FakeBackend(seed=1, malformed_rate=1.0).generate(prompt))[0]
    ideas = parse_ideas(bad.text, cards_by_slug={c.slug: uuid4() for c in CARDS}, ref_map=ref_map)
    assert ideas and not any(i.format_ok for i in ideas)
