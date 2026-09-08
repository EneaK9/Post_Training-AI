import asyncio
from uuid import uuid4

from outlier_ai.generation.preship import (
    check_brand_constraints,
    check_lengths,
    check_policy_keywords,
    preship_check,
)
from outlier_ai.generation.verifier import (
    CardRef,
    HeuristicVerifier,
    parse_slug_json,
    tag_agreement,
)
from outlier_schemas.config import ImagesConfig
from outlier_schemas.models import AdCopy

CARDS = [
    CardRef(uuid4(), "contrarian", "Contrarian", "strategy", "Opposite."),
    CardRef(uuid4(), "objection-flip", "Objection Flip", "strategy", "Objection."),
    CardRef(uuid4(), "specific-numbers", "Specific Numbers", "principle", "Numbers."),
    CardRef(uuid4(), "meme-format", "Meme Format", "style", "Meme."),
]


def test_heuristic_verifier_finds_named_cards():
    v = HeuristicVerifier(CARDS)
    copy = AdCopy(
        primary_text="Everything you know is wrong. Executed as Specific Numbers, Meme Format.",
        headline="Contrarian: serum",
        description="20% off",
        cta="Shop",
    )
    res = asyncio.run(v.tag("Hook: contrarian take", copy, "serum on a counter"))
    assert set(res.card_slugs) == {"contrarian", "specific-numbers", "meme-format"}
    assert res.version.startswith("verifier-heuristic")
    ids_by_slug = {c.slug: c.id for c in CARDS}
    match, jaccard = tag_agreement(
        [ids_by_slug["contrarian"], ids_by_slug["meme-format"]], res.card_ids
    )
    assert not match and abs(jaccard - 2 / 3) < 1e-9
    assert tag_agreement([], []) == (True, 1.0)


def test_parse_slug_json_is_tolerant():
    assert parse_slug_json('Sure. {"cards": ["Contrarian", "meme-format", "meme-format", 3]}') == [
        "contrarian",
        "meme-format",
    ]
    assert parse_slug_json("no json here") == []
    assert parse_slug_json('{"cards": "oops"}') == []


def test_policy_and_brand_checks():
    text = "guaranteed results with our miracle serum. before/after photos of real customers."
    flags = check_policy_keywords(text)
    assert any("unrealistic_outcomes" in f for f in flags)
    brand = check_brand_constraints(
        [
            "no before/after photos of real customers",
            "include FDA disclaimer",
            "show product in use",
        ],
        text,
    )
    assert any("constraint violated" in f for f in brand)
    assert any("required element missing" in f for f in brand)
    clean = check_brand_constraints(["no medical claims"], "a nice serum for your evening routine")
    assert clean == []


def test_lengths_and_report():
    cfg = ImagesConfig()
    copy = AdCopy(primary_text="x" * 200, headline="h" * 50, description="d" * 10, cta="Shop")
    warnings = check_lengths(copy, cfg)
    assert len(warnings) == 2 and warnings[0].startswith("primary_text")
    report = asyncio.run(
        preship_check(
            copy=AdCopy(primary_text="Clean copy", headline="Hi", description="d", cta="Shop"),
            angle="Hook: a / Enemy: b",
            visual_brief="a bottle",
            constraints=["no medical claims"],
            images_cfg=cfg,
        )
    )
    assert report.clean and report.length_warnings == []
    bad = asyncio.run(
        preship_check(
            copy=AdCopy(
                primary_text="A miracle cure for acne", headline="Hi", description="d", cta="Shop"
            ),
            angle="",
            visual_brief="",
            constraints=[],
            images_cfg=cfg,
        )
    )
    assert not bad.policy_ok and not bad.clean
