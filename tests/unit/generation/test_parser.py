from uuid import UUID, uuid4

from hypothesis import given, settings
from hypothesis import strategies as st

from outlier_ai.generation.parser import parse_ideas, set_uses_distinct_combinations
from outlier_schemas.enums import Typicality

CARDS: dict[str, UUID] = {
    "contrarian": uuid4(),
    "objection-flip": uuid4(),
    "specific-numbers": uuid4(),
    "meme-format": uuid4(),
}
REFS: dict[str, UUID] = {"a1b2c3d4": uuid4(), "deadbeef": uuid4()}


def block(
    cards: str = "contrarian, specific-numbers",
    typ: str = "rare",
    reasoning: str = "Responds to [signal:deadbeef] on [history:a1b2c3d4]; uses [card:contrarian].",
    copy: str = (
        "primary_text: Everything you know is wrong.\nAnd here is why.\n"
        "headline: Wrong\ndescription: 20% off\ncta: Shop Now"
    ),
    close: bool = True,
) -> str:
    return (
        "<idea>\n"
        f"<cards>{cards}</cards>\n"
        f"<typicality>{typ}</typicality>\n"
        f"<reasoning>{reasoning}</reasoning>\n"
        "<angle>Hook: x / Enemy: y\nPromise: z</angle>\n"
        f"<copy>{copy}</copy>\n"
        "<visual_brief>A serum bottle on a kitchen counter.</visual_brief>\n"
        + ("</idea>" if close else "")
    )


def test_well_formed_set_parses_cleanly():
    text = (
        "Here are the ideas.\n"
        + block()
        + "\n\n"
        + block(cards="objection-flip, meme-format", typ="common")
    )
    ideas = parse_ideas(text, cards_by_slug=CARDS, ref_map=REFS, expected_k=2)
    assert len(ideas) == 2
    a, b = ideas
    assert a.format_ok and b.format_ok, (a.errors, b.errors)
    assert a.card_slugs == ["contrarian", "specific-numbers"]
    assert a.card_ids == [CARDS["contrarian"], CARDS["specific-numbers"]]
    assert a.typicality == Typicality.rare and b.typicality == Typicality.common
    assert a.copy.primary_text == "Everything you know is wrong.\nAnd here is why."
    assert a.copy.headline == "Wrong" and a.copy.cta == "Shop Now"
    assert a.citations.history == [REFS["a1b2c3d4"]]
    assert a.citations.signals == [REFS["deadbeef"]]
    assert a.citations.cards == ["contrarian"]
    assert set_uses_distinct_combinations(ideas) == []


def test_errors_are_reported_not_raised():
    bad = parse_ideas(
        block(cards="contrarian", typ="wild", copy="headline: only", close=False),
        cards_by_slug=CARDS,
        ref_map=REFS,
    )[0]
    assert not bad.format_ok
    joined = " | ".join(bad.errors)
    assert "missing </idea>" in joined
    assert "need at least 2" in joined
    assert "typicality: invalid" in joined
    assert "copy: missing primary_text" in joined and "copy: missing cta" in joined


def test_unknown_and_duplicate_cards_and_citations():
    idea = parse_ideas(
        block(cards="contrarian, contrarian, not-a-card", reasoning="[card:nope] [history:zzzz]"),
        cards_by_slug=CARDS,
        ref_map=REFS,
    )[0]
    assert "cards: duplicate contrarian" in idea.errors
    assert "cards: unknown or inactive card not-a-card" in idea.errors
    assert idea.card_ids == [CARDS["contrarian"]]
    assert idea.citations.unresolved == ["card:nope", "history:zzzz"]


def test_set_level_k_mismatch_and_duplicate_combinations():
    text = block() + block()
    ideas = parse_ideas(text, cards_by_slug=CARDS, ref_map=REFS, expected_k=3)
    assert all("set: expected 3 ideas, got 2" in i.errors for i in ideas)
    assert set_uses_distinct_combinations(ideas) == [1]


def test_card_prefix_and_case_are_tolerated():
    idea = parse_ideas(
        block(cards="[card:Contrarian], CARD:specific-numbers"), cards_by_slug=CARDS
    )[0]
    assert idea.card_slugs == ["contrarian", "specific-numbers"]
    assert idea.format_ok


def test_empty_and_garbage_input():
    assert parse_ideas("", cards_by_slug=CARDS) == []
    assert parse_ideas("no ideas here", cards_by_slug=CARDS) == []


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=400))
def test_never_raises_on_arbitrary_text(text: str):
    ideas = parse_ideas(text, cards_by_slug=CARDS, ref_map=REFS)
    assert isinstance(ideas, list)
    for i in ideas:
        assert isinstance(i.format_ok, bool)


slug_list = st.lists(st.sampled_from(sorted(CARDS)), min_size=2, max_size=4, unique=True)
plain = st.text(
    alphabet=st.characters(blacklist_characters="<>[]", blacklist_categories=("Cs",)),
    min_size=1,
    max_size=40,
).filter(lambda s: s.strip() != "")


@settings(max_examples=100, deadline=None)
@given(
    slugs=slug_list,
    typ=st.sampled_from(list(Typicality)),
    headline=plain,
    primary=plain,
    desc=plain,
    cta=plain,
)
def test_valid_blocks_round_trip(slugs, typ, headline, primary, desc, cta):
    copy = f"primary_text: {primary}\nheadline: {headline}\ndescription: {desc}\ncta: {cta}"
    idea = parse_ideas(
        block(cards=", ".join(slugs), typ=typ.value, copy=copy), cards_by_slug=CARDS
    )[0]
    assert idea.format_ok, idea.errors
    assert idea.card_slugs == slugs
    assert idea.typicality == typ
    assert idea.copy.headline == headline.strip()
