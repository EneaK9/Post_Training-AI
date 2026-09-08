from uuid import uuid4

import pytest
from pydantic import ValidationError

from outlier_schemas.enums import CardKind, Tier
from outlier_schemas.models import AdCopy, Card, Combination, Trajectory, short_ref


def test_card_slug_must_be_kebab():
    Card(
        slug="objection-flip",
        name="Objection Flip",
        kind=CardKind.strategy,
        definition="x",
        contributed_by="seed",
    )
    with pytest.raises(ValidationError):
        Card(
            slug="Objection Flip",
            name="x",
            kind=CardKind.strategy,
            definition="x",
            contributed_by="s",
        )


def test_combination_sorts_and_rejects_duplicates():
    a, b = uuid4(), uuid4()
    c = Combination(card_ids=[b, a], niche_key="k")
    assert c.card_ids == sorted([a, b])
    with pytest.raises(ValidationError):
        Combination(card_ids=[a, a], niche_key="k")


def test_trajectory_copy_alias_round_trips():
    t = Trajectory.model_validate(
        {
            "brief_id": str(uuid4()),
            "author_id": "m",
            "author_kind": "model",
            "copy": {"primary_text": "p", "headline": "h", "description": "d", "cta": "Shop"},
        }
    )
    assert isinstance(t.ad_copy, AdCopy)
    assert t.ad_copy.headline == "h"
    dumped = t.model_dump(by_alias=True)
    assert "copy" in dumped and "ad_copy" not in dumped


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        Card(slug="a-b", name="x", kind=CardKind.style, definition="x", contributed_by="s", bogus=1)  # type: ignore[call-arg]


def test_tier_is_int_enum():
    assert Tier(2) == Tier.two
    assert int(Tier.three) == 3


def test_short_ref_is_eight_hex():
    ref = short_ref(uuid4())
    assert len(ref) == 8
    int(ref, 16)
