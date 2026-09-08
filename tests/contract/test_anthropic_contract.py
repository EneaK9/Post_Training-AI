"""Contract test for the Anthropic backend: one real generation must parse under the strict
grammar. Runs only with ANTHROPIC_API_KEY set; costs a few cents."""

from __future__ import annotations

import os
import uuid

import pytest

from outlier_schemas.config import AppConfig

pytestmark = pytest.mark.contract

needs_key = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no API key")


@needs_key
async def test_generate_one_idea_parses(app_config: AppConfig):
    from outlier_ai.generation.backends.anthropic_backend import AnthropicBackend
    from outlier_ai.generation.parser import parse_ideas
    from outlier_ai.generation.verifier import CardRef

    cards = [
        CardRef(
            id=uuid.uuid4(), slug="contrarian", name="Contrarian", kind="strategy", definition=""
        ),
        CardRef(id=uuid.uuid4(), slug="ugc-testimonial", name="UGC", kind="style", definition=""),
    ]
    prompt = (
        "Propose exactly 1 feed ad idea for a protein bar brief. Active cards: contrarian, "
        "ugc-testimonial. Use this exact block grammar:\n<idea>\n<cards>slug, slug</cards>\n"
        "<typicality>common</typicality>\n<reasoning>...</reasoning>\n<angle>...</angle>\n"
        "<copy>primary_text: ...\nheadline: ...\ndescription: ...\ncta: SHOP_NOW</copy>\n"
        "<visual_brief>...</visual_brief>\n</idea>"
    )
    backend = AnthropicBackend(app_config.generation.anthropic_model)
    outputs = await backend.generate(prompt, 1)
    result = parse_ideas(outputs[0].text, cards_by_slug={c.slug: c.id for c in cards})
    ideas = getattr(result, "ideas", result)
    assert ideas and ideas[0].format_ok, result
