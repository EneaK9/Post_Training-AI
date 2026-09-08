"""Contract test for the Anthropic backend: one real generation must parse under the strict
grammar. Runs only with ANTHROPIC_API_KEY set; costs a few cents."""

from __future__ import annotations

import os

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
        CardRef(id=__import__("uuid").uuid4(), slug="contrarian", kind="strategy", name="Contrarian"),
        CardRef(id=__import__("uuid").uuid4(), slug="ugc-testimonial", kind="style", name="UGC"),
    ]
    prompt = (
        "Propose exactly 1 feed ad idea for a protein bar brief. Active cards: contrarian, "
        "ugc-testimonial. Use this exact block grammar:\n<idea>\n<cards>slug, slug</cards>\n"
        "<typicality>common</typicality>\n<reasoning>...</reasoning>\n<angle>...</angle>\n"
        "<copy>primary_text: ...\nheadline: ...\ndescription: ...\ncta: SHOP_NOW</copy>\n"
        "<visual_brief>...</visual_brief>\n</idea>"
    )
    backend = AnthropicBackend(model=app_config.generation.anthropic_model, max_tokens=1500)
    outputs = await backend.generate(prompt, 1)
    parsed = parse_ideas(outputs[0].text, {c.slug: c for c in cards})
    assert parsed and parsed[0].format_ok, parsed
