"""Templated idea composition for synthetic trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from outlier_ai.synthetic.briefs import CATEGORIES
from outlier_ai.synthetic.cards import CardSpec
from outlier_schemas.enums import CardKind, Typicality

DESIRES = [
    "to stop thinking about it",
    "to look like you have it together",
    "to be the one who found it first",
]
TENX = ["10x fewer steps.", "Lasts 10x longer.", "Set up in 1/10th the time."]
JOKES = [
    ("We asked our lawyer if we could say this.", "He said no. So here's the product instead."),
    ("This ad has no music, no dance, no discount code.", "Just the thing. Turns out that works."),
]
OBJECTIONS = [
    "It's too expensive.",
    "I don't need another one of these.",
    "This looks like every other brand.",
]
PROCESS = ["11 days", "three tries", "a 40-step process nobody copies"]


def _fill(template: str, ctx: dict[str, str]) -> str:
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


def compose_idea(
    rng: np.random.Generator, brief: dict[str, Any], cards: list[CardSpec]
) -> dict[str, Any]:
    cat = CATEGORIES[brief["category"]]
    strategy = next((c for c in cards if c.kind == CardKind.strategy), cards[0])
    joke = JOKES[int(rng.integers(0, len(JOKES)))]
    ctx = {
        "incumbent": str(rng.choice(cat["incumbents"])),
        "trend": str(rng.choice(cat["trends"])),
        "category_noun": str(rng.choice(cat["nouns"])),
        "desire": str(rng.choice(DESIRES)),
        "tenx_claim": str(rng.choice(TENX)),
        "joke_setup": joke[0],
        "joke_payoff": joke[1],
        "objection": str(rng.choice(OBJECTIONS)),
        "process_detail": str(rng.choice(PROCESS)),
        "product": brief["product"],
        "company": brief["company"],
        "offer": brief["offer"],
    }
    hook = _fill(strategy.hook or "{product}", ctx)
    enemy = _fill(strategy.enemy or "the usual", ctx)
    promise = _fill(strategy.promise or "{offer}", ctx)
    others = [c.name for c in cards if c is not strategy]
    style_note = f" Executed as {', '.join(others)}." if others else ""
    angle = f"Hook: {hook} / Enemy: {enemy}\nPromise: {promise}"
    primary = (
        f"{hook} {brief['company']} makes {brief['product']}. {promise} {brief['offer']}."
        f"{style_note}"
    )
    headline = hook[:38] if len(hook) > 38 else hook
    description = brief["offer"][:28]
    reasoning = (
        f"Chose [card:{strategy.slug}] because {strategy.qualifying_condition.rstrip('.')} for "
        f"{brief['company']}." + "".join(f" [card:{c.slug}]" for c in cards if c is not strategy)
    )
    visual = f"{brief['product']} shown in a real setting; text overlay reads '{headline}'."
    return {
        "angle": angle,
        "copy": {
            "primary_text": primary,
            "headline": headline,
            "description": description,
            "cta": str(rng.choice(["Shop Now", "Learn More", "Get Offer"])),
        },
        "visual_brief": visual,
        "reasoning": reasoning,
    }


def _sample(rng: np.random.Generator, pool: list[CardSpec], n: int) -> list[CardSpec]:
    if n <= 0 or not pool:
        return []
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[int(i)] for i in idx]


def pick_cards(rng: np.random.Generator, by_kind: dict[CardKind, list[CardSpec]]) -> list[CardSpec]:
    n_strat = int(rng.choice([1, 1, 1, 2]))
    strategies = _sample(rng, by_kind[CardKind.strategy], n_strat)
    others_pool = by_kind[CardKind.style] + by_kind[CardKind.principle] + by_kind[CardKind.mechanic]
    n_other = int(rng.choice([1, 1, 2, 2, 3])) if others_pool else 0
    n_other = max(n_other, 2 - n_strat)
    return strategies + _sample(rng, others_pool, n_other)


def pick_typicality(rng: np.random.Generator) -> Typicality:
    return Typicality(str(rng.choice(["common", "common", "uncommon", "uncommon", "rare"])))
