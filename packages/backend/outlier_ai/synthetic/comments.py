"""Synthetic comments with themes correlated to the hidden outcome.

Each comment has a raw form (as the fake Meta client would return it, occasionally with
planted PII) and a clean form (as the seed stores it). Themes map to signal kinds so the
seed can write realistic Signal rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from outlier_schemas.enums import Sentiment, SignalKind


@dataclass(frozen=True)
class Theme:
    key: str
    kind: SignalKind
    sentiment: Sentiment
    summary: str
    templates: tuple[str, ...]


NEGATIVE_THEMES: list[Theme] = [
    Theme(
        "price_objection",
        SignalKind.objection,
        Sentiment.neg,
        "Price is seen as too high",
        (
            "The price is a joke.",
            "Who pays this much for this?",
            "Way overpriced.",
            "$$$ for what exactly",
        ),
    ),
    Theme(
        "seen_before",
        SignalKind.objection,
        Sentiment.neg,
        "Audience says it is nothing new",
        ("Seen this a hundred times.", "Every brand says this.", "Same ad different logo."),
    ),
    Theme(
        "scam_suspicion",
        SignalKind.objection,
        Sentiment.neg,
        "Suspicion the offer is a scam",
        ("Looks like a scam.", "Too good to be true.", "Reported."),
    ),
    Theme(
        "shipping_complaint",
        SignalKind.objection,
        Sentiment.neg,
        "Complaints about shipping or delivery",
        ("Took 3 weeks to arrive.", "Shipping cost more than the product."),
    ),
    Theme(
        "misread_offer",
        SignalKind.misreading,
        Sentiment.neg,
        "Offer is misread as something else",
        (
            "Wait is this a subscription??",
            "So it's not actually free?",
            "I thought this was for kids",
        ),
    ),
]

POSITIVE_THEMES: list[Theme] = [
    Theme(
        "praise_quality",
        SignalKind.praise,
        Sentiment.pos,
        "Praise for product quality",
        ("Honestly the best I've tried.", "Been using for a month, love it.", "Quality is unreal."),
    ),
    Theme(
        "where_to_buy",
        SignalKind.question,
        Sentiment.pos,
        "People asking where to buy",
        ("Where can I get this?", "Link?", "Do you ship to Canada?"),
    ),
    Theme(
        "tag_friend",
        SignalKind.share_pattern,
        Sentiment.pos,
        "Tagging friends",
        ("@friend this is you", "@partner we need this", "tagging my sister"),
    ),
    Theme(
        "worked_for_me",
        SignalKind.quote,
        Sentiment.pos,
        "Testimonial-style comments",
        ("This actually fixed my problem.", "Ordered twice already.", "Converted a skeptic here."),
    ),
]

NEUTRAL_THEMES: list[Theme] = [
    Theme(
        "question_details",
        SignalKind.question,
        Sentiment.neu,
        "Questions about details",
        ("What sizes does it come in?", "Is it vegan?", "How long does it last?"),
    ),
    Theme(
        "joke",
        SignalKind.joke,
        Sentiment.neu,
        "Jokes riffing on the ad",
        (
            "My wallet just left the chat.",
            "The ad is better than the product I bet.",
            "Ok that hook got me.",
        ),
    ),
    Theme(
        "competitor",
        SignalKind.competitor_mention,
        Sentiment.neu,
        "Competitor mentions",
        ("BrandX does this cheaper.", "How is this different from BrandY?"),
    ),
]

ALL_THEMES: dict[str, Theme] = {
    t.key: t for t in NEGATIVE_THEMES + POSITIVE_THEMES + NEUTRAL_THEMES
}

# Planted PII for the stripper to catch. Synthetic only.
PII_SNIPPETS: tuple[str, ...] = (
    " email me at jane.doe@example.com",
    " call me 555-867-5309",
    " my name is Marcus Whitfield btw",
    " dm me +1 (415) 555-0142",
)


@dataclass(frozen=True)
class SynthComment:
    theme_key: str
    raw_text: str
    clean_text: str
    commenter_ext_id: str
    hours_after_launch: float
    like_count: int


def theme_weights(roas_multiple: float, polarity: float) -> dict[str, float]:
    """Higher ROAS shifts mass to positive themes; polarity raises negatives and jokes."""
    neg = 1.0 if roas_multiple < 1.0 else 0.4 if roas_multiple < 2.0 else 0.2
    pos = 0.2 if roas_multiple < 1.0 else 0.6 if roas_multiple < 2.0 else 1.2
    neu = 0.5
    weights: dict[str, float] = {}
    for t in NEGATIVE_THEMES:
        weights[t.key] = neg * (1.0 + polarity)
    weights["price_objection"] *= 1.6 if roas_multiple < 1.0 else 1.0
    for t in POSITIVE_THEMES:
        weights[t.key] = pos
    for t in NEUTRAL_THEMES:
        weights[t.key] = neu * (1.5 if t.key == "joke" and polarity > 0.5 else 1.0)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def generate_comments(
    rng: np.random.Generator,
    n: int,
    roas_multiple: float,
    polarity: float,
    window_hours: float,
    pii_rate: float = 0.03,
) -> list[SynthComment]:
    if n <= 0:
        return []
    weights = theme_weights(roas_multiple, polarity)
    keys = list(weights)
    probs = np.array([weights[k] for k in keys])
    out: list[SynthComment] = []
    for _ in range(n):
        theme = ALL_THEMES[keys[int(rng.choice(len(keys), p=probs))]]
        clean = str(rng.choice(theme.templates))
        raw = clean
        if rng.random() < pii_rate:
            raw = clean + str(rng.choice(PII_SNIPPETS))
        out.append(
            SynthComment(
                theme_key=theme.key,
                raw_text=raw,
                clean_text=clean,
                commenter_ext_id=f"fbuser_{int(rng.integers(1, 200_000))}",
                hours_after_launch=float(rng.uniform(0, window_hours)),
                like_count=int(rng.poisson(0.6)),
            )
        )
    return out


@dataclass(frozen=True)
class SignalDraft:
    kind: SignalKind
    sentiment: Sentiment
    text: str
    count: int
    excerpts: tuple[str, ...]


def summarize_signals(comments: list[SynthComment], min_count: int = 2) -> list[SignalDraft]:
    by_theme: dict[str, list[SynthComment]] = {}
    for c in comments:
        by_theme.setdefault(c.theme_key, []).append(c)
    drafts: list[SignalDraft] = []
    for key, items in by_theme.items():
        if len(items) < min_count:
            continue
        theme = ALL_THEMES[key]
        excerpts = tuple(dict.fromkeys(c.clean_text for c in items))[:3]
        drafts.append(SignalDraft(theme.kind, theme.sentiment, theme.summary, len(items), excerpts))
    drafts.sort(key=lambda d: d.count, reverse=True)
    return drafts
