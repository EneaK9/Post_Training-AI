"""Synthetic briefs across product categories with world states."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

CATEGORIES: dict[str, dict[str, list[str]]] = {
    "skincare": {
        "companies": ["Lumen Labs", "Barefield", "Nocturne Skin"],
        "products": ["a fragrance-free retinol serum", "a two-step barrier repair kit"],
        "offers": [
            "20% off first order",
            "free travel size with purchase",
            "subscribe and save 15%",
        ],
        "audiences": ["women 28-45 with sensitive skin", "men 25-40 new to skincare"],
        "constraints": ["no before/after photos of real customers", "no medical claims"],
        "nouns": ["skincare", "serums"],
        "incumbents": ["ten-step routines", "drugstore moisturizer"],
        "trends": ["glass-skin routines", "influencer skincare"],
    },
    "supplements": {
        "companies": ["Northstar Nutrition", "Plainlabel"],
        "products": ["a magnesium glycinate sleep formula", "a single-ingredient creatine"],
        "offers": ["buy 2 get 1", "first month $9"],
        "audiences": ["adults 30-55 with poor sleep", "gym-goers 22-40"],
        "constraints": ["no disease claims", "include FDA disclaimer"],
        "nouns": ["supplements", "sleep aids"],
        "incumbents": ["melatonin", "proprietary blends"],
        "trends": ["30-ingredient greens powders", "biohacking stacks"],
    },
    "fitness_equipment": {
        "companies": ["Ironroot", "Kettleform"],
        "products": ["an adjustable kettlebell", "a folding squat rack"],
        "offers": ["free shipping", "$50 off this week"],
        "audiences": ["home gym owners 28-50", "parents with no time for the gym"],
        "constraints": ["show product in a real home", "no shirtless models"],
        "nouns": ["home gyms", "kettlebells"],
        "incumbents": ["the gym membership", "a wall of dumbbells"],
        "trends": ["connected fitness screens", "boutique fitness classes"],
    },
    "apparel": {
        "companies": ["Fieldnote Apparel", "Common Thread"],
        "products": ["a merino travel tee", "a 365-day work pant"],
        "offers": ["two for $120", "free returns for 90 days"],
        "audiences": ["frequent travelers 30-50", "office workers who hate dry cleaning"],
        "constraints": ["no discount language over 25%", "show fabric close-up"],
        "nouns": ["clothes", "basics"],
        "incumbents": ["fast fashion", "the dry cleaner"],
        "trends": ["drop culture", "logo-heavy streetwear"],
    },
    "home_goods": {
        "companies": ["Stillwater Home", "Hearthline"],
        "products": ["a cast-iron pan that needs no seasoning", "a linen duvet set"],
        "offers": ["bundle and save 20%", "lifetime warranty"],
        "audiences": ["homeowners 30-60", "new couples furnishing a first place"],
        "constraints": ["no lifestyle stock photos", "show product in use"],
        "nouns": ["cookware", "bedding"],
        "incumbents": ["nonstick pans", "polyester sheets"],
        "trends": ["viral kitchen gadgets", "matching sets"],
    },
    "pet_food": {
        "companies": ["Good Hound", "Whisker Works"],
        "products": ["fresh dog food delivered weekly", "a single-protein cat kibble"],
        "offers": ["50% off first box", "free vet nutrition call"],
        "audiences": ["dog owners 25-45 in cities", "cat owners with picky eaters"],
        "constraints": ["no vet endorsement claims", "show real pets"],
        "nouns": ["pet food", "kibble"],
        "incumbents": ["big-bag kibble", "grocery store cans"],
        "trends": ["raw feeding", "grain-free everything"],
    },
    "coffee": {
        "companies": ["Ninth Street Roasters", "Halfcaf Co"],
        "products": ["a single-origin subscription", "a low-acid cold brew concentrate"],
        "offers": ["first bag $5", "free grinder with 6-month plan"],
        "audiences": ["remote workers 25-45", "people with acid reflux who love coffee"],
        "constraints": ["no health claims", "show brewing ritual"],
        "nouns": ["coffee", "cold brew"],
        "incumbents": ["the pod machine", "the $7 latte"],
        "trends": ["ultra-light roasts", "coffee tiktok gadgets"],
    },
    "saas_tools": {
        "companies": ["Ledgerly", "Inboxzero.ai"],
        "products": ["bookkeeping software for solo founders", "an email triage assistant"],
        "offers": ["free for 30 days", "first year 40% off"],
        "audiences": ["solo founders 28-45", "consultants drowning in email"],
        "constraints": ["no competitor logos", "show the actual UI"],
        "nouns": ["software", "productivity tools"],
        "incumbents": ["the spreadsheet", "the shared inbox"],
        "trends": ["AI in everything", "all-in-one platforms"],
    },
}

# (tag, description). Tags feed the latent model's world boost; descriptions go in the brief.
WORLD_STATES: list[tuple[str, str]] = [
    ("none", "Nothing unusual in the market this week."),
    (
        "black_friday_approach",
        "Black Friday is three weeks out; feeds are filling with discount ads.",
    ),
    ("heatwave", "A heatwave is dominating news and social feeds."),
    ("new_year", "First week of January; resolution content everywhere."),
    ("back_to_school", "Back-to-school season; parents are the loudest audience."),
    ("competitor_scandal", "A major competitor is in the news for a product recall."),
    (
        "platform_shift",
        "A viral format shift on the platform is rewarding raw, unedited video stills.",
    ),
]


def make_brief(
    rng: np.random.Generator, index: int, category: str, created_at: datetime
) -> dict[str, Any]:
    c = CATEGORIES[category]
    world_tag, world_desc = WORLD_STATES[int(rng.integers(0, len(WORLD_STATES)))]
    company = str(rng.choice(c["companies"]))
    product = str(rng.choice(c["products"]))
    offer = str(rng.choice(c["offers"]))
    audience = str(rng.choice(c["audiences"]))
    constraints = [
        str(x)
        for x in rng.choice(c["constraints"], size=min(2, len(c["constraints"])), replace=False)
    ]
    raw_text = (
        f"{company} sells {product}. Offer: {offer}. Audience: {audience}. "
        f"Constraints: {'; '.join(constraints)}. World: {world_desc}"
    )
    return {
        "company": company,
        "product": product,
        "offer": offer,
        "audience": audience,
        "category": category,
        "world_state": world_desc,
        "world_state_tag": world_tag,
        "world_state_at": created_at,
        "constraints": constraints,
        "raw_text": raw_text,
        "created_at": created_at,
        "meta": {
            "landing_url": f"https://example.com/{category}/{index}",
            "default_targeting_spec": {
                "geo_locations": {"countries": ["US"]},
                "age_min": 25,
                "age_max": 55,
            },
        },
    }


def brief_schedule(
    rng: np.random.Generator, n: int, days_back: int, now: datetime | None = None
) -> list[datetime]:
    now = now or datetime.now(UTC)
    offsets = np.sort(rng.uniform(days_back * 0.4, days_back, size=n))[::-1]
    return [now - timedelta(days=float(o)) for o in offsets]
