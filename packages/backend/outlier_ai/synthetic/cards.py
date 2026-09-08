"""Seed strategy cards from spec section 3.1 plus clearly-marked synthetic filler cards.

The eight strategy cards are real seeds. Style, principle, and mechanic cards are NOT shipped
with the system per the spec; the ones below exist only so the synthetic archive has
combinations to work with. They are tagged `contributed_by = "synthetic"` and should be
retired before real experts author theirs.
"""

from __future__ import annotations

from dataclasses import dataclass

from outlier_schemas.enums import CardKind

SEED_CONTRIBUTOR = "seed"
SYNTHETIC_CONTRIBUTOR = "synthetic"


@dataclass(frozen=True)
class CardSpec:
    slug: str
    name: str
    kind: CardKind
    definition: str
    qualifying_condition: str
    source: str | None
    contributed_by: str
    # Flavor used by the synthetic idea composer, not stored on the card.
    hook: str = ""
    enemy: str = ""
    promise: str = ""


SEED_STRATEGY_CARDS: list[CardSpec] = [
    CardSpec(
        slug="were-killing-x",
        name="We're Killing X",
        kind=CardKind.strategy,
        definition=(
            "Announce the end of an incumbent category, habit, or product. The ad names what is "
            "being killed, says why it deserved to die, and positions the product as the thing "
            "that replaces it."
        ),
        qualifying_condition=(
            "The audience shares a frustration with an incumbent they use out of habit, and the "
            "product is a credible replacement rather than an improvement."
        ),
        source=None,
        contributed_by=SEED_CONTRIBUTOR,
        hook="{incumbent} is over.",
        enemy="{incumbent}",
        promise="Never go back to {incumbent}.",
    ),
    CardSpec(
        slug="process-moats",
        name="Process Moats",
        kind=CardKind.strategy,
        definition=(
            "Show the hard, slow, or unusual process behind the product as the reason it cannot be "
            "copied. The process is the proof and the story."
        ),
        qualifying_condition=(
            "The product has a genuinely distinctive production, sourcing, or service process that "
            "can be shown, not just claimed."
        ),
        source=None,
        contributed_by=SEED_CONTRIBUTOR,
        hook="It takes us {process_detail}. Here's why.",
        enemy="shortcuts",
        promise="You can taste, feel, or see the difference.",
    ),
    CardSpec(
        slug="anti-movements",
        name="Anti-Movements",
        kind=CardKind.strategy,
        definition=(
            "Position against a trend the audience is tired of and rally the people who want to "
            "opt out. The ad gives the fatigued minority a flag."
        ),
        qualifying_condition=(
            "A visible trend in the category with a vocal minority who resent it."
        ),
        source=None,
        contributed_by=SEED_CONTRIBUTOR,
        hook="Tired of {trend}? So are we.",
        enemy="{trend}",
        promise="For people who are done with {trend}.",
    ),
    CardSpec(
        slug="human-desires",
        name="Human Desires",
        kind=CardKind.strategy,
        definition=(
            "Anchor the ad in a primal desire such as status, belonging, safety, ease, or mastery "
            "rather than in a feature. The product is the route to the desire."
        ),
        qualifying_condition=(
            "Always applicable; strongest when the feature story is weak or undifferentiated."
        ),
        source=None,
        contributed_by=SEED_CONTRIBUTOR,
        hook="What you actually want is {desire}.",
        enemy="settling",
        promise="{desire}, finally.",
    ),
    CardSpec(
        slug="10x-product",
        name="10x Product",
        kind=CardKind.strategy,
        definition=(
            "Claim an order-of-magnitude difference on one dimension and prove it inside the ad. "
            "One number, one dimension, one proof."
        ),
        qualifying_condition=(
            "A measurable dimension on which the product is dramatically, provably better."
        ),
        source=None,
        contributed_by=SEED_CONTRIBUTOR,
        hook="{tenx_claim}",
        enemy="incremental",
        promise="Not a little better. {tenx_claim}",
    ),
    CardSpec(
        slug="contrarian",
        name="Contrarian",
        kind=CardKind.strategy,
        definition=(
            "State the opposite of the category's received wisdom and defend it. The ad picks a "
            "fight with a belief, not a competitor."
        ),
        qualifying_condition=(
            "A category with strong received wisdom that the product actually contradicts."
        ),
        source=None,
        contributed_by=SEED_CONTRIBUTOR,
        hook="Everything you were told about {category_noun} is wrong.",
        enemy="received wisdom",
        promise="The {category_noun} that ignores the rules.",
    ),
    CardSpec(
        slug="creativity-and-humor",
        name="Creativity and Humor",
        kind=CardKind.strategy,
        definition=(
            "Earn attention with an unexpected joke or creative device. The product is the "
            "resolution of the punchline, not an interruption of it."
        ),
        qualifying_condition=(
            "A brand voice that tolerates humor and a purchase where levity does not undercut "
            "trust."
        ),
        source=None,
        contributed_by=SEED_CONTRIBUTOR,
        hook="{joke_setup}",
        enemy="boring ads",
        promise="{joke_payoff}",
    ),
    CardSpec(
        slug="objection-flip",
        name="Objection Flip",
        kind=CardKind.strategy,
        definition=(
            "Take what critics say and make it the ad. Lead with the objection verbatim and "
            "turn it into the reason to buy."
        ),
        qualifying_condition=(
            "A recurring objection visible in comments, reviews, or support tickets."
        ),
        source=None,
        contributed_by=SEED_CONTRIBUTOR,
        hook="\"{objection}\" Yes. Here's why that's the point.",
        enemy="the objection itself",
        promise="The thing they complain about is the thing that works.",
    ),
]


def _syn(slug: str, name: str, kind: CardKind, definition: str, qual: str) -> CardSpec:
    return CardSpec(
        slug=slug,
        name=name,
        kind=kind,
        definition=definition,
        qualifying_condition=qual,
        source="synthetic placeholder; replace with expert-authored card",
        contributed_by=SYNTHETIC_CONTRIBUTOR,
    )


SYNTHETIC_CARDS: list[CardSpec] = [
    # styles
    _syn(
        "long-copy-editorial",
        "Long-Copy Editorial",
        CardKind.style,
        "Primary text runs several paragraphs and reads like an article.",
        "Considered purchases where the reader wants to be convinced.",
    ),
    _syn(
        "ugly-ad-native",
        "Ugly Native Ad",
        CardKind.style,
        "Deliberately unpolished, looks like a user post rather than an ad.",
        "Feeds saturated with polished brand creative.",
    ),
    _syn(
        "founder-letter",
        "Founder Letter",
        CardKind.style,
        "Written in the first person by the founder, signed.",
        "Brands with a real founder story and a face.",
    ),
    _syn(
        "meme-format",
        "Meme Format",
        CardKind.style,
        "Uses a recognizable meme structure as the creative frame.",
        "Younger audiences; low-stakes purchases.",
    ),
    _syn(
        "testimonial-screenshot",
        "Testimonial Screenshot",
        CardKind.style,
        "The visual is a screenshot of a real customer message or review.",
        "Products with strong word of mouth.",
    ),
    _syn(
        "before-after-split",
        "Before/After Split",
        CardKind.style,
        "Two-panel visual contrasting the state before and after.",
        "Visible transformations that can be shown honestly.",
    ),
    # principles
    _syn(
        "specific-numbers",
        "Specific Numbers",
        CardKind.principle,
        "Every claim carries an exact figure rather than an adjective.",
        "Any claim that can be quantified.",
    ),
    _syn(
        "one-idea-per-ad",
        "One Idea Per Ad",
        CardKind.principle,
        "The ad makes exactly one point and repeats it.",
        "Always.",
    ),
    _syn(
        "name-the-enemy",
        "Name the Enemy",
        CardKind.principle,
        "The ad explicitly names what it is against.",
        "When there is a clear enemy the audience recognizes.",
    ),
    _syn(
        "show-dont-tell",
        "Show, Don't Tell",
        CardKind.principle,
        "Demonstrate the benefit visually instead of asserting it.",
        "Physical products and visible outcomes.",
    ),
    _syn(
        "price-anchor",
        "Price Anchor",
        CardKind.principle,
        "Set an expensive reference point before revealing the price.",
        "Premium-priced products.",
    ),
    _syn(
        "social-proof-first",
        "Social Proof First",
        CardKind.principle,
        "Open with how many people already bought or agree.",
        "Products with real volume or ratings.",
    ),
    _syn(
        "curiosity-gap",
        "Curiosity Gap",
        CardKind.principle,
        "Withhold the key detail so the reader clicks to close the gap.",
        "Ads driving to a landing page that pays off the gap.",
    ),
    # mechanics
    _syn(
        "countdown-urgency",
        "Countdown Urgency",
        CardKind.mechanic,
        "A deadline or limited quantity stated in the copy.",
        "Real deadlines only.",
    ),
    _syn(
        "comparison-table",
        "Comparison Table",
        CardKind.mechanic,
        "A visual grid comparing the product to alternatives.",
        "Products that win on several visible dimensions.",
    ),
    _syn(
        "quiz-hook",
        "Quiz Hook",
        CardKind.mechanic,
        "Opens with a question the reader answers about themselves.",
        "Personalized or multi-variant products.",
    ),
    _syn(
        "comment-bait-question",
        "Comment-Bait Question",
        CardKind.mechanic,
        "Ends with a question designed to draw comments.",
        "When engagement signals are wanted.",
    ),
    _syn(
        "founder-face",
        "Founder Face",
        CardKind.mechanic,
        "The visual features the founder looking at camera.",
        "Brands with a founder willing to be the face.",
    ),
    _syn(
        "bundle-offer",
        "Bundle Offer",
        CardKind.mechanic,
        "The offer is a bundle framed as a saving.",
        "Products with natural complements.",
    ),
    _syn(
        "guarantee-lead",
        "Guarantee Lead",
        CardKind.mechanic,
        "The guarantee is the headline.",
        "Products with a strong guarantee and low return rates.",
    ),
]

ALL_CARDS: list[CardSpec] = SEED_STRATEGY_CARDS + SYNTHETIC_CARDS


def cards_by_kind(specs: list[CardSpec] | None = None) -> dict[CardKind, list[CardSpec]]:
    out: dict[CardKind, list[CardSpec]] = {k: [] for k in CardKind}
    for c in specs or ALL_CARDS:
        out[c.kind].append(c)
    return out
