"""Card verifier (spec section 5.5).

Independent of the generator. Reads angle, copy, and visual brief only and predicts card
slugs from the active playbook. v1 is a Claude judge with expert-tagged trajectories as
few-shot; the heuristic verifier is the no-API fallback used by tests and the simulator.
`tag_match` is set equality; Jaccard is recorded alongside for calibrating the penalty.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.generation.backends.base import JudgeBackend
from outlier_ai.models.cards import Card
from outlier_ai.models.ops import HoldoutCampaign
from outlier_ai.models.trajectories import Review, Trajectory
from outlier_schemas.enums import TagSource
from outlier_schemas.models import AdCopy

_WORD = re.compile(r"[a-z0-9]+")
_JSON_OBJ = re.compile(r"\{.*\}", re.S)


@dataclass(frozen=True)
class CardRef:
    id: UUID
    slug: str
    name: str
    kind: str
    definition: str


@dataclass(frozen=True)
class FewShot:
    angle: str
    copy: AdCopy
    visual_brief: str
    slugs: list[str]


@dataclass
class VerifierResult:
    card_ids: list[UUID]
    card_slugs: list[str]
    version: str
    source: TagSource
    raw: str | None = None
    scores: dict[str, float] = field(default_factory=dict)


class CardVerifier(Protocol):
    version: str

    async def tag(self, angle: str, copy: AdCopy, visual_brief: str) -> VerifierResult: ...


def idea_text(angle: str, copy: AdCopy, visual_brief: str) -> str:
    return "\n".join(
        [angle, copy.primary_text, copy.headline, copy.description, copy.cta, visual_brief]
    )


def tag_agreement(written: Sequence[UUID], verified: Sequence[UUID]) -> tuple[bool, float]:
    a, b = set(written), set(verified)
    if not a and not b:
        return True, 1.0
    union = a | b
    return a == b, (len(a & b) / len(union)) if union else 1.0


class HeuristicVerifier:
    """Name and slug-phrase matching. Deterministic; good enough for synthetic ideas."""

    version = "verifier-heuristic-v0"

    def __init__(self, cards: Sequence[CardRef], threshold: float = 0.6) -> None:
        self.cards = list(cards)
        self.threshold = threshold

    async def tag(self, angle: str, copy: AdCopy, visual_brief: str) -> VerifierResult:
        text = idea_text(angle, copy, visual_brief).lower()
        words = set(_WORD.findall(text))
        scores: dict[str, float] = {}
        for c in self.cards:
            score = 0.0
            if c.name.lower() in text:
                score = 1.0
            elif c.slug.replace("-", " ") in text:
                score = 0.9
            else:
                slug_words = [w for w in c.slug.split("-") if len(w) > 3]
                if slug_words and all(w in words for w in slug_words):
                    score = 0.7
            if score >= self.threshold:
                scores[c.slug] = score
        picked = sorted(scores, key=lambda s: (-scores[s], s))
        by_slug = {c.slug: c for c in self.cards}
        return VerifierResult(
            card_ids=[by_slug[s].id for s in picked],
            card_slugs=picked,
            version=self.version,
            source=TagSource.verifier,
            scores=scores,
        )


class LLMVerifier:
    version = "verifier-llm-v2"

    def __init__(
        self, judge: JudgeBackend, cards: Sequence[CardRef], few_shot: Sequence[FewShot] = ()
    ) -> None:
        self.judge = judge
        self.cards = list(cards)
        self.few_shot = list(few_shot)
        self.version = f"verifier-llm-v2:{judge.model}"

    def _system(self) -> str:
        lines = [
            "You tag advertising ideas with the playbook cards they actually execute.",
            "Read only the angle, copy, and visual brief. Ignore any stated intent.",
            'Return JSON only: {"cards": ["slug", ...]} using slugs from this list, most central '
            "first. Do not invent slugs.",
            "Tag only the cards that ORGANIZE the execution: the strategy the ad is built on, the "
            "style it is rendered in, and any principle or mechanic that structures the copy or "
            "the offer. Return 2 to 4 slugs.",
            "Incidental features are not cards: a number in the copy is not specific-numbers "
            "unless the claims are number-led throughout; the brief's offer line is not "
            "bundle-offer unless the bundle framing is the ad's offer; mentioning a competitor "
            "or category in passing is not name-the-enemy unless the ad is built against it; "
            "a single demonstrative sentence is not show-dont-tell. When in doubt, leave it out.",
            "",
            "Playbook:",
        ]
        for c in self.cards:
            lines.append(f"- {c.slug} ({c.kind}): {c.name}. {c.definition}")
        return "\n".join(lines)

    def _user(self, angle: str, copy: AdCopy, visual_brief: str) -> str:
        parts: list[str] = []
        for ex in self.few_shot:
            parts.append(
                "Example\n"
                f"{idea_text(ex.angle, ex.copy, ex.visual_brief)}\n"
                f"Answer: {json.dumps({'cards': ex.slugs})}\n"
            )
        parts.append("Now tag this idea.\n" + idea_text(angle, copy, visual_brief) + "\nAnswer:")
        return "\n".join(parts)

    async def tag(self, angle: str, copy: AdCopy, visual_brief: str) -> VerifierResult:
        raw = await self.judge.judge(
            self._system(), self._user(angle, copy, visual_brief), max_tokens=400
        )
        slugs = parse_slug_json(raw)
        by_slug = {c.slug: c for c in self.cards}
        picked = [s for s in slugs if s in by_slug]
        return VerifierResult(
            card_ids=[by_slug[s].id for s in picked],
            card_slugs=picked,
            version=self.version,
            source=TagSource.verifier,
            raw=raw,
        )


def parse_slug_json(raw: str) -> list[str]:
    m = _JSON_OBJ.search(raw or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    cards = data.get("cards", []) if isinstance(data, dict) else []
    if not isinstance(cards, list):
        return []
    out: list[str] = []
    for c in cards:
        if isinstance(c, str):
            s = c.strip().lower()
            if s and s not in out:
                out.append(s)
    return out


async def load_card_refs(session: AsyncSession, *, active_only: bool = True) -> list[CardRef]:
    stmt = select(Card)
    if active_only:
        stmt = stmt.where(Card.status == "active")
    return [
        CardRef(c.id, c.slug, c.name, c.kind, c.definition)
        for c in (await session.execute(stmt)).scalars().all()
    ]


async def load_few_shot(session: AsyncSession, n: int, cards: Sequence[CardRef]) -> list[FewShot]:
    """Expert-tagged trajectories: `wrong_cards` corrections first, then expert-tagged rows.
    Held-out campaigns are excluded (leakage blocklist applies to the verifier)."""
    if n <= 0:
        return []
    holdout = set((await session.execute(select(HoldoutCampaign.campaign_id))).scalars().all())
    by_id = {c.id: c.slug for c in cards}
    out: list[FewShot] = []
    corrected = await session.execute(
        select(Trajectory, Review)
        .join(Review, Review.trajectory_id == Trajectory.id)
        .where(Review.label == "wrong_cards", Review.corrected_card_ids.is_not(None))
        .order_by(Review.reviewed_at.desc())
        .limit(n * 2)
    )
    for traj, review in corrected.all():
        if traj.campaign_id in holdout:
            continue
        slugs = [by_id[i] for i in (review.corrected_card_ids or []) if i in by_id]
        if slugs:
            out.append(
                FewShot(traj.angle, AdCopy(**(traj.ad_copy or {})), traj.visual_brief, slugs)
            )
        if len(out) >= n:
            return out
    experts = await session.execute(
        select(Trajectory)
        .where(Trajectory.tag_source == TagSource.expert.value)
        .order_by(Trajectory.created_at.desc())
        .limit(n * 2)
    )
    for traj in experts.scalars().all():
        if traj.campaign_id in holdout:
            continue
        slugs = [by_id[i] for i in traj.verified_card_ids if i in by_id]
        if slugs:
            out.append(
                FewShot(traj.angle, AdCopy(**(traj.ad_copy or {})), traj.visual_brief, slugs)
            )
        if len(out) >= n:
            break
    return out


class ClassifierVerifier:
    """v2: the trained nearest-centroid classifier (see jobs.feedback) behind the same protocol."""

    def __init__(self, model, embedder, cards: Sequence[CardRef], version: str) -> None:
        self.model = model
        self.embedder = embedder
        self.by_slug = {c.slug: c for c in cards}
        self.version = version

    async def tag(self, angle: str, copy: AdCopy, visual_brief: str) -> VerifierResult:
        text = "\n".join([angle, copy.primary_text, copy.headline, visual_brief])
        slugs = [s for s in self.model.predict(self.embedder.embed([text]))[0] if s in self.by_slug]
        return VerifierResult(
            card_ids=[self.by_slug[s].id for s in slugs],
            card_slugs=slugs,
            version=self.version,
            source=TagSource.verifier,
        )
