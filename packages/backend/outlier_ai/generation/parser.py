"""Strict multi-idea parser for the section 5.4 grammar.

Tolerant in what it accepts as input (never raises), strict in what it reports: every
deviation is an error on the idea, `format_ok` is false, and the idea earns zero reward plus
the format penalty downstream. Card slugs are resolved against the active playbook; history
and signal refs against the prompt trace's ref map.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import UUID

from outlier_schemas.enums import Typicality
from outlier_schemas.models import AdCopy

_IDEA = re.compile(r"<idea>(.*?)(?:</idea>|(?=<idea>)|\Z)", re.IGNORECASE | re.DOTALL)
_FIELDS = ("cards", "typicality", "reasoning", "angle", "copy", "visual_brief")
_FIELD_RE = {f: re.compile(rf"<{f}>(.*?)</{f}>", re.IGNORECASE | re.DOTALL) for f in _FIELDS}
_CITATION = re.compile(r"\[(card|history|signal):([^\]\s]+)\]", re.IGNORECASE)
_COPY_KEYS = ("primary_text", "headline", "description", "cta")
_COPY_KEY_RE = re.compile(
    r"^\s*(primary_text|headline|description|cta)\s*:\s*", re.IGNORECASE | re.MULTILINE
)
_SLUG_SPLIT = re.compile(r"[,\n;|]+|\s{2,}")


@dataclass
class Citations:
    cards: list[str] = field(default_factory=list)
    history: list[UUID] = field(default_factory=list)
    signals: list[UUID] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def ids(self) -> list[UUID]:
        return [*self.history, *self.signals]


@dataclass
class ParsedIdea:
    index: int
    raw: str
    card_slugs: list[str]
    card_ids: list[UUID]
    typicality: Typicality | None
    reasoning: str
    citations: Citations
    angle: str
    copy: AdCopy
    visual_brief: str
    errors: list[str]

    @property
    def format_ok(self) -> bool:
        return not self.errors


def _clean_slug(token: str) -> str:
    t = token.strip().strip("[]").strip()
    if t.lower().startswith("card:"):
        t = t[5:]
    return t.strip().lower()


def parse_copy(block: str) -> tuple[AdCopy, list[str]]:
    """`key: value` lines; primary_text may span lines until the next key."""
    errors: list[str] = []
    values: dict[str, str] = {}
    matches = list(_COPY_KEY_RE.finditer(block))
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        value = block[m.end() : end].strip()
        if key in values:
            errors.append(f"copy: duplicate key {key}")
        values[key] = value
    for key in _COPY_KEYS:
        if key not in values:
            errors.append(f"copy: missing {key}")
        elif not values[key]:
            errors.append(f"copy: empty {key}")
    return AdCopy(**{k: values.get(k, "") for k in _COPY_KEYS}), errors


def parse_ideas(
    text: str,
    *,
    cards_by_slug: Mapping[str, UUID],
    ref_map: Mapping[str, UUID] | None = None,
    min_cards: int = 2,
    expected_k: int | None = None,
) -> list[ParsedIdea]:
    ref_map = ref_map or {}
    ideas: list[ParsedIdea] = []
    blocks = list(_IDEA.finditer(text or ""))
    for index, m in enumerate(blocks):
        raw = m.group(0)
        body = m.group(1)
        errors: list[str] = []
        if not raw.rstrip().lower().endswith("</idea>"):
            errors.append("missing </idea>")

        fields: dict[str, str] = {}
        for f in _FIELDS:
            fm = _FIELD_RE[f].search(body)
            if fm is None:
                errors.append(f"missing <{f}>")
                fields[f] = ""
            else:
                fields[f] = fm.group(1).strip()
                if not fields[f]:
                    errors.append(f"empty <{f}>")

        # cards
        slugs: list[str] = []
        for tok in _SLUG_SPLIT.split(fields["cards"]):
            s = _clean_slug(tok)
            if s:
                slugs.append(s)
        seen: set[str] = set()
        unique: list[str] = []
        for s in slugs:
            if s in seen:
                errors.append(f"cards: duplicate {s}")
                continue
            seen.add(s)
            unique.append(s)
        card_ids: list[UUID] = []
        for s in unique:
            cid = cards_by_slug.get(s)
            if cid is None:
                errors.append(f"cards: unknown or inactive card {s}")
            else:
                card_ids.append(cid)
        if len(card_ids) < min_cards:
            errors.append(f"cards: need at least {min_cards}, got {len(card_ids)}")

        # typicality
        typ: Typicality | None = None
        raw_typ = fields["typicality"].strip().lower()
        if raw_typ:
            try:
                typ = Typicality(raw_typ)
            except ValueError:
                errors.append(f"typicality: invalid value {raw_typ!r}")

        # citations from reasoning
        cits = Citations()
        for kind, ref in _CITATION.findall(fields["reasoning"]):
            kind = kind.lower()
            ref = ref.strip()
            if kind == "card":
                s = _clean_slug(ref)
                if s in cards_by_slug:
                    cits.cards.append(s)
                else:
                    cits.unresolved.append(f"card:{ref}")
            else:
                rid = ref_map.get(ref)
                if rid is None:
                    try:
                        rid = UUID(ref)
                    except ValueError:
                        rid = None
                if rid is None:
                    cits.unresolved.append(f"{kind}:{ref}")
                elif kind == "history":
                    cits.history.append(rid)
                else:
                    cits.signals.append(rid)

        copy, copy_errors = parse_copy(fields["copy"])
        errors.extend(copy_errors)

        ideas.append(
            ParsedIdea(
                index=index,
                raw=raw,
                card_slugs=unique,
                card_ids=card_ids,
                typicality=typ,
                reasoning=fields["reasoning"],
                citations=cits,
                angle=fields["angle"],
                copy=copy,
                visual_brief=fields["visual_brief"],
                errors=errors,
            )
        )

    if expected_k is not None and len(ideas) != expected_k:
        # Set-level deviation is recorded on every idea so no idea silently benefits.
        for idea in ideas:
            idea.errors.append(f"set: expected {expected_k} ideas, got {len(ideas)}")
    return ideas


def set_uses_distinct_combinations(ideas: list[ParsedIdea]) -> list[int]:
    """Indices of ideas whose card set duplicates an earlier idea's set."""
    seen: dict[tuple[UUID, ...], int] = {}
    dupes: list[int] = []
    for idea in ideas:
        key = tuple(sorted(idea.card_ids))
        if not key:
            continue
        if key in seen:
            dupes.append(idea.index)
        else:
            seen[key] = idea.index
    return dupes
