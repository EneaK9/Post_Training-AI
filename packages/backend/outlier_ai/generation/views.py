"""Plain views the renderer consumes. Built from ORM rows by `assemble.py`, or by hand in tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class CardView:
    slug: str
    name: str
    kind: str
    definition: str
    qualifying_condition: str


@dataclass(frozen=True)
class BriefView:
    id: UUID
    company: str
    product: str
    offer: str
    audience: str
    category: str
    goal_metric: str
    channel: str
    world_state: str
    world_state_at: datetime | None
    constraints: list[str]
    raw_text: str


@dataclass(frozen=True)
class SignalView:
    ref: str
    kind: str
    text: str
    sentiment: str
    count: int
    status: str


@dataclass(frozen=True)
class HistoryItemView:
    ref: str
    reason: str
    same_brief: bool
    brief_company: str
    card_slugs: list[str]
    verified_card_slugs: list[str]
    niche: str
    typicality: str | None
    angle: str
    copy: dict[str, Any]
    tier: int | None
    ratio: float | None
    screening_ratio: float | None
    days_ago: int | None
    signals: list[SignalView] = field(default_factory=list)


@dataclass(frozen=True)
class BatchView:
    index: int
    state: str
    items: list[HistoryItemView]


@dataclass(frozen=True)
class AccountHistoryView:
    """How often each niche has been run on the account; drives the typicality labels."""

    niche_uses: dict[str, int]
