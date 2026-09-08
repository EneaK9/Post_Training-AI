from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from outlier_schemas.enums import RewardModelKind
from outlier_schemas.models import AdCopy


@dataclass(frozen=True)
class IdeaFeatures:
    brief_text: str
    world_state: str
    angle: str
    copy: AdCopy
    visual_brief: str
    verified_card_slugs: list[str]
    typicality: str | None
    novelty_distance: float | None
    cited_signal_count: int
    format_ok: bool
    tag_match: bool | None
    cited_signal_texts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RewardScore:
    mean: float
    std: float
    version: str
    kind: RewardModelKind

    def rm_score(self, lambda_pess: float) -> float:
        return self.mean - lambda_pess * self.std


class RewardModel(Protocol):
    version: str
    kind: RewardModelKind

    async def score(self, features: IdeaFeatures) -> RewardScore: ...
