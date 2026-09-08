"""The Meta client protocol and its data types. [v3-assumed] surface, see docs/assumptions.md.

Only the operations the spec needs: image upload, creative, campaign, ad set, ad, review
polling, daily insights with engagement action types, post comments, budget change, pause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

AD_NAME_PREFIX = "OAI"


def ad_object_name(episode_id: str, trajectory_id: str, render_id: str) -> str:
    """`OAI|episode|trajectory|render` per spec section 8."""
    return f"{AD_NAME_PREFIX}|{episode_id}|{trajectory_id}|{render_id}"


def parse_ad_object_name(name: str) -> tuple[str, str, str] | None:
    parts = name.split("|")
    if len(parts) != 4 or parts[0] != AD_NAME_PREFIX:
        return None
    return parts[1], parts[2], parts[3]


@dataclass(frozen=True)
class AdImageRef:
    image_hash: str
    url: str


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    objective: str = "OUTCOME_SALES"
    status: str = "ACTIVE"
    special_ad_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdSetSpec:
    name: str
    campaign_id: str
    daily_budget_cents: int
    targeting: dict[str, Any]
    pixel_id: str | None
    attribution_setting: str
    optimization_goal: str = "OFFSITE_CONVERSIONS"
    billing_event: str = "IMPRESSIONS"
    placements: tuple[str, ...] = ("facebook_feed", "instagram_feed")
    status: str = "ACTIVE"


@dataclass(frozen=True)
class CreativeSpec:
    name: str
    page_id: str
    image_hash: str
    primary_text: str
    headline: str
    description: str
    cta: str
    link_url: str


@dataclass(frozen=True)
class AdSpec:
    name: str
    adset_id: str
    creative_id: str
    status: str = "ACTIVE"


@dataclass(frozen=True)
class ReviewStatus:
    ad_id: str
    effective_status: str  # PENDING_REVIEW | ACTIVE | DISAPPROVED | PAUSED | ARCHIVED
    review_feedback: str | None = None

    @property
    def approved(self) -> bool:
        return self.effective_status == "ACTIVE"

    @property
    def rejected(self) -> bool:
        return self.effective_status == "DISAPPROVED"


@dataclass(frozen=True)
class InsightRow:
    ad_id: str
    day: date
    impressions: int
    link_clicks: int
    spend: float
    purchases: int
    revenue: float
    frequency: float | None
    reactions: int
    comments: int
    shares: int
    saves: int
    attribution_setting: str


@dataclass(frozen=True)
class PostComment:
    id: str
    post_id: str
    from_id: str
    message: str
    created_time: datetime
    like_count: int = 0


@dataclass(frozen=True)
class CreativeResult:
    creative_id: str
    effective_object_story_id: str


@dataclass
class AccountInfo:
    account_id: str
    name: str
    currency: str = "USD"
    timezone_name: str = "UTC"
    api_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class MetaClient(Protocol):
    """Everything the system does against Meta. Async so the fake can hit the database."""

    account_id: str

    async def get_account_info(self) -> AccountInfo: ...
    async def upload_image(self, image_bytes: bytes, name: str) -> AdImageRef: ...
    async def create_campaign(self, spec: CampaignSpec) -> str: ...
    async def create_adset(self, spec: AdSetSpec) -> str: ...
    async def create_creative(self, spec: CreativeSpec) -> CreativeResult: ...
    async def create_ad(self, spec: AdSpec) -> str: ...
    async def get_ad_review_status(self, ad_ids: list[str]) -> list[ReviewStatus]: ...
    async def get_daily_insights(self, ad_ids: list[str], day: date) -> list[InsightRow]: ...
    async def get_post_comments(
        self, effective_object_story_id: str, since: datetime | None = None
    ) -> list[PostComment]: ...
    async def update_adset_budget(self, adset_id: str, daily_budget_cents: int) -> None: ...
    async def pause_ad(self, ad_id: str) -> None: ...
