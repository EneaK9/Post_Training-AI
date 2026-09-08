"""Domain objects from spec section 3, as pydantic models.

These are the API and internal transfer shapes. The ORM tables in the backend mirror
them; derived objects (ScreeningStats, Outcome, Combination) are always recomputed
from raw rows and config, never edited by hand.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from outlier_schemas.enums import (
    AccountStatus,
    AuthorKind,
    BackendKind,
    BaselineKind,
    BatchState,
    CardKind,
    CardStatus,
    Channel,
    EpisodeStatus,
    EvidenceSource,
    GoalMetric,
    ImageMode,
    OutcomeMetric,
    RelationKind,
    RelationSource,
    RelationStatus,
    RenderStatus,
    ReviewLabel,
    Sentiment,
    SignalKind,
    SignalStatus,
    TagSource,
    Tier,
    Typicality,
)


class Model(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        use_enum_values=False,
        extra="forbid",
    )


def new_id() -> UUID:
    return uuid4()


def short_ref(object_id: UUID) -> str:
    """Eight-hex-character reference used inside prompts and citations.

    Full ids are long for a language model to copy. PromptTrace keeps the ref -> id map
    so the parser can resolve citations back to full ids.
    """
    return object_id.hex[:8]


# --------------------------------------------------------------------------------------
# Column 1: Playbook
# --------------------------------------------------------------------------------------


class Card(Model):
    id: UUID = Field(default_factory=new_id)
    slug: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(min_length=1, max_length=120)
    kind: CardKind
    definition: str = Field(min_length=1)
    qualifying_condition: str = ""
    source: str | None = None
    contributed_by: str
    status: CardStatus = CardStatus.draft
    version: int = 1
    example_trajectory_ids: list[UUID] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CardRelation(Model):
    id: UUID = Field(default_factory=new_id)
    from_id: UUID
    to_id: UUID
    kind: RelationKind
    source: RelationSource
    status: RelationStatus = RelationStatus.pending
    created_by: str
    created_at: datetime | None = None


class Combination(Model):
    """Derived. One row per distinct set of card ids ever used."""

    id: UUID = Field(default_factory=new_id)
    card_ids: list[UUID]
    niche_key: str
    uses: int = 0
    tier_counts: dict[int, int] = Field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    tier2_rate: float | None = None
    first_used: datetime | None = None
    last_used: datetime | None = None
    named_as_card_id: UUID | None = None

    @field_validator("card_ids")
    @classmethod
    def _sorted_unique(cls, v: list[UUID]) -> list[UUID]:
        if len(set(v)) != len(v):
            raise ValueError("card_ids must be unique")
        return sorted(v)


# --------------------------------------------------------------------------------------
# Column 2: Brief
# --------------------------------------------------------------------------------------


class BriefMeta(Model):
    ad_account_id: str | None = None
    pixel_id: str | None = None
    page_id: str | None = None
    default_targeting_spec: dict[str, Any] = Field(default_factory=dict)
    landing_url: str | None = None


class Brief(Model):
    id: UUID = Field(default_factory=new_id)
    created_by: str
    created_at: datetime | None = None
    company: str
    product: str
    offer: str
    audience: str
    goal_metric: GoalMetric = GoalMetric.purchases
    channel: Channel = Channel.meta_feed_image
    category: str
    world_state: str = ""
    world_state_at: datetime | None = None
    constraints: list[str] = Field(default_factory=list)
    brand_assets: list[str] = Field(default_factory=list)
    raw_text: str = ""
    meta: BriefMeta = Field(default_factory=BriefMeta)


# --------------------------------------------------------------------------------------
# Column 3: History
# --------------------------------------------------------------------------------------


class AdCopy(Model):
    primary_text: str = ""
    headline: str = ""
    description: str = ""
    cta: str = ""


class Engagement(Model):
    reactions: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0


class ScreeningStats(Model):
    """Derived from daily rows over the screening window."""

    impressions: int
    link_clicks: int
    spend: float
    ctr: float
    cpc: float | None
    ctr_lower_bound: float
    account_median_ctr_90d: float
    screening_ratio: float
    passed: bool
    window_complete: bool
    engagement: Engagement = Field(default_factory=Engagement)
    config_hash: str | None = None


class OutcomeGates(Model):
    volume_ok: bool
    durability_ok: bool
    category_ok: bool


class Outcome(Model):
    """Derived from daily rows at scale. The only training reward."""

    metric: OutcomeMetric
    value: float
    impressions: int
    conversions: int
    spend: float
    revenue: float
    days_at_scale: int
    account_median_90d: float
    account_median_human_90d: float | None
    category_median: float
    baseline: BaselineKind
    ratio: float
    ratio_lower_bound: float
    outlier_tier: Tier
    gates: OutcomeGates
    attribution_setting: str
    measured_at: datetime
    config_hash: str | None = None


class Review(Model):
    label: ReviewLabel
    corrected_card_ids: list[UUID] | None = None
    reviewer_id: str
    note: str = ""
    reviewed_at: datetime | None = None


class Note(Model):
    id: UUID = Field(default_factory=new_id)
    trajectory_id: UUID
    author_id: str
    text: str
    created_at: datetime | None = None


class Render(Model):
    id: UUID = Field(default_factory=new_id)
    trajectory_id: UUID
    image_url: str | None = None
    image_backend: str
    seed: int
    size: tuple[int, int]
    meta_ad_id: str | None = None
    meta_adset_id: str | None = None
    meta_creative_id: str | None = None
    effective_object_story_id: str | None = None
    status: RenderStatus = RenderStatus.draft
    rejection_reason: str | None = None
    screening: ScreeningStats | None = None
    scale: Outcome | None = None


class Evidence(Model):
    source: EvidenceSource
    ref: str
    excerpt: str


class Signal(Model):
    id: UUID = Field(default_factory=new_id)
    trajectory_id: UUID
    render_id: UUID | None = None
    kind: SignalKind
    text: str
    evidence: list[Evidence] = Field(default_factory=list)
    sentiment: Sentiment = Sentiment.neu
    count: int = 1
    extracted_by: AuthorKind
    status: SignalStatus = SignalStatus.proposed
    created_at: datetime | None = None


class PreshipReport(Model):
    policy_ok: bool = True
    policy_flags: list[str] = Field(default_factory=list)
    brand_ok: bool = True
    brand_flags: list[str] = Field(default_factory=list)
    length_warnings: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.policy_ok and self.brand_ok


class Trajectory(Model):
    id: UUID = Field(default_factory=new_id)
    episode_id: UUID | None = None
    batch_id: UUID | None = None
    brief_id: UUID
    campaign_id: str | None = None
    attempt_index: int = 0
    author_id: str
    author_kind: AuthorKind
    backend: BackendKind | None = None
    card_ids: list[UUID] = Field(default_factory=list)
    verified_card_ids: list[UUID] = Field(default_factory=list)
    tag_source: TagSource | None = None
    tag_match: bool | None = None
    tag_jaccard: float | None = None
    typicality: Typicality | None = None
    format_ok: bool = True
    format_errors: list[str] = Field(default_factory=list)
    reasoning: str = ""
    cited_ids: list[UUID] = Field(default_factory=list)
    angle: str = ""
    ad_copy: AdCopy = Field(default_factory=AdCopy, alias="copy")
    visual_brief: str = ""
    renders: list[Render] = Field(default_factory=list)
    outcome: Outcome | None = None
    signals: list[UUID] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    review: Review | None = None
    preship: PreshipReport | None = None
    rm_score: float | None = None
    rm_version: str | None = None
    library_version: int = 0
    config_hash: str | None = None
    created_at: datetime | None = None


# --------------------------------------------------------------------------------------
# Search episodes and batches [v3-assumed]
# --------------------------------------------------------------------------------------


class SearchEpisode(Model):
    id: UUID = Field(default_factory=new_id)
    brief_id: UUID
    ad_account_id: UUID | None = None
    backend: BackendKind
    budget_cap: float
    spent: float = 0.0
    status: EpisodeStatus = EpisodeStatus.searching
    keep_running_after_outlier: bool = False
    config_hash: str | None = None
    created_by: str
    created_at: datetime | None = None
    ended_at: datetime | None = None


class Batch(Model):
    id: UUID = Field(default_factory=new_id)
    episode_id: UUID
    index: int
    state: BatchState = BatchState.proposed
    prompt_trace: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


# --------------------------------------------------------------------------------------
# Raw facts
# --------------------------------------------------------------------------------------


class DailyInsight(Model):
    """One raw row per render per day. Everything derived starts here."""

    render_id: UUID
    day: date
    impressions: int = 0
    link_clicks: int = 0
    spend: float = 0.0
    purchases: int = 0
    revenue: float = 0.0
    frequency: float | None = None
    reactions: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    attribution_setting: str = ""
    daily_budget: float | None = None
    phase: str = "screening"  # screening | scale


class Comment(Model):
    id: UUID = Field(default_factory=new_id)
    render_id: UUID
    commenter_hash: str
    text: str
    created_time: datetime
    like_count: int = 0
    filtered_reason: str | None = None
    external_id: str | None = None


class AdAccount(Model):
    id: UUID = Field(default_factory=new_id)
    name: str
    meta_account_id: str
    page_id: str | None = None
    pixel_id: str | None = None
    status: AccountStatus = AccountStatus.active
    attribution_setting: str
    api_version: str
    category: str | None = None
    daily_cap_usd: float | None = None


class GenerationRequest(Model):
    brief_id: UUID
    episode_id: UUID | None = None
    backend: BackendKind = BackendKind.fake
    k: int = Field(default=8, ge=1, le=64)
    image_mode: ImageMode = ImageMode.brand_assets
    renders_per_idea: int = Field(default=3, ge=1, le=6)
    size: tuple[int, int] = (1080, 1080)
