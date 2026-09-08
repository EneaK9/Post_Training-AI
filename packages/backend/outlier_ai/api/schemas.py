"""Request and response shapes for the API. ORM rows are converted in `api/views.py`."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from outlier_schemas.enums import (
    BackendKind,
    CardKind,
    RelationKind,
    ReviewLabel,
    Sentiment,
    SignalKind,
)
from outlier_schemas.models import AdCopy, PreshipReport


class Resp(BaseModel):
    """Response models: every field is always present in output, so the OpenAPI schema marks
    defaulted fields as required and the generated TypeScript types are non-optional."""

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_serialization_defaults_required=True,
    )


class Out(Resp):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        serialize_by_alias=True,
        json_schema_serialization_defaults_required=True,
    )


# ---- auth ----------------------------------------------------------------------------
class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(Out):
    id: UUID
    email: str
    display_name: str
    role: str


# ---- cards ---------------------------------------------------------------------------
class CardStats(Resp):
    uses: int = 0
    measured: int = 0
    tier2_count: int = 0
    tier2_rate: float | None = None
    verifier_agreement: float | None = None


class CardOut(Out):
    id: UUID
    slug: str
    name: str
    kind: str
    definition: str
    qualifying_condition: str
    source: str | None
    contributed_by: str
    status: str
    version: int
    created_at: datetime
    updated_at: datetime
    stats: CardStats = Field(default_factory=CardStats)


class CardCreate(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=64)
    name: str = Field(min_length=1, max_length=120)
    kind: CardKind
    definition: str = Field(min_length=1)
    qualifying_condition: str = ""
    source: str | None = None
    status: str = Field(default="draft", pattern=r"^(draft|active)$")


class CardUpdate(BaseModel):
    name: str | None = None
    kind: CardKind | None = None
    definition: str | None = None
    qualifying_condition: str | None = None
    source: str | None = None


class CardVersionOut(Out):
    id: UUID
    card_id: UUID
    version: int
    name: str
    kind: str
    definition: str
    qualifying_condition: str
    source: str | None
    status: str
    edited_by: str
    created_at: datetime


class RelationOut(Out):
    id: UUID
    from_id: UUID
    to_id: UUID
    kind: str
    source: str
    status: str
    created_by: str
    evidence: dict[str, Any] | None
    created_at: datetime


class RelationCreate(BaseModel):
    from_id: UUID
    to_id: UUID
    kind: RelationKind


class RelationDecision(BaseModel):
    status: str = Field(pattern=r"^(accepted|rejected)$")


# ---- combinations --------------------------------------------------------------------
class CombinationOut(Out):
    id: UUID
    card_ids: list[UUID]
    card_slugs: list[str] = Field(default_factory=list)
    niche_key: str
    uses: int
    tier_counts: dict[str, int]
    tier2_rate: float | None
    first_used: datetime | None
    last_used: datetime | None
    named_as_card_id: UUID | None


class NameAsCardIn(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=64)
    name: str
    kind: CardKind = CardKind.strategy
    definition: str
    qualifying_condition: str = ""


# ---- briefs --------------------------------------------------------------------------
class BriefCreate(BaseModel):
    company: str
    product: str
    offer: str
    audience: str
    category: str
    goal_metric: str = "purchases"
    channel: str = "meta_feed_image"
    world_state: str = ""
    constraints: list[str] = Field(default_factory=list)
    brand_assets: list[str] = Field(default_factory=list)
    raw_text: str = ""
    ad_account_id: UUID | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class BriefUpdate(BaseModel):
    company: str | None = None
    product: str | None = None
    offer: str | None = None
    audience: str | None = None
    category: str | None = None
    world_state: str | None = None
    constraints: list[str] | None = None
    brand_assets: list[str] | None = None
    raw_text: str | None = None
    meta: dict[str, Any] | None = None


class BriefOut(Out):
    id: UUID
    created_by: str
    created_at: datetime
    ad_account_id: UUID | None
    company: str
    product: str
    offer: str
    audience: str
    goal_metric: str
    channel: str
    category: str
    world_state: str
    world_state_at: datetime | None
    constraints: list[str]
    brand_assets: list[str]
    raw_text: str
    meta: dict[str, Any]
    episode_count: int = 0
    trajectory_count: int = 0


# ---- trajectories --------------------------------------------------------------------
class ScreeningOut(Out):
    impressions: int
    link_clicks: int
    spend: float
    ctr: float
    ctr_lower_bound: float
    account_median_ctr_90d: float
    screening_ratio: float
    passed: bool
    window_complete: bool
    engagement: dict[str, Any]


class OutcomeOut(Out):
    metric: str
    value: float
    conversions: int
    spend: float
    revenue: float
    days_at_scale: int
    account_median_90d: float
    category_median: float
    baseline: str
    ratio: float
    ratio_lower_bound: float
    outlier_tier: int
    gates: dict[str, Any]
    attribution_setting: str
    measured_at: datetime


class RenderOut(Out):
    id: UUID
    seed: int
    width: int
    height: int
    image_uri: str | None
    image_backend: str
    status: str
    rejection_reason: str | None
    meta_ad_id: str | None
    shipped_at: datetime | None
    screening: ScreeningOut | None = None
    outcome: OutcomeOut | None = None


class ReviewOut(Out):
    label: str
    corrected_card_ids: list[UUID] | None
    reviewer_id: str
    note: str
    reviewed_at: datetime


class ReviewIn(BaseModel):
    label: ReviewLabel
    corrected_card_ids: list[UUID] | None = None
    note: str = ""


class NoteOut(Out):
    id: UUID
    author_id: str
    text: str
    created_at: datetime


class NoteIn(BaseModel):
    text: str = Field(min_length=1)


class SignalOut(Out):
    id: UUID
    trajectory_id: UUID
    render_id: UUID | None
    kind: str
    text: str
    evidence: list[dict[str, Any]]
    sentiment: str
    count: int
    extracted_by: str
    status: str
    decided_by: str | None
    decided_at: datetime | None
    created_at: datetime


class SignalCreate(BaseModel):
    trajectory_id: UUID
    render_id: UUID | None = None
    kind: SignalKind = SignalKind.note
    text: str = Field(min_length=1)
    sentiment: Sentiment = Sentiment.neu
    count: int = 1


class CommentOut(Out):
    id: UUID
    render_id: UUID
    text: str
    created_time: datetime
    like_count: int
    filtered_reason: str | None


class TrajectoryOut(Out):
    id: UUID
    brief_id: UUID
    episode_id: UUID | None
    batch_id: UUID | None
    campaign_id: str | None
    attempt_index: int
    author_id: str
    author_kind: str
    backend: str | None
    card_ids: list[UUID]
    verified_card_ids: list[UUID]
    card_slugs: list[str] = Field(default_factory=list)
    verified_card_slugs: list[str] = Field(default_factory=list)
    tag_source: str | None
    tag_match: bool | None
    tag_jaccard: float | None
    typicality: str | None
    format_ok: bool
    format_errors: list[str]
    reasoning: str
    cited_ids: list[UUID]
    angle: str
    ad_copy: AdCopy = Field(validation_alias="copy", serialization_alias="copy")
    visual_brief: str
    preship: PreshipReport | None
    outlier_tier: int | None
    screening_ratio: float | None = None
    ratio: float | None = None
    rm_score: float | None
    rm_version: str | None
    library_version: int
    config_hash: str | None
    created_at: datetime
    review: ReviewOut | None = None
    renders: list[RenderOut] = Field(default_factory=list)
    signals: list[SignalOut] = Field(default_factory=list)
    notes: list[NoteOut] = Field(default_factory=list)


class TrajectoryDetail(TrajectoryOut):
    comments: list[CommentOut] = Field(default_factory=list)


class CopyEdit(BaseModel):
    primary_text: str
    headline: str
    description: str
    cta: str


class Paged(Resp):
    total: int
    items: list[Any]


# ---- episodes ------------------------------------------------------------------------
class EpisodeCreate(BaseModel):
    brief_id: UUID
    budget_cap: float | None = Field(default=None, gt=0)
    backend: BackendKind = BackendKind.fake
    ad_account_id: UUID | None = None
    keep_running_after_outlier: bool = False


class BatchOut(Out):
    id: UUID
    index: int
    state: str
    created_at: datetime
    trajectory_ids: list[UUID] = Field(default_factory=list)
    prompt_trace: dict[str, Any] | None = None


class EpisodeOut(Out):
    id: UUID
    brief_id: UUID
    ad_account_id: UUID | None
    backend: str
    budget_cap: float
    spent: float
    status: str
    keep_running_after_outlier: bool
    campaign_id: str | None
    config_hash: str | None
    created_by: str
    created_at: datetime
    ended_at: datetime | None
    stop_reason: str | None
    batches: list[BatchOut] = Field(default_factory=list)
    live_ads: int = 0
    best_tier: int | None = None
    new_signals: int = 0


# ---- accounts ------------------------------------------------------------------------
class AccountCreate(BaseModel):
    name: str
    meta_account_id: str
    page_id: str | None = None
    pixel_id: str | None = None
    attribution_setting: str = "7d_click_1d_view"
    category: str | None = None
    daily_cap_usd: float | None = None
    access_token: str | None = None
    page_token: str | None = None
    is_fake: bool = False


class AccountUpdate(BaseModel):
    name: str | None = None
    status: str | None = Field(default=None, pattern=r"^(active|needs_reauth|disabled)$")
    daily_cap_usd: float | None = None
    access_token: str | None = None
    page_token: str | None = None
    category: str | None = None


class AccountOut(Out):
    id: UUID
    name: str
    meta_account_id: str
    page_id: str | None
    pixel_id: str | None
    status: str
    attribution_setting: str
    api_version: str
    category: str | None
    daily_cap_usd: float | None
    is_fake: bool
    has_access_token: bool = False
    has_page_token: bool = False
    last_sync_at: datetime | None
    last_error: str | None


# ---- config --------------------------------------------------------------------------
class ConfigOut(Resp):
    hash: str
    applied_by: str | None
    applied_at: datetime | None
    note: str | None
    config: dict[str, Any]
    yaml: str
    file_hash: str


class ConfigIn(BaseModel):
    yaml: str | None = None
    config: dict[str, Any] | None = None
    note: str = ""


class ConfigValidateOut(Resp):
    ok: bool
    errors: list[str]
    hash: str | None
    diff: dict[str, Any]
    recompute_required: bool


class ConfigVersionOut(Out):
    hash: str
    applied_by: str
    applied_at: datetime
    note: str


# ---- misc ----------------------------------------------------------------------------
class SearchHit(Resp):
    type: str
    id: UUID
    label: str
    sub: str = ""


class SearchOut(Resp):
    q: str
    hits: list[SearchHit]


class AuditOut(Out):
    id: int
    actor_id: str
    action: str
    object_type: str
    object_id: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    at: datetime


class GenerateIn(BaseModel):
    brief_id: UUID
    episode_id: UUID | None = None
    backend: BackendKind = BackendKind.fake
    k: int = Field(default=8, ge=1, le=64)
    renders_per_idea: int = Field(default=3, ge=1, le=6)
    no_llm: bool = False
    seed: int = 0


class GeneratedIdeaOut(Resp):
    status: str
    trajectory: TrajectoryOut
    novelty_reason: str | None
    novelty_distance: float | None
    verifier_version: str


class GenerateOut(Resp):
    brief_id: UUID
    episode_id: UUID | None
    batch_id: UUID | None
    backend: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    refused: bool
    trace: dict[str, Any]
    ideas: list[GeneratedIdeaOut]


class ArchiveSampleOut(Resp):
    brief_id: UUID
    similar_brief_ids: list[UUID]
    excluded_holdout: int
    niche_uses: dict[str, int]
    items: list[dict[str, Any]]


class PromptOut(Resp):
    prompt: str
    trace: dict[str, Any]


class ImportResult(Resp):
    created: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = Field(default_factory=list)


class KillSwitchOut(Resp):
    shipping_enabled: bool
    reason: str
    changed_by: str
    changed_at: datetime


class KillSwitchIn(BaseModel):
    shipping_enabled: bool
    reason: str = ""


# ---- Phase 6: snapshots, reward model training, training runs, eval, feedback --------------
class SnapshotOut(Out):
    id: UUID
    hash: str
    uri: str
    n_trajectories: int
    n_tier2: int
    config_hash: str | None
    created_at: datetime


class RMTrainIn(BaseModel):
    kind: str | None = Field(default=None, pattern=r"^(rm_cold|rm_outcome)$")
    activate: bool = True
    seed: int = 0


class RMTrainOut(Resp):
    trained: bool
    version: str | None = None
    kind: str | None = None
    n_rows: int = 0
    n_positive: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class GoldGapOut(Resp):
    reference_score: float | None = None
    reference_rate: float | None = None
    recent_score: float | None = None
    recent_rate: float | None = None
    gap: float | None = None
    tripped: bool = False
    detail: str = ""
    threshold: float


class TrainingRunIn(BaseModel):
    stage: str = Field(pattern=r"^(rft|dpo|grpo_offpolicy|grpo_onpolicy)$")
    base_model: str = "Qwen/Qwen3-8B"
    adapter_from: UUID | None = None
    smoke: bool = False
    dry: bool = False
    simulator: bool = False
    allow_rm_reward: bool = False


class TrainingRunOut(Out):
    id: UUID
    stage: str
    status: str
    config_hash: str | None
    snapshot_hash: str | None
    rm_version: str | None
    verifier_version: str | None
    base_model: str | None
    checkpoint_uri: str | None
    metrics: dict[str, Any]
    stop_reason: str | None
    created_by: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class EvalLaunchIn(BaseModel):
    kind: str = Field(default="online", pattern=r"^(online|loop_b_vs_loop_a)$")
    systems: list[str] = Field(default_factory=lambda: ["loop_a_fake", "random_fake"])
    n_briefs: int = Field(default=3, ge=1, le=200)
    budget_cap: float = Field(default=2000.0, gt=0)
    seed: int = 0
    max_days: int = Field(default=120, ge=1, le=400)
    inline: bool = True


class EvalArmOut(Out):
    id: UUID
    system: str
    blind_label: str
    n_briefs: int
    tier2_rate: float | None
    ci_low: float | None
    ci_high: float | None
    details: dict[str, Any]


class EvalRunOut(Out):
    id: UUID
    kind: str
    status: str
    holdout_brief_ids: list[UUID]
    attribution_setting: str | None
    config_hash: str | None
    summary: dict[str, Any]
    created_by: str
    created_at: datetime
    finished_at: datetime | None
    arms: list[EvalArmOut] = Field(default_factory=list)
    job_id: UUID | None = None


class FeedbackOut(Resp):
    action: str
    count: int
    version: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
