"""Enumerations shared across the system. String values are what the database stores."""

from enum import IntEnum, StrEnum


class CardKind(StrEnum):
    strategy = "strategy"
    style = "style"
    principle = "principle"
    mechanic = "mechanic"


class CardStatus(StrEnum):
    draft = "draft"
    active = "active"
    retired = "retired"


class RelationKind(StrEnum):
    complements = "complements"
    conflicts = "conflicts"
    prerequisite = "prerequisite"


class RelationSource(StrEnum):
    expert = "expert"
    suggested = "suggested"


class RelationStatus(StrEnum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"


class AuthorKind(StrEnum):
    human = "human"
    model = "model"


class TagSource(StrEnum):
    expert = "expert"
    model = "model"
    model_verified = "model_verified"
    verifier = "verifier"


class ReviewLabel(StrEnum):
    run = "run"
    skip = "skip"
    wrong_cards = "wrong_cards"


class RenderStatus(StrEnum):
    draft = "draft"
    pending_review = "pending_review"
    rejected = "rejected"
    screening = "screening"
    scaled = "scaled"
    stopped = "stopped"


class SignalKind(StrEnum):
    comment_theme = "comment_theme"
    objection = "objection"
    praise = "praise"
    joke = "joke"
    misreading = "misreading"
    question = "question"
    competitor_mention = "competitor_mention"
    quote = "quote"
    hook_stat = "hook_stat"
    share_pattern = "share_pattern"
    note = "note"


class SignalStatus(StrEnum):
    proposed = "proposed"
    confirmed = "confirmed"
    rejected = "rejected"


class Sentiment(StrEnum):
    neg = "neg"
    neu = "neu"
    pos = "pos"


class EvidenceSource(StrEnum):
    meta_comment = "meta_comment"
    reaction_stats = "reaction_stats"
    insight = "insight"
    manual = "manual"


class OutcomeMetric(StrEnum):
    roas = "roas"
    purchases_per_dollar = "purchases_per_dollar"
    leads_per_dollar = "leads_per_dollar"


class BaselineKind(StrEnum):
    account = "account"
    category_fallback = "category_fallback"


class EpisodeStatus(StrEnum):
    searching = "searching"
    outlier_found = "outlier_found"
    budget_exhausted = "budget_exhausted"
    stopped = "stopped"


class BatchState(StrEnum):
    proposed = "proposed"
    approved = "approved"
    shipping = "shipping"
    in_review = "in_review"
    screening = "screening"
    scaling = "scaling"
    stopped = "stopped"
    measured = "measured"


class Typicality(StrEnum):
    common = "common"
    uncommon = "uncommon"
    rare = "rare"


class Role(StrEnum):
    operator = "operator"
    researcher = "researcher"
    expert = "expert"


class AccountStatus(StrEnum):
    active = "active"
    needs_reauth = "needs_reauth"
    disabled = "disabled"


class Tier(IntEnum):
    zero = 0
    one = 1
    two = 2
    three = 3


class Channel(StrEnum):
    meta_feed_image = "meta_feed_image"


class GoalMetric(StrEnum):
    purchases = "purchases"
    leads = "leads"


class ImageMode(StrEnum):
    generate = "generate"
    brand_assets = "brand_assets"


class ModelWeights(StrEnum):
    base = "base"
    instruct = "instruct"


class NicheProjection(StrEnum):
    full_combination = "full_combination"
    strategy_only = "strategy_only"
    strategy_plus_style = "strategy_plus_style"


class BackendKind(StrEnum):
    fake = "fake"
    anthropic = "anthropic"
    local = "local"


class RewardModelKind(StrEnum):
    rm_cold = "rm_cold"
    rm_outcome = "rm_outcome"


class VerifierKind(StrEnum):
    llm = "llm"
    classifier = "classifier"


class JobState(StrEnum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class TrainingStage(StrEnum):
    rft = "rft"
    dpo = "dpo"
    grpo_offpolicy = "grpo_offpolicy"
    grpo_onpolicy = "grpo_onpolicy"


class TrainingRunStatus(StrEnum):
    queued = "queued"
    running = "running"
    paused = "paused"
    completed = "completed"
    stopped = "stopped"
    failed = "failed"


class EvalKind(StrEnum):
    online = "online"
    loop_b_vs_loop_a = "loop_b_vs_loop_a"
