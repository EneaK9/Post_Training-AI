"""Typed configuration tree for config/config.yaml.

Every knob the spec puts in `config.yaml` is here with a validator. The backend's
`core.config` loads the YAML into `AppConfig`, computes a canonical hash, and stores
each applied version. Runs, episodes, and trajectories pin the hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from outlier_schemas.enums import (
    BackendKind,
    ImageMode,
    ModelWeights,
    NicheProjection,
    VerifierKind,
)


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TierMultiples(_Cfg):
    tier1: float = Field(default=1.5, gt=1.0)
    tier2: float = Field(default=3.0, gt=1.0)
    tier3: float = Field(default=10.0, gt=1.0)

    @model_validator(mode="after")
    def _monotonic(self) -> TierMultiples:
        if not (self.tier1 < self.tier2 < self.tier3):
            raise ValueError("tier multiples must be strictly increasing")
        return self


class TierRewards(_Cfg):
    tier0: float = 0.0
    tier1: float = 0.0
    tier2: float = 1.0
    tier3: float = 1.0


class ScreeningConfig(_Cfg):
    ctr_multiple: float = Field(default=1.5, gt=0)
    min_impressions: int = Field(default=5000, ge=0)
    window_days: int = Field(default=7, ge=1)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)


class OutlierConfig(_Cfg):
    tier_multiples: TierMultiples = Field(default_factory=TierMultiples)
    tier_rewards: TierRewards = Field(default_factory=TierRewards)
    baseline_window_days: int = Field(default=90, ge=7)
    baseline_min_ads: int = Field(default=20, ge=1)
    min_purchases_at_scale: int = Field(default=30, ge=1)
    durability_days: int = Field(default=7, ge=1)
    bootstrap_samples: int = Field(default=2000, ge=100)
    bootstrap_confidence: float = Field(default=0.90, gt=0.5, lt=1.0)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    category_medians_file: str = "category_medians.yaml"


class EpisodeConfig(_Cfg):
    default_budget_cap_usd: float = Field(default=5000, gt=0)
    screening_budget_per_ad_usd: float = Field(default=20, gt=0)
    scale_budget_per_ad_usd: float = Field(default=100, gt=0)
    screening_days: int = Field(default=7, ge=1)
    scale_days: int = Field(default=7, ge=1)
    renders_per_idea: int = Field(default=3, ge=1, le=6)
    ideas_per_batch: int = Field(default=4, ge=1, le=16)
    max_batches: int = Field(default=6, ge=1)
    overlap_batches: bool = True
    review_poll_minutes: int = Field(default=30, ge=1)
    keep_running_after_outlier: bool = False


class ArchiveConfig(_Cfg):
    n_elite: int = Field(default=8, ge=0)
    n_rare: int = Field(default=3, ge=0)
    novelty_threshold: float = Field(default=0.12, ge=0.0, le=2.0)
    niche_projection: NicheProjection = NicheProjection.strategy_only
    history_days: int = Field(default=90, ge=1)
    similar_brief_min_cosine: float = Field(default=0.75, ge=-1.0, le=1.0)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dims: int = Field(default=384, ge=8)


class GenerationConfig(_Cfg):
    k: int = Field(default=8, ge=1, le=64)
    typicality_prompting: bool = True
    weights: ModelWeights = ModelWeights.instruct
    max_prompt_tokens: int = Field(default=24000, ge=2000)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    default_backend: BackendKind = BackendKind.fake
    anthropic_model: str = "claude-sonnet-5"
    local_model: str = "Qwen/Qwen3-8B"
    local_base_model: str = "Qwen/Qwen3-8B-Base"
    vllm_url: str = "http://localhost:8000/v1"
    min_cards_per_idea: int = Field(default=2, ge=1)


class VerifierConfig(_Cfg):
    kind: VerifierKind = VerifierKind.llm
    model: str = "claude-sonnet-5"
    few_shot_n: int = Field(default=12, ge=0)
    classifier_min_labels: int = Field(default=500, ge=1)


class RewardModelConfig(_Cfg):
    lambda_pess: float = Field(default=1.0, ge=0)
    heads: int = Field(default=3, ge=1)
    cold_until_positives: int = Field(default=50, ge=1)
    focal_gamma: float = Field(default=2.0, ge=0)
    gold_gap_threshold: float = Field(default=0.15, gt=0)
    judge_model: str = "claude-sonnet-5"


class LoraConfig(_Cfg):
    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)


class GuardsConfig(_Cfg):
    swing_rate_min: float = Field(default=0.15, ge=0, le=1)
    entropy_floor: float = Field(default=0.6, ge=0)
    max_steps_between_outcomes: int = Field(default=50, ge=1)


class RLConfig(_Cfg):
    k: int = Field(default=32, ge=2)
    tau_start: float = Field(default=0.1, gt=0)
    tau_target: float = Field(default=1.0, gt=0)
    tau_anneal_steps: int = Field(default=200, ge=1)
    eps_mean: float = Field(default=0.1, ge=0, le=1)
    beta: float = Field(default=0.5, ge=0)
    kl_coef: float = Field(default=0.05, ge=0)
    clip_eps: float = Field(default=0.2, gt=0)
    tag_penalty: float = Field(default=0.2, ge=0)
    format_penalty: float = Field(default=0.5, ge=0)
    shaping_weight: float = Field(default=0.3, ge=0)
    learning_rate: float = Field(default=1e-5, gt=0)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    guards: GuardsConfig = Field(default_factory=GuardsConfig)
    min_tier2_trajectories_to_start: int = Field(default=50, ge=1)


class ActionTypes(_Cfg):
    reactions: str = "post_reaction"
    comments: str = "comment"
    shares: str = "post"
    saves: str = "onsite_conversion.post_save"


class MetaConfig(_Cfg):
    api_version: str = Field(default="v26.0", pattern=r"^v\d+\.\d+$")
    permissions: list[str] = Field(
        default_factory=lambda: [
            "ads_management",
            "ads_read",
            "pages_show_list",
            "pages_read_user_content",
        ]
    )
    attribution_setting: str = "7d_click_1d_view"
    insights_sync_hour_utc: int = Field(default=6, ge=0, le=23)
    comments_sync_hour_utc: int = Field(default=7, ge=0, le=23)
    daily_account_cap_usd: float = Field(default=500, gt=0)
    action_types: ActionTypes = Field(default_factory=ActionTypes)


class ImagesConfig(_Cfg):
    mode: ImageMode = ImageMode.brand_assets
    sizes: list[tuple[int, int]] = Field(default_factory=lambda: [(1080, 1080), (1080, 1350)])
    primary_text_warn_chars: int = 125
    headline_warn_chars: int = 40
    description_warn_chars: int = 30


class SignalsConfig(_Cfg):
    extraction_model: str = "claude-sonnet-5"
    min_comments_for_theme: int = Field(default=2, ge=1)


class CategoryMedian(_Cfg):
    roas: float = Field(gt=0)
    ctr: float = Field(gt=0, lt=1)


class AppConfig(_Cfg):
    version: int = 1
    outlier: OutlierConfig = Field(default_factory=OutlierConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    archive: ArchiveConfig = Field(default_factory=ArchiveConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    reward_model: RewardModelConfig = Field(default_factory=RewardModelConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    meta: MetaConfig = Field(default_factory=MetaConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    category_medians: dict[str, CategoryMedian] = Field(default_factory=dict)

    # ---- hashing and IO -------------------------------------------------------------

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def diff(self, other: AppConfig) -> dict[str, tuple[Any, Any]]:
        """Flat dotted-path diff: {path: (self_value, other_value)}."""
        out: dict[str, tuple[Any, Any]] = {}
        _walk_diff(self.model_dump(mode="json"), other.model_dump(mode="json"), "", out)
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        medians_file = raw.get("outlier", {}).get("category_medians_file")
        if medians_file and "category_medians" not in raw:
            mpath = path.parent / medians_file
            if mpath.exists():
                with mpath.open() as mf:
                    raw["category_medians"] = (yaml.safe_load(mf) or {}).get("categories", {})
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        return cls.model_validate(raw)


def _walk_diff(a: Any, b: Any, prefix: str, out: dict[str, tuple[Any, Any]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}.{key}" if prefix else str(key)
            _walk_diff(a.get(key), b.get(key), path, out)
    elif a != b:
        out[prefix] = (a, b)
