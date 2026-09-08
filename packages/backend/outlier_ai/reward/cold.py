"""`rm_cold`: ranks the review queue before real outcomes exist.

Heuristic version for tests and the simulator; Claude-judge version when an API key is set.
Both are bounded to [0, 1] with a fixed std so `rm_score = mean - lambda_pess * std` behaves.
"""

from __future__ import annotations

import json
import re

from outlier_ai.generation.backends.base import JudgeBackend
from outlier_ai.reward.base import IdeaFeatures, RewardScore
from outlier_schemas.enums import RewardModelKind

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


class HeuristicColdRewardModel:
    version = "rm_cold-heuristic-v0"
    kind = RewardModelKind.rm_cold
    std = 0.15

    async def score(self, features: IdeaFeatures) -> RewardScore:
        f = features
        mean = 0.35
        if f.typicality == "rare":
            mean += 0.15
        elif f.typicality == "uncommon":
            mean += 0.07
        mean += 0.10 * min(f.cited_signal_count, 3) / 3
        if f.novelty_distance is not None:
            mean += 0.15 * max(0.0, min(f.novelty_distance / 0.5, 1.0))
        if not f.format_ok:
            mean -= 0.40
        if f.tag_match is False:
            mean -= 0.10
        if len(f.verified_card_slugs) < 2:
            mean -= 0.10
        return RewardScore(max(0.0, min(1.0, mean)), self.std, self.version, self.kind)


JUDGE_SYSTEM = (
    "You are a senior performance marketer reviewing ad ideas before they get paid budget. "
    "Estimate the probability (0 to 1) that a top expert would choose to RUN this idea rather "
    "than skip it, given the brief and what the audience said before. Reward unusual angles that "
    "respond to real audience signals; penalize generic copy and ideas that repeat what failed. "
    'Return JSON only: {"p_run": 0.0-1.0, "why": "one sentence"}.'
)


class LLMColdRewardModel:
    kind = RewardModelKind.rm_cold
    std = 0.20

    def __init__(self, judge: JudgeBackend) -> None:
        self.judge = judge
        self.version = f"rm_cold-llm-v1:{judge.model}"

    async def score(self, features: IdeaFeatures) -> RewardScore:
        f = features
        user = (
            f"Brief: {f.brief_text}\nWorld state: {f.world_state}\n"
            f"Cards: {', '.join(f.verified_card_slugs)} | typicality: {f.typicality}\n"
            f"Signals cited: {'; '.join(f.cited_signal_texts) or 'none'}\n"
            f"Angle: {f.angle}\nHeadline: {f.copy.headline}\nPrimary: {f.copy.primary_text}\n"
            f"Description: {f.copy.description} | CTA: {f.copy.cta}\nVisual: {f.visual_brief}\n"
            f"Format valid: {f.format_ok} | verifier agrees with stated cards: {f.tag_match}\nJSON:"
        )
        raw = await self.judge.judge(JUDGE_SYSTEM, user, max_tokens=200)
        p = 0.5
        m = _JSON_OBJ.search(raw or "")
        if m:
            try:
                p = float(json.loads(m.group(0)).get("p_run", 0.5))
            except (json.JSONDecodeError, TypeError, ValueError):
                p = 0.5
        if not f.format_ok:
            p = min(p, 0.1)
        return RewardScore(max(0.0, min(1.0, p)), self.std, self.version, self.kind)
