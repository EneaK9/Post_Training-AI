"""Training guards (spec section 7.3): stop on swing-rate drop, entropy floor, gold-gap trip,
and cap the steps between real outcome batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuardConfig:
    swing_rate_min: float = 0.15
    entropy_floor: float = 0.6
    max_steps_between_outcomes: int = 50
    gold_gap_threshold: float = 0.15


@dataclass
class GuardDecision:
    stop: bool
    reasons: list[str] = field(default_factory=list)


def swing_rate(typicalities: Sequence[str | None]) -> float:
    """Fraction of rollouts the policy itself labels uncommon or rare: the swing it is taking."""
    if not typicalities:
        return 0.0
    return sum(1 for t in typicalities if t in ("uncommon", "rare")) / len(typicalities)


def combination_entropy(keys: Sequence[object]) -> float:
    """Normalized entropy of verified combinations across a batch of rollouts (0 = collapsed)."""
    import math

    if not keys:
        return 0.0
    counts: dict[object, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    n = len(keys)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    max_h = math.log(len(keys)) if len(keys) > 1 else 1.0
    return h / max_h if max_h > 0 else 0.0


def check_guards(
    *,
    swing: float,
    entropy: float,
    gold_gap: float | None,
    steps_since_outcome: int,
    cfg: GuardConfig,
) -> GuardDecision:
    reasons: list[str] = []
    if swing < cfg.swing_rate_min:
        reasons.append(f"swing rate {swing:.2f} < {cfg.swing_rate_min}")
    if entropy < cfg.entropy_floor:
        reasons.append(f"combination entropy {entropy:.2f} < {cfg.entropy_floor}")
    if gold_gap is not None and gold_gap > cfg.gold_gap_threshold:
        reasons.append(f"gold gap {gold_gap:.2f} > {cfg.gold_gap_threshold} (reward model hacked)")
    if steps_since_outcome > cfg.max_steps_between_outcomes:
        reasons.append(f"{steps_since_outcome} steps without a real outcome batch")
    return GuardDecision(stop=bool(reasons), reasons=reasons)
