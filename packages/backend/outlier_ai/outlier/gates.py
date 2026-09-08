"""Gates that a tier must clear (spec section 4).

- volume: at least `min_purchases_at_scale` conversions.
- durability: the tier held for `durability_days` consecutive days at scale spend. Measured
  on the cumulative-to-date ROAS ratio, so a decaying ad whose running ratio drops under the
  tier boundary fails (scenario row 13).
- category: value above the category median.

If any gate fails the tier is recorded as 0 with the flags stored (scenario row 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from outlier_ai.outlier.tiers import multiple_for_tier
from outlier_schemas.config import OutlierConfig
from outlier_schemas.enums import Tier


@dataclass(frozen=True)
class GateResult:
    volume_ok: bool
    durability_ok: bool
    category_ok: bool

    @property
    def all_ok(self) -> bool:
        return self.volume_ok and self.durability_ok and self.category_ok

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


def cumulative_ratios(
    revenue_by_day: Sequence[float], spend_by_day: Sequence[float], baseline: float
) -> list[float]:
    """Running ROAS / baseline after each day."""
    out: list[float] = []
    rev = spd = 0.0
    for r, s in zip(revenue_by_day, spend_by_day, strict=True):
        rev += r
        spd += s
        out.append(rev / spd / baseline if spd > 0 and baseline > 0 else 0.0)
    return out


def check_gates(
    *,
    conversions: int,
    value: float,
    category_median: float,
    cumulative: Sequence[float],
    tier: Tier,
    cfg: OutlierConfig,
) -> GateResult:
    volume_ok = conversions >= cfg.min_purchases_at_scale
    enough_days = len(cumulative) >= cfg.durability_days
    if tier == Tier.zero:
        durability_ok = enough_days
    else:
        boundary = multiple_for_tier(tier, cfg.tier_multiples)
        tail = list(cumulative[-cfg.durability_days :])
        durability_ok = enough_days and all(c >= boundary for c in tail)
    category_ok = value > category_median
    return GateResult(volume_ok=volume_ok, durability_ok=durability_ok, category_ok=category_ok)
