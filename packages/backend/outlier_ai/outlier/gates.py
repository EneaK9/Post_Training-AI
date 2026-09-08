"""Gates that a tier must clear (spec section 4).

- volume: at least `min_purchases_at_scale` conversions.
- durability: the tier held for `durability_days` consecutive days at scale spend. The
  boundary must be cleared by the aggregate ratio over the last `durability_days` days AND by
  the aggregate over the most recent half of that window, so an ad that opens strong and
  decays (scenario row 13) fails even though its running total still looks good.
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


def window_ratio(
    revenue_by_day: Sequence[float], spend_by_day: Sequence[float], baseline: float
) -> float:
    spend = float(sum(spend_by_day))
    if spend <= 0 or baseline <= 0:
        return 0.0
    return float(sum(revenue_by_day)) / spend / baseline


def durability_holds(
    revenue_by_day: Sequence[float],
    spend_by_day: Sequence[float],
    baseline: float,
    boundary: float,
    durability_days: int,
) -> bool:
    if len(revenue_by_day) < durability_days or len(revenue_by_day) != len(spend_by_day):
        return False
    rev = list(revenue_by_day[-durability_days:])
    spd = list(spend_by_day[-durability_days:])
    half = len(rev) // 2
    full = window_ratio(rev, spd, baseline)
    recent = window_ratio(rev[half:], spd[half:], baseline)
    return full >= boundary and recent >= boundary


def check_gates(
    *,
    conversions: int,
    value: float,
    category_median: float,
    revenue_by_day: Sequence[float],
    spend_by_day: Sequence[float],
    baseline: float,
    tier: Tier,
    cfg: OutlierConfig,
) -> GateResult:
    volume_ok = conversions >= cfg.min_purchases_at_scale
    if tier == Tier.zero:
        durability_ok = len(revenue_by_day) >= cfg.durability_days
    else:
        durability_ok = durability_holds(
            revenue_by_day,
            spend_by_day,
            baseline,
            multiple_for_tier(tier, cfg.tier_multiples),
            cfg.durability_days,
        )
    category_ok = value > category_median
    return GateResult(volume_ok=volume_ok, durability_ok=durability_ok, category_ok=category_ok)
