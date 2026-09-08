"""Baselines: account medians over a trailing window, with category fallback.

Everything here is pure. `RenderAgg` is the per-render aggregate the recompute loader
builds once per account; baselines are then computed in memory for any `as_of` date.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median
from uuid import UUID

from outlier_schemas.config import CategoryMedian, OutlierConfig
from outlier_schemas.enums import AuthorKind, BaselineKind

MIN_IMPRESSIONS_FOR_CTR_BASELINE = 1000


@dataclass(frozen=True)
class DayRow:
    day: date
    phase: str
    impressions: int
    link_clicks: int
    spend: float
    purchases: int
    revenue: float
    reactions: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    frequency: float | None = None


@dataclass
class RenderAgg:
    render_id: UUID
    trajectory_id: UUID
    brief_id: UUID
    campaign_id: str | None
    author_kind: str
    category: str
    goal_metric: str
    attribution_setting: str
    rows: list[DayRow] = field(default_factory=list)

    @property
    def screening_rows(self) -> list[DayRow]:
        return [r for r in self.rows if r.phase == "screening"]

    @property
    def scale_rows(self) -> list[DayRow]:
        return sorted((r for r in self.rows if r.phase == "scale"), key=lambda r: r.day)

    @property
    def first_day(self) -> date | None:
        return min((r.day for r in self.rows), default=None)

    @property
    def last_day(self) -> date | None:
        return max((r.day for r in self.rows), default=None)

    @property
    def screening_impressions(self) -> int:
        return sum(r.impressions for r in self.screening_rows)

    @property
    def screening_clicks(self) -> int:
        return sum(r.link_clicks for r in self.screening_rows)

    @property
    def scale_spend(self) -> float:
        return float(sum(r.spend for r in self.scale_rows))

    @property
    def scale_revenue(self) -> float:
        return float(sum(r.revenue for r in self.scale_rows))

    @property
    def scale_conversions(self) -> int:
        return sum(r.purchases for r in self.scale_rows)

    @property
    def scale_value(self) -> float | None:
        """ROAS for purchases; conversions per dollar for leads."""
        if self.scale_spend <= 0:
            return None
        if self.goal_metric == "leads":
            return self.scale_conversions / self.scale_spend
        return self.scale_revenue / self.scale_spend


@dataclass(frozen=True)
class Baseline:
    value: float
    kind: BaselineKind
    n_ads: int


@dataclass(frozen=True)
class Baselines:
    roas: Baseline
    ctr: Baseline
    human_roas: float | None


def category_median_for(
    category: str, medians: Mapping[str, CategoryMedian]
) -> CategoryMedian | None:
    return medians.get(category) or medians.get("default")


def median_or_none(values: Sequence[float], min_n: int) -> tuple[float | None, int]:
    if len(values) < min_n or not values:
        return None, len(values)
    return float(median(values)), len(values)


def _in_window(agg: RenderAgg, as_of: date, window_days: int) -> bool:
    last = agg.last_day
    return last is not None and as_of - timedelta(days=window_days) <= last <= as_of


def compute_baselines(
    aggs: Sequence[RenderAgg],
    *,
    as_of: date,
    category: str,
    goal_metric: str,
    attribution_setting: str,
    cfg: OutlierConfig,
    category_medians: Mapping[str, CategoryMedian],
    exclude_trajectory_id: UUID | None = None,
    holdout_campaigns: frozenset[str] = frozenset(),
) -> Baselines:
    """Account medians over the trailing window, same metric family and attribution setting.

    Falls back to the category median when fewer than `baseline_min_ads` ads qualify. The
    human-only median is reported alongside and is None when too few human ads exist.
    """
    eligible = [
        a
        for a in aggs
        if a.goal_metric == goal_metric
        and a.attribution_setting == attribution_setting
        and a.trajectory_id != exclude_trajectory_id
        and (a.campaign_id is None or a.campaign_id not in holdout_campaigns)
        and _in_window(a, as_of, cfg.baseline_window_days)
    ]
    cat = category_median_for(category, category_medians)

    scale_values = [v for a in eligible if a.scale_rows and (v := a.scale_value) is not None]
    roas_med, n_roas = median_or_none(scale_values, cfg.baseline_min_ads)
    if roas_med is not None:
        roas = Baseline(roas_med, BaselineKind.account, n_roas)
    else:
        roas = Baseline(cat.roas if cat else 0.0, BaselineKind.category_fallback, n_roas)

    ctr_values = [
        a.screening_clicks / a.screening_impressions
        for a in eligible
        if a.screening_impressions >= MIN_IMPRESSIONS_FOR_CTR_BASELINE
    ]
    ctr_med, n_ctr = median_or_none(ctr_values, cfg.baseline_min_ads)
    if ctr_med is not None:
        ctr = Baseline(ctr_med, BaselineKind.account, n_ctr)
    else:
        ctr = Baseline(cat.ctr if cat else 0.0, BaselineKind.category_fallback, n_ctr)

    human_values = [
        v
        for a in eligible
        if a.author_kind == AuthorKind.human and a.scale_rows and (v := a.scale_value) is not None
    ]
    human_med, _ = median_or_none(human_values, cfg.baseline_min_ads)

    return Baselines(roas=roas, ctr=ctr, human_roas=human_med)
