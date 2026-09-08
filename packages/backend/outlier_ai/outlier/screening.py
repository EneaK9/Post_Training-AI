"""Screening stage: CTR lower bound against the account CTR median. Assigns no reward."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from outlier_ai.outlier.stats import wilson_lower_bound
from outlier_schemas.config import ScreeningConfig


class EngagementRow(Protocol):
    impressions: int
    link_clicks: int
    spend: float
    reactions: int
    comments: int
    shares: int
    saves: int


@dataclass(frozen=True)
class ScreeningResult:
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
    engagement: dict[str, int]

    def as_row(self) -> dict:
        return asdict(self)


def compute_screening(
    rows: Sequence[EngagementRow],
    account_median_ctr: float,
    cfg: ScreeningConfig,
    *,
    window_complete: bool | None = None,
) -> ScreeningResult | None:
    """Aggregate screening-phase rows and decide the pass.

    `passed` requires the full window, the impression floor, and
    `ctr_lower_bound >= ctr_multiple * account_median_ctr`. Returns None with no rows.
    """
    if not rows:
        return None
    impressions = sum(r.impressions for r in rows)
    clicks = sum(r.link_clicks for r in rows)
    spend = float(sum(r.spend for r in rows))
    ctr = clicks / impressions if impressions else 0.0
    cpc = spend / clicks if clicks else None
    lb = wilson_lower_bound(clicks, impressions, cfg.confidence)
    ratio = lb / account_median_ctr if account_median_ctr > 0 else 0.0
    complete = len(rows) >= cfg.window_days if window_complete is None else window_complete
    passed = bool(
        complete
        and impressions >= cfg.min_impressions
        and account_median_ctr > 0
        and lb >= cfg.ctr_multiple * account_median_ctr
    )
    return ScreeningResult(
        impressions=impressions,
        link_clicks=clicks,
        spend=round(spend, 2),
        ctr=ctr,
        cpc=cpc,
        ctr_lower_bound=lb,
        account_median_ctr_90d=account_median_ctr,
        screening_ratio=ratio,
        passed=passed,
        window_complete=complete,
        engagement={
            "reactions": sum(r.reactions for r in rows),
            "comments": sum(r.comments for r in rows),
            "shares": sum(r.shares for r in rows),
            "saves": sum(r.saves for r in rows),
        },
    )
