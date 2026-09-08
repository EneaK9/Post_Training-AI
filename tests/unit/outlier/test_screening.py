from dataclasses import dataclass

from outlier_ai.outlier.screening import compute_screening
from outlier_schemas.config import ScreeningConfig

CFG = ScreeningConfig()
MEDIAN_CTR = 0.012


@dataclass
class Row:
    impressions: int
    link_clicks: int
    spend: float = 20.0
    reactions: int = 1
    comments: int = 1
    shares: int = 0
    saves: int = 0


def rows(n: int, impressions: int, clicks: int) -> list[Row]:
    return [Row(impressions, clicks) for _ in range(n)]


def test_passes_with_high_ctr_full_window():
    r = compute_screening(rows(7, 1000, 30), MEDIAN_CTR, CFG)
    assert r is not None
    assert r.impressions == 7000 and r.link_clicks == 210
    assert r.ctr == 0.03
    assert r.ctr_lower_bound < r.ctr
    assert r.window_complete and r.passed
    assert r.screening_ratio >= CFG.ctr_multiple


def test_fails_with_ctr_near_median():
    r = compute_screening(rows(7, 1000, 13), MEDIAN_CTR, CFG)
    assert r is not None and not r.passed
    assert r.screening_ratio < CFG.ctr_multiple


def test_incomplete_window_never_passes():
    r = compute_screening(rows(3, 1000, 40), MEDIAN_CTR, CFG)
    assert r is not None and not r.window_complete and not r.passed
    forced = compute_screening(rows(3, 1000, 40), MEDIAN_CTR, CFG, window_complete=True)
    assert forced is not None and not forced.passed  # 3000 impressions < 5000


def test_min_impressions_gate():
    r = compute_screening(rows(7, 500, 25), MEDIAN_CTR, CFG)
    assert r is not None and r.impressions == 3500 and not r.passed


def test_no_rows_returns_none_and_engagement_sums():
    assert compute_screening([], MEDIAN_CTR, CFG) is None
    r = compute_screening(rows(7, 1000, 30), MEDIAN_CTR, CFG)
    assert r is not None
    assert r.engagement == {"reactions": 7, "comments": 7, "shares": 0, "saves": 0}
    assert r.cpc == r.spend / r.link_clicks


def test_zero_median_cannot_pass():
    r = compute_screening(rows(7, 1000, 30), 0.0, CFG)
    assert r is not None and not r.passed and r.screening_ratio == 0.0
