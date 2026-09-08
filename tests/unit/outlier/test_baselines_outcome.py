from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from outlier_ai.outlier.baselines import DayRow, RenderAgg, compute_baselines
from outlier_ai.outlier.recompute import compute_outcome
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BaselineKind, Tier

AS_OF = date(2026, 9, 1)


def make_agg(
    *,
    roas: float,
    days_ago: int = 10,
    scale_days: int = 7,
    purchases_per_day: int = 8,
    author: str = "model",
    campaign: str | None = None,
    category: str = "skincare",
    ctr: float = 0.012,
    revenue_pattern: list[float] | None = None,
) -> RenderAgg:
    tid = uuid4()
    start = AS_OF - timedelta(days=days_ago + scale_days + 7)
    rows: list[DayRow] = []
    for d in range(7):
        rows.append(
            DayRow(start + timedelta(days=d), "screening", 1500, int(1500 * ctr), 20.0, 0, 0.0)
        )
    for d in range(scale_days):
        rev = revenue_pattern[d] if revenue_pattern else roas * 100.0
        rows.append(
            DayRow(start + timedelta(days=7 + d), "scale", 8000, 96, 100.0, purchases_per_day, rev)
        )
    return RenderAgg(
        render_id=uuid4(),
        trajectory_id=tid,
        brief_id=uuid4(),
        campaign_id=campaign,
        author_kind=author,
        category=category,
        goal_metric="purchases",
        attribution_setting="7d_click_1d_view",
        rows=rows,
    )


def baselines_for(aggs, cfg: AppConfig, **kw):
    return compute_baselines(
        aggs,
        as_of=AS_OF,
        category="skincare",
        goal_metric="purchases",
        attribution_setting="7d_click_1d_view",
        cfg=cfg.outlier,
        category_medians=cfg.category_medians,
        **kw,
    )


def test_account_median_when_enough_ads(app_config: AppConfig):
    aggs = [make_agg(roas=2.0 + 0.02 * i) for i in range(25)]
    b = baselines_for(aggs, app_config)
    assert b.roas.kind == BaselineKind.account and b.roas.n_ads == 25
    assert abs(b.roas.value - 2.24) < 1e-9
    assert b.ctr.kind == BaselineKind.account and abs(b.ctr.value - 0.012) < 1e-9


def test_category_fallback_when_few_ads(app_config: AppConfig):
    aggs = [make_agg(roas=5.0) for _ in range(5)]
    b = baselines_for(aggs, app_config)
    assert b.roas.kind == BaselineKind.category_fallback
    assert b.roas.value == app_config.category_medians["skincare"].roas
    assert b.ctr.value == app_config.category_medians["skincare"].ctr


def test_window_holdout_and_self_exclusion(app_config: AppConfig):
    fresh = [make_agg(roas=2.0) for _ in range(20)]
    old = [make_agg(roas=9.0, days_ago=120) for _ in range(10)]
    held = [make_agg(roas=9.0, campaign="camp_holdout") for _ in range(10)]
    b = baselines_for(fresh + old + held, app_config, holdout_campaigns=frozenset({"camp_holdout"}))
    assert b.roas.n_ads == 20 and b.roas.value == 2.0
    b2 = baselines_for(fresh, app_config, exclude_trajectory_id=fresh[0].trajectory_id)
    assert b2.roas.n_ads == 19 and b2.roas.kind == BaselineKind.category_fallback


def test_human_only_median_reported_alongside(app_config: AppConfig):
    humans = [make_agg(roas=3.0, author="human") for _ in range(20)]
    models = [make_agg(roas=1.0) for _ in range(20)]
    b = baselines_for(humans + models, app_config)
    assert b.human_roas == 3.0
    assert b.roas.value == 2.0
    few = baselines_for(models + humans[:3], app_config)
    assert few.human_roas is None


def test_strong_ad_is_tier_two_with_gates_ok(app_config: AppConfig):
    cohort = [make_agg(roas=2.2) for _ in range(25)]
    target = make_agg(roas=8.0, purchases_per_day=10)
    b = baselines_for(cohort, app_config)
    out = compute_outcome(target, b, 2.4, app_config, measured_at=datetime.now(UTC))
    assert out is not None
    assert out.outlier_tier == Tier.two
    assert out.gates.all_ok
    assert abs(out.ratio - 8.0 / 2.2) < 1e-9
    assert out.ratio_lower_bound <= out.ratio + 1e-9
    assert out.conversions == 70 and out.days_at_scale == 7
    assert out.baseline == BaselineKind.account


def test_gate_failure_records_tier_zero_with_flags(app_config: AppConfig):
    cohort = [make_agg(roas=2.2) for _ in range(25)]
    b = baselines_for(cohort, app_config)
    thin = make_agg(roas=8.0, purchases_per_day=2)
    out = compute_outcome(thin, b, 2.4, app_config, measured_at=datetime.now(UTC))
    assert out is not None and out.outlier_tier == Tier.zero
    assert not out.gates.volume_ok and out.gates.category_ok
    decaying = make_agg(
        roas=8.0, purchases_per_day=10, revenue_pattern=[2000, 1200, 700, 400, 300, 250, 200]
    )
    out2 = compute_outcome(decaying, b, 2.4, app_config, measured_at=datetime.now(UTC))
    assert out2 is not None and out2.outlier_tier == Tier.zero
    assert not out2.gates.durability_ok


def test_no_scale_rows_means_no_outcome(app_config: AppConfig):
    cohort = [make_agg(roas=2.2) for _ in range(25)]
    b = baselines_for(cohort, app_config)
    unscaled = make_agg(roas=8.0, scale_days=0)
    assert compute_outcome(unscaled, b, 2.4, app_config, measured_at=datetime.now(UTC)) is None


def test_thresholds_change_tiers(app_config: AppConfig):
    cohort = [make_agg(roas=2.2) for _ in range(25)]
    b = baselines_for(cohort, app_config)
    target = make_agg(roas=8.0, purchases_per_day=10)
    strict = app_config.model_copy(deep=True)
    strict.outlier.tier_multiples.tier2 = 4.0
    strict.outlier.tier_multiples.tier3 = 12.0
    out = compute_outcome(target, b, 2.4, strict, measured_at=datetime.now(UTC))
    assert out is not None and out.outlier_tier == Tier.one
