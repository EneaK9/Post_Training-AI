import pytest

from outlier_ai.outlier.gates import check_gates, cumulative_ratios
from outlier_ai.outlier.tiers import is_outlier, multiple_for_tier, reward_for_tier, tier_for_ratio
from outlier_schemas.config import OutlierConfig, TierMultiples, TierRewards
from outlier_schemas.enums import Tier

M = TierMultiples()
CFG = OutlierConfig()


@pytest.mark.parametrize(
    ("ratio", "tier"),
    [(0.0, 0), (1.49, 0), (1.5, 1), (2.99, 1), (3.0, 2), (9.99, 2), (10.0, 3), (50.0, 3)],
)
def test_tier_boundaries(ratio: float, tier: int):
    assert tier_for_ratio(ratio, M) == Tier(tier)


def test_default_rewards_only_pay_the_tail():
    r = TierRewards()
    assert [reward_for_tier(Tier(t), r) for t in range(4)] == [0.0, 0.0, 1.0, 1.0]
    assert [is_outlier(Tier(t)) for t in range(4)] == [False, False, True, True]


def test_multiple_for_tier_is_the_lower_boundary():
    assert multiple_for_tier(Tier.zero, M) == 0.0
    assert multiple_for_tier(Tier.one, M) == 1.5
    assert multiple_for_tier(Tier.two, M) == 3.0
    assert multiple_for_tier(Tier.three, M) == 10.0


def test_cumulative_ratios_are_running_totals():
    assert cumulative_ratios([300, 300, 300], [100, 100, 100], 2.0) == [1.5, 1.5, 1.5]
    assert cumulative_ratios([600, 0], [100, 100], 2.0) == [3.0, 1.5]
    assert cumulative_ratios([], [], 2.0) == []


SPEND7 = [100.0] * 7


def gates(*, conversions=60, value=8.0, category_median=2.4, revenue, spend=SPEND7, tier=Tier.two):
    return check_gates(
        conversions=conversions,
        value=value,
        category_median=category_median,
        revenue_by_day=revenue,
        spend_by_day=spend,
        baseline=2.2,
        tier=tier,
        cfg=CFG,
    )


def test_volume_gate():
    steady = [770.0] * 7  # 3.5x of a 2.2 baseline
    assert not gates(conversions=29, revenue=steady).volume_ok
    assert gates(conversions=30, revenue=steady).all_ok


def test_durability_fails_when_the_ad_decays():
    # Strong open, collapse in the back half: total still 4.6x but the recent half is 1.2x.
    decaying = [2000.0, 2000.0, 2000.0, 300.0, 300.0, 250.0, 200.0]
    g = gates(revenue=decaying)
    assert not g.durability_ok and g.volume_ok and g.category_ok
    held = gates(revenue=[700.0] * 7)  # 3.18x throughout
    assert held.durability_ok


def test_durability_needs_enough_days():
    assert not gates(revenue=[770.0] * 6, spend=[100.0] * 6).durability_ok
    zero = gates(conversions=0, value=1.0, revenue=[100.0] * 7, tier=Tier.zero)
    assert zero.durability_ok  # tier 0 has no boundary to hold


def test_durability_checks_the_boundary_of_the_candidate_tier():
    borderline = [700.0] * 7  # 3.18x: clears tier 2 but not tier 3
    assert gates(revenue=borderline, tier=Tier.two).durability_ok
    assert not gates(revenue=borderline, tier=Tier.three).durability_ok


def test_category_gate():
    g = gates(value=2.4, revenue=[770.0] * 7)
    assert not g.category_ok
    assert g.as_dict() == {"volume_ok": True, "durability_ok": True, "category_ok": False}
