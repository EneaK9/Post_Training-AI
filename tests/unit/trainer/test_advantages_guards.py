import numpy as np
import pytest

from outlier_trainer.advantages import (
    RewardConfig,
    advantages,
    compose_reward,
    group_advantages,
    risk_transform,
    shaping_decay,
    tau_schedule,
    uniqueness_scale,
)
from outlier_trainer.guards import GuardConfig, check_guards, combination_entropy, swing_rate

CFG = RewardConfig()


def test_reward_composition_matches_spec():
    assert (
        compose_reward(
            outlier_tier=2, rm_score=0.9, tag_match=True, format_ok=True, n_positives=0, cfg=CFG
        )
        == 1.0
    )
    assert (
        compose_reward(
            outlier_tier=1, rm_score=0.9, tag_match=True, format_ok=True, n_positives=0, cfg=CFG
        )
        == 0.0
    )
    assert compose_reward(
        outlier_tier=None, rm_score=0.5, tag_match=True, format_ok=True, n_positives=0, cfg=CFG
    ) == pytest.approx(0.15)
    # shaping decays to zero as real positives accumulate
    assert (
        compose_reward(
            outlier_tier=None,
            rm_score=0.5,
            tag_match=True,
            format_ok=True,
            n_positives=100,
            cfg=CFG,
        )
        == 0.0
    )
    assert shaping_decay(25, 50) == 0.75
    # penalties
    assert compose_reward(
        outlier_tier=2, rm_score=None, tag_match=False, format_ok=True, n_positives=0, cfg=CFG
    ) == pytest.approx(0.8)
    assert compose_reward(
        outlier_tier=None, rm_score=None, tag_match=None, format_ok=False, n_positives=0, cfg=CFG
    ) == pytest.approx(-0.5)


def test_risk_transform_is_convex_and_tends_to_identity():
    r = np.array([0.0, 0.5, 1.0])
    u_low = risk_transform(r, 0.1)
    assert np.all(np.diff(u_low) > 0)
    assert u_low[2] - u_low[1] > u_low[1] - u_low[0]  # convex: the top gains more
    u_high = risk_transform(r, 1000.0)
    assert np.allclose(u_high, r, atol=1e-3)
    with pytest.raises(ValueError):
        risk_transform(r, 0.0)


def test_group_advantages_are_zero_mean_and_reward_the_tail():
    r = np.array([0.0, 0.0, 0.0, 1.0])
    a = group_advantages(r, tau=0.2, eps_mean=0.1)
    assert abs(a.mean()) < 1e-9
    assert a[3] > 0 and all(x < 0 for x in a[:3])
    # graded rewards: the mean objective gives the middle idea zero advantage; the risk-seeking
    # objective penalizes it, because only the best of the group is worth chasing
    graded = np.array([0.0, 0.5, 1.0])
    a_mean = group_advantages(graded, tau=1000.0, eps_mean=0.0)
    a_risk = group_advantages(graded, tau=0.2, eps_mean=0.0)
    assert abs(a_mean[1]) < 1e-3
    assert a_risk[1] < 0 < a_risk[2]
    assert a_risk[2] / abs(a_risk[0]) > a_mean[2] / abs(a_mean[0])


def test_uniqueness_scaling_favors_rare_combinations():
    a = np.array([1.0, 1.0, 1.0, 1.0])
    scaled = uniqueness_scale(a, ["combo-a", "combo-a", "combo-a", "combo-b"], beta=0.5)
    assert scaled[3] == 1.0 and scaled[0] == pytest.approx(1 / np.sqrt(3))
    full = advantages([0, 0, 1, 1], ["a", "a", "a", "b"], tau=0.5, eps_mean=0.1, beta=0.5)
    assert full[3] > full[2]  # same reward, rarer combination, more signal


def test_tau_schedule_anneals_geometrically():
    assert tau_schedule(0, 0.1, 1.0, 200) == pytest.approx(0.1)
    assert tau_schedule(100, 0.1, 1.0, 200) == pytest.approx(np.sqrt(0.1))
    assert tau_schedule(200, 0.1, 1.0, 200) == 1.0
    assert tau_schedule(999, 0.1, 1.0, 200) == 1.0


def test_guards_trip_on_collapse_and_hacking():
    cfg = GuardConfig()
    assert swing_rate(["common", "rare", "uncommon", None]) == 0.5
    assert combination_entropy(["a", "a", "a", "a"]) == 0.0
    assert combination_entropy(["a", "b", "c", "d"]) == pytest.approx(1.0)
    ok = check_guards(swing=0.4, entropy=0.9, gold_gap=0.05, steps_since_outcome=3, cfg=cfg)
    assert not ok.stop
    collapsed = check_guards(swing=0.05, entropy=0.1, gold_gap=None, steps_since_outcome=3, cfg=cfg)
    assert collapsed.stop and len(collapsed.reasons) == 2
    hacked = check_guards(swing=0.4, entropy=0.9, gold_gap=0.3, steps_since_outcome=60, cfg=cfg)
    assert (
        hacked.stop
        and any("gold gap" in r for r in hacked.reasons)
        and any("without a real outcome" in r for r in hacked.reasons)
    )
