"""Reward model pieces: the ensemble learns a separable synthetic task, pessimism lowers the
score, calibration buckets sum up, and the gold gap trips only on drift."""

from __future__ import annotations

import numpy as np

from outlier_ai.reward.calibration import calibration
from outlier_ai.reward.ensemble import Ensemble, train_ensemble
from outlier_ai.reward.gold_gap import gold_gap


def _toy(n: int = 300, dims: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dims)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0.8).astype(np.float32)  # ~20% positives
    return x, y


def test_ensemble_learns_and_round_trips():
    x, y = _toy()
    ens, report = train_ensemble(x, y, heads=3, hidden=32, epochs=150, seed=1, meta={"k": "v"})
    mean, std = ens.score(x)
    assert mean.shape == (len(x),) and std.shape == (len(x),)
    # ranks positives above negatives far better than chance
    pos, neg = mean[y == 1], mean[y == 0]
    auc = float(np.mean([p > q for p in pos[:60] for q in neg[:60]]))
    assert auc > 0.85, auc
    assert report.positives == int(y.sum())
    clone = Ensemble.from_bytes(ens.to_bytes())
    mean2, _ = clone.score(x)
    assert np.allclose(mean, mean2, atol=1e-5)
    assert clone.meta["k"] == "v"


def test_pessimism_and_calibration():
    x, y = _toy(seed=2)
    ens, _ = train_ensemble(x, y, heads=3, hidden=32, epochs=100, seed=3)
    mean, std = ens.score(x)
    assert np.all(std >= 0)
    assert np.all(mean - 0.5 * std <= mean)
    cal = calibration(mean, y, n_bins=5)
    assert sum(b["n"] for b in cal.bins) == len(y)
    assert cal.expected_calibration_error is not None
    assert 0 <= cal.expected_calibration_error <= 1
    # highest-scored bucket has a higher realized rate than the lowest one
    rates = [b["tier2_rate"] for b in cal.bins if b["n"] > 0]
    assert rates[-1] >= rates[0]


def test_gold_gap_trips_only_on_drift():
    rng = np.random.default_rng(0)
    ref_scores = rng.uniform(size=200)
    ref_pos = (rng.uniform(size=200) < ref_scores).astype(float)  # calibrated
    steady = gold_gap(ref_scores, ref_pos, ref_scores, ref_pos, threshold=0.15)
    assert not steady.tripped and abs(steady.gap or 0.0) < 1e-9
    # recent window: same scores but almost no realized positives -> proxy is lying
    drifted = gold_gap(ref_scores, ref_pos, ref_scores, np.zeros(200), threshold=0.15)
    assert drifted.tripped and (drifted.gap or 0) > 0.15
    empty = gold_gap(np.array([]), np.array([]), ref_scores, ref_pos, threshold=0.15)
    assert not empty.tripped
