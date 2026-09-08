"""Calibration on tier 2+: does a higher score mean a higher realized outlier rate?"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationReport:
    bins: list[dict[str, float]] = field(default_factory=list)
    expected_calibration_error: float | None = None
    auc: float | None = None
    n: int = 0
    positives: int = 0


def calibration(scores: np.ndarray, positives: np.ndarray, n_bins: int = 5) -> CalibrationReport:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(positives, dtype=float)
    rep = CalibrationReport(n=len(s), positives=int(y.sum()))
    if len(s) == 0:
        return rep
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi) if i == n_bins - 1 else (s >= lo) & (s < hi)
        if not mask.any():
            continue
        conf, acc = float(s[mask].mean()), float(y[mask].mean())
        rep.bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(mask.sum()),
                "mean_score": conf,
                "tier2_rate": acc,
            }
        )
        ece += mask.mean() * abs(conf - acc)
    rep.expected_calibration_error = float(ece)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) and len(neg):
        order = np.argsort(np.concatenate([pos, neg]))
        ranks = np.empty(len(order))
        ranks[order] = np.arange(1, len(order) + 1)
        rep.auc = float(
            (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        )
    return rep
