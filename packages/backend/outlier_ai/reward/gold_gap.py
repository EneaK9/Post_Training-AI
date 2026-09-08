"""Gold gap monitor: proxy reward rising while real outcomes do not (Gao et al. 2023).

Compares the reward model's mean score on recently measured ideas with their realized tier 2
rate, against the same statistics on an earlier reference window. A gap above the threshold
pauses Loop B; Loop A keeps running (scenario table row 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GoldGap:
    reference_score: float | None
    reference_rate: float | None
    recent_score: float | None
    recent_rate: float | None
    gap: float | None
    tripped: bool
    detail: str


def gold_gap(
    ref_scores: np.ndarray,
    ref_positive: np.ndarray,
    recent_scores: np.ndarray,
    recent_positive: np.ndarray,
    *,
    threshold: float,
) -> GoldGap:
    def stats(s: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
        if len(s) == 0:
            return None, None
        return float(np.mean(s)), float(np.mean(y))

    rs, rr = stats(np.asarray(ref_scores, float), np.asarray(ref_positive, float))
    cs, cr = stats(np.asarray(recent_scores, float), np.asarray(recent_positive, float))
    if rs is None or cs is None or rr is None or cr is None:
        return GoldGap(rs, rr, cs, cr, None, False, "not enough data")
    # proxy moved up relative to the real rate by more than the threshold: either the score
    # rose while outcomes did not, or outcomes fell while the score held (both mean the reward
    # model is now over-scoring what actually ships)
    gap = (cs - rs) - (cr - rr)
    tripped = gap > threshold
    return GoldGap(
        rs,
        rr,
        cs,
        cr,
        float(gap),
        bool(tripped),
        "proxy diverging from real outcomes" if tripped else "ok",
    )
