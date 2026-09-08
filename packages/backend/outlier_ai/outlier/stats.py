"""Small statistics helpers with no scipy dependency."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()


def z_for_confidence(confidence: float) -> float:
    """Two-sided z for a lower bound at `confidence` (0.95 -> 1.645 one-sided)."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1.0)")
    return _NORMAL.inv_cdf(confidence)


def wilson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided Wilson score lower bound for a binomial proportion.

    Used for the screening CTR check: `ctr_lower_bound >= multiple * account_median_ctr`.
    Returns 0.0 when there are no trials.
    """
    if trials <= 0:
        return 0.0
    if successes < 0 or successes > trials:
        raise ValueError("successes must be within [0, trials]")
    z = z_for_confidence(confidence)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    margin = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float(max(0.0, (center - margin) / denom))


def bootstrap_ratio_lower_bound(
    revenue_by_day: np.ndarray,
    spend_by_day: np.ndarray,
    baseline: float,
    n_samples: int = 2000,
    confidence: float = 0.90,
    rng: np.random.Generator | None = None,
) -> float:
    """Lower bound of ROAS / baseline by bootstrapping days.

    Resamples days with replacement, recomputes total revenue / total spend, divides by the
    baseline, and returns the (1 - confidence) quantile. With a single day the bound equals
    the point estimate.
    """
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    revenue_by_day = np.asarray(revenue_by_day, dtype=float)
    spend_by_day = np.asarray(spend_by_day, dtype=float)
    if revenue_by_day.shape != spend_by_day.shape or revenue_by_day.ndim != 1:
        raise ValueError("revenue and spend must be 1-d arrays of equal length")
    n = revenue_by_day.size
    if n == 0 or spend_by_day.sum() <= 0:
        return 0.0
    if n == 1:
        return float(revenue_by_day.sum() / spend_by_day.sum() / baseline)
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_samples, n))
    rev = revenue_by_day[idx].sum(axis=1)
    spd = spend_by_day[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(spd > 0, rev / spd / baseline, 0.0)
    return float(np.quantile(ratios, 1.0 - confidence))
