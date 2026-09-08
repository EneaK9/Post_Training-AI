from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RateCI:
    rate: float
    low: float
    high: float
    n: int


def bootstrap_rate(
    successes: np.ndarray | list[int],
    *,
    n_samples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RateCI:
    x = np.asarray(successes, dtype=float)
    if x.size == 0:
        return RateCI(0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_samples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return RateCI(float(x.mean()), float(lo), float(hi), int(x.size))


def intervals_disjoint(a: RateCI, b: RateCI) -> bool:
    return a.low > b.high or b.low > a.high


def paired_difference(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    *,
    n_samples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RateCI:
    """Bootstrap CI of the per-brief difference a - b (same briefs, paired)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        return RateCI(0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_samples, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return RateCI(float(d.mean()), float(lo), float(hi), int(d.size))
