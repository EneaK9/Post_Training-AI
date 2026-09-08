"""Reward composition and advantage shaping for Loop B (spec sections 7.2 and 7.3).

Pure numpy so the same math runs in unit tests, in the trainer, and in the simulator.

    reward = outcome_reward (1 if tier >= 2 else 0; None if never shipped)
           + shaping_weight * rm_score            (only when outcome is None; decays with positives)
           - tag_penalty * (1 - tag_match) - format_penalty * (1 - format_ok)

    u = risk_transform(r, tau)                     exponential utility, tau annealed low -> target
    A = (1 - eps_mean) * (u - mean(u)) + eps_mean * (r - mean(r))
    A = A * (1 / cluster_size(verified_card_ids)) ** beta
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    tier2_reward: float = 1.0
    shaping_weight: float = 0.3
    tag_penalty: float = 0.2
    format_penalty: float = 0.5
    cold_until_positives: int = 50


def shaping_decay(n_positives: int, cold_until_positives: int) -> float:
    """1.0 with no real outcomes, linearly down to 0 once the archive has 2x the cold threshold."""
    if cold_until_positives <= 0:
        return 0.0
    return float(max(0.0, 1.0 - n_positives / (2.0 * cold_until_positives)))


def compose_reward(
    *,
    outlier_tier: int | None,
    rm_score: float | None,
    tag_match: bool | None,
    format_ok: bool,
    n_positives: int,
    cfg: RewardConfig,
) -> float:
    if outlier_tier is not None:
        r = cfg.tier2_reward if outlier_tier >= 2 else 0.0
    elif rm_score is not None:
        r = (
            cfg.shaping_weight
            * shaping_decay(n_positives, cfg.cold_until_positives)
            * float(rm_score)
        )
    else:
        r = 0.0
    if tag_match is False:
        r -= cfg.tag_penalty
    if not format_ok:
        r -= cfg.format_penalty
    return float(r)


def tau_schedule(step: int, tau_start: float, tau_target: float, anneal_steps: int) -> float:
    """Geometric anneal from tau_start to tau_target over anneal_steps, then constant."""
    if anneal_steps <= 0 or step >= anneal_steps:
        return float(tau_target)
    frac = step / anneal_steps
    return float(np.exp(np.log(tau_start) + frac * (np.log(tau_target) - np.log(tau_start))))


def risk_transform(r: np.ndarray | Sequence[float], tau: float) -> np.ndarray:
    """Risk-seeking exponential utility u = tau * (exp(r / tau) - 1).

    Convex in r, so the best of a group earns disproportionately more; as tau grows it tends to
    r itself (the mean objective). Small tau approximates max@k.
    """
    x = np.asarray(r, dtype=float)
    if tau <= 0:
        raise ValueError("tau must be positive")
    return tau * (np.exp(x / tau) - 1.0)


def group_advantages(r: np.ndarray | Sequence[float], tau: float, eps_mean: float) -> np.ndarray:
    x = np.asarray(r, dtype=float)
    if x.size == 0:
        return x
    u = risk_transform(x, tau)
    return (1.0 - eps_mean) * (u - u.mean()) + eps_mean * (x - x.mean())


def cluster_sizes(keys: Sequence[Any]) -> np.ndarray:
    counts: dict[Any, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return np.asarray([counts[k] for k in keys], dtype=float)


def uniqueness_scale(a: np.ndarray, keys: Sequence[Any], beta: float) -> np.ndarray:
    """Down-weight advantages of rollouts that share a verified combination with many others
    (Uniqueness-Aware RL): A * (1 / cluster_size) ** beta."""
    sizes = cluster_sizes(keys)
    return np.asarray(a, dtype=float) * (1.0 / sizes) ** beta


def advantages(
    rewards: Sequence[float],
    combo_keys: Sequence[Any],
    *,
    tau: float,
    eps_mean: float,
    beta: float,
) -> np.ndarray:
    return uniqueness_scale(group_advantages(rewards, tau, eps_mean), combo_keys, beta)
