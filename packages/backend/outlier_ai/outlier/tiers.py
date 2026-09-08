"""Tiers by multiple of the account median (spec section 4)."""

from __future__ import annotations

from outlier_schemas.config import TierMultiples, TierRewards
from outlier_schemas.enums import Tier


def tier_for_ratio(ratio: float, multiples: TierMultiples) -> Tier:
    if ratio >= multiples.tier3:
        return Tier.three
    if ratio >= multiples.tier2:
        return Tier.two
    if ratio >= multiples.tier1:
        return Tier.one
    return Tier.zero


def multiple_for_tier(tier: Tier, multiples: TierMultiples) -> float:
    """The lower boundary of a tier. Tier 0 has no boundary and returns 0."""
    return {
        Tier.zero: 0.0,
        Tier.one: multiples.tier1,
        Tier.two: multiples.tier2,
        Tier.three: multiples.tier3,
    }[tier]


def reward_for_tier(tier: Tier, rewards: TierRewards) -> float:
    return {
        Tier.zero: rewards.tier0,
        Tier.one: rewards.tier1,
        Tier.two: rewards.tier2,
        Tier.three: rewards.tier3,
    }[tier]


def is_outlier(tier: Tier) -> bool:
    return tier >= Tier.two
