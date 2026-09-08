"""Hidden ground truth for synthetic evaluation.

`LatentOutcomeModel` maps (niche, brief category, world state) to a heavy-tailed ROAS
multiple relative to the account median, with correlated CTR and a polarity that drives
comment volume and sentiment. A few (niche, category) pairs are "hot": their mean sits at
tier 2 territory so there is something for Loop A to find. Everything else is noise around
below-median, mirroring the roughly 5 percent tail the spec describes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MU = float(np.log(0.85))  # typical ad lands below the account median
HOT_MU_RANGE = (float(np.log(3.2)), float(np.log(4.5)))
WARM_MU_RANGE = (float(np.log(1.3)), float(np.log(1.9)))


@dataclass(frozen=True)
class AccountProfile:
    median_roas: float = 2.2
    median_ctr: float = 0.012
    cpm_usd: float = 12.0
    aov_usd: float = 60.0
    attribution_setting: str = "7d_click_1d_view"


@dataclass(frozen=True)
class AdTruth:
    roas_multiple: float
    ctr_multiple: float
    polarity: float  # 0..1, how much the ad divides the audience


@dataclass(frozen=True)
class DailyRow:
    impressions: int
    link_clicks: int
    spend: float
    purchases: int
    revenue: float
    frequency: float
    reactions: int
    comments: int
    shares: int
    saves: int


class LatentOutcomeModel:
    def __init__(
        self,
        seed: int,
        categories: list[str],
        strategy_slugs: list[str],
        world_tags: list[str] | None = None,
        n_hot_pairs: int = 6,
        n_warm_pairs: int = 10,
        sigma: float = 0.55,
        tail_prob: float = 0.03,
    ) -> None:
        self.seed = seed
        self.sigma = sigma
        self.tail_prob = tail_prob
        rng = np.random.default_rng(seed)
        niches = self._candidate_niches(strategy_slugs)
        pairs = [(n, c) for n in niches for c in categories]
        idx = rng.permutation(len(pairs))
        self.affinity: dict[tuple[str, str], float] = {}
        for i in idx[:n_hot_pairs]:
            self.affinity[pairs[i]] = float(rng.uniform(*HOT_MU_RANGE))
        for i in idx[n_hot_pairs : n_hot_pairs + n_warm_pairs]:
            self.affinity[pairs[i]] = float(rng.uniform(*WARM_MU_RANGE))
        self.world_boost: dict[tuple[str, str], float] = {}
        for tag in world_tags or []:
            for c in categories:
                if rng.random() < 0.25:
                    self.world_boost[(tag, c)] = float(rng.normal(0.0, 0.25))

    @staticmethod
    def _candidate_niches(strategy_slugs: list[str]) -> list[str]:
        singles = sorted(strategy_slugs)
        pairs = ["+".join(sorted((a, b))) for i, a in enumerate(singles) for b in singles[i + 1 :]]
        return singles + pairs

    def hot_pairs(self) -> list[tuple[str, str]]:
        return [k for k, mu in self.affinity.items() if mu >= HOT_MU_RANGE[0]]

    def mean_multiple(self, niche: str, category: str, world_tag: str = "none") -> float:
        mu = self.affinity.get((niche, category), DEFAULT_MU) + self.world_boost.get(
            (world_tag, category), 0.0
        )
        return float(np.exp(mu + self.sigma**2 / 2))

    def sample_truth(
        self, niche: str, category: str, world_tag: str, rng: np.random.Generator
    ) -> AdTruth:
        mu = self.affinity.get((niche, category), DEFAULT_MU) + self.world_boost.get(
            (world_tag, category), 0.0
        )
        multiple = float(np.exp(mu + self.sigma * rng.normal()))
        if rng.random() < self.tail_prob:
            multiple *= 1.0 + float(rng.pareto(1.5))
        ctr_multiple = float(np.exp(0.5 * np.log(multiple) + 0.3 * rng.normal()))
        polarity = float(
            np.clip(0.25 + 0.35 * (ctr_multiple > 1.3) + 0.15 * rng.normal(), 0.0, 1.0)
        )
        return AdTruth(roas_multiple=multiple, ctr_multiple=ctr_multiple, polarity=polarity)

    @staticmethod
    def fatigue(day_index: int, phase: str) -> float:
        if phase != "scale":
            return 1.0
        return max(0.6, 1.0 - 0.02 * day_index)

    def simulate_day(
        self,
        truth: AdTruth,
        daily_budget: float,
        account: AccountProfile,
        day_index: int,
        phase: str,
        rng: np.random.Generator,
    ) -> DailyRow:
        spend = daily_budget * float(rng.uniform(0.9, 1.0))
        impressions = int(rng.poisson(spend / account.cpm_usd * 1000.0))
        fat = self.fatigue(day_index, phase)
        ctr = min(0.2, account.median_ctr * truth.ctr_multiple * fat)
        clicks = int(rng.binomial(impressions, ctr)) if impressions > 0 else 0
        target_roas = account.median_roas * truth.roas_multiple * fat
        expected_purchases = target_roas * spend / account.aov_usd
        purchases = int(rng.poisson(expected_purchases))
        revenue = (
            float(purchases * account.aov_usd * np.exp(0.15 * rng.normal())) if purchases else 0.0
        )
        frequency = 1.0 + 0.15 * day_index
        reactions = int(rng.poisson(impressions * 0.004 * (0.5 + truth.ctr_multiple)))
        comments = int(rng.poisson(impressions * 0.0008 * (0.5 + 2.0 * truth.polarity)))
        shares = int(rng.poisson(impressions * 0.0005 * truth.ctr_multiple))
        saves = int(rng.poisson(impressions * 0.0006 * min(truth.roas_multiple, 5.0)))
        return DailyRow(
            impressions=impressions,
            link_clicks=clicks,
            spend=round(spend, 2),
            purchases=purchases,
            revenue=round(revenue, 2),
            frequency=round(frequency, 3),
            reactions=reactions,
            comments=comments,
            shares=shares,
            saves=saves,
        )
