"""Does Loop A find the hidden outliers faster than random selection?

Arms share the same synthetic history and the same latent truth (same seed):
- `loop_a`: archive-aware fake generator + reward-model ranking of the queue
- `random`: history-blind fake generator + random selection

Each (seed, arm) re-seeds the database, then runs one simulated episode per brief. Reports
the tier-2 found rate, mean batches to the first outlier, mean spend, and the paired
difference. This is the step-12 gate of the spec and a permanent regression check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.embeddings import HashEmbedder
from outlier_ai.core.storage import Storage
from outlier_ai.episodes.loop_a import LoopAResult, LoopARunner
from outlier_ai.generation.backends.fake_backend import FakeBackend
from outlier_ai.generation.service import GenerationService
from outlier_ai.generation.verifier import HeuristicVerifier, load_card_refs
from outlier_ai.images import get_image_backend
from outlier_ai.meta.factory import default_latent
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.meta import AdAccount
from outlier_ai.outlier.recompute import recompute_all
from outlier_ai.reward.cold import HeuristicColdRewardModel
from outlier_ai.synthetic.seed import FAKE_ACCOUNT_ID
from outlier_ai.synthetic.seed import seed as seed_db
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import ImageMode


@dataclass
class ArmResult:
    arm: str
    seed: int
    runs: list[LoopAResult] = field(default_factory=list)

    @property
    def found_rate(self) -> float:
        return (
            mean(1.0 if r.outlier_trajectory_id else 0.0 for r in self.runs) if self.runs else 0.0
        )

    @property
    def mean_batches(self) -> float:
        return mean(r.batches for r in self.runs) if self.runs else 0.0

    @property
    def mean_spend(self) -> float:
        return mean(r.spent for r in self.runs) if self.runs else 0.0

    @property
    def mean_days_to_outlier(self) -> float | None:
        hits = [r.days for r in self.runs if r.outlier_trajectory_id]
        return mean(hits) if hits else None


@dataclass
class ExperimentReport:
    arms: dict[str, list[ArmResult]]
    n_briefs: int
    seeds: list[int]

    def summary(self) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        for arm, results in self.arms.items():
            runs = [r for res in results for r in res.runs]
            found = [1.0 if r.outlier_trajectory_id else 0.0 for r in runs]
            days = [r.days for r in runs if r.outlier_trajectory_id]
            out[arm] = {
                "runs": float(len(runs)),
                "found_rate": mean(found) if found else 0.0,
                "mean_batches": mean(r.batches for r in runs) if runs else 0.0,
                "mean_spend": mean(r.spent for r in runs) if runs else 0.0,
                "mean_days_to_outlier": mean(days) if days else None,
                "mean_ideas_shipped": mean(r.ideas_shipped for r in runs) if runs else 0.0,
            }
        return out

    def as_dict(self) -> dict:
        return {
            "n_briefs": self.n_briefs,
            "seeds": self.seeds,
            "summary": self.summary(),
            "runs": {
                arm: [asdict(r) for res in results for r in res.runs]
                for arm, results in self.arms.items()
            },
        }


def _tune(
    cfg: AppConfig, *, renders_per_idea: int, k: int, ideas_per_batch: int, max_batches: int
) -> AppConfig:
    c = cfg.model_copy(deep=True)
    c.episode.renders_per_idea = renders_per_idea
    c.generation.k = k
    c.episode.ideas_per_batch = ideas_per_batch
    c.episode.max_batches = max_batches
    return c


async def run_arm(
    session: AsyncSession,
    *,
    arm: str,
    seed: int,
    cfg: AppConfig,
    storage: Storage,
    n_briefs: int,
    history_trajectories: int,
    budget_cap: float,
    max_days: int,
) -> ArmResult:
    await seed_db(
        session,
        n_briefs=max(n_briefs, 3),
        n_trajectories=history_trajectories,
        seed=seed,
        days_back=100,
        config=cfg,
    )
    await recompute_all(session, cfg)
    await session.commit()
    account = (
        await session.execute(select(AdAccount).where(AdAccount.meta_account_id == FAKE_ACCOUNT_ID))
    ).scalar_one()
    briefs = (
        (await session.execute(select(Brief).order_by(Brief.created_at).limit(n_briefs)))
        .scalars()
        .all()
    )
    cards = await load_card_refs(session)
    policy = "archive" if arm == "loop_a" else "random"
    approval = "top_rm" if arm == "loop_a" else "random"
    result = ArmResult(arm=arm, seed=seed)
    for i, brief in enumerate(briefs):
        client = FakeMetaClient(
            session,
            FAKE_ACCOUNT_ID,
            default_latent(seed),
            niche_projection=cfg.archive.niche_projection,
            seed=seed * 1000 + i,
        )
        service = GenerationService(
            backend=FakeBackend(seed=seed * 100 + i, policy=policy),
            verifier=HeuristicVerifier(cards),
            reward_model=HeuristicColdRewardModel(),
            image_backend=get_image_backend(ImageMode.brand_assets),
            storage=storage,
            embedder=HashEmbedder(cfg.archive.embedding_dims),
            cfg=cfg,
        )
        episode = SearchEpisode(
            brief_id=brief.id,
            ad_account_id=account.id,
            backend="fake",
            budget_cap=budget_cap,
            created_by=f"experiment:{arm}",
            config_hash=cfg.hash,
        )
        session.add(episode)
        await session.flush()
        runner = LoopARunner(
            session=session,
            cfg=cfg,
            service=service,
            client=client,
            storage=storage,
            approval=approval,
            actor=f"experiment:{arm}",
            seed=seed * 7 + i,
        )  # type: ignore[arg-type]
        run = await runner.simulate(episode, max_days=max_days)
        await session.commit()
        result.runs.append(run)
    return result


async def run_experiment(
    session: AsyncSession,
    *,
    cfg: AppConfig,
    storage: Storage,
    n_briefs: int = 6,
    seeds: list[int] | None = None,
    history_trajectories: int = 120,
    budget_cap: float = 4000.0,
    max_days: int = 90,
    renders_per_idea: int = 1,
    k: int = 6,
    ideas_per_batch: int = 3,
    max_batches: int = 5,
    arms: tuple[str, ...] = ("loop_a", "random"),
) -> ExperimentReport:
    seeds = seeds or [1]
    tuned = _tune(
        cfg,
        renders_per_idea=renders_per_idea,
        k=k,
        ideas_per_batch=ideas_per_batch,
        max_batches=max_batches,
    )
    results: dict[str, list[ArmResult]] = {a: [] for a in arms}
    for s in seeds:
        for arm in arms:
            results[arm].append(
                await run_arm(
                    session,
                    arm=arm,
                    seed=s,
                    cfg=tuned,
                    storage=storage,
                    n_briefs=n_briefs,
                    history_trajectories=history_trajectories,
                    budget_cap=budget_cap,
                    max_days=max_days,
                )
            )
    return ExperimentReport(arms=results, n_briefs=n_briefs, seeds=seeds)


def episode_ids(report: ExperimentReport) -> list[UUID]:
    return [r.episode_id for res in report.arms.values() for a in res for r in a.runs]
