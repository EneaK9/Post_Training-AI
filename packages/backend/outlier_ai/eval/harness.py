"""Run the online eval: every system on every held-out brief, equal budgets, blind arm labels.

Fake accounts run the simulated loop to completion; real accounts create the episodes and let the
worker cadence drive them (results fill in over weeks). `tier2_rate` per system with bootstrap
intervals; `loop_b_vs_loop_a` is the paired difference between `loop_b` and `loop_a_local`.
Success (spec section 11): Outlier AI beats the mean-objective baseline with non-overlapping
intervals on at least 20 briefs. Offline proxies are reported, never used as success.
"""

from __future__ import annotations

import string
from dataclasses import asdict, dataclass, field
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.core.storage import Storage
from outlier_ai.episodes.loop_a import LoopARunner
from outlier_ai.eval.holdout import blocklist_campaigns, select_holdout_briefs
from outlier_ai.eval.stats import RateCI, bootstrap_rate, intervals_disjoint, paired_difference
from outlier_ai.eval.systems import SYSTEMS, System
from outlier_ai.generation.factory import build_service
from outlier_ai.meta.factory import client_for_account
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.meta import AdAccount
from outlier_ai.models.ml import EvalArm, EvalRun
from outlier_ai.models.trajectories import Trajectory
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import EvalKind

MIN_BRIEFS_FOR_SUCCESS = 20


@dataclass
class ArmOutcome:
    system: str
    blind_label: str
    episode_ids: list[UUID] = field(default_factory=list)
    successes: list[int] = field(default_factory=list)  # per brief: 1 if tier 2+ found
    spend: list[float] = field(default_factory=list)
    ci: RateCI | None = None


@dataclass
class EvalResult:
    eval_id: UUID
    kind: EvalKind
    brief_ids: list[UUID]
    arms: dict[str, ArmOutcome]
    loop_b_vs_loop_a: RateCI | None
    success: bool | None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "n_briefs": len(self.brief_ids),
            "arms": {
                k: {
                    "blind_label": a.blind_label,
                    "tier2_rate": a.ci.rate if a.ci else None,
                    "ci": [a.ci.low, a.ci.high] if a.ci else None,
                    "n": len(a.successes),
                    "mean_spend": float(np.mean(a.spend)) if a.spend else None,
                }
                for k, a in self.arms.items()
            },
            "loop_b_vs_loop_a": asdict(self.loop_b_vs_loop_a) if self.loop_b_vs_loop_a else None,
            "success": self.success,
            "notes": self.notes,
        }


async def _account_for(session: AsyncSession, brief: Brief) -> AdAccount | None:
    return await session.get(AdAccount, brief.ad_account_id) if brief.ad_account_id else None


async def run_eval(
    session: AsyncSession,
    *,
    cfg: AppConfig,
    storage: Storage,
    systems: list[str],
    n_briefs: int,
    budget_cap: float,
    created_by: str,
    kind: EvalKind = EvalKind.online,
    seed: int = 0,
    max_days: int = 120,
    attribution_setting: str | None = None,
    settings: Settings | None = None,
) -> EvalResult:
    s = settings or get_settings()
    unknown = [x for x in systems if x not in SYSTEMS]
    if unknown:
        raise ValueError(f"unknown systems {unknown}")
    briefs = await select_holdout_briefs(session, n=n_briefs, seed=seed)
    brief_ids = [b.id for b in briefs]
    run = EvalRun(
        kind=kind.value,
        status="running",
        holdout_brief_ids=brief_ids,
        attribution_setting=attribution_setting or cfg.meta.attribution_setting,
        config_hash=cfg.hash,
        created_by=created_by,
    )
    session.add(run)
    await session.flush()
    await blocklist_campaigns(session, brief_ids)
    labels = list(string.ascii_uppercase)
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)  # blind labels: A, B, C... assigned at random
    arms: dict[str, ArmOutcome] = {
        name: ArmOutcome(system=name, blind_label=labels[i]) for i, name in enumerate(systems)
    }
    notes: list[str] = []
    for brief in briefs:
        account = await _account_for(session, brief)
        if account is None:
            notes.append(f"brief {brief.id} has no ad account; skipped")
            continue
        for i, name in enumerate(systems):
            system: System = SYSTEMS[name]
            episode = SearchEpisode(
                brief_id=brief.id,
                ad_account_id=account.id,
                backend=system.backend.value,
                budget_cap=budget_cap,
                created_by=f"eval:{run.id}:{arms[name].blind_label}",
                config_hash=cfg.hash,
            )
            session.add(episode)
            await session.flush()
            arms[name].episode_ids.append(episode.id)
            client = await client_for_account(
                session, account, cfg, settings=s, fake_seed=seed * 100 + i, require_shippable=True
            )
            service = await build_service(
                session,
                cfg,
                backend_kind=system.backend,
                settings=s,
                use_llm_judges=system.backend.value != "fake",
                seed=seed * 10 + i,
                adapter=system.adapter,
                fake_policy=system.fake_policy,
                storage=storage,
            )
            runner = LoopARunner(
                session=session,
                cfg=cfg,
                service=service,
                client=client,
                storage=storage,
                settings=s,
                approval=system.approval,
                actor=f"eval:{run.id}",
                seed=seed,
            )
            if isinstance(client, FakeMetaClient):
                result = await runner.simulate(episode, max_days=max_days)
                arms[name].successes.append(1 if result.outlier_trajectory_id else 0)
                arms[name].spend.append(result.spent)
            else:
                await runner.step(episode)
                notes.append(
                    f"episode {episode.id} started on a live account; "
                    "results arrive with the daily cadence"
                )
            # every campaign an eval episode creates is held out from training
            if episode.campaign_id:
                await blocklist_campaigns(
                    session, [], campaign_ids=[episode.campaign_id], added_by=f"eval:{run.id}"
                )
    for a in arms.values():
        if a.successes:
            a.ci = bootstrap_rate(a.successes, seed=seed)
    lb_vs_la: RateCI | None = None
    if (
        "loop_b" in arms
        and "loop_a_local" in arms
        and arms["loop_b"].successes
        and len(arms["loop_b"].successes) == len(arms["loop_a_local"].successes)
    ):
        lb_vs_la = paired_difference(
            arms["loop_b"].successes, arms["loop_a_local"].successes, seed=seed
        )
    success: bool | None = None
    ours = next(
        (
            arms[n]
            for n in ("loop_b", "loop_a_api", "loop_a_local", "loop_a_fake")
            if n in arms and arms[n].ci
        ),
        None,
    )
    base = next(
        (arms[n] for n in ("mean_rl_baseline", "random_fake") if n in arms and arms[n].ci), None
    )
    if ours and base and ours.ci and base.ci:
        enough = len(ours.successes) >= MIN_BRIEFS_FOR_SUCCESS
        if enough:
            success = bool(ours.ci.rate > base.ci.rate and intervals_disjoint(ours.ci, base.ci))
        else:
            # no verdict either way: the rule needs 20 briefs before intervals mean anything
            success = None
            notes.append(
                f"{len(ours.successes)} briefs < {MIN_BRIEFS_FOR_SUCCESS} required "
                "for a success verdict"
            )
    result = EvalResult(
        eval_id=run.id,
        kind=kind,
        brief_ids=brief_ids,
        arms=arms,
        loop_b_vs_loop_a=lb_vs_la,
        success=success,
        notes=notes,
    )
    for a in arms.values():
        session.add(
            EvalArm(
                eval_run_id=run.id,
                system=a.system,
                blind_label=a.blind_label,
                n_briefs=len(a.successes),
                tier2_rate=a.ci.rate if a.ci else None,
                ci_low=a.ci.low if a.ci else None,
                ci_high=a.ci.high if a.ci else None,
                details={
                    "episode_ids": [str(e) for e in a.episode_ids],
                    "successes": a.successes,
                    "spend": a.spend,
                },
            )
        )
    run.status = "completed" if all(a.successes for a in arms.values()) else "running"
    run.summary = result.summary()
    from datetime import UTC, datetime

    if run.status == "completed":
        run.finished_at = datetime.now(UTC)
    await session.flush()
    return result


async def refresh_live_eval(session: AsyncSession, eval_id: UUID, *, seed: int = 0) -> dict:
    """For live accounts: recompute per-arm tier 2 rates from the episodes' current state."""
    run = await session.get(EvalRun, eval_id)
    if run is None:
        raise ValueError("eval not found")
    arms = (
        (await session.execute(select(EvalArm).where(EvalArm.eval_run_id == eval_id)))
        .scalars()
        .all()
    )
    for a in arms:
        ids = [UUID(x) for x in a.details.get("episode_ids", [])]
        if not ids:
            continue
        rows = (
            await session.execute(
                select(SearchEpisode.id, SearchEpisode.status, SearchEpisode.spent).where(
                    SearchEpisode.id.in_(ids)
                )
            )
        ).all()
        done = [r for r in rows if r.status != "searching"]
        if not done:
            continue
        succ = []
        for r in done:
            best = (
                await session.execute(
                    select(Trajectory.outlier_tier)
                    .where(Trajectory.episode_id == r.id, Trajectory.outlier_tier.is_not(None))
                    .order_by(Trajectory.outlier_tier.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            succ.append(1 if (best or 0) >= 2 else 0)
        ci = bootstrap_rate(succ, seed=seed)
        a.n_briefs, a.tier2_rate, a.ci_low, a.ci_high = len(succ), ci.rate, ci.low, ci.high
        a.details = {**a.details, "successes": succ, "spend": [float(r.spent) for r in done]}
    run.summary = {
        **(run.summary or {}),
        "arms": {
            a.system: {
                "blind_label": a.blind_label,
                "tier2_rate": a.tier2_rate,
                "ci": [a.ci_low, a.ci_high],
                "n": a.n_briefs,
            }
            for a in arms
        },
    }
    await session.flush()
    return run.summary
