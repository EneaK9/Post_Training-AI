"""Model screen data: Loop A stats, review stats, verifier panel, RM panel, architecture."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from outlier_ai.api.deps import DB, Config, CurrentUser
from outlier_ai.api.schemas import GoldGapOut
from outlier_ai.api.views import cards_map
from outlier_ai.core.settings import get_settings
from outlier_ai.models.cards import Combination
from outlier_ai.models.meta import AdAccount
from outlier_ai.models.ml import EvalRun, RewardModelVersion, TrainingRun, VerifierVersion
from outlier_ai.models.trajectories import Render, Review, ScreeningStats, Trajectory
from outlier_ai.reward.train import gold_gap_check

router = APIRouter(prefix="/stats", tags=["stats"])


def _rate(num: int, den: int) -> float | None:
    return (num / den) if den else None


@router.get("/loop_a")
async def loop_a_stats(db: DB, _: CurrentUser) -> dict[str, Any]:
    """Ideas generated, novelty rejected, shipped, screening pass rate, tier 2 rate per backend,
    per combination, per typicality label. Answers whether "rare" ideas actually hit more."""
    trajs = (
        await db.execute(
            select(
                Trajectory.id,
                Trajectory.backend,
                Trajectory.typicality,
                Trajectory.format_ok,
                Trajectory.outlier_tier,
                Trajectory.verified_card_ids,
                Trajectory.card_ids,
            )
        )
    ).all()
    reviews = {r.trajectory_id: r for r in (await db.execute(select(Review))).scalars().all()}
    shipped_ids = {
        tid
        for (tid,) in (
            await db.execute(
                select(Render.trajectory_id).where(Render.shipped_at.is_not(None)).distinct()
            )
        ).all()
    }
    screening = (
        await db.execute(
            select(Render.trajectory_id, ScreeningStats.passed).join(
                ScreeningStats, ScreeningStats.render_id == Render.id
            )
        )
    ).all()
    passed_by: dict[Any, bool] = {}
    for tid, passed in screening:
        passed_by[tid] = passed_by.get(tid, False) or bool(passed)

    def bucket():
        return {
            "generated": 0,
            "format_rejected": 0,
            "novelty_rejected": 0,
            "shipped": 0,
            "screened": 0,
            "screen_passed": 0,
            "measured": 0,
            "tier2": 0,
        }

    by_backend: dict[str, dict[str, int]] = defaultdict(bucket)
    by_typ: dict[str, dict[str, int]] = defaultdict(bucket)
    for tid, backend, typ, fmt_ok, tier, _verified, _written in trajs:
        for key, table in ((backend or "human", by_backend), (typ or "unlabeled", by_typ)):
            b = table[key]
            b["generated"] += 1
            if not fmt_ok:
                b["format_rejected"] += 1
            r = reviews.get(tid)
            if r and r.label == "skip" and r.note.startswith("novelty_reject"):
                b["novelty_rejected"] += 1
            if tid in shipped_ids:
                b["shipped"] += 1
            if tid in passed_by:
                b["screened"] += 1
                b["screen_passed"] += 1 if passed_by[tid] else 0
            if tier is not None:
                b["measured"] += 1
                b["tier2"] += 1 if tier >= 2 else 0

    def enrich(table: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
        return {
            k: {
                **v,
                "screen_pass_rate": _rate(v["screen_passed"], v["screened"]),
                "tier2_rate": _rate(v["tier2"], v["measured"]),
            }
            for k, v in table.items()
        }

    cards = await cards_map(db)
    combos = (await db.execute(select(Combination).where(Combination.uses > 0))).scalars().all()
    combo_rows = sorted(
        (
            {
                "cards": [cards[i].slug for i in c.card_ids if i in cards],
                "niche": c.niche_key,
                "uses": c.uses,
                "measured": sum(int(v) for v in (c.tier_counts or {}).values()),
                "tier2": int((c.tier_counts or {}).get("2", 0))
                + int((c.tier_counts or {}).get("3", 0)),
                "tier2_rate": c.tier2_rate,
            }
            for c in combos
        ),
        key=lambda r: (-(r["tier2_rate"] or 0.0), -r["uses"]),
    )[:50]
    return {
        "by_backend": enrich(by_backend),
        "by_typicality": enrich(by_typ),
        "by_combination": combo_rows,
        "totals": enrich({"all": _merge(by_backend.values())})["all"],
    }


def _merge(buckets) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for b in buckets:
        for k, v in b.items():
            out[k] += v
    return dict(out)


@router.get("/reviews")
async def review_stats(db: DB, _: CurrentUser) -> dict[str, Any]:
    rows = (
        await db.execute(
            select(Trajectory.backend, Review.label, Review.note).join(
                Review, Review.trajectory_id == Trajectory.id
            )
        )
    ).all()
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {"run": 0, "skip": 0, "wrong_cards": 0, "novelty_rejects": 0, "format_rejects": 0}
    )
    for backend, label, note in rows:
        b = out[backend or "human"]
        if label == "skip" and note.startswith("novelty_reject"):
            b["novelty_rejects"] += 1
        elif label == "skip" and note.startswith("format_error"):
            b["format_rejects"] += 1
        else:
            b[label] += 1
    return dict(out)


@router.get("/verifier")
async def verifier_stats(db: DB, _: CurrentUser) -> dict[str, Any]:
    """Agreement between written and verified tags plus, per card, how often experts corrected
    the verifier (the confusion the Model screen shows)."""
    cards = await cards_map(db)
    rows = (
        await db.execute(
            select(
                Trajectory.card_ids,
                Trajectory.verified_card_ids,
                Trajectory.tag_match,
                Trajectory.tag_jaccard,
            ).where(Trajectory.tag_match.is_not(None))
        )
    ).all()
    agree = sum(1 for r in rows if r[2])
    jacc = [r[3] for r in rows if r[3] is not None]
    per_card: dict[str, dict[str, int]] = defaultdict(
        lambda: {"written": 0, "verified": 0, "both": 0, "corrected_in": 0, "corrected_out": 0}
    )
    for written, verified, _, _ in rows:
        w, v = set(written), set(verified)
        for cid in w | v:
            slug = cards[cid].slug if cid in cards else str(cid)
            per_card[slug]["written"] += cid in w
            per_card[slug]["verified"] += cid in v
            per_card[slug]["both"] += cid in w and cid in v
    corrections = (
        await db.execute(
            select(Trajectory.verified_card_ids, Review.corrected_card_ids)
            .join(Review, Review.trajectory_id == Trajectory.id)
            .where(Review.label == "wrong_cards")
        )
    ).all()
    for verified, corrected in corrections:
        v, c = set(verified), set(corrected or [])
        for cid in c - v:
            per_card[cards[cid].slug if cid in cards else str(cid)]["corrected_in"] += 1
        for cid in v - c:
            per_card[cards[cid].slug if cid in cards else str(cid)]["corrected_out"] += 1
    versions = [
        {
            "version": v.version,
            "kind": v.kind,
            "active": v.is_active,
            "labels": v.trained_on_labels,
            "metrics": v.metrics,
        }
        for v in (
            await db.execute(select(VerifierVersion).order_by(VerifierVersion.created_at.desc()))
        )
        .scalars()
        .all()
    ]
    return {
        "tagged": len(rows),
        "agreement_rate": _rate(agree, len(rows)),
        "mean_jaccard": (sum(jacc) / len(jacc)) if jacc else None,
        "wrong_cards_labels": len(corrections),
        "per_card": dict(per_card),
        "versions": versions,
    }


@router.get("/reward_model")
async def reward_model_stats(db: DB, cfg: Config, _: CurrentUser) -> dict[str, Any]:
    versions = [
        {
            "version": v.version,
            "kind": v.kind,
            "active": v.is_active,
            "metrics": v.metrics,
            "created_at": v.created_at.isoformat(),
        }
        for v in (
            await db.execute(
                select(RewardModelVersion).order_by(RewardModelVersion.created_at.desc())
            )
        )
        .scalars()
        .all()
    ]
    scored = (
        await db.execute(
            select(Trajectory.rm_version, Trajectory.rm_score, Trajectory.outlier_tier).where(
                Trajectory.rm_score.is_not(None)
            )
        )
    ).all()
    by_version: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scored": 0,
            "measured": 0,
            "tier2": 0,
            "mean_score_tier2": [],
            "mean_score_other": [],
        }
    )
    for version, score, tier in scored:
        b = by_version[version or "unknown"]
        b["scored"] += 1
        if tier is not None:
            b["measured"] += 1
            if tier >= 2:
                b["tier2"] += 1
                b["mean_score_tier2"].append(score)
            else:
                b["mean_score_other"].append(score)
    for b in by_version.values():
        b["mean_score_tier2"] = (
            (sum(b["mean_score_tier2"]) / len(b["mean_score_tier2"]))
            if b["mean_score_tier2"]
            else None
        )
        b["mean_score_other"] = (
            (sum(b["mean_score_other"]) / len(b["mean_score_other"]))
            if b["mean_score_other"]
            else None
        )
    positives = int(
        (
            await db.execute(
                select(func.count())
                .select_from(Trajectory)
                .where(Trajectory.outlier_tier >= 2, Trajectory.verified_card_ids != [])
            )
        ).scalar_one()
    )
    return {
        "versions": versions,
        "by_version": dict(by_version),
        "tier2_positives": positives,
        "cold_until_positives": cfg.reward_model.cold_until_positives,
        "lambda_pess": cfg.reward_model.lambda_pess,
        "loop_b_ready": positives >= cfg.rl.min_tier2_trajectories_to_start,
    }


@router.get("/architecture")
async def architecture(db: DB, cfg: Config, _: CurrentUser) -> dict[str, Any]:
    """Nodes and edges for the live architecture diagram; every node names its config path."""
    accounts = (await db.execute(select(AdAccount))).scalars().all()
    rm = (
        (await db.execute(select(RewardModelVersion).where(RewardModelVersion.is_active.is_(True))))
        .scalars()
        .first()
    )
    verifier = (
        (await db.execute(select(VerifierVersion).where(VerifierVersion.is_active.is_(True))))
        .scalars()
        .first()
    )
    s = get_settings()
    nodes = [
        {
            "id": "brief",
            "group": "loop_a",
            "label": "Brief (column 2)",
            "detail": "company, product, offer, audience, world state",
            "config": None,
        },
        {
            "id": "playbook",
            "group": "loop_a",
            "label": "Playbook (column 1)",
            "detail": "active cards, grouped by kind",
            "config": None,
        },
        {
            "id": "archive",
            "group": "loop_a",
            "label": "Archive sample",
            "detail": (
                f"n_elite {cfg.archive.n_elite}, n_rare {cfg.archive.n_rare}, "
                f"{cfg.archive.history_days}d, niche {cfg.archive.niche_projection.value}"
            ),
            "config": "archive",
        },
        {
            "id": "prompt",
            "group": "loop_a",
            "label": "Prompt renderer",
            "detail": (
                f"K {cfg.generation.k}, typicality "
                f"{'on' if cfg.generation.typicality_prompting else 'off'}, "
                f"{cfg.generation.max_prompt_tokens} tokens"
            ),
            "config": "generation",
        },
        {
            "id": "backend",
            "group": "loop_a",
            "label": "Generator",
            "detail": (
                f"default {cfg.generation.default_backend.value}; anthropic "
                f"{cfg.generation.anthropic_model} effort {cfg.generation.anthropic_effort}; "
                f"local {cfg.generation.local_model} ({cfg.generation.weights.value})"
            ),
            "config": "generation",
        },
        {
            "id": "verifier",
            "group": "loop_a",
            "label": "Card verifier",
            "detail": (
                f"{cfg.verifier.kind.value} ({cfg.verifier.model}); active "
                f"{verifier.version if verifier else 'heuristic/llm v1'}"
            ),
            "config": "verifier",
        },
        {
            "id": "novelty",
            "group": "loop_a",
            "label": "Novelty rejection",
            "detail": f"threshold {cfg.archive.novelty_threshold}",
            "config": "archive",
        },
        {
            "id": "rm",
            "group": "loop_a",
            "label": "Reward model",
            "detail": (
                f"{rm.version if rm else 'rm_cold'}; lambda {cfg.reward_model.lambda_pess}; "
                f"cold until {cfg.reward_model.cold_until_positives}"
            ),
            "config": "reward_model",
        },
        {
            "id": "images",
            "group": "loop_a",
            "label": "Images",
            "detail": f"{cfg.images.mode.value}, {cfg.episode.renders_per_idea} renders",
            "config": "images",
        },
        {
            "id": "review",
            "group": "loop_a",
            "label": "Review queue",
            "detail": "human run label is the approval",
            "config": None,
        },
        {
            "id": "ship",
            "group": "evaluator",
            "label": "Ship",
            "detail": (
                f"budget governor: daily cap ${cfg.meta.daily_account_cap_usd:.0f}; "
                f"dry_run {s.dry_run}"
            ),
            "config": "meta",
        },
        {
            "id": "meta",
            "group": "evaluator",
            "label": "Meta (evaluator)",
            "detail": f"{len(accounts)} accounts: "
            + ", ".join(f"{a.name} [{a.status}{', fake' if a.is_fake else ''}]" for a in accounts),
            "config": "meta",
        },
        {
            "id": "screening",
            "group": "evaluator",
            "label": "Screening",
            "detail": (
                f"CTR lb >= {cfg.outlier.screening.ctr_multiple}x median, "
                f"{cfg.outlier.screening.min_impressions} impr, "
                f"{cfg.outlier.screening.window_days}d"
            ),
            "config": "outlier",
        },
        {
            "id": "scale",
            "group": "evaluator",
            "label": "Scale + tiers",
            "detail": (
                f"tiers {cfg.outlier.tier_multiples.tier1}/{cfg.outlier.tier_multiples.tier2}/"
                f"{cfg.outlier.tier_multiples.tier3}x; gates {cfg.outlier.min_purchases_at_scale} "
                f"purchases, {cfg.outlier.durability_days}d"
            ),
            "config": "outlier",
        },
        {
            "id": "signals",
            "group": "evaluator",
            "label": "Comments -> signals",
            "detail": f"extraction {cfg.signals.extraction_model}; PII stripped",
            "config": "signals",
        },
        {
            "id": "snapshot",
            "group": "loop_b",
            "label": "Archive snapshot",
            "detail": "Parquet export, holdout excluded",
            "config": None,
        },
        {
            "id": "trainer",
            "group": "loop_b",
            "label": "Loop B trainer",
            "detail": (
                f"RFT -> DPO -> off-policy GRPO; K {cfg.rl.k}, tau {cfg.rl.tau_start}->"
                f"{cfg.rl.tau_target}, beta {cfg.rl.beta}, KL {cfg.rl.kl_coef}"
            ),
            "config": "rl",
        },
        {
            "id": "guards",
            "group": "loop_b",
            "label": "Guards",
            "detail": (
                f"swing >= {cfg.rl.guards.swing_rate_min}, "
                f"entropy >= {cfg.rl.guards.entropy_floor}, "
                f"gold gap {cfg.reward_model.gold_gap_threshold}"
            ),
            "config": "rl",
        },
        {
            "id": "policy",
            "group": "loop_b",
            "label": "LocalPolicy checkpoint",
            "detail": "LoRA adapter served by vLLM",
            "config": "generation",
        },
    ]
    edges = [
        ("brief", "prompt"),
        ("playbook", "prompt"),
        ("archive", "prompt"),
        ("prompt", "backend"),
        ("backend", "verifier"),
        ("verifier", "novelty"),
        ("novelty", "rm"),
        ("rm", "review"),
        ("backend", "images"),
        ("images", "review"),
        ("review", "ship"),
        ("ship", "meta"),
        ("meta", "screening"),
        ("screening", "scale"),
        ("meta", "signals"),
        ("signals", "archive"),
        ("scale", "archive"),
        ("archive", "snapshot"),
        ("snapshot", "trainer"),
        ("rm", "trainer"),
        ("trainer", "guards"),
        ("trainer", "policy"),
        ("policy", "backend"),
    ]
    runs = [
        {
            "id": str(r.id),
            "stage": r.stage,
            "status": r.status,
            "metrics": r.metrics,
            "stop_reason": r.stop_reason,
            "created_at": r.created_at.isoformat(),
        }
        for r in (
            await db.execute(select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(20))
        )
        .scalars()
        .all()
    ]
    evals = [
        {
            "id": str(e.id),
            "kind": e.kind,
            "status": e.status,
            "summary": e.summary,
            "created_at": e.created_at.isoformat(),
        }
        for e in (await db.execute(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(20)))
        .scalars()
        .all()
    ]
    return {
        "nodes": nodes,
        "edges": [{"source": a, "target": b} for a, b in edges],
        "config_hash": cfg.hash,
        "training_runs": runs,
        "eval_runs": evals,
        "dry_run": s.dry_run,
    }


@router.get("/gold_gap", response_model=GoldGapOut)
async def gold_gap_stats(db: DB, cfg: Config, _: CurrentUser, window: int = 100) -> GoldGapOut:
    """Proxy-vs-real drift of the active reward model (spec section 6). Tripped pauses Loop B."""
    gg = await gold_gap_check(db, cfg, window=window)
    return GoldGapOut(**gg, threshold=cfg.reward_model.gold_gap_threshold)
