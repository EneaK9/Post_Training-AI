"""Train, save, load, and serve the reward-model ensemble.

`rm_cold` learns from expert `run` / `skip` labels until the archive holds enough real tier 2+
positives; then `rm_outcome` learns from realized tiers. Held-out campaigns are excluded. The
artifact is stored in object storage and registered in `rm_models`; the active version scores
new ideas through `EnsembleRewardModel`, which satisfies the same `RewardModel` protocol as the
cold heuristics, so the generation service does not care which one it has.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.embeddings import Embedder
from outlier_ai.core.storage import Storage, key_from_uri
from outlier_ai.models.briefs import Brief
from outlier_ai.models.ml import RewardModelVersion
from outlier_ai.models.ops import HoldoutCampaign
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Review, Trajectory
from outlier_ai.reward.base import IdeaFeatures, RewardScore
from outlier_ai.reward.calibration import CalibrationReport, calibration
from outlier_ai.reward.ensemble import Ensemble, TrainReport, train_ensemble
from outlier_ai.reward.features import FeatureBuilder
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import RewardModelKind
from outlier_schemas.models import AdCopy


@dataclass
class RMTrainResult:
    version: str
    kind: RewardModelKind
    uri: str
    report: TrainReport
    calibration: CalibrationReport
    n_rows: int
    n_positive: int
    val_auc: float | None = None
    activated: bool = False
    note: str = ""


def meets_activation_guards(cfg: AppConfig, n_rows: int, n_positive: int) -> str | None:
    """Reason the data is too thin to trust a trained ensemble, or None when it qualifies."""
    g = cfg.reward_model
    n_negative = n_rows - n_positive
    if n_rows < g.min_rows:
        return f"{n_rows} rows < min_rows {g.min_rows}"
    if n_positive < g.min_per_class or n_negative < g.min_per_class:
        return f"{n_positive} positive / {n_negative} negative < min_per_class {g.min_per_class}"
    return None


async def _features_for(
    session: AsyncSession, trajs: list[Trajectory], cards_by_id: dict
) -> list[IdeaFeatures]:
    briefs = {
        b.id: b
        for b in (
            await session.execute(select(Brief).where(Brief.id.in_({t.brief_id for t in trajs})))
        )
        .scalars()
        .all()
    }
    sig_ids = {i for t in trajs for i in (t.cited_ids or [])}
    sig_text = (
        {
            s.id: s.text
            for s in (
                await session.execute(
                    select(Signal).where(Signal.id.in_(sig_ids), Signal.status == "confirmed")
                )
            )
            .scalars()
            .all()
        }
        if sig_ids
        else {}
    )
    out: list[IdeaFeatures] = []
    for t in trajs:
        b = briefs.get(t.brief_id)
        cited = [sig_text[i] for i in (t.cited_ids or []) if i in sig_text]
        out.append(
            IdeaFeatures(
                brief_text=b.raw_text if b else "",
                world_state=b.world_state if b else "",
                angle=t.angle,
                copy=AdCopy(**(t.ad_copy or {})),
                visual_brief=t.visual_brief,
                verified_card_slugs=[
                    cards_by_id[i].slug for i in t.verified_card_ids if i in cards_by_id
                ],
                typicality=t.typicality,
                novelty_distance=None,
                cited_signal_count=len(cited),
                format_ok=t.format_ok,
                tag_match=t.tag_match,
                cited_signal_texts=cited,
            )
        )
    return out


async def build_dataset(
    session: AsyncSession, kind: RewardModelKind
) -> tuple[list[Trajectory], np.ndarray]:
    """rm_cold: label 1 = expert `run`, 0 = `skip` (novelty/format skips excluded).
    rm_outcome: label 1 = tier 2+, 0 = measured lower. Held-out campaigns excluded."""
    holdout = set((await session.execute(select(HoldoutCampaign.campaign_id))).scalars().all())
    if kind == RewardModelKind.rm_cold:
        rows = (
            await session.execute(
                select(Trajectory, Review.label, Review.note)
                .join(Review, Review.trajectory_id == Trajectory.id)
                .where(Review.label.in_(["run", "skip"]), Trajectory.format_ok.is_(True))
            )
        ).all()
        trajs, labels = [], []
        for t, label, note in rows:
            if t.campaign_id in holdout or note.startswith(
                ("novelty_reject", "format_error", "auto:")
            ):
                continue
            trajs.append(t)
            labels.append(1.0 if label == "run" else 0.0)
        return trajs, np.asarray(labels, dtype=np.float32)
    rows = (
        (
            await session.execute(
                select(Trajectory).where(
                    Trajectory.outlier_tier.is_not(None), Trajectory.verified_card_ids != []
                )
            )
        )
        .scalars()
        .all()
    )
    trajs = [t for t in rows if t.campaign_id not in holdout]
    return trajs, np.asarray(
        [1.0 if (t.outlier_tier or 0) >= 2 else 0.0 for t in trajs], dtype=np.float32
    )


async def train_reward_model(
    session: AsyncSession,
    *,
    cfg: AppConfig,
    embedder: Embedder,
    storage: Storage,
    kind: RewardModelKind | None = None,
    activate: bool = True,
    seed: int = 0,
    force: bool = False,
) -> RMTrainResult | None:
    """Train the ensemble on the labeled archive. Activation needs the held-out AUC guard unless
    `force` is set (a researcher choosing to try the model anyway; the note is still recorded)."""
    from outlier_ai.archive.combinations import load_cards_by_id

    cards_by_id = await load_cards_by_id(session)
    if kind is None:
        _, y_out = await build_dataset(session, RewardModelKind.rm_outcome)
        kind = (
            RewardModelKind.rm_outcome
            if int(y_out.sum()) >= cfg.reward_model.cold_until_positives
            else RewardModelKind.rm_cold
        )
    trajs, y = await build_dataset(session, kind)
    if meets_activation_guards(cfg, len(trajs), int(y.sum())) is not None:
        return None
    fb = FeatureBuilder(embedder)
    x = fb.build(await _features_for(session, trajs, cards_by_id))
    # timestamp plus a short random suffix: two trainings in the same second must not collide
    version = f"{kind.value}-ens-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}"
    # held-out check first: a model that cannot rank unseen labeled ideas must not steer the queue
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_val = max(4, len(y) // 5)
    val_idx, tr_idx = order[:n_val], order[n_val:]
    val_auc: float | None = None
    if y[tr_idx].sum() >= 2 and (len(tr_idx) - y[tr_idx].sum()) >= 2 and len(set(y[val_idx])) == 2:
        ens_v, _ = train_ensemble(
            x[tr_idx],
            y[tr_idx],
            heads=cfg.reward_model.heads,
            gamma=cfg.reward_model.focal_gamma,
            seed=seed,
        )
        mean_v, _ = ens_v.score(x[val_idx])
        val_auc = calibration(mean_v, y[val_idx]).auc
    note = ""
    if val_auc is None:
        note = "no held-out AUC (validation split lacked both classes)"
    elif val_auc < cfg.reward_model.min_val_auc:
        note = f"held-out AUC {val_auc:.2f} < min_val_auc {cfg.reward_model.min_val_auc}"
    activate = activate and (force or not note)
    ens, report = train_ensemble(
        x,
        y,
        heads=cfg.reward_model.heads,
        gamma=cfg.reward_model.focal_gamma,
        seed=seed,
        meta={"kind": kind.value, "embedder": embedder.name, "dims": fb.dims, "version": version},
    )
    mean, _ = ens.score(x)
    cal = calibration(mean, y)
    uri = storage.put(f"models/rm/{version}.npz", ens.to_bytes(), "application/octet-stream")
    if activate:
        for row in (
            (
                await session.execute(
                    select(RewardModelVersion).where(RewardModelVersion.is_active.is_(True))
                )
            )
            .scalars()
            .all()
        ):
            row.is_active = False
    session.add(
        RewardModelVersion(
            version=version,
            kind=kind.value,
            artifact_uri=uri,
            is_active=activate,
            metrics={
                "train": asdict(report),
                "calibration": asdict(cal),
                "n_rows": len(trajs),
                "n_positive": int(y.sum()),
                "val_auc": val_auc,
                "note": note,
            },
        )
    )
    await session.flush()
    return RMTrainResult(
        version=version,
        kind=kind,
        uri=uri,
        report=report,
        calibration=cal,
        n_rows=len(trajs),
        n_positive=int(y.sum()),
        val_auc=val_auc,
        activated=activate,
        note=note,
    )


class EnsembleRewardModel:
    """RewardModel protocol backed by a trained ensemble artifact."""

    def __init__(
        self, ensemble: Ensemble, embedder: Embedder, version: str, kind: RewardModelKind
    ) -> None:
        self.ensemble = ensemble
        self.fb = FeatureBuilder(embedder)
        self.version = version
        self.kind = kind

    async def score(self, features: IdeaFeatures) -> RewardScore:
        x = self.fb.build([features])
        mean, std = self.ensemble.score(x)
        return RewardScore(float(mean[0]), float(std[0]), self.version, self.kind)


async def load_active_reward_model(
    session: AsyncSession, storage: Storage, embedder: Embedder, cfg: AppConfig | None = None
) -> EnsembleRewardModel | None:
    row = (
        (
            await session.execute(
                select(RewardModelVersion)
                .where(RewardModelVersion.is_active.is_(True))
                .order_by(RewardModelVersion.created_at.desc())
            )
        )
        .scalars()
        .first()
    )
    if row is None or not row.artifact_uri:
        return None
    if cfg is not None and meets_activation_guards(
        cfg, int(row.metrics.get("n_rows", 0)), int(row.metrics.get("n_positive", 0))
    ):
        return None  # trained on too little data for the current guards: stay on the cold model
    ens = Ensemble.from_bytes(storage.get(key_from_uri(row.artifact_uri)))
    if ens.meta.get("embedder") != embedder.name:
        return None  # features would not line up
    return EnsembleRewardModel(ens, embedder, row.version, RewardModelKind(row.kind))


async def gold_gap_check(session: AsyncSession, cfg: AppConfig, *, window: int = 100) -> dict:
    """Compare the active RM's scores vs realized tier 2 rate on the most recent measured ideas
    against the previous window. Returns the GoldGap as a dict for the API and the trainer."""
    from outlier_ai.reward.gold_gap import gold_gap

    rows = (
        await session.execute(
            select(Trajectory.rm_score, Trajectory.outlier_tier, Trajectory.outcome_measured_at)
            .where(Trajectory.outlier_tier.is_not(None), Trajectory.rm_score.is_not(None))
            .order_by(Trajectory.outcome_measured_at.desc())
            .limit(window * 2)
        )
    ).all()
    recent = rows[:window]
    ref = rows[window:]
    gg = gold_gap(
        np.asarray([r[0] for r in ref]),
        np.asarray([1.0 if (r[1] or 0) >= 2 else 0.0 for r in ref]),
        np.asarray([r[0] for r in recent]),
        np.asarray([1.0 if (r[1] or 0) >= 2 else 0.0 for r in recent]),
        threshold=cfg.reward_model.gold_gap_threshold,
    )
    return asdict(gg)


def positives_count(y: np.ndarray) -> int:
    return int(np.asarray(y).sum())


def uuid_list(ids: list[UUID]) -> list[str]:
    return [str(i) for i in ids]


async def open_ideas(session: AsyncSession) -> list[Trajectory]:
    """Ideas still in the review queue: format ok, not reviewed, no shipped render."""
    from outlier_ai.models.trajectories import Render

    shipped = select(Render.trajectory_id).where(Render.shipped_at.is_not(None))
    reviewed = select(Review.trajectory_id)
    stmt = (
        select(Trajectory)
        .where(
            Trajectory.format_ok.is_(True),
            Trajectory.id.not_in(shipped),
            Trajectory.id.not_in(reviewed),
        )
        .order_by(Trajectory.created_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def rescore_open_ideas(
    session: AsyncSession, *, cfg: AppConfig, reward_model, limit: int | None = None
) -> int:
    """Re-score queued ideas with the given reward model so the review queue order follows it."""
    from outlier_ai.archive.combinations import load_cards_by_id

    trajs = await open_ideas(session)
    if limit:
        trajs = trajs[:limit]
    if not trajs:
        return 0
    feats = await _features_for(session, trajs, await load_cards_by_id(session))
    for t, f in zip(trajs, feats, strict=True):
        score = await reward_model.score(f)
        t.rm_score = score.rm_score(cfg.reward_model.lambda_pess)
        t.rm_version = score.version
    await session.flush()
    return len(trajs)
