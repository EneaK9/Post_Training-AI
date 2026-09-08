"""Feedback loops (spec build step 15): suggested relations, verifier retraining, RM retraining."""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.audit import record_audit
from outlier_ai.core.embeddings import Embedder
from outlier_ai.core.storage import Storage, key_from_uri
from outlier_ai.models.cards import Card, CardRelation
from outlier_ai.models.ml import VerifierVersion
from outlier_ai.models.ops import HoldoutCampaign
from outlier_ai.models.trajectories import Review, Trajectory
from outlier_schemas.config import AppConfig


@dataclass
class RelationSuggestion:
    a: UUID
    b: UUID
    observed: int
    expected: float
    lift: float


async def suggest_relations(
    session: AsyncSession, *, min_observed: int = 2, min_lift: float = 1.5, actor: str = "feedback"
) -> list[RelationSuggestion]:
    """Card pairs that co-occur in tier 2+ combinations above chance become pending `suggested`
    relations for an expert to accept or reject (scenario row 4)."""
    rows = (
        await session.execute(
            select(Trajectory.verified_card_ids).where(Trajectory.outlier_tier >= 2)
        )
    ).all()
    combos = [tuple(sorted(set(ids))) for (ids,) in rows if ids]
    n = len(combos)
    if n < 2:
        return []
    card_freq: Counter = Counter(c for combo in combos for c in combo)
    pair_freq: Counter = Counter(pair for combo in combos for pair in combinations(combo, 2))
    existing = {
        tuple(sorted((r.from_id, r.to_id)))
        for r in (await session.execute(select(CardRelation))).scalars().all()
    }
    out: list[RelationSuggestion] = []
    for (a, b), observed in pair_freq.items():
        expected = card_freq[a] * card_freq[b] / n
        lift = observed / expected if expected > 0 else 0.0
        if observed >= min_observed and lift >= min_lift and (a, b) not in existing:
            session.add(
                CardRelation(
                    from_id=a,
                    to_id=b,
                    kind="complements",
                    source="suggested",
                    status="pending",
                    created_by=actor,
                    evidence={
                        "observed": observed,
                        "expected": expected,
                        "lift": lift,
                        "tier2_ideas": n,
                    },
                )
            )
            out.append(RelationSuggestion(a, b, observed, expected, lift))
    if out:
        await session.flush()
        await record_audit(
            session,
            actor_id=actor,
            action="relations.suggested",
            object_type="card_relation",
            object_id="batch",
            after={"count": len(out)},
        )
    return out


# ---- classifier verifier (v2) -----------------------------------------------------------


class ClassifierVerifierModel:
    """Nearest-centroid multi-label tagger over hashed text features. Small, deterministic,
    and trained only on expert-confirmed tags; good enough to replace the LLM judge once labels
    are plentiful, cheap enough to run on every idea."""

    def __init__(
        self, slugs: list[str], centroids: np.ndarray, thresholds: np.ndarray, embedder_name: str
    ) -> None:
        self.slugs = slugs
        self.centroids = centroids
        self.thresholds = thresholds
        self.embedder_name = embedder_name

    def predict(self, x: np.ndarray) -> list[list[str]]:
        sims = x @ self.centroids.T  # embeddings are L2-normalized
        out: list[list[str]] = []
        for row in sims:
            picked = [self.slugs[j] for j in np.argsort(-row) if row[j] >= self.thresholds[j]]
            out.append(picked[:4])
        return out

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            centroids=self.centroids,
            thresholds=self.thresholds,
            slugs=np.array(self.slugs),
            embedder=np.array([self.embedder_name]),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> ClassifierVerifierModel:
        z = np.load(io.BytesIO(data), allow_pickle=False)
        return cls(
            [str(s) for s in z["slugs"]], z["centroids"], z["thresholds"], str(z["embedder"][0])
        )


def train_classifier(
    texts: list[str], labels: list[list[str]], embedder: Embedder
) -> ClassifierVerifierModel:
    x = embedder.embed(texts)
    slugs = sorted({s for ls in labels for s in ls})
    centroids = np.zeros((len(slugs), x.shape[1]), dtype=np.float32)
    thresholds = np.zeros(len(slugs), dtype=np.float32)
    for j, slug in enumerate(slugs):
        pos = np.array([i for i, ls in enumerate(labels) if slug in ls])
        c = x[pos].mean(axis=0)
        c /= np.linalg.norm(c) + 1e-9
        centroids[j] = c
        sims_pos = x[pos] @ c
        neg = np.array([i for i, ls in enumerate(labels) if slug not in ls])
        sims_neg = x[neg] @ c if len(neg) else np.array([-1.0])
        # threshold halfway between typical positive and typical negative similarity
        thresholds[j] = float((np.median(sims_pos) + np.quantile(sims_neg, 0.9)) / 2)
    return ClassifierVerifierModel(slugs, centroids, thresholds, embedder.name)


async def expert_tagged(session: AsyncSession) -> tuple[list[str], list[list[str]]]:
    """Expert labels: `wrong_cards` corrections and expert-tagged imports, holdout excluded."""
    holdout = set((await session.execute(select(HoldoutCampaign.campaign_id))).scalars().all())
    cards = {c.id: c.slug for c in (await session.execute(select(Card))).scalars().all()}
    texts: list[str] = []
    labels: list[list[str]] = []
    rows = (
        await session.execute(
            select(Trajectory, Review)
            .join(Review, Review.trajectory_id == Trajectory.id, isouter=True)
            .where(Trajectory.format_ok.is_(True))
        )
    ).all()
    for t, r in rows:
        if t.campaign_id in holdout:
            continue
        ids: list[UUID] | None = None
        if r is not None and r.label == "wrong_cards" and r.corrected_card_ids:
            ids = list(r.corrected_card_ids)
        elif t.tag_source == "expert" and t.verified_card_ids:
            ids = list(t.verified_card_ids)
        if not ids:
            continue
        copy = t.ad_copy or {}
        texts.append(
            "\n".join(
                [
                    t.angle,
                    str(copy.get("primary_text", "")),
                    str(copy.get("headline", "")),
                    t.visual_brief,
                ]
            )
        )
        labels.append([cards[i] for i in ids if i in cards])
    return texts, labels


async def retrain_verifier(
    session: AsyncSession,
    *,
    cfg: AppConfig,
    embedder: Embedder,
    storage: Storage,
    min_labels: int | None = None,
    actor: str = "feedback",
) -> VerifierVersion | None:
    texts, labels = await expert_tagged(session)
    need = cfg.verifier.classifier_min_labels if min_labels is None else min_labels
    if len(texts) < need or len({s for ls in labels for s in ls}) < 2:
        return None
    model = train_classifier(texts, labels, embedder)
    # training-set agreement as the panel metric
    preds = model.predict(embedder.embed(texts))
    exact = sum(1 for p, ls in zip(preds, labels, strict=True) if set(p) == set(ls)) / len(texts)
    jacc = float(
        np.mean(
            [
                len(set(p) & set(ls)) / max(1, len(set(p) | set(ls)))
                for p, ls in zip(preds, labels, strict=True)
            ]
        )
    )
    version = f"verifier-classifier-v2-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    uri = storage.put(
        f"models/verifier/{version}.npz", model.to_bytes(), "application/octet-stream"
    )
    for row in (
        (await session.execute(select(VerifierVersion).where(VerifierVersion.is_active.is_(True))))
        .scalars()
        .all()
    ):
        row.is_active = False
    vv = VerifierVersion(
        version=version,
        kind="classifier",
        artifact_uri=uri,
        metrics={"train_exact": exact, "train_jaccard": jacc, "cards": len(model.slugs)},
        trained_on_labels=len(texts),
        is_active=True,
    )
    session.add(vv)
    await session.flush()
    await record_audit(
        session,
        actor_id=actor,
        action="verifier.retrained",
        object_type="verifier",
        object_id=version,
        after=vv.metrics,
    )
    return vv


async def load_active_classifier(
    session: AsyncSession, storage: Storage
) -> ClassifierVerifierModel | None:
    row = (
        (
            await session.execute(
                select(VerifierVersion)
                .where(VerifierVersion.is_active.is_(True), VerifierVersion.kind == "classifier")
                .order_by(VerifierVersion.created_at.desc())
            )
        )
        .scalars()
        .first()
    )
    if row is None or not row.artifact_uri:
        return None
    return ClassifierVerifierModel.from_bytes(storage.get(key_from_uri(row.artifact_uri)))
