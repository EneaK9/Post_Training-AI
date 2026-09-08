"""Feature builder for the reward-model ensemble.

Frozen text embeddings of the brief, the idea (angle, copy, visual brief), the cited confirmed
signals, and the world state, concatenated with a hashed indicator of the verified combination
and a few scalar cues. Everything is deterministic given the embedder, so a saved model can be
re-applied to new ideas.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from outlier_ai.core.embeddings import Embedder
from outlier_ai.reward.base import IdeaFeatures

COMBO_BUCKETS = 64
SCALARS = 8


def combo_indicator(slugs: Sequence[str], buckets: int = COMBO_BUCKETS) -> np.ndarray:
    """Hashed multi-hot over card slugs plus a bucket for the whole combination."""
    v = np.zeros(buckets, dtype=np.float32)
    for s in slugs:
        v[int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big") % buckets] = (
            1.0
        )
    if slugs:
        key = "+".join(sorted(slugs))
        v[
            int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "big") % buckets
        ] += 0.5
    return v


def scalar_cues(f: IdeaFeatures) -> np.ndarray:
    return np.asarray(
        [
            1.0 if f.typicality == "rare" else 0.0,
            1.0 if f.typicality == "uncommon" else 0.0,
            min(f.cited_signal_count, 5) / 5.0,
            (f.novelty_distance if f.novelty_distance is not None else 0.5),
            1.0 if f.format_ok else 0.0,
            1.0 if f.tag_match else 0.0 if f.tag_match is False else 0.5,
            min(len(f.verified_card_slugs), 6) / 6.0,
            min(len(f.copy.primary_text), 500) / 500.0,
        ],
        dtype=np.float32,
    )


class FeatureBuilder:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.dims = embedder.dims * 3 + COMBO_BUCKETS + SCALARS

    def texts(self, f: IdeaFeatures) -> list[str]:
        idea = "\n".join(
            [f.angle, f.copy.headline, f.copy.primary_text, f.copy.description, f.visual_brief]
        )
        context = "\n".join([f.brief_text, f"world: {f.world_state}"])
        signals = "\n".join(f.cited_signal_texts) or "no signals cited"
        return [idea, context, signals]

    def build(self, feats: Sequence[IdeaFeatures]) -> np.ndarray:
        if not feats:
            return np.zeros((0, self.dims), dtype=np.float32)
        flat = [t for f in feats for t in self.texts(f)]
        emb = self.embedder.embed(flat).reshape(len(feats), -1)
        combos = np.stack([combo_indicator(f.verified_card_slugs) for f in feats])
        scalars = np.stack([scalar_cues(f) for f in feats])
        return np.concatenate([emb, combos, scalars], axis=1).astype(np.float32)
