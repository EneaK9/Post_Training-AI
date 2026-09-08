"""Text embeddings for brief similarity and novelty rejection.

`HashEmbedder` is deterministic, dependency-free feature hashing over word unigrams and
bigrams. It is the default for tests and development. `SentenceTransformerEmbedder` wraps
`sentence-transformers` (install the `ml` extra) for production quality. Both produce
L2-normalized vectors so cosine distance is `1 - dot`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np

from outlier_ai.core.settings import Settings, get_settings

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dims: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    name = "hash-v1"

    def __init__(self, dims: int = 384) -> None:
        self.dims = dims

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _TOKEN.findall(text.lower())
            grams = toks + [f"{a}_{b}" for a, b in pairwise(toks)]
            for g in grams:
                h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
                idx = h % self.dims
                sign = 1.0 if (h >> 62) & 1 else -1.0
                out[i, idx] += sign
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dims = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(
    settings: Settings | None = None, *, dims: int = 384, model_name: str | None = None
) -> Embedder:
    s = settings or get_settings()
    if s.embedder == "sentence_transformers" and model_name:
        return SentenceTransformerEmbedder(model_name)
    return HashEmbedder(dims)


def cosine_distance(a: np.ndarray | Sequence[float], b: np.ndarray | Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))
