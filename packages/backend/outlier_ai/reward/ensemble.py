"""Pessimistic three-head ensemble (spec section 6).

Each head is a small MLP trained with focal loss on frozen features; heads differ by seed and
by bootstrap resample. `rm_score = mean - lambda_pess * std` across heads, so ideas the heads
disagree on are scored down (Coste et al. 2024). Pure numpy: no GPU, no torch, trains in seconds
on thousands of rows, and the artifact is a single .npz file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HeadWeights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _forward(h: HeadWeights, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z1 = x @ h.w1 + h.b1
    a1 = np.maximum(z1, 0.0)
    logit = a1 @ h.w2 + h.b2
    return _sigmoid(logit).ravel(), a1


def focal_loss_grad(
    p: np.ndarray, y: np.ndarray, gamma: float, alpha: float = 0.5
) -> tuple[float, np.ndarray]:
    """Focal loss and d(loss)/d(logit) averaged over the batch. Down-weights easy negatives so
    the rare positives (tier 2+) shape the model."""
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    pt = np.where(y == 1, p, 1 - p)
    a_t = np.where(y == 1, alpha, 1 - alpha)
    loss = -a_t * (1 - pt) ** gamma * np.log(pt)
    # d/dlogit of focal loss (standard derivation)
    dl_dp = a_t * ((1 - pt) ** gamma) * (gamma * pt * np.log(pt) / (1 - pt + eps) - 1) / pt
    sign = np.where(y == 1, 1.0, -1.0)
    d_logit = dl_dp * sign * p * (1 - p)
    return float(loss.mean()), d_logit / len(y)


class Ensemble:
    def __init__(
        self, heads: list[HeadWeights], *, mean: np.ndarray, std: np.ndarray, meta: dict[str, Any]
    ) -> None:
        self.heads = heads
        self.mean = mean
        self.std = std
        self.meta = meta

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)

    def predict_heads(self, x: np.ndarray) -> np.ndarray:
        xn = self._norm(np.asarray(x, dtype=np.float32))
        return np.stack([_forward(h, xn)[0] for h in self.heads])  # (heads, n)

    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ph = self.predict_heads(x)
        return ph.mean(axis=0), ph.std(axis=0)

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        arrays: dict[str, np.ndarray] = {"mean": self.mean, "std": self.std}
        for i, h in enumerate(self.heads):
            arrays[f"h{i}_w1"], arrays[f"h{i}_b1"], arrays[f"h{i}_w2"], arrays[f"h{i}_b2"] = (
                h.w1,
                h.b1,
                h.w2,
                h.b2,
            )
        arrays["meta"] = np.frombuffer(repr(self.meta).encode(), dtype=np.uint8)
        arrays["n_heads"] = np.asarray([len(self.heads)])
        np.savez_compressed(buf, **arrays)  # type: ignore[arg-type]
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> Ensemble:
        import ast

        z = np.load(io.BytesIO(data), allow_pickle=False)
        n = int(z["n_heads"][0])
        heads = [
            HeadWeights(z[f"h{i}_w1"], z[f"h{i}_b1"], z[f"h{i}_w2"], z[f"h{i}_b2"])
            for i in range(n)
        ]
        meta = ast.literal_eval(z["meta"].tobytes().decode()) if z["meta"].size else {}
        return cls(heads, mean=z["mean"], std=z["std"], meta=meta)


@dataclass
class TrainReport:
    n: int
    positives: int
    heads: int
    epochs: int
    final_loss: float
    train_auc: float | None
    history: list[float] = field(default_factory=list)


def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based AUC
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    *,
    heads: int = 3,
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 3e-3,
    gamma: float = 2.0,
    weight_decay: float = 1e-4,
    seed: int = 0,
    meta: dict[str, Any] | None = None,
) -> tuple[Ensemble, TrainReport]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n, d = x.shape
    mean, std = x.mean(axis=0), x.std(axis=0)
    xn = (x - mean) / (std + 1e-6)
    rng = np.random.default_rng(seed)
    trained: list[HeadWeights] = []
    history: list[float] = []
    for hidx in range(heads):
        idx = rng.integers(0, n, size=n)  # bootstrap resample per head
        xb, yb = xn[idx], y[idx]
        h = HeadWeights(
            w1=(rng.normal(0, 1, (d, hidden)) * np.sqrt(2.0 / d)).astype(np.float32),
            b1=np.zeros(hidden, dtype=np.float32),
            w2=(rng.normal(0, 1, (hidden, 1)) * np.sqrt(1.0 / hidden)).astype(np.float32),
            b2=np.zeros(1, dtype=np.float32),
        )
        # Adam
        m = {k: np.zeros_like(v) for k, v in vars(h).items()}
        v_ = {k: np.zeros_like(v) for k, v in vars(h).items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, epochs + 1):
            p, a1 = _forward(h, xb)
            loss, d_logit = focal_loss_grad(p, yb, gamma)
            g_w2 = a1.T @ d_logit[:, None] + weight_decay * h.w2
            g_b2 = np.asarray([d_logit.sum()])
            d_a1 = d_logit[:, None] @ h.w2.T
            d_z1 = d_a1 * (a1 > 0)
            g_w1 = xb.T @ d_z1 + weight_decay * h.w1
            g_b1 = d_z1.sum(axis=0)
            for k, g in (("w1", g_w1), ("b1", g_b1), ("w2", g_w2), ("b2", g_b2)):
                m[k] = b1 * m[k] + (1 - b1) * g
                v_[k] = b2 * v_[k] + (1 - b2) * g * g
                mhat = m[k] / (1 - b1**t)
                vhat = v_[k] / (1 - b2**t)
                setattr(
                    h, k, (getattr(h, k) - lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)
                )
            if hidx == 0 and t % max(1, epochs // 10) == 0:
                history.append(loss)
        trained.append(h)
    ens = Ensemble(trained, mean=mean, std=std, meta=meta or {})
    p_mean, _ = ens.score(x)
    final_loss, _ = focal_loss_grad(p_mean, y, gamma)
    return ens, TrainReport(
        n=n,
        positives=int(y.sum()),
        heads=heads,
        epochs=epochs,
        final_loss=final_loss,
        train_auc=_auc(p_mean, y),
        history=history,
    )
