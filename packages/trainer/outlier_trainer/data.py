"""Archive snapshot access for the trainer. Reads the Parquet file the backend exported; never
touches the database."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from outlier_trainer.advantages import RewardConfig, compose_reward


@dataclass(frozen=True)
class Row:
    trajectory_id: str
    brief_id: str
    combination_key: str
    typicality: str | None
    tag_match: bool | None
    format_ok: bool
    shipped: bool
    outlier_tier: int | None
    rm_score: float | None
    prompt: str | None
    completion: str


def load_snapshot(path: str | Path) -> list[Row]:
    table = pq.read_table(path)
    rows: list[Row] = []
    for r in table.to_pylist():
        if not r.get("trajectory_id"):
            continue
        rows.append(
            Row(
                trajectory_id=r["trajectory_id"],
                brief_id=r["brief_id"],
                combination_key=r.get("combination_key") or "",
                typicality=r.get("typicality"),
                tag_match=r.get("tag_match"),
                format_ok=bool(r.get("format_ok", True)),
                shipped=bool(r.get("shipped", False)),
                outlier_tier=r.get("outlier_tier"),
                rm_score=r.get("rm_score"),
                prompt=r.get("prompt"),
                completion=r.get("completion") or "",
            )
        )
    return rows


def with_prompts(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.prompt]


def rft_examples(rows: list[Row]) -> list[tuple[str, str]]:
    """Prompt -> completion pairs for tier 2+ ideas (rejection-sampling fine-tuning)."""
    return [
        (r.prompt or "", r.completion) for r in with_prompts(rows) if (r.outlier_tier or 0) >= 2
    ]


def dpo_pairs(rows: list[Row]) -> list[tuple[str, str, str]]:
    """(prompt, chosen, rejected): tier 2+ vs tier 0 on the same brief."""
    by_brief: dict[str, list[Row]] = defaultdict(list)
    for r in with_prompts(rows):
        if r.outlier_tier is not None:
            by_brief[r.brief_id].append(r)
    pairs: list[tuple[str, str, str]] = []
    for group in by_brief.values():
        winners = [r for r in group if (r.outlier_tier or 0) >= 2]
        losers = [r for r in group if r.outlier_tier == 0]
        for w in winners:
            for loser in losers:
                pairs.append((w.prompt or "", w.completion, loser.completion))
    return pairs


def offpolicy_groups(
    rows: list[Row], *, cfg: RewardConfig, n_positives: int, min_size: int = 2
) -> list[list[tuple[Row, float]]]:
    """Per-brief groups of (row, reward) for off-policy GRPO. Rewards follow section 7.2:
    real tier when measured, RM shaping when not, minus tag and format penalties."""
    by_brief: dict[str, list[tuple[Row, float]]] = defaultdict(list)
    for r in with_prompts(rows):
        reward = compose_reward(
            outlier_tier=r.outlier_tier,
            rm_score=r.rm_score,
            tag_match=r.tag_match,
            format_ok=r.format_ok,
            n_positives=n_positives,
            cfg=cfg,
        )
        by_brief[r.brief_id].append((r, reward))
    return [g for g in by_brief.values() if len(g) >= min_size]


def count_positives(rows: list[Row]) -> int:
    return sum(1 for r in rows if (r.outlier_tier or 0) >= 2)
