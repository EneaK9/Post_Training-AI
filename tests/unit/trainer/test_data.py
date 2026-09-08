"""Trainer data access reads the Parquet snapshot the backend exports and builds the three
training views (RFT examples, DPO pairs, off-policy groups) without a database."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from outlier_trainer.advantages import RewardConfig
from outlier_trainer.data import (
    count_positives,
    dpo_pairs,
    load_snapshot,
    offpolicy_groups,
    rft_examples,
)
from outlier_trainer.run import main


def _write(tmp_path):
    rows = [
        {
            "trajectory_id": f"t{i}",
            "brief_id": "b1" if i < 4 else "b2",
            "combination_key": "a+b" if i % 2 else "c+d",
            "typicality": "rare" if i % 3 == 0 else "common",
            "tag_match": True,
            "format_ok": True,
            "shipped": True,
            "outlier_tier": [2, 0, 3, 0, 0, 1][i],
            "rm_score": 0.5,
            "prompt": "PROMPT" if i != 5 else None,
            "completion": f"<idea>{i}</idea>",
        }
        for i in range(6)
    ]
    path = tmp_path / "snap.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_views(tmp_path):
    path = _write(tmp_path)
    rows = load_snapshot(path)
    assert len(rows) == 6 and count_positives(rows) == 2
    assert len(rft_examples(rows)) == 2
    pairs = dpo_pairs(rows)
    # brief b1: winners t0, t2 vs losers t1, t3 -> 4 pairs; b2 has no winner
    assert len(pairs) == 4 and all(p[0] == "PROMPT" for p in pairs)
    groups = offpolicy_groups(rows, cfg=RewardConfig(), n_positives=2)
    assert [len(g) for g in groups] == [4]  # b2 has only one row with a prompt
    rewards = [r for _, r in groups[0]]
    assert max(rewards) > min(rewards)


def test_run_dry_writes_report(tmp_path):
    path = _write(tmp_path)
    out = tmp_path / "out"
    assert main(["--stage", "rft", "--snapshot", str(path), "--out", str(out), "--dry"]) == 0
    report = (out / "run.json").read_text()
    assert '"rows": 6' in report and '"tier2_positives": 2' in report


def test_onpolicy_is_gated(tmp_path):
    path = _write(tmp_path)
    out = tmp_path / "out2"
    assert main(["--stage", "grpo_onpolicy", "--snapshot", str(path), "--out", str(out)]) == 2
    assert "pass --simulator" in (out / "run.json").read_text()
