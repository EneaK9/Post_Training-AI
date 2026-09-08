"""Archive snapshot for Loop B: one Parquet file, one hash, holdout excluded.

Each row is a trajectory with its brief, cards, verified cards, prompt (from the batch trace when
the idea came out of Loop A, otherwise re-rendered as of its creation time), the idea block in
the section 5.4 grammar, outcome tier, rm_score, tag match, and typicality. The trainer never
touches the database; it reads this file.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.combinations import load_cards_by_id
from outlier_ai.core.storage import Storage
from outlier_ai.generation.assemble import build_prompt
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import Batch
from outlier_ai.models.ml import ArchiveSnapshot
from outlier_ai.models.ops import HoldoutCampaign
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_schemas.config import AppConfig


@dataclass
class SnapshotResult:
    hash: str
    uri: str
    n_trajectories: int
    n_tier2: int
    n_with_prompt: int


def idea_block(t: Trajectory, slugs: list[str]) -> str:
    copy = t.ad_copy or {}
    return (
        "<idea>\n"
        f"<cards>{', '.join(slugs)}</cards>\n"
        f"<typicality>{t.typicality or 'common'}</typicality>\n"
        f"<reasoning>{t.reasoning}</reasoning>\n"
        f"<angle>{t.angle}</angle>\n"
        f"<copy>primary_text: {copy.get('primary_text', '')}\n"
        f"headline: {copy.get('headline', '')}\n"
        f"description: {copy.get('description', '')}\ncta: {copy.get('cta', '')}</copy>\n"
        f"<visual_brief>{t.visual_brief}</visual_brief>\n"
        "</idea>"
    )


async def export_snapshot(
    session: AsyncSession,
    *,
    cfg: AppConfig,
    storage: Storage,
    include_unmeasured: bool = False,
    rerender_prompts: bool = True,
) -> SnapshotResult:
    holdout = set((await session.execute(select(HoldoutCampaign.campaign_id))).scalars().all())
    cards = await load_cards_by_id(session)
    stmt = select(Trajectory).order_by(Trajectory.created_at)
    if not include_unmeasured:
        stmt = stmt.where(Trajectory.outlier_tier.is_not(None))
    trajs = [
        t
        for t in (await session.execute(stmt)).scalars().all()
        if t.campaign_id not in holdout and t.verified_card_ids
    ]
    shipped = {
        tid
        for (tid,) in (
            await session.execute(
                select(Render.trajectory_id).where(Render.shipped_at.is_not(None)).distinct()
            )
        ).all()
    }
    briefs = {b.id: b for b in (await session.execute(select(Brief))).scalars().all()}
    batch_prompts: dict = {}
    for b in (await session.execute(select(Batch))).scalars().all():
        if b.prompt_trace and b.prompt_trace.get("prompt"):
            batch_prompts[b.id] = b.prompt_trace["prompt"]

    rows: list[dict] = []
    n_with_prompt = 0
    for t in trajs:
        slugs = [cards[i].slug for i in t.verified_card_ids if i in cards]
        prompt = batch_prompts.get(t.batch_id) if t.batch_id else None
        if prompt is None and rerender_prompts:
            try:
                prompt, _, _, _ = await build_prompt(session, t.brief_id, cfg, now=t.created_at)
            except Exception:  # a missing brief must not sink the export
                prompt = None
        if prompt:
            n_with_prompt += 1
        b = briefs.get(t.brief_id)
        rows.append(
            {
                "trajectory_id": str(t.id),
                "brief_id": str(t.brief_id),
                "episode_id": str(t.episode_id) if t.episode_id else None,
                "campaign_id": t.campaign_id,
                "brief_category": b.category if b else None,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "author_kind": t.author_kind,
                "backend": t.backend,
                "card_slugs": [cards[i].slug for i in t.card_ids if i in cards],
                "verified_card_slugs": slugs,
                "combination_key": "+".join(sorted(slugs)),
                "typicality": t.typicality,
                "tag_match": t.tag_match,
                "format_ok": t.format_ok,
                "shipped": t.id in shipped,
                "outlier_tier": t.outlier_tier,
                "rm_score": t.rm_score,
                "rm_version": t.rm_version,
                "prompt": prompt,
                "completion": idea_block(t, slugs),
                "config_hash": t.config_hash,
            }
        )
    table = (
        pa.Table.from_pylist(rows)
        if rows
        else pa.table({"trajectory_id": pa.array([], pa.string())})
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    data = buf.getvalue()
    h = hashlib.sha256(data).hexdigest()[:16]
    uri = storage.put(f"snapshots/archive-{h}.parquet", data, "application/octet-stream")
    n_tier2 = sum(1 for r in rows if (r["outlier_tier"] or 0) >= 2)
    existing = (
        await session.execute(select(ArchiveSnapshot).where(ArchiveSnapshot.hash == h))
    ).scalar_one_or_none()
    if existing is None:
        session.add(
            ArchiveSnapshot(
                hash=h, uri=uri, n_trajectories=len(rows), n_tier2=n_tier2, config_hash=cfg.hash
            )
        )
        await session.flush()
    return SnapshotResult(
        hash=h, uri=uri, n_trajectories=len(rows), n_tier2=n_tier2, n_with_prompt=n_with_prompt
    )


def snapshot_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")
