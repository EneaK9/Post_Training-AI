from __future__ import annotations

from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter

from outlier_ai.api.deps import DB, Config, CurrentUser
from outlier_ai.api.schemas import ArchiveSampleOut, PromptOut
from outlier_ai.archive.sampling import sample_archive
from outlier_ai.generation.assemble import build_prompt

router = APIRouter(prefix="/archive", tags=["archive"])


@router.get("/sample", response_model=ArchiveSampleOut)
async def archive_sample(brief_id: UUID, db: DB, cfg: Config, _: CurrentUser) -> ArchiveSampleOut:
    """Exactly what render() would include for this brief."""
    sample = await sample_archive(db, brief_id, cfg)
    items = []
    for i in sample.items:
        d = asdict(i)
        d["trajectory_id"] = str(i.trajectory_id)
        d["brief_id"] = str(i.brief_id)
        d["created_at"] = i.created_at.isoformat() if i.created_at else None
        d["signals"] = [{**asdict(s), "id": str(s.id)} for s in i.signals]
        items.append(d)
    return ArchiveSampleOut(
        brief_id=sample.brief_id,
        similar_brief_ids=sample.similar_brief_ids,
        excluded_holdout=sample.excluded_holdout,
        niche_uses=sample.niche_uses,
        items=items,
    )


@router.get("/prompt", response_model=PromptOut)
async def archive_prompt(
    brief_id: UUID,
    db: DB,
    cfg: Config,
    _: CurrentUser,
    episode_id: UUID | None = None,
    k: int | None = None,
) -> PromptOut:
    prompt, trace, _sample, _slugs = await build_prompt(
        db, brief_id, cfg, episode_id=episode_id, k=k
    )
    return PromptOut(prompt=prompt, trace=trace.as_dict())
