from __future__ import annotations

from fastapi import APIRouter, Depends

from outlier_ai.api.deps import DB, Config, require
from outlier_ai.api.schemas import GeneratedIdeaOut, GenerateIn, GenerateOut
from outlier_ai.api.views import trajectory_outs
from outlier_ai.core.audit import record_audit
from outlier_ai.generation.factory import build_service
from outlier_ai.models.auth import User
from outlier_schemas.models import GenerationRequest

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("", response_model=GenerateOut)
async def generate(
    body: GenerateIn, db: DB, cfg: Config, user: User = Depends(require("generate"))
) -> GenerateOut:
    service = await build_service(
        db, cfg, backend_kind=body.backend, use_llm_judges=not body.no_llm, seed=body.seed
    )
    req = GenerationRequest(
        brief_id=body.brief_id,
        episode_id=body.episode_id,
        backend=body.backend,
        k=body.k,
        renders_per_idea=body.renders_per_idea,
    )
    result = await service.generate_batch(db, req)
    outs = await trajectory_outs(db, [i.trajectory for i in result.ideas])
    by_id = {o.id: o for o in outs}
    await record_audit(
        db,
        actor_id=user.email,
        action="generate",
        object_type="brief",
        object_id=body.brief_id,
        after={
            "backend": body.backend.value,
            "k": body.k,
            "queued": len(result.queued),
            "rejected": len(result.rejected),
            "batch_id": str(result.batch_id) if result.batch_id else None,
        },
    )
    return GenerateOut(
        brief_id=result.brief_id,
        episode_id=result.episode_id,
        batch_id=result.batch_id,
        backend=result.raw.backend,
        model=result.raw.model,
        input_tokens=result.raw.input_tokens,
        output_tokens=result.raw.output_tokens,
        cost_usd=result.raw.cost_usd,
        refused=result.raw.refused,
        trace=result.trace.as_dict(),
        ideas=[
            GeneratedIdeaOut(
                status=i.status,
                trajectory=by_id[i.trajectory.id],
                novelty_reason=i.novelty.reason if i.novelty else None,
                novelty_distance=i.novelty.distance if i.novelty else None,
                verifier_version=i.verifier.version,
            )
            for i in result.ideas
        ],
    )
