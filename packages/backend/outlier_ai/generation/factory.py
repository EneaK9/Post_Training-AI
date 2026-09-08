"""Assemble a GenerationService from config and settings."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.embeddings import get_embedder
from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.core.storage import get_storage
from outlier_ai.generation.backends import get_generator, get_judge
from outlier_ai.generation.service import GenerationService
from outlier_ai.generation.verifier import (
    HeuristicVerifier,
    LLMVerifier,
    load_card_refs,
    load_few_shot,
)
from outlier_ai.images import get_image_backend
from outlier_ai.reward.cold import HeuristicColdRewardModel, LLMColdRewardModel
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BackendKind, VerifierKind


async def build_service(
    session: AsyncSession,
    cfg: AppConfig,
    *,
    backend_kind: BackendKind,
    settings: Settings | None = None,
    use_llm_judges: bool | None = None,
    seed: int = 0,
    adapter: str | None = None,
) -> GenerationService:
    s = settings or get_settings()
    backend = get_generator(backend_kind, cfg, s, seed=seed, adapter=adapter)
    judge = get_judge(cfg, s) if (use_llm_judges is None or use_llm_judges) else None
    cards = await load_card_refs(session)
    if judge is not None and cfg.verifier.kind == VerifierKind.llm:
        few_shot = await load_few_shot(session, cfg.verifier.few_shot_n, cards)
        verifier = LLMVerifier(judge, cards, few_shot)
        reward_model = LLMColdRewardModel(judge)
    else:
        verifier = HeuristicVerifier(cards)
        reward_model = HeuristicColdRewardModel()
    return GenerationService(
        backend=backend,
        verifier=verifier,
        reward_model=reward_model,
        image_backend=get_image_backend(cfg.images.mode),
        storage=get_storage(s),
        embedder=get_embedder(
            s, dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
        ),
        cfg=cfg,
        judge=judge,
    )
