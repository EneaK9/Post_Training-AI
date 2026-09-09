"""Assemble a GenerationService from config and settings."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.embeddings import get_embedder
from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.core.storage import Storage, get_storage
from outlier_ai.generation.backends import get_generator, get_judge
from outlier_ai.generation.service import GenerationService
from outlier_ai.generation.verifier import (
    CardVerifier,
    ClassifierVerifier,
    HeuristicVerifier,
    LLMVerifier,
    load_card_refs,
    load_few_shot,
)
from outlier_ai.images import get_image_backend
from outlier_ai.jobs.feedback import load_active_classifier
from outlier_ai.models.ml import VerifierVersion
from outlier_ai.reward.base import RewardModel
from outlier_ai.reward.cold import HeuristicColdRewardModel, LLMColdRewardModel
from outlier_ai.reward.train import load_active_reward_model
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
    fake_policy: str = "archive",
    storage: Storage | None = None,
) -> GenerationService:
    s = settings or get_settings()
    backend = get_generator(
        backend_kind, cfg, s, seed=seed, adapter=adapter, fake_policy=fake_policy
    )
    judge = get_judge(cfg, s) if (use_llm_judges is None or use_llm_judges) else None
    cards = await load_card_refs(session)
    storage = storage or get_storage(s)
    embedder = get_embedder(
        s, dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
    )
    verifier: CardVerifier
    classifier = await load_active_classifier(session, storage)
    if classifier is not None and classifier.embedder_name == embedder.name:
        active = (
            (
                await session.execute(
                    select(VerifierVersion).where(VerifierVersion.is_active.is_(True))
                )
            )
            .scalars()
            .first()
        )
        verifier = ClassifierVerifier(
            classifier, embedder, cards, active.version if active else "verifier-classifier-v2"
        )
    elif judge is not None and cfg.verifier.kind == VerifierKind.llm:
        few_shot = await load_few_shot(session, cfg.verifier.few_shot_n, cards)
        verifier = LLMVerifier(judge, cards, few_shot)
    else:
        verifier = HeuristicVerifier(cards)
    reward_model: RewardModel
    trained = await load_active_reward_model(session, storage, embedder, cfg)
    if trained is not None:
        reward_model = trained
    elif judge is not None:
        reward_model = LLMColdRewardModel(judge)
    else:
        reward_model = HeuristicColdRewardModel()
    return GenerationService(
        backend=backend,
        verifier=verifier,
        reward_model=reward_model,
        image_backend=get_image_backend(cfg.images.mode),
        storage=storage,
        embedder=get_embedder(
            s, dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
        ),
        cfg=cfg,
        judge=judge,
    )
