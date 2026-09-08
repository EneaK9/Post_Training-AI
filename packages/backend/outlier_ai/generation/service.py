"""Generation service: brief in, K parsed, verified, novelty-checked, scored ideas out.

Pipeline (spec sections 5.3 to 5.8): archive sample -> prompt -> backend -> parse ->
verifier -> novelty -> cold reward model -> pre-ship -> persist trajectories -> renders.
Novelty rejects and malformed ideas are stored with a `skip` review and never reach the
queue. Everything the model saw is kept on the batch as the prompt trace.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.combinations import load_cards_by_id, niche_key, upsert_combination
from outlier_ai.archive.novelty import NoveltyDecision, check_novelty, embed_combo_angle
from outlier_ai.core.embeddings import Embedder
from outlier_ai.core.errors import NotFoundError, ValidationError
from outlier_ai.core.storage import Storage, key_from_uri
from outlier_ai.generation.assemble import build_prompt
from outlier_ai.generation.backends.base import GeneratorBackend, JudgeBackend, RawOutput
from outlier_ai.generation.parser import ParsedIdea, parse_ideas
from outlier_ai.generation.preship import preship_check
from outlier_ai.generation.renderer import PromptTrace
from outlier_ai.generation.verifier import CardVerifier, VerifierResult, tag_agreement
from outlier_ai.images.base import ImageBackend, ImageSpec
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import CardVersion
from outlier_ai.models.episodes import Batch, SearchEpisode
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Render, Review, Trajectory
from outlier_ai.reward.base import IdeaFeatures, RewardModel, RewardScore
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import AuthorKind, EpisodeStatus, RenderStatus, ReviewLabel
from outlier_schemas.models import GenerationRequest, PreshipReport


@dataclass
class GeneratedIdea:
    trajectory: Trajectory
    parsed: ParsedIdea
    verifier: VerifierResult
    novelty: NoveltyDecision | None
    reward: RewardScore | None
    preship: PreshipReport | None
    renders: list[Render] = field(default_factory=list)
    status: str = "queued"  # queued | novelty_rejected | format_rejected


@dataclass
class GenerationResult:
    brief_id: UUID
    episode_id: UUID | None
    batch_id: UUID | None
    prompt: str
    trace: PromptTrace
    raw: RawOutput
    ideas: list[GeneratedIdea]

    @property
    def queued(self) -> list[GeneratedIdea]:
        return [i for i in self.ideas if i.status == "queued"]

    @property
    def rejected(self) -> list[GeneratedIdea]:
        return [i for i in self.ideas if i.status != "queued"]


class GenerationService:
    def __init__(
        self,
        *,
        backend: GeneratorBackend,
        verifier: CardVerifier,
        reward_model: RewardModel,
        image_backend: ImageBackend,
        storage: Storage,
        embedder: Embedder,
        cfg: AppConfig,
        judge: JudgeBackend | None = None,
    ) -> None:
        self.backend = backend
        self.verifier = verifier
        self.reward_model = reward_model
        self.image_backend = image_backend
        self.storage = storage
        self.embedder = embedder
        self.cfg = cfg
        self.judge = judge

    async def _library_version(self, session: AsyncSession) -> int:
        return int((await session.execute(select(func.count(CardVersion.id)))).scalar_one())

    async def _next_attempt_index(self, session: AsyncSession, brief_id: UUID) -> int:
        return int(
            (
                await session.execute(
                    select(func.count(Trajectory.id)).where(Trajectory.brief_id == brief_id)
                )
            ).scalar_one()
        )

    async def _open_batch(
        self, session: AsyncSession, episode_id: UUID, trace: PromptTrace, prompt: str = ""
    ) -> tuple[SearchEpisode, Batch]:
        episode = await session.get(SearchEpisode, episode_id)
        if episode is None:
            raise NotFoundError(f"episode {episode_id} not found")
        if episode.status != EpisodeStatus.searching.value:
            raise ValidationError(
                f"episode is {episode.status}; only searching episodes accept batches"
            )
        n = int(
            (
                await session.execute(
                    select(func.count(Batch.id)).where(Batch.episode_id == episode_id)
                )
            ).scalar_one()
        )
        if n >= self.cfg.episode.max_batches:
            raise ValidationError(
                f"episode already has {n} batches (max {self.cfg.episode.max_batches})"
            )
        batch = Batch(
            id=uuid.uuid4(),
            episode_id=episode_id,
            index=n,
            state="proposed",
            prompt_trace={**trace.as_dict(), "prompt": prompt},
        )
        session.add(batch)
        await session.flush()
        return episode, batch

    async def _cited_signal_texts(self, session: AsyncSession, ids: list[UUID]) -> list[str]:
        if not ids:
            return []
        rows = await session.execute(
            select(Signal.text).where(Signal.id.in_(ids), Signal.status == "confirmed")
        )
        return [t for (t,) in rows.all()]

    def _brand_asset_bytes(self, brief: Brief) -> bytes | None:
        for uri in brief.brand_assets or []:
            try:
                key = key_from_uri(uri)
                if self.storage.exists(key):
                    return self.storage.get(key)
            except ValueError:
                continue
        return None

    async def generate_batch(
        self, session: AsyncSession, req: GenerationRequest, *, now: datetime | None = None
    ) -> GenerationResult:
        now = now or datetime.now(UTC)
        cfg = self.cfg
        brief = await session.get(Brief, req.brief_id)
        if brief is None:
            raise NotFoundError(f"brief {req.brief_id} not found")

        prompt, trace, _sample, slug_map = await build_prompt(
            session, req.brief_id, cfg, episode_id=req.episode_id, k=req.k, now=now
        )
        episode: SearchEpisode | None = None
        batch: Batch | None = None
        if req.episode_id is not None:
            episode, batch = await self._open_batch(session, req.episode_id, trace, prompt)

        raw = (await self.backend.generate(prompt, n=1))[0]
        ref_map = {r: UUID(u) for r, u in trace.ref_map.items()}
        parsed = parse_ideas(
            raw.text,
            cards_by_slug=slug_map,
            ref_map=ref_map,
            min_cards=cfg.generation.min_cards_per_idea,
            expected_k=req.k,
        )
        if raw.refused:
            for p in parsed:
                p.errors.append("backend refused")

        cards_by_id = await load_cards_by_id(session)
        library_version = await self._library_version(session)
        attempt_base = await self._next_attempt_index(session, brief.id)
        brand_asset = self._brand_asset_bytes(brief)
        ideas: list[GeneratedIdea] = []

        for i, p in enumerate(parsed):
            verified = await self.verifier.tag(p.angle, p.copy, p.visual_brief)
            match, jaccard = tag_agreement(p.card_ids, verified.card_ids)
            embed_slugs = verified.card_slugs or p.card_slugs
            embedding = embed_combo_angle(self.embedder, embed_slugs, p.angle)
            novelty: NoveltyDecision | None = None
            if p.format_ok:
                novelty = await check_novelty(
                    session,
                    brief_id=brief.id,
                    verified_card_ids=verified.card_ids or p.card_ids,
                    embedding=embedding,
                    cfg=cfg,
                    as_of=now,
                )
            signal_texts = await self._cited_signal_texts(session, p.citations.signals)
            features = IdeaFeatures(
                brief_text=brief.raw_text,
                world_state=brief.world_state,
                angle=p.angle,
                copy=p.copy,
                visual_brief=p.visual_brief,
                verified_card_slugs=verified.card_slugs,
                typicality=p.typicality.value if p.typicality else None,
                novelty_distance=novelty.distance if novelty else None,
                cited_signal_count=len(p.citations.signals),
                format_ok=p.format_ok,
                tag_match=match if p.format_ok else None,
                cited_signal_texts=signal_texts,
            )
            reward = await self.reward_model.score(features)
            preship = (
                await preship_check(
                    copy=p.copy,
                    angle=p.angle,
                    visual_brief=p.visual_brief,
                    constraints=list(brief.constraints or []),
                    images_cfg=cfg.images,
                    judge=self.judge,
                )
                if p.format_ok
                else None
            )

            traj = Trajectory(
                id=uuid.uuid4(),
                episode_id=episode.id if episode else None,
                batch_id=batch.id if batch else None,
                brief_id=brief.id,
                campaign_id=episode.campaign_id if episode else None,
                attempt_index=attempt_base + i,
                author_id=self.backend.name,
                author_kind=AuthorKind.model.value,
                backend=self.backend.kind.value,
                card_ids=p.card_ids,
                verified_card_ids=verified.card_ids,
                tag_source=verified.source.value,
                tag_match=match,
                tag_jaccard=jaccard,
                typicality=p.typicality.value if p.typicality else None,
                format_ok=p.format_ok,
                format_errors=list(p.errors),
                reasoning=p.reasoning,
                cited_ids=p.citations.ids,
                angle=p.angle,
                ad_copy=p.copy.model_dump(),
                visual_brief=p.visual_brief,
                preship=preship.model_dump() if preship else None,
                rm_score=reward.rm_score(cfg.reward_model.lambda_pess),
                rm_version=reward.version,
                library_version=library_version,
                config_hash=cfg.hash,
                combo_angle_embedding=embedding,
                created_at=now,
            )
            session.add(traj)
            idea = GeneratedIdea(traj, p, verified, novelty, reward, preship)

            if not p.format_ok:
                idea.status = "format_rejected"
                session.add(
                    Review(
                        trajectory_id=traj.id,
                        label=ReviewLabel.skip.value,
                        reviewer_id="system",
                        note="format_error: " + "; ".join(p.errors)[:500],
                        reviewed_at=now,
                    )
                )
            elif novelty is not None and novelty.rejected:
                idea.status = "novelty_rejected"
                session.add(
                    Review(
                        trajectory_id=traj.id,
                        label=ReviewLabel.skip.value,
                        reviewer_id="system",
                        note=novelty.note,
                        reviewed_at=now,
                    )
                )
            else:
                for seed in range(req.renders_per_idea):
                    size = (
                        req.size
                        if seed < 2
                        else (cfg.images.sizes[-1] if cfg.images.sizes else req.size)
                    )
                    rendered = self.image_backend.render(
                        ImageSpec(
                            headline=p.copy.headline,
                            primary_text=p.copy.primary_text,
                            brand_name=brief.company,
                            visual_brief=p.visual_brief,
                            size=(int(size[0]), int(size[1])),
                            seed=seed,
                            brand_asset=brand_asset,
                        ),
                        self.storage,
                        key_prefix=f"renders/{brief.id}/{traj.id}",
                    )
                    render = Render(
                        id=uuid.uuid4(),
                        trajectory_id=traj.id,
                        image_uri=rendered.uri,
                        image_backend=rendered.backend,
                        seed=seed,
                        width=rendered.width,
                        height=rendered.height,
                        status=RenderStatus.draft.value,
                    )
                    session.add(render)
                    idea.renders.append(render)

            combo_ids = verified.card_ids or p.card_ids
            if combo_ids:
                await upsert_combination(
                    session,
                    combo_ids,
                    niche_key(combo_ids, cards_by_id, cfg.archive.niche_projection),
                    now,
                )
            ideas.append(idea)

        await session.flush()
        return GenerationResult(
            brief_id=brief.id,
            episode_id=episode.id if episode else None,
            batch_id=batch.id if batch else None,
            prompt=prompt,
            trace=trace,
            raw=raw,
            ideas=ideas,
        )
