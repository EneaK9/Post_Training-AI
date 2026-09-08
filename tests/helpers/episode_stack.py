"""Shared episode test stack: seeded DB, fake Meta client, generation service, ship helpers."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.embeddings import HashEmbedder
from outlier_ai.core.storage import LocalStorage
from outlier_ai.episodes.controller import tick_episode
from outlier_ai.generation.backends.fake_backend import FakeBackend
from outlier_ai.generation.service import GenerationService
from outlier_ai.generation.verifier import HeuristicVerifier, load_card_refs
from outlier_ai.images import get_image_backend
from outlier_ai.jobs.handlers import run_sync_insights
from outlier_ai.meta.factory import default_latent
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.meta.ship import ship_batch
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.meta import AdAccount
from outlier_ai.models.trajectories import Review
from outlier_ai.outlier.recompute import recompute_all
from outlier_ai.reward.cold import HeuristicColdRewardModel
from outlier_ai.synthetic.seed import FAKE_ACCOUNT_ID, seed
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BackendKind, ImageMode
from outlier_schemas.models import GenerationRequest

pytestmark = pytest.mark.integration

START = date(2026, 9, 1)


def dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 12, tzinfo=UTC)


class Stack:
    def __init__(
        self, session: AsyncSession, cfg: AppConfig, storage_dir, *, fake_seed=3, rejection_rate=0.0
    ):
        self.session = session
        self.cfg = cfg
        self.storage = LocalStorage(storage_dir)
        self.fake_seed = fake_seed
        self.rejection_rate = rejection_rate

    async def setup(self, n=60, s=4):
        await seed(
            self.session, n_briefs=3, n_trajectories=n, seed=s, days_back=90, config=self.cfg
        )
        await recompute_all(self.session, self.cfg)
        self.account = (
            await self.session.execute(
                select(AdAccount).where(AdAccount.meta_account_id == FAKE_ACCOUNT_ID)
            )
        ).scalar_one()
        brief = (
            (await self.session.execute(select(Brief).order_by(Brief.created_at))).scalars().first()
        )
        assert brief is not None
        self.brief = brief
        self.client = FakeMetaClient(
            self.session,
            FAKE_ACCOUNT_ID,
            default_latent(),
            seed=self.fake_seed,
            rejection_rate=self.rejection_rate,
            review_delay_days=1,
            start_day=START,
        )
        self.episode = SearchEpisode(
            brief_id=self.brief.id,
            ad_account_id=self.account.id,
            backend="fake",
            budget_cap=self.cfg.episode.default_budget_cap_usd,
            created_by="test",
        )
        self.session.add(self.episode)
        await self.session.commit()
        cards = await load_card_refs(self.session)
        self.service = GenerationService(
            backend=FakeBackend(seed=self.fake_seed),
            verifier=HeuristicVerifier(cards),
            reward_model=HeuristicColdRewardModel(),
            image_backend=get_image_backend(ImageMode.brand_assets),
            storage=self.storage,
            embedder=HashEmbedder(self.cfg.archive.embedding_dims),
            cfg=self.cfg,
        )
        return self

    async def generate_and_approve(self, k=4, renders=1, now=None):
        result = await self.service.generate_batch(
            self.session,
            GenerationRequest(
                brief_id=self.brief.id,
                episode_id=self.episode.id,
                backend=BackendKind.fake,
                k=k,
                renders_per_idea=renders,
            ),
            now=now,
        )
        for idea in result.queued:
            self.session.add(
                Review(
                    trajectory_id=idea.trajectory.id,
                    label="run",
                    reviewer_id="expert@example.com",
                    note="",
                )
            )
        await self.session.commit()
        return result

    async def ship(self, batch_id, now):
        assert batch_id is not None
        results = await ship_batch(
            self.session,
            batch_id,
            client=self.client,
            storage=self.storage,
            cfg=self.cfg,
            actor="test",
            now=now,
        )
        await self.session.commit()
        return results

    async def advance(self, days):
        today = await self.client.advance_days(days)
        await self.session.commit()
        return today

    async def sync_and_tick(self, today: date):
        await run_sync_insights(
            self.session,
            self.cfg,
            account_id=self.account.id,
            today=today + timedelta(days=1),
            clients={self.account.id: self.client},
        )
        report = await tick_episode(
            self.session, self.episode, client=self.client, cfg=self.cfg, now=dt(today)
        )
        await self.session.commit()
        return report


async def _count(session, model, *where):
    return int(
        (await session.execute(select(func.count()).select_from(model).where(*where))).scalar_one()
    )
