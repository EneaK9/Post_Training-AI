"""Fake Meta client backed by the `fake_meta_objects` table and the latent outcome model.

Behaves like Meta from the outside: creates objects and returns ids, holds ads in review,
approves or disapproves them, and, as its clock advances, produces daily insight rows and
comments drawn from the hidden truth of the render behind each ad. The sync jobs read from
it exactly as they would read from the real API, so the whole pipeline is exercised.

The truth for an ad is derived from the render id encoded in the ad name
(`OAI|episode|trajectory|render`): render -> trajectory -> verified cards + brief -> niche,
category, world state -> `LatentOutcomeModel.sample_truth`. Ads with unparseable names get the
cold default so tests that do not care about outcomes still work.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.combinations import niche_key
from outlier_ai.meta.base import (
    AccountInfo,
    AdImageRef,
    AdSetSpec,
    AdSpec,
    CampaignSpec,
    CreativeResult,
    CreativeSpec,
    InsightRow,
    PostComment,
    ReviewStatus,
    parse_ad_object_name,
)
from outlier_ai.meta.errors import MetaError, MetaObjectNotFoundError, MetaValidationError
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import Card
from outlier_ai.models.meta import FakeMetaObject
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_ai.synthetic.comments import generate_comments
from outlier_ai.synthetic.latent import AccountProfile, AdTruth, LatentOutcomeModel
from outlier_schemas.enums import NicheProjection

KIND_CLOCK = "clock"
KIND_IMAGE = "image"
KIND_CAMPAIGN = "campaign"
KIND_ADSET = "adset"
KIND_CREATIVE = "creative"
KIND_AD = "ad"
KIND_INSIGHT = "insight"
KIND_COMMENT = "comment"

DISAPPROVAL_REASONS = (
    "Personal attributes: the ad implies knowledge of the viewer's personal characteristics.",
    "Unrealistic outcomes: the ad promises results that cannot be guaranteed.",
    "Low quality or disruptive content: excessive text on the image.",
)


class FakeMetaClient:
    """Implements `MetaClient`. One instance per (session, account)."""

    def __init__(
        self,
        session: AsyncSession,
        account_id: str,
        latent: LatentOutcomeModel,
        *,
        account_profile: AccountProfile | None = None,
        niche_projection: NicheProjection = NicheProjection.strategy_only,
        seed: int = 0,
        rejection_rate: float = 0.05,
        review_delay_days: int = 1,
        start_day: date | None = None,
        api_version: str = "v26.0",
    ) -> None:
        self.session = session
        self.account_id = account_id
        self.latent = latent
        self.profile = account_profile or AccountProfile()
        self.niche_projection = niche_projection
        self.rng = np.random.default_rng(seed)
        self.rejection_rate = rejection_rate
        self.review_delay_days = review_delay_days
        self._start_day = start_day or datetime.now(UTC).date()
        self.api_version = api_version
        self._fail_next: type[MetaError] | None = None
        self._counter = 0

    # ---- test hooks -----------------------------------------------------------------------

    def fail_next(self, error: type[MetaError]) -> None:
        """Make the next API call raise `error` (scenario tests for auth, rate limit, version)."""
        self._fail_next = error

    def _maybe_fail(self) -> None:
        if self._fail_next is not None:
            err, self._fail_next = self._fail_next, None
            raise err(f"fake meta: injected {err.__name__}")

    # ---- state helpers --------------------------------------------------------------------

    def _new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    async def _get(self, kind: str, external_id: str) -> FakeMetaObject | None:
        res = await self.session.execute(
            select(FakeMetaObject).where(
                FakeMetaObject.account_id == self.account_id,
                FakeMetaObject.kind == kind,
                FakeMetaObject.external_id == external_id,
            )
        )
        return res.scalar_one_or_none()

    async def _require(self, kind: str, external_id: str) -> FakeMetaObject:
        obj = await self._get(kind, external_id)
        if obj is None:
            raise MetaObjectNotFoundError(f"fake meta: {kind} {external_id} not found")
        return obj

    async def _put(self, kind: str, external_id: str, payload: dict[str, Any]) -> FakeMetaObject:
        obj = FakeMetaObject(
            account_id=self.account_id, kind=kind, external_id=external_id, payload=payload
        )
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def _list(self, kind: str) -> list[FakeMetaObject]:
        res = await self.session.execute(
            select(FakeMetaObject).where(
                FakeMetaObject.account_id == self.account_id, FakeMetaObject.kind == kind
            )
        )
        return list(res.scalars().all())

    @staticmethod
    def _update(obj: FakeMetaObject, **changes: Any) -> None:
        # Reassign so SQLAlchemy sees the JSONB change.
        obj.payload = {**obj.payload, **changes}

    async def today(self) -> date:
        clock = await self._get(KIND_CLOCK, self.account_id)
        if clock is None:
            clock = await self._put(
                KIND_CLOCK, self.account_id, {"today": self._start_day.isoformat()}
            )
        return date.fromisoformat(clock.payload["today"])

    # ---- truth ----------------------------------------------------------------------------

    async def _truth_for_ad_name(self, name: str) -> AdTruth:
        parsed = parse_ad_object_name(name)
        if parsed is None:
            return self.latent.sample_truth("__unknown__", "default", "none", self.rng)
        _, _, render_id = parsed
        try:
            rid = uuid.UUID(render_id)
        except ValueError:
            return self.latent.sample_truth("__unknown__", "default", "none", self.rng)
        render = await self.session.get(Render, rid)
        if render is None:
            return self.latent.sample_truth("__unknown__", "default", "none", self.rng)
        traj = await self.session.get(Trajectory, render.trajectory_id)
        brief = await self.session.get(Brief, traj.brief_id) if traj else None
        if traj is None or brief is None:
            return self.latent.sample_truth("__unknown__", "default", "none", self.rng)
        card_ids = traj.verified_card_ids or traj.card_ids
        res = await self.session.execute(select(Card).where(Card.id.in_(card_ids)))
        cards_by_id = {c.id: c for c in res.scalars().all()}
        niche = niche_key(card_ids, cards_by_id, self.niche_projection)
        world_tag = str(brief.meta.get("world_state_tag", "none"))
        base = self.latent.sample_truth(niche, brief.category, world_tag, self.rng)
        # per-render variation, same as the seed
        return AdTruth(
            roas_multiple=base.roas_multiple * float(np.exp(0.10 * self.rng.normal())),
            ctr_multiple=base.ctr_multiple * float(np.exp(0.10 * self.rng.normal())),
            polarity=base.polarity,
        )

    # ---- MetaClient ------------------------------------------------------------------------

    async def get_account_info(self) -> AccountInfo:
        self._maybe_fail()
        return AccountInfo(
            account_id=self.account_id, name="Fake Account", api_version=self.api_version
        )

    async def upload_image(self, image_bytes: bytes, name: str) -> AdImageRef:
        self._maybe_fail()
        if not image_bytes:
            raise MetaValidationError("fake meta: empty image")
        image_hash = uuid.uuid5(uuid.NAMESPACE_URL, f"{name}:{len(image_bytes)}").hex
        if await self._get(KIND_IMAGE, image_hash) is None:
            await self._put(KIND_IMAGE, image_hash, {"name": name, "bytes": len(image_bytes)})
        return AdImageRef(image_hash=image_hash, url=f"https://fake.meta/images/{image_hash}")

    async def create_campaign(self, spec: CampaignSpec) -> str:
        self._maybe_fail()
        cid = self._new_id("camp")
        await self._put(
            KIND_CAMPAIGN, cid, {**asdict(spec), "created_day": (await self.today()).isoformat()}
        )
        return cid

    async def create_adset(self, spec: AdSetSpec) -> str:
        self._maybe_fail()
        await self._require(KIND_CAMPAIGN, spec.campaign_id)
        if spec.daily_budget_cents < 100:
            raise MetaValidationError("fake meta: daily budget below minimum")
        aid = self._new_id("adset")
        await self._put(
            KIND_ADSET, aid, {**asdict(spec), "created_day": (await self.today()).isoformat()}
        )
        return aid

    async def create_creative(self, spec: CreativeSpec) -> CreativeResult:
        self._maybe_fail()
        await self._require(KIND_IMAGE, spec.image_hash)
        crid = self._new_id("cr")
        story_id = f"{spec.page_id}_{uuid.uuid4().hex[:16]}"
        await self._put(
            KIND_CREATIVE, crid, {**asdict(spec), "effective_object_story_id": story_id}
        )
        return CreativeResult(creative_id=crid, effective_object_story_id=story_id)

    async def create_ad(self, spec: AdSpec) -> str:
        self._maybe_fail()
        adset = await self._require(KIND_ADSET, spec.adset_id)
        creative = await self._require(KIND_CREATIVE, spec.creative_id)
        today = await self.today()
        truth = await self._truth_for_ad_name(spec.name)
        ad_id = self._new_id("ad")
        await self._put(
            KIND_AD,
            ad_id,
            {
                "name": spec.name,
                "adset_id": spec.adset_id,
                "creative_id": spec.creative_id,
                "story_id": creative.payload["effective_object_story_id"],
                "status": spec.status,
                "effective_status": "PENDING_REVIEW",
                "review_feedback": None,
                "submitted_day": today.isoformat(),
                "activated_day": None,
                "truth": asdict(truth),
                "daily_budget_cents": adset.payload["daily_budget_cents"],
                "phase": "screening",
                "phase_day": 0,
                "attribution_setting": adset.payload["attribution_setting"],
            },
        )
        return ad_id

    async def get_ad_review_status(self, ad_ids: list[str]) -> list[ReviewStatus]:
        self._maybe_fail()
        out: list[ReviewStatus] = []
        for ad_id in ad_ids:
            ad = await self._require(KIND_AD, ad_id)
            p = ad.payload
            out.append(ReviewStatus(ad_id, p["effective_status"], p.get("review_feedback")))
        return out

    async def get_daily_insights(self, ad_ids: list[str], day: date) -> list[InsightRow]:
        self._maybe_fail()
        rows: list[InsightRow] = []
        for ad_id in ad_ids:
            obj = await self._get(KIND_INSIGHT, f"{ad_id}:{day.isoformat()}")
            if obj is None:
                continue
            p = obj.payload
            rows.append(
                InsightRow(
                    ad_id=ad_id,
                    day=day,
                    impressions=p["impressions"],
                    link_clicks=p["link_clicks"],
                    spend=p["spend"],
                    purchases=p["purchases"],
                    revenue=p["revenue"],
                    frequency=p.get("frequency"),
                    reactions=p["reactions"],
                    comments=p["comments"],
                    shares=p["shares"],
                    saves=p["saves"],
                    attribution_setting=p["attribution_setting"],
                )
            )
        return rows

    async def get_post_comments(
        self, effective_object_story_id: str, since: datetime | None = None
    ) -> list[PostComment]:
        self._maybe_fail()
        out: list[PostComment] = []
        for obj in await self._list(KIND_COMMENT):
            p = obj.payload
            if p["post_id"] != effective_object_story_id:
                continue
            created = datetime.fromisoformat(p["created_time"])
            if since is not None and created <= since:
                continue
            out.append(
                PostComment(
                    id=obj.external_id,
                    post_id=p["post_id"],
                    from_id=p["from_id"],
                    message=p["message"],
                    created_time=created,
                    like_count=p.get("like_count", 0),
                )
            )
        out.sort(key=lambda c: c.created_time)
        return out

    async def update_adset_budget(self, adset_id: str, daily_budget_cents: int) -> None:
        self._maybe_fail()
        adset = await self._require(KIND_ADSET, adset_id)
        old = adset.payload["daily_budget_cents"]
        self._update(adset, daily_budget_cents=daily_budget_cents)
        for ad in await self._list(KIND_AD):
            if ad.payload["adset_id"] == adset_id:
                changes: dict[str, Any] = {"daily_budget_cents": daily_budget_cents}
                if daily_budget_cents > old:
                    changes.update(phase="scale", phase_day=0)
                self._update(ad, **changes)
        await self.session.flush()

    async def pause_ad(self, ad_id: str) -> None:
        self._maybe_fail()
        ad = await self._require(KIND_AD, ad_id)
        self._update(ad, status="PAUSED", effective_status="PAUSED")
        await self.session.flush()

    # ---- simulation -----------------------------------------------------------------------

    async def advance_days(self, n: int = 1) -> date:
        """Move the clock forward, resolving reviews and emitting insights and comments."""
        today = await self.today()
        for _ in range(n):
            today = today + timedelta(days=1)
            ads = await self._list(KIND_AD)
            for ad in ads:
                p = ad.payload
                if p["effective_status"] == "PENDING_REVIEW":
                    submitted = date.fromisoformat(p["submitted_day"])
                    if (today - submitted).days >= self.review_delay_days:
                        if self.rng.random() < self.rejection_rate:
                            reason = str(self.rng.choice(DISAPPROVAL_REASONS))
                            self._update(ad, effective_status="DISAPPROVED", review_feedback=reason)
                        else:
                            self._update(
                                ad, effective_status="ACTIVE", activated_day=today.isoformat()
                            )
                    continue
                if p["effective_status"] != "ACTIVE":
                    continue
                await self._emit_day(ad, today)
            clock = await self._require(KIND_CLOCK, self.account_id)
            self._update(clock, today=today.isoformat())
            await self.session.flush()
        return today

    async def _emit_day(self, ad: FakeMetaObject, day: date) -> None:
        p = ad.payload
        truth = AdTruth(**p["truth"])
        budget_usd = p["daily_budget_cents"] / 100.0
        row = self.latent.simulate_day(
            truth, budget_usd, self.profile, p["phase_day"], p["phase"], self.rng
        )
        await self._put(
            KIND_INSIGHT,
            f"{ad.external_id}:{day.isoformat()}",
            {
                "ad_id": ad.external_id,
                "impressions": row.impressions,
                "link_clicks": row.link_clicks,
                "spend": row.spend,
                "purchases": row.purchases,
                "revenue": row.revenue,
                "frequency": row.frequency,
                "reactions": row.reactions,
                "comments": row.comments,
                "shares": row.shares,
                "saves": row.saves,
                "attribution_setting": p["attribution_setting"],
            },
        )
        n_comments = min(30, row.comments)
        day_start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        for c in generate_comments(self.rng, n_comments, truth.roas_multiple, truth.polarity, 24.0):
            await self._put(
                KIND_COMMENT,
                self._new_id("c"),
                {
                    "post_id": p["story_id"],
                    "from_id": c.commenter_ext_id,
                    "message": c.raw_text,  # raw: PII stripping happens in the sync job
                    "created_time": (day_start + timedelta(hours=c.hours_after_launch)).isoformat(),
                    "like_count": c.like_count,
                },
            )
        self._update(ad, phase_day=p["phase_day"] + 1)
