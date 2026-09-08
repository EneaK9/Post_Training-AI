from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.meta.base import (
    AdSetSpec,
    AdSpec,
    CampaignSpec,
    CreativeSpec,
    ad_object_name,
    parse_ad_object_name,
)
from outlier_ai.meta.errors import MetaAuthError, MetaObjectNotFoundError
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.models.trajectories import Render
from outlier_ai.synthetic.briefs import CATEGORIES
from outlier_ai.synthetic.cards import SEED_STRATEGY_CARDS
from outlier_ai.synthetic.latent import LatentOutcomeModel
from outlier_ai.synthetic.seed import FAKE_ACCOUNT_ID, FAKE_PAGE_ID, seed

pytestmark = pytest.mark.integration


def _latent() -> LatentOutcomeModel:
    return LatentOutcomeModel(1, list(CATEGORIES), [c.slug for c in SEED_STRATEGY_CARDS])


async def _ship_one(client: FakeMetaClient, session: AsyncSession, budget_cents: int = 2000):
    render = (await session.execute(select(Render).limit(1))).scalars().first()
    assert render is not None
    img = await client.upload_image(b"\x89PNG fake bytes", "r.png")
    camp = await client.create_campaign(CampaignSpec(name="OAI|ep|test"))
    adset = await client.create_adset(
        AdSetSpec(
            name="adset",
            campaign_id=camp,
            daily_budget_cents=budget_cents,
            targeting={"geo_locations": {"countries": ["US"]}},
            pixel_id="px",
            attribution_setting="7d_click_1d_view",
        )
    )
    creative = await client.create_creative(
        CreativeSpec(
            name="cr",
            page_id=FAKE_PAGE_ID,
            image_hash=img.image_hash,
            primary_text="p",
            headline="h",
            description="d",
            cta="SHOP_NOW",
            link_url="https://example.com",
        )
    )
    name = ad_object_name("ep", str(render.trajectory_id), str(render.id))
    ad_id = await client.create_ad(
        AdSpec(name=name, adset_id=adset, creative_id=creative.creative_id)
    )
    return ad_id, adset, creative.effective_object_story_id


def test_ad_object_name_round_trips():
    name = ad_object_name("e", "t", "r")
    assert name == "OAI|e|t|r"
    assert parse_ad_object_name(name) == ("e", "t", "r")
    assert parse_ad_object_name("something else") is None


async def test_fake_lifecycle_review_insights_comments(db_session: AsyncSession):
    await seed(db_session, n_briefs=3, n_trajectories=12, seed=2, days_back=60)
    client = FakeMetaClient(
        db_session,
        FAKE_ACCOUNT_ID,
        _latent(),
        seed=3,
        rejection_rate=0.0,
        review_delay_days=1,
        start_day=date(2026, 9, 1),
    )
    ad_id, adset_id, story_id = await _ship_one(client, db_session)

    status = (await client.get_ad_review_status([ad_id]))[0]
    assert status.effective_status == "PENDING_REVIEW"
    assert await client.get_daily_insights([ad_id], date(2026, 9, 2)) == []

    await client.advance_days(1)  # review resolves, no insights yet on the approval day
    assert (await client.get_ad_review_status([ad_id]))[0].approved

    await client.advance_days(7)
    days = [date(2026, 9, 3 + i) for i in range(7)]
    rows = [r for d in days for r in await client.get_daily_insights([ad_id], d)]
    assert len(rows) == 7
    assert all(0 <= r.link_clicks <= r.impressions for r in rows)
    assert all(r.spend <= 20.0 for r in rows)

    comments = await client.get_post_comments(story_id)
    assert comments, "an active ad with impressions should draw comments"
    assert all(c.post_id == story_id for c in comments)
    later = await client.get_post_comments(story_id, since=comments[0].created_time)
    assert len(later) == len(comments) - 1

    # scaling raises the budget and switches the ad into the scale phase
    await client.update_adset_budget(adset_id, 10000)
    await client.advance_days(1)
    scale_rows = await client.get_daily_insights([ad_id], date(2026, 9, 10))
    assert scale_rows and scale_rows[0].spend > 50

    await client.pause_ad(ad_id)
    await client.advance_days(1)
    assert await client.get_daily_insights([ad_id], date(2026, 9, 11)) == []


async def test_fake_rejects_and_injects_errors(db_session: AsyncSession):
    await seed(db_session, n_briefs=2, n_trajectories=6, seed=5, days_back=60)
    client = FakeMetaClient(
        db_session, FAKE_ACCOUNT_ID, _latent(), seed=1, rejection_rate=1.0, review_delay_days=0
    )
    ad_id, _, _ = await _ship_one(client, db_session)
    await client.advance_days(1)
    status = (await client.get_ad_review_status([ad_id]))[0]
    assert status.rejected and status.review_feedback

    client.fail_next(MetaAuthError)
    with pytest.raises(MetaAuthError):
        await client.get_account_info()
    # error is one-shot
    assert (await client.get_account_info()).account_id == FAKE_ACCOUNT_ID

    with pytest.raises(MetaObjectNotFoundError):
        await client.get_ad_review_status(["ad_missing"])


async def test_truth_follows_render_niche(db_session: AsyncSession):
    """Two ads on the same render share the hidden mean; the fake is not pure noise."""
    await seed(db_session, n_briefs=2, n_trajectories=6, seed=8, days_back=60)
    client = FakeMetaClient(db_session, FAKE_ACCOUNT_ID, _latent(), seed=2, rejection_rate=0.0)
    ad_a, _, _ = await _ship_one(client, db_session)
    ad_b, _, _ = await _ship_one(client, db_session)
    a = (await client._require("ad", ad_a)).payload["truth"]
    b = (await client._require("ad", ad_b)).payload["truth"]
    assert a["roas_multiple"] > 0 and b["roas_multiple"] > 0
    assert a != b  # per-render variation keeps ads distinct
