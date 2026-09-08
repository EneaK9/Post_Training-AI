"""Campaign structure [v3-assumed]: one campaign per episode, one ad set per ad, identical
targeting from the brief, equal fixed daily budgets, no CBO, feed placements only, named
`OAI|episode|trajectory|render`."""

from __future__ import annotations

from typing import Any

from outlier_ai.meta.base import AdSetSpec, AdSpec, CampaignSpec, CreativeSpec, ad_object_name
from outlier_ai.models.briefs import Brief
from outlier_ai.models.episodes import SearchEpisode
from outlier_ai.models.meta import AdAccount
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_schemas.config import AppConfig

CTA_MAP = {
    "shop now": "SHOP_NOW",
    "learn more": "LEARN_MORE",
    "get offer": "GET_OFFER",
    "sign up": "SIGN_UP",
    "subscribe": "SUBSCRIBE",
    "buy now": "BUY_NOW",
    "order now": "ORDER_NOW",
}

DEFAULT_TARGETING: dict[str, Any] = {
    "geo_locations": {"countries": ["US"]},
    "age_min": 18,
    "age_max": 65,
}

FEED_ONLY: dict[str, Any] = {
    "publisher_platforms": ["facebook", "instagram"],
    "facebook_positions": ["feed"],
    "instagram_positions": ["stream"],
}


def cta_type(cta: str) -> str:
    return CTA_MAP.get(cta.strip().lower(), "LEARN_MORE")


def attribution_spec(setting: str) -> list[dict[str, Any]]:
    """`7d_click_1d_view` -> Meta attribution_spec entries."""
    spec: list[dict[str, Any]] = []
    tokens = setting.lower().replace("-", "_").split("_")
    i = 0
    while i + 1 < len(tokens):
        days_token, kind = tokens[i], tokens[i + 1]
        if days_token.endswith("d") and days_token[:-1].isdigit():
            days = int(days_token[:-1])
            if kind == "click":
                spec.append({"event_type": "CLICK_THROUGH", "window_days": days})
            elif kind == "view":
                spec.append({"event_type": "VIEW_THROUGH", "window_days": days})
        i += 2
    return spec or [{"event_type": "CLICK_THROUGH", "window_days": 7}]


def campaign_spec(episode: SearchEpisode, brief: Brief) -> CampaignSpec:
    return CampaignSpec(
        name=f"OAI|{episode.id}|{brief.id}", objective="OUTCOME_SALES", status="ACTIVE"
    )


def adset_spec(
    *,
    episode: SearchEpisode,
    trajectory: Trajectory,
    render: Render,
    brief: Brief,
    account: AdAccount,
    campaign_id: str,
    daily_budget_usd: float,
) -> AdSetSpec:
    targeting = {
        **DEFAULT_TARGETING,
        **(brief.meta or {}).get("default_targeting_spec", {}),
        **FEED_ONLY,
    }
    return AdSetSpec(
        name=ad_object_name(str(episode.id), str(trajectory.id), str(render.id)),
        campaign_id=campaign_id,
        daily_budget_cents=round(daily_budget_usd * 100),
        targeting=targeting,
        pixel_id=(brief.meta or {}).get("pixel_id") or account.pixel_id,
        attribution_setting=account.attribution_setting,
        optimization_goal="OFFSITE_CONVERSIONS",
        billing_event="IMPRESSIONS",
        placements=("facebook_feed", "instagram_feed"),
        status="ACTIVE",
    )


def creative_spec(
    *, trajectory: Trajectory, render: Render, brief: Brief, account: AdAccount, image_hash: str
) -> CreativeSpec:
    copy = trajectory.ad_copy or {}
    page_id = (brief.meta or {}).get("page_id") or account.page_id or ""
    return CreativeSpec(
        name=ad_object_name("creative", str(trajectory.id), str(render.id)),
        page_id=page_id,
        image_hash=image_hash,
        primary_text=str(copy.get("primary_text", "")),
        headline=str(copy.get("headline", "")),
        description=str(copy.get("description", "")),
        cta=cta_type(str(copy.get("cta", ""))),
        link_url=(brief.meta or {}).get("landing_url") or "https://example.com",
    )


def ad_spec(
    *,
    episode: SearchEpisode,
    trajectory: Trajectory,
    render: Render,
    adset_id: str,
    creative_id: str,
) -> AdSpec:
    return AdSpec(
        name=ad_object_name(str(episode.id), str(trajectory.id), str(render.id)),
        adset_id=adset_id,
        creative_id=creative_id,
        status="ACTIVE",
    )


def screening_budget(cfg: AppConfig) -> float:
    return float(cfg.episode.screening_budget_per_ad_usd)


def scale_budget(cfg: AppConfig) -> float:
    return float(cfg.episode.scale_budget_per_ad_usd)
