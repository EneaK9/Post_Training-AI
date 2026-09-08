"""Real Meta Marketing API client behind the `MetaClient` protocol.

Wraps `facebook_business` (extra `meta`). The SDK is synchronous, so every call runs in a
worker thread. Errors are mapped to typed exceptions so the sync jobs can act on them:
expired tokens mark the account `needs_reauth` (scenario row 12), rate limits back off, a
retired API version is surfaced explicitly.

The client is only ever constructed by `meta.factory.client_for_account`, which refuses when
DRY_RUN is on, the kill switch is closed, or the account is not active.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, TypeVar

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
)
from outlier_ai.meta.campaign import attribution_spec
from outlier_ai.meta.errors import (
    MetaApiVersionError,
    MetaAuthError,
    MetaError,
    MetaObjectNotFoundError,
    MetaRateLimitError,
    MetaValidationError,
)

T = TypeVar("T")

AUTH_CODES = {190, 102, 10}
RATE_CODES = {4, 17, 32, 613, 80000, 80004}
VERSION_CODES = {2635}
NOT_FOUND_CODES = {803}
PURCHASE_ACTIONS = ("purchase", "offsite_conversion.fb_pixel_purchase", "omni_purchase")


def map_facebook_error(exc: Any) -> MetaError:
    """Translate a FacebookRequestError (or anything with the same attributes) to a MetaError."""
    code = getattr(exc, "api_error_code", lambda: None)
    code = code() if callable(code) else code
    subcode = getattr(exc, "api_error_subcode", lambda: None)
    subcode = subcode() if callable(subcode) else subcode
    message = getattr(exc, "api_error_message", lambda: str(exc))
    message = message() if callable(message) else message
    text = f"meta api error {code}/{subcode}: {message}"
    if code in AUTH_CODES:
        return MetaAuthError(text)
    if code in RATE_CODES:
        return MetaRateLimitError(text)
    if code in VERSION_CODES or (
        "version" in str(message).lower() and "deprecat" in str(message).lower()
    ):
        return MetaApiVersionError(text)
    if code in NOT_FOUND_CODES or "does not exist" in str(message).lower():
        return MetaObjectNotFoundError(text)
    if code == 100:
        return MetaValidationError(text)
    return MetaError(text)


class MetaGraphClient:
    """Implements `MetaClient` against the live Graph API."""

    def __init__(
        self,
        *,
        account_id: str,
        access_token: str,
        page_token: str | None,
        api_version: str,
        app_id: str | None,
        app_secret: str | None,
        action_types: dict[str, str] | None = None,
    ) -> None:
        try:
            from facebook_business.api import FacebookAdsApi
            from facebook_business.exceptions import FacebookRequestError
        except ImportError as e:  # pragma: no cover - only without the extra
            raise MetaError("facebook_business is not installed; install outlier-ai[meta]") from e
        self._api = FacebookAdsApi.init(app_id, app_secret, access_token, api_version=api_version)
        self._page_api = (
            FacebookAdsApi(None) if False else None  # placeholder for a page-token session below
        )
        self._page_token = page_token
        self._access_token = access_token
        self._app_id = app_id
        self._app_secret = app_secret
        self._api_version = api_version
        self._RequestError = FacebookRequestError
        self.account_id = account_id if account_id.startswith("act_") else f"act_{account_id}"
        self.action_types = action_types or {
            "reactions": "post_reaction",
            "comments": "comment",
            "shares": "post",
            "saves": "onsite_conversion.post_save",
        }

    async def _call(self, fn: Callable[[], T]) -> T:
        try:
            return await asyncio.to_thread(fn)
        except self._RequestError as e:
            raise map_facebook_error(e) from e

    # ---- MetaClient ------------------------------------------------------------------------

    async def get_account_info(self) -> AccountInfo:
        from facebook_business.adobjects.adaccount import AdAccount

        def fn() -> Any:
            return AdAccount(self.account_id).api_get(fields=["name", "currency", "timezone_name"])

        acc = await self._call(fn)
        return AccountInfo(
            account_id=self.account_id,
            name=str(acc.get("name", "")),
            currency=str(acc.get("currency", "USD")),
            timezone_name=str(acc.get("timezone_name", "UTC")),
            api_version=self._api_version,
        )

    async def upload_image(self, image_bytes: bytes, name: str) -> AdImageRef:
        from facebook_business.adobjects.adaccount import AdAccount

        def fn() -> Any:
            return AdAccount(self.account_id).create_ad_image(
                params={"bytes": base64.b64encode(image_bytes).decode(), "name": name}
            )

        img = await self._call(fn)
        return AdImageRef(image_hash=str(img["hash"]), url=str(img.get("url", "")))

    async def create_campaign(self, spec: CampaignSpec) -> str:
        from facebook_business.adobjects.adaccount import AdAccount

        def fn() -> Any:
            return AdAccount(self.account_id).create_campaign(
                params={
                    "name": spec.name,
                    "objective": spec.objective,
                    "status": spec.status,
                    "special_ad_categories": list(spec.special_ad_categories),
                    # no CBO: budgets live on the ad sets
                    "is_adset_budget_sharing_enabled": False,
                }
            )

        return str((await self._call(fn))["id"])

    async def create_adset(self, spec: AdSetSpec) -> str:
        from facebook_business.adobjects.adaccount import AdAccount

        params: dict[str, Any] = {
            "name": spec.name,
            "campaign_id": spec.campaign_id,
            "daily_budget": spec.daily_budget_cents,
            "billing_event": spec.billing_event,
            "optimization_goal": spec.optimization_goal,
            "bid_strategy": "LOWEST_COST_WITHOUT_CAP",
            "targeting": spec.targeting,
            "status": spec.status,
            "attribution_spec": attribution_spec(spec.attribution_setting),
        }
        if spec.pixel_id:
            params["promoted_object"] = {"pixel_id": spec.pixel_id, "custom_event_type": "PURCHASE"}

        def fn() -> Any:
            return AdAccount(self.account_id).create_ad_set(params=params)

        return str((await self._call(fn))["id"])

    async def create_creative(self, spec: CreativeSpec) -> CreativeResult:
        from facebook_business.adobjects.adaccount import AdAccount
        from facebook_business.adobjects.adcreative import AdCreative

        params = {
            "name": spec.name,
            "object_story_spec": {
                "page_id": spec.page_id,
                "link_data": {
                    "image_hash": spec.image_hash,
                    "link": spec.link_url,
                    "message": spec.primary_text,
                    "name": spec.headline,
                    "description": spec.description,
                    "call_to_action": {"type": spec.cta, "value": {"link": spec.link_url}},
                },
            },
        }

        def fn() -> Any:
            created = AdAccount(self.account_id).create_ad_creative(params=params)
            detail = AdCreative(created["id"]).api_get(fields=["effective_object_story_id"])
            return created["id"], detail.get("effective_object_story_id", "")

        creative_id, story_id = await self._call(fn)
        return CreativeResult(creative_id=str(creative_id), effective_object_story_id=str(story_id))

    async def create_ad(self, spec: AdSpec) -> str:
        from facebook_business.adobjects.adaccount import AdAccount

        def fn() -> Any:
            return AdAccount(self.account_id).create_ad(
                params={
                    "name": spec.name,
                    "adset_id": spec.adset_id,
                    "creative": {"creative_id": spec.creative_id},
                    "status": spec.status,
                }
            )

        return str((await self._call(fn))["id"])

    async def get_ad_review_status(self, ad_ids: list[str]) -> list[ReviewStatus]:
        from facebook_business.adobjects.ad import Ad

        def fn() -> list[ReviewStatus]:
            out: list[ReviewStatus] = []
            for ad_id in ad_ids:
                ad = Ad(ad_id).api_get(fields=["effective_status", "ad_review_feedback"])
                feedback = ad.get("ad_review_feedback") or {}
                reasons = feedback.get("global") if isinstance(feedback, dict) else None
                out.append(
                    ReviewStatus(
                        ad_id=ad_id,
                        effective_status=str(ad.get("effective_status", "PENDING_REVIEW")),
                        review_feedback="; ".join(f"{k}: {v}" for k, v in (reasons or {}).items())
                        or None,
                    )
                )
            return out

        return await self._call(fn)

    async def get_daily_insights(self, ad_ids: list[str], day: date) -> list[InsightRow]:
        from facebook_business.adobjects.ad import Ad

        fields = [
            "ad_id",
            "impressions",
            "inline_link_clicks",
            "spend",
            "frequency",
            "actions",
            "action_values",
        ]
        params = {
            "time_range": {"since": day.isoformat(), "until": day.isoformat()},
            "use_unified_attribution_setting": True,
            "level": "ad",
        }

        def fn() -> list[InsightRow]:
            rows: list[InsightRow] = []
            for ad_id in ad_ids:
                for ins in Ad(ad_id).get_insights(fields=fields, params=params):
                    actions = {
                        a["action_type"]: float(a.get("value", 0))
                        for a in ins.get("actions", []) or []
                    }
                    values = {
                        a["action_type"]: float(a.get("value", 0))
                        for a in ins.get("action_values", []) or []
                    }
                    purchases = int(next((actions[k] for k in PURCHASE_ACTIONS if k in actions), 0))
                    revenue = float(next((values[k] for k in PURCHASE_ACTIONS if k in values), 0.0))
                    rows.append(
                        InsightRow(
                            ad_id=ad_id,
                            day=day,
                            impressions=int(float(ins.get("impressions", 0))),
                            link_clicks=int(float(ins.get("inline_link_clicks", 0))),
                            spend=float(ins.get("spend", 0.0)),
                            purchases=purchases,
                            revenue=revenue,
                            frequency=float(ins["frequency"]) if ins.get("frequency") else None,
                            reactions=int(actions.get(self.action_types["reactions"], 0)),
                            comments=int(actions.get(self.action_types["comments"], 0)),
                            shares=int(actions.get(self.action_types["shares"], 0)),
                            saves=int(actions.get(self.action_types["saves"], 0)),
                            attribution_setting="unified",
                        )
                    )
            return rows

        return await self._call(fn)

    async def get_post_comments(
        self, effective_object_story_id: str, since: datetime | None = None
    ) -> list[PostComment]:
        """Comments need the Page token (`pages_read_user_content`)."""
        from facebook_business.api import FacebookAdsApi, FacebookRequest

        if not self._page_token:
            raise MetaAuthError(
                "no page token on this account; comments need pages_read_user_content"
            )
        page_api = FacebookAdsApi.init(
            self._app_id, self._app_secret, self._page_token, api_version=self._api_version
        )

        def fn() -> list[PostComment]:
            out: list[PostComment] = []
            params: dict[str, Any] = {
                "fields": "id,from,message,created_time,like_count",
                "limit": 100,
                "order": "chronological",
            }
            if since is not None:
                params["since"] = int(since.timestamp())
            after: str | None = None
            while True:
                if after:
                    params["after"] = after
                req = FacebookRequest(
                    node_id=effective_object_story_id,
                    method="GET",
                    endpoint="/comments",
                    api=page_api,
                )
                req.add_params(params)
                resp = req.execute()
                body = resp.json() if hasattr(resp, "json") else resp
                for c in body.get("data", []):
                    created = datetime.fromisoformat(
                        str(c["created_time"]).replace("+0000", "+00:00")
                    )
                    out.append(
                        PostComment(
                            id=str(c["id"]),
                            post_id=effective_object_story_id,
                            from_id=str((c.get("from") or {}).get("id", "")),
                            message=str(c.get("message", "")),
                            created_time=created.astimezone(UTC),
                            like_count=int(c.get("like_count", 0)),
                        )
                    )
                after = ((body.get("paging") or {}).get("cursors") or {}).get("after")
                if not after or not (body.get("paging") or {}).get("next"):
                    break
            return out

        return await self._call(fn)

    async def update_adset_budget(self, adset_id: str, daily_budget_cents: int) -> None:
        from facebook_business.adobjects.adset import AdSet

        await self._call(
            lambda: AdSet(adset_id).api_update(params={"daily_budget": daily_budget_cents})
        )

    async def pause_ad(self, ad_id: str) -> None:
        from facebook_business.adobjects.ad import Ad

        await self._call(lambda: Ad(ad_id).api_update(params={"status": "PAUSED"}))
