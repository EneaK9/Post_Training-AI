"""Contract tests against a Meta sandbox ad account. They run only when META_SANDBOX_TOKEN and
META_SANDBOX_ACCOUNT_ID are set (CI job `contract`, manual). They pin the API version in
config.meta and the shape of the calls the client makes; when `config.meta.api_version` changes,
run these first (plan Phase 7: schema-drift check)."""

from __future__ import annotations

import os

import pytest

from outlier_schemas.config import AppConfig

pytestmark = pytest.mark.contract

TOKEN = os.environ.get("META_SANDBOX_TOKEN")
ACCOUNT = os.environ.get("META_SANDBOX_ACCOUNT_ID")
needs_sandbox = pytest.mark.skipif(
    not (TOKEN and ACCOUNT), reason="META_SANDBOX_TOKEN / META_SANDBOX_ACCOUNT_ID not set"
)


def _client(cfg: AppConfig):
    from outlier_ai.meta.client import MetaGraphClient

    assert TOKEN and ACCOUNT
    return MetaGraphClient(
        account_id=ACCOUNT,
        access_token=TOKEN,
        page_token=os.environ.get("META_SANDBOX_PAGE_TOKEN"),
        api_version=cfg.meta.api_version,
        app_id=os.environ.get("META_APP_ID"),
        app_secret=os.environ.get("META_APP_SECRET"),
        action_types=cfg.meta.action_types.model_dump(),
    )


@needs_sandbox
async def test_account_info_matches_pinned_api_version(app_config: AppConfig):
    info = await _client(app_config).get_account_info()
    assert info.account_id.endswith(ACCOUNT.removeprefix("act_"))  # type: ignore[union-attr]
    assert info.name and info.currency


@needs_sandbox
async def test_daily_insights_shape_for_no_ads(app_config: AppConfig):
    """An empty insights call must return an empty list, not raise: the sync job relies on it."""
    from datetime import UTC, datetime, timedelta

    rows = await _client(app_config).get_daily_insights(
        ["0"], (datetime.now(UTC) - timedelta(days=2)).date()
    )
    assert rows == []


def test_pinned_versions_are_documented(app_config: AppConfig, repo_root):
    """The runbook must mention the pinned Graph API version so a bump is a visible decision."""
    text = (repo_root / "docs" / "runbook.md").read_text()
    assert app_config.meta.api_version in text
