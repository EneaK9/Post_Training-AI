"""Self-service sign-up: creates the account, signs it in, refuses duplicates, and can be turned off."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.api.app import create_app
from outlier_ai.core.settings import get_settings

pytestmark = pytest.mark.integration


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test")


async def test_signup_creates_and_signs_in(db_session: AsyncSession):
    client = _client()
    opts = await client.get("/api/auth/options")
    assert opts.status_code == 200 and opts.json()["signup_enabled"] is True
    assert set(opts.json()["roles"]) == {"operator", "researcher", "expert"}

    email = f"New.Person-{uuid.uuid4().hex[:6]}@Example.com"
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "longenough1", "display_name": " Neo ", "role": "expert"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["email"] == email.lower() and body["display_name"] == "Neo"
    assert body["role"] == "expert"
    me = await client.get("/api/auth/me")  # the sign-up response set the session cookie
    assert me.status_code == 200 and me.json()["email"] == email.lower()

    dup = await _client().post(
        "/api/auth/signup", json={"email": email.upper(), "password": "longenough1"}
    )
    assert dup.status_code == 409
    weak = await _client().post("/api/auth/signup", json={"email": "a@b.co", "password": "short"})
    assert weak.status_code == 422
    bad_role = await _client().post(
        "/api/auth/signup", json={"email": "c@d.co", "password": "longenough1", "role": "admin"}
    )
    assert bad_role.status_code == 422
    # and the new account can log in normally
    login = await _client().post(
        "/api/auth/login", json={"email": email, "password": "longenough1"}
    )
    assert login.status_code == 200


async def test_signup_can_be_disabled(db_session: AsyncSession, monkeypatch):
    monkeypatch.setenv("SIGNUP_ENABLED", "false")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        client = _client()
        assert (await client.get("/api/auth/options")).json()["signup_enabled"] is False
        r = await client.post(
            "/api/auth/signup", json={"email": "x@y.co", "password": "longenough1"}
        )
        assert r.status_code == 403
    finally:
        monkeypatch.delenv("SIGNUP_ENABLED", raising=False)
        get_settings.cache_clear()  # type: ignore[attr-defined]
