"""The only place a Meta client is constructed. Safety gates live here."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.crypto import get_cipher
from outlier_ai.core.errors import SafetyError
from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.episodes.budget import kill_switch_open
from outlier_ai.meta.base import MetaClient
from outlier_ai.meta.fake import FakeMetaClient
from outlier_ai.models.meta import AdAccount
from outlier_ai.synthetic.briefs import CATEGORIES, WORLD_STATES
from outlier_ai.synthetic.cards import SEED_STRATEGY_CARDS
from outlier_ai.synthetic.latent import LatentOutcomeModel
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import AccountStatus

_FAKE_CLIENTS: dict[str, FakeMetaClient] = {}


def default_latent(seed: int = 1) -> LatentOutcomeModel:
    return LatentOutcomeModel(
        seed=seed,
        categories=list(CATEGORIES),
        strategy_slugs=[c.slug for c in SEED_STRATEGY_CARDS],
        world_tags=[t for t, _ in WORLD_STATES],
    )


async def client_for_account(
    session: AsyncSession,
    account: AdAccount,
    cfg: AppConfig,
    *,
    settings: Settings | None = None,
    latent: LatentOutcomeModel | None = None,
    fake_seed: int = 0,
    require_shippable: bool = False,
) -> MetaClient:
    """Return the fake for fake accounts; the real client only when every gate allows it.

    `require_shippable` adds the kill switch check (reads never need it, ships do).
    """
    s = settings or get_settings()
    if account.is_fake:
        client = FakeMetaClient(
            session,
            account.meta_account_id,
            latent or default_latent(),
            niche_projection=cfg.archive.niche_projection,
            seed=fake_seed,
            api_version=cfg.meta.api_version,
        )
        return client
    if s.dry_run:
        raise SafetyError("DRY_RUN is on: refusing to construct a real Meta client")
    if account.status != AccountStatus.active.value:
        raise SafetyError(f"account {account.meta_account_id} is {account.status}")
    if require_shippable and not await kill_switch_open(session):
        raise SafetyError("kill switch: shipping is disabled")
    if not account.access_token_enc:
        raise SafetyError(f"account {account.meta_account_id} has no access token")
    cipher = get_cipher(s)
    from outlier_ai.meta.client import MetaGraphClient  # imported lazily: needs the `meta` extra

    return MetaGraphClient(
        account_id=account.meta_account_id,
        access_token=cipher.decrypt(account.access_token_enc),
        page_token=cipher.decrypt(account.page_token_enc) if account.page_token_enc else None,
        api_version=cfg.meta.api_version,
        app_id=s.meta_app_id,
        app_secret=s.meta_app_secret,
        action_types=cfg.meta.action_types.model_dump(),
    )
