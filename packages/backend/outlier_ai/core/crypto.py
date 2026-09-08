"""Encryption at rest for Meta tokens and HMAC hashing for commenter identities."""

from __future__ import annotations

import hashlib
import hmac

from cryptography.fernet import Fernet, InvalidToken

from outlier_ai.core.errors import ConfigError
from outlier_ai.core.settings import Settings, get_settings


class TokenCipher:
    def __init__(self, key: str) -> None:
        try:
            self._f = Fernet(key.encode() if isinstance(key, str) else key)
        except (ValueError, TypeError) as e:
            raise ConfigError("FERNET_KEY is not a valid Fernet key") from e

    def encrypt(self, plaintext: str) -> str:
        return self._f.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._f.decrypt(ciphertext.encode()).decode()
        except InvalidToken as e:
            raise ConfigError("token could not be decrypted with the configured FERNET_KEY") from e


def get_cipher(settings: Settings | None = None) -> TokenCipher:
    s = settings or get_settings()
    if not s.fernet_key:
        raise ConfigError("FERNET_KEY is required to store or read Meta tokens")
    return TokenCipher(s.fernet_key)


def hash_identity(external_id: str, settings: Settings | None = None) -> str:
    """Stable, keyed one-way hash for commenter ids. Never store the raw id."""
    s = settings or get_settings()
    return hmac.new(s.pii_hmac_key.encode(), external_id.encode(), hashlib.sha256).hexdigest()
